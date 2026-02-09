import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoPriorHead2D(nn.Module):
    def __init__(self, c2d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c2d, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def pseudo_vesselness_from_mip(mip_bk1hw: torch.Tensor) -> torch.Tensor:
    """
    输入形状: [B, K, 1, H, W]
    """
    B, K, C, H, W = mip_bk1hw.shape
    x = mip_bk1hw
    
    # 局部归一化
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    x = (x - x_min) / (x_max - x_min).clamp_min(1e-6)

    # 简化的血管增强
    x_flat = x.view(B * K, 1, H, W)
    bg3 = F.avg_pool2d(x_flat, 3, stride=1, padding=1)
    bg7 = F.avg_pool2d(x_flat, 7, stride=1, padding=3)
    hi = x_flat - 0.5 * (bg3 + bg7)

    return torch.sigmoid(4.0 * (2.0 * hi + 0.5)).view(B, K, 1, H, W)

class SlabCrossAttention3D2D(nn.Module):
    def __init__(self, c3d: int, c2d: int, embed_dim: int = 256, num_heads: int = 4, 
                 beta_adj: float = 0.5, lambda_topo_init: float = 0.3, use_topo_bias: bool = True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.beta_adj = beta_adj
        self.use_topo_bias = use_topo_bias

        self.q = nn.Linear(c3d, self.embed_dim, bias=False)
        self.k = nn.Linear(c2d, self.embed_dim, bias=False)
        self.v = nn.Linear(c2d, self.embed_dim, bias=False)
        self.proj = nn.Linear(self.embed_dim, c3d, bias=False)
        
        self.ln_q = nn.LayerNorm(self.embed_dim)
        self.ln_kv = nn.LayerNorm(self.embed_dim)

        if self.use_topo_bias:
            self.topo_head = TopoPriorHead2D(c2d)
            self.lambda_topo = nn.Parameter(torch.tensor(lambda_topo_init))
            kernel = torch.ones(1, 1, 3, 3)
            kernel[0, 0, 1, 1] = 0.0
            self.register_buffer("_nei_kernel", kernel, persistent=False)

    @staticmethod
    def _z_to_slab_index(D_s, D0, slab, K, device):
        z0 = torch.round(torch.linspace(0, D0 - 1, steps=D_s, device=device)).long()
        k = torch.clamp(torch.div(z0, slab, rounding_mode="floor"), 0, K - 1)
        return k

    def forward(self, F3, F2, D0, slab, mip=None, return_aux=False):
        B, C3, D, H, W = F3.shape
        B, C2, H2, W2, K = F2.shape
        HW, device = H * W, F3.device
        
        # 1. 计算 3D 层深度到 2D Slab 的映射
        k_index = self._z_to_slab_index(D, D0, slab, K, device)
        aux = {}

        # 2. 计算拓扑偏置
        if self.use_topo_bias:
            # 防御性尺寸对齐：确保来自 SlidingWindowInferer 的 MIP 与当前特征图 H, W 一致
            if mip is not None and mip.shape[-2:] != (H, W):
                mip_resized = F.interpolate(
                    mip.view(B * K, 1, mip.shape[-2], mip.shape[-1]), 
                    size=(H, W), mode='bilinear', align_corners=False
                ).view(B, K, 1, H, W)
            else:
                mip_resized = mip

            f2_bkchw = F2.permute(0, 4, 1, 2, 3).reshape(B * K, C2, H, W)
            p_pred = self.topo_head(f2_bkchw).view(B, K, 1, H, W)
            aux["P_pred"] = p_pred
            
            if mip_resized is not None:
                aux["P_pseudo"] = pseudo_vesselness_from_mip(mip_resized)
            
            # 计算局部响应增强
            ps_pred = F.conv2d(p_pred.view(B*K, 1, H, W), self._nei_kernel, padding=1).view(B, K, 1, H, W)
            topo_map = (p_pred + 0.5 * ps_pred).view(B, K, HW)

        # 3. 准备注意力 Token
        x2 = F2.permute(0, 4, 2, 3, 1).reshape(B, K, HW, C2)
        Kall = self.ln_kv(self.k(x2)).view(B, K, HW, self.num_heads, self.head_dim)
        Vall = self.ln_kv(self.v(x2)).view(B, K, HW, self.num_heads, self.head_dim)
        
        x3 = F3.permute(0, 2, 3, 4, 1).reshape(B, D, HW, C3)
        Qall = self.ln_q(self.q(x3)).view(B, D, HW, self.num_heads, self.head_dim)

        # ✅ 修复：Oh_all 类型必须与 Qall 严格一致，防止 AMP 下的赋值错误
        Oh_all = torch.zeros_like(Qall, dtype=Qall.dtype, device=device)
        
        # 4. 分层分片注意力核心循环
        for k in range(K):
            z_ids = torch.nonzero(k_index == k).squeeze(-1)
            if z_ids.numel() == 0: continue
            
            cand = [kk for kk in (k-1, k, k+1) if 0 <= kk < K]
            L_cand = len(cand)
            slab_bias = torch.cat([torch.full((HW,), 0.0 if kk==k else -self.beta_adj, device=device) for kk in cand])
            
            Qg = Qall[:, z_ids]
            Kh_cat = torch.cat([Kall[:, kk] for kk in cand], dim=1)
            Vh_cat = torch.cat([Vall[:, kk] for kk in cand], dim=1)

            # 计算基础注意力 [B, n, d, h, m]
            logits = torch.einsum("bd h n c, b m n c -> b n d h m", Qg, Kh_cat) / (self.head_dim ** 0.5)
            logits = logits + slab_bias.view(1, 1, 1, 1, -1)

            if self.use_topo_bias:
                D_sub = len(z_ids)
                # ✅ 修复：避开高级索引重排风险，使用简洁的对角线注入
                for j, _ in enumerate(cand):
                    # diag_boost 形状: [B, HW]
                    diag_boost = (self.lambda_topo * topo_map[:, cand[j]]) 
                    
                    # 确定当前分片在 KV 序列中的偏移位置
                    offset = j * HW
                    hw_idx = torch.arange(HW, device=device)
                    m_idx = offset + hw_idx
                    
                    # 注入拓扑偏置，使用 view(B, 1, 1, HW) 确保在 Heads 和 Depth 维度正确广播
                    logits[..., hw_idx, m_idx] += diag_boost.view(B, 1, 1, HW)

            attn = torch.softmax(logits, dim=-1)
            
            # ✅ 修复：将 Og 类型转回容器 dtype，防止 AMP 导致的 Half/Float 冲突
            Og = torch.einsum("b n d h m, b m n c -> b n d h c", attn, Vh_cat).to(Oh_all.dtype)
            
            # 还原形状存入容器: [B, D_sub, HW, num_heads, head_dim]
            Oh_all[:, z_ids] = Og.permute(0, 2, 3, 1, 4).contiguous()

        # 5. 投影回 3D 空间并应用残差连接
        Oall = self.proj(Oh_all.reshape(B, D, HW, self.embed_dim))
        Oall = Oall.view(B, D, H, W, C3).permute(0, 4, 1, 2, 3)
        
        out = F3 + Oall
        return (out, aux) if return_aux else out

class SlabFusionStack(nn.Module):
    def __init__(self, depth: int, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([SlabCrossAttention3D2D(**kwargs) for _ in range(depth)])
    def forward(self, F3, F2, D0, slab, mip=None, return_aux=False):
        x = F3
        last_aux = {}
        for blk in self.blocks:
            x, last_aux = blk(x, F2, D0, slab, mip, return_aux=True) if return_aux else (blk(x, F2, D0, slab, mip), {})
        return (x, last_aux) if return_aux else x