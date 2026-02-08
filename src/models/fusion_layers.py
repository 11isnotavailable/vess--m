import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1) 2D 拓扑先验头：通过 2D 分支预测血管概率 P(x,y)
# ============================================================
class TopoPriorHead2D(nn.Module):
    def __init__(self, c2d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c2d, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2) 伪标签生成器：从 MIP 图像中提取血管中心性作为监督信号
# ============================================================
@torch.no_grad()
def pseudo_vesselness_from_mip(mip_bk1hw: torch.Tensor) -> torch.Tensor:
    """
    遵循原版逻辑：从 2D MIP 自动生成伪标签
    """
    B, K, C, H, W = mip_bk1hw.shape
    x = mip_bk1hw
    
    # 局部归一化
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    # 背景消除与特征增强
    bg3 = F.avg_pool2d(x.view(B * K, 1, H, W), 3, stride=1, padding=1).view(B, K, 1, H, W)
    bg7 = F.avg_pool2d(x.view(B * K, 1, H, W), 7, stride=1, padding=3).view(B, K, 1, H, W)
    hi = x - 0.5 * (bg3 + bg7)

    dx = hi[..., :, 1:] - hi[..., :, :-1]
    dy = hi[..., 1:, :] - hi[..., :-1, :]
    g = torch.sqrt(F.pad(dx, (0,1,0,0))**2 + F.pad(dy, (0,0,0,1))**2 + 1e-6)

    return torch.sigmoid(4.0 * (2.0 * hi + 1.0 * g)).clamp(0.0, 1.0)


# ============================================================
# 3) 核心分片交叉注意力层 (Slab Cross-Attention)
# ============================================================
class SlabCrossAttention3D2D(nn.Module):
    def __init__(
        self,
        c3d: int,
        c2d: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        beta_adj: float = 0.5,           # 统一为原版 0.5
        lambda_topo_init: float = 0.3,    # 统一为原版 0.3
        use_topo_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.beta_adj = beta_adj
        self.use_topo_bias = use_topo_bias

        # 线性投射层
        self.q = nn.Linear(c3d, self.embed_dim, bias=False)
        self.k = nn.Linear(c2d, self.embed_dim, bias=False)
        self.v = nn.Linear(c2d, self.embed_dim, bias=False)
        self.proj = nn.Linear(self.embed_dim, c3d, bias=False)
        
        self.ln_q = nn.LayerNorm(self.embed_dim)
        self.ln_kv = nn.LayerNorm(self.embed_dim)

        if self.use_topo_bias:
            self.topo_head = TopoPriorHead2D(c2d)
            self.lambda_topo = nn.Parameter(torch.tensor(lambda_topo_init))
            # 邻域增强算子
            kernel = torch.ones(1, 1, 3, 3)
            kernel[0, 0, 1, 1] = 0.0
            self.register_buffer("_nei_kernel", kernel, persistent=False)

    @staticmethod
    def _z_to_slab_index(D_s, D0, slab, K, device):
        """
        完美复现原版深度到分片的映射逻辑
        """
        z0 = torch.round(torch.linspace(0, D0 - 1, steps=D_s, device=device)).long()
        k = torch.clamp(torch.div(z0, slab, rounding_mode="floor"), 0, K - 1)
        return k

    def forward(self, F3, F2, D0, slab, mip=None, return_aux=False):
        B, C3, D, H, W = F3.shape
        B, C2, H2, W2, K = F2.shape
        HW, device = H * W, F3.device
        
        # 计算 3D 层深度到 2D Slab 的映射
        k_index = self._z_to_slab_index(D, D0, slab, K, device)

        aux = {}
        # 计算拓扑偏置
        if self.use_topo_bias:
            f2_bkchw = F2.permute(0, 4, 1, 2, 3).reshape(B * K, C2, H, W)
            p_pred = self.topo_head(f2_bkchw).view(B, K, 1, H, W)
            aux["P_pred"] = p_pred
            if mip is not None:
                aux["P_pseudo"] = pseudo_vesselness_from_mip(mip)
            
            # 计算局部响应增强
            ps_pred = F.conv2d(p_pred.view(B*K, 1, H, W), self._nei_kernel, padding=1).view(B, K, 1, H, W)
            topo_map = (p_pred + 0.5 * ps_pred).view(B, K, HW)

        # 准备注意力 Token
        # 2D Key/Value (B, K, HW, H, D_head)
        x2 = F2.permute(0, 4, 2, 3, 1).reshape(B, K, HW, C2)
        Kall = self.ln_kv(self.k(x2)).view(B, K, HW, self.num_heads, self.head_dim)
        Vall = self.ln_kv(self.v(x2)).view(B, K, HW, self.num_heads, self.head_dim)
        
        # 3D Query (B, D, HW, H, D_head)
        x3 = F3.permute(0, 2, 3, 4, 1).reshape(B, D, HW, C3)
        Qall = self.ln_q(self.q(x3)).view(B, D, HW, self.num_heads, self.head_dim)

        Oh_all = torch.zeros_like(Qall)
        
        # 分层分片注意力 (核心向量化优化)
        for k in range(K):
            z_ids = torch.nonzero(k_index == k).squeeze(-1)
            if z_ids.numel() == 0: continue
            
            # 仅允许关注相邻分片 {k-1, k, k+1}
            cand = [kk for kk in (k-1, k, k+1) if 0 <= kk < K]
            # 完美复现 beta_adj 惩罚：非当前分片减去 beta_adj
            slab_bias = torch.cat([torch.full((HW,), 0.0 if kk==k else -self.beta_adj, device=device) for kk in cand])
            
            Qg = Qall[:, z_ids].reshape(B, -1, self.num_heads, self.head_dim)
            Kh_cat = torch.cat([Kall[:, kk] for kk in cand], dim=1)
            Vh_cat = torch.cat([Vall[:, kk] for kk in cand], dim=1)

            # 计算注意力评分
            logits = torch.einsum("bnhd,bmhd->bnhm", Qg, Kh_cat) / (self.head_dim ** 0.5)
            logits = logits + slab_bias.view(1, 1, 1, -1)

            if self.use_topo_bias:
                # 将 2D 拓扑概率作为对角线增益注入 Logits
                for j, kk in enumerate(cand):
                    diag_boost = self.lambda_topo * topo_map[:, kk]
                    l_view = logits.view(B, len(z_ids), HW, self.num_heads, len(cand), HW)
                    l_view[:, :, torch.arange(HW), :, j, torch.arange(HW)] += diag_boost[:, None, None, :]

            attn = torch.softmax(logits, dim=-1)
            Og = torch.einsum("bnhm,bmhd->bnhd", attn, Vh_cat)
            Oh_all[:, z_ids] = Og.view(B, len(z_ids), HW, self.num_heads, self.head_dim)

        # 投影回 3D 特征空间并应用残差连接
        Oall = self.proj(Oh_all.reshape(B, D, HW, self.embed_dim))
        Oall = Oall.view(B, D, H, W, C3).permute(0, 4, 1, 2, 3)
        
        out = F3 + Oall
        return (out, aux) if return_aux else out


# ============================================================
# 4) 融合层堆叠容器
# ============================================================
class SlabFusionStack(nn.Module):
    def __init__(self, depth: int, **kwargs):
        super().__init__()
        # 深度由配置文件的 fuse_depth_L3/L4 决定，默认为 2
        self.blocks = nn.ModuleList([SlabCrossAttention3D2D(**kwargs) for _ in range(depth)])

    def forward(self, F3, F2, D0, slab, mip=None, return_aux=False):
        x = F3
        last_aux = {}
        for blk in self.blocks:
            if return_aux:
                x, last_aux = blk(x, F2, D0, slab, mip, return_aux=True)
            else:
                x = blk(x, F2, D0, slab, mip, return_aux=False)
        return (x, last_aux) if return_aux else x