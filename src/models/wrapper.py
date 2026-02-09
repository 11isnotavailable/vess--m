import torch
import torch.nn as nn
from .backbone import DynUNetBackbone
from .encoders import UNet2D_MIP_Encoder
from .fusion_layers import SlabFusionStack 

class VesselFM_Refactored(DynUNetBackbone):
    def __init__(self, use_fusion=True, slab_thickness=4, **kwargs):
        super().__init__(**kwargs)
        self.use_fusion = use_fusion
        self.slab_thickness = slab_thickness
        
        if self.use_fusion:
            self.mip_encoder = UNet2D_MIP_Encoder(in_ch=1)
            c3 = self.filters # [32, 64, 128, 256, 320, 320]
            # 融合位置严格锁定在 L3 和 L4
            self.fuse_L3 = SlabFusionStack(depth=2, c3d=c3[3], c2d=256, embed_dim=256, num_heads=4)
            self.fuse_L4 = SlabFusionStack(depth=2, c3d=c3[4], c2d=320, embed_dim=384, num_heads=8)

    def _slab_mip(self, x3d, slab=4):
        """
        修正后的 MIP 逻辑：返回 [B, K, 1, H, W] 格式
        """
        B, C, D, H, W = x3d.shape
        pad = (slab - (D % slab)) % slab
        if pad > 0: x3d = torch.nn.functional.pad(x3d, (0, 0, 0, 0, 0, pad))
        K = (D + pad) // slab
        
        # [B, C, K, slab, H, W] -> [B, C, K, H, W]
        x = x3d.view(B, C, K, slab, H, W)
        x_mip = torch.amax(x, dim=3)
        
        # 调整为 [B, K, C, H, W] 以配合后续处理
        return x_mip.permute(0, 2, 1, 3, 4).contiguous()
    def forward(self, x):
        D0 = x.shape[2]
        feats_3d = self.encoder_forward(x)
        
        if not self.use_fusion:
            return self.decoder_forward(feats_3d), {}

        # 原始图像的 MIP，用于 2D 支路作为输入
        mip_bk = self._slab_mip(x, self.slab_thickness)
        B, K, C, H, W = mip_bk.shape
        
        # 2D 支路特征提取
        feats_2d = self.mip_encoder(mip_bk.view(B * K, C, H, W))
        f2_L3 = feats_2d[3].view(B, K, -1, H//8, W//8).permute(0, 2, 3, 4, 1).contiguous()
        f2_L4 = feats_2d[4].view(B, K, -1, H//16, W//16).permute(0, 2, 3, 4, 1).contiguous()

        # 融合层
        L3, aux3 = self.fuse_L3(feats_3d[3], f2_L3, D0=D0, slab=self.slab_thickness, mip=mip_bk, return_aux=True)
        L4, aux4 = self.fuse_L4(feats_3d[4], f2_L4, D0=D0, slab=self.slab_thickness, mip=mip_bk, return_aux=True)

        feats_3d[3], feats_3d[4] = L3, L4
        
        # 获得最终融合后的 3D 预测结果
        pred_3d = self.decoder_forward(feats_3d)
        
        # ✅ 核心新增：计算 3D 预测结果的投影图 [B, K, 1, H, W]
        # 这将用于在 pl_module 中计算与金标 MIP 的一致性损失
        pred_3d_mip = self._slab_mip(pred_3d, self.slab_thickness)

        return pred_3d, {
            "aux3": aux3, 
            "aux4": aux4, 
            "pred_3d_mip": pred_3d_mip  # 传出投影结果
        }
    # def forward(self, x):
    #     D0 = x.shape[2]
    #     feats_3d = self.encoder_forward(x)
        
    #     if not self.use_fusion:
    #         return self.decoder_forward(feats_3d), {}

    #     # mip_bk 现在的形状是 [B, K, 1, H, W]
    #     mip_bk = self._slab_mip(x, self.slab_thickness)
    #     B, K, C, H, W = mip_bk.shape
        
    #     # 2D 支路特征提取：reshape 为 [B*K, 1, H, W]
    #     feats_2d = self.mip_encoder(mip_bk.view(B * K, C, H, W))
        
    #     # 将特征还原为 [B, K, C_feat, H_feat, W_feat] 并转换顺序为 [B, C_feat, H_feat, W_feat, K] 适配 Attention
    #     f2_L3 = feats_2d[3].view(B, K, -1, H//8, W//8).permute(0, 2, 3, 4, 1).contiguous()
    #     f2_L4 = feats_2d[4].view(B, K, -1, H//16, W//16).permute(0, 2, 3, 4, 1).contiguous()

    #     # 传入 D0 和 slab 确保注意力范围计算正确
    #     L3, aux3 = self.fuse_L3(feats_3d[3], f2_L3, D0=D0, slab=self.slab_thickness, mip=mip_bk, return_aux=True)
    #     L4, aux4 = self.fuse_L4(feats_3d[4], f2_L4, D0=D0, slab=self.slab_thickness, mip=mip_bk, return_aux=True)

    #     feats_3d[3], feats_3d[4] = L3, L4
    #     return self.decoder_forward(feats_3d), {"aux3": aux3, "aux4": aux4}