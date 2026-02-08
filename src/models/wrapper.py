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
        # 完美复现原版 slab-MIP 逻辑
        B, C, D, H, W = x3d.shape
        pad = (slab - (D % slab)) % slab
        if pad > 0: x3d = torch.nn.functional.pad(x3d, (0, 0, 0, 0, 0, pad))
        K = (D + pad) // slab
        x = x3d.view(B, C, K, slab, H, W)
        return torch.amax(x, dim=3).permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x):
        # 记录原始深度 D0 用于索引映射
        D0 = x.shape[2]
        feats_3d = self.encoder_forward(x)
        
        if not self.use_fusion:
            return self.decoder_forward(feats_3d), {}

        mip_bk = self._slab_mip(x, self.slab_thickness)
        B, C, H, W, K = mip_bk.shape
        # 2D 支路特征提取
        feats_2d = self.mip_encoder(mip_bk.permute(0, 4, 1, 2, 3).reshape(B*K, C, H, W))
        
        # 将 K 维还原并进行分片融合
        f2_L3 = feats_2d[3].view(B, K, -1, H//8, W//8).permute(0, 2, 3, 4, 1).contiguous()
        f2_L4 = feats_2d[4].view(B, K, -1, H//16, W//16).permute(0, 2, 3, 4, 1).contiguous()

        # 传入 D0 和 slab 确保注意力范围计算正确
        L3, aux3 = self.fuse_L3(feats_3d[3], f2_L3, D0=D0, slab=self.slab_thickness, mip=mip_bk, return_aux=True)
        L4, aux4 = self.fuse_L4(feats_3d[4], f2_L4, D0=D0, slab=self.slab_thickness, mip=mip_bk, return_aux=True)

        feats_3d[3], feats_3d[4] = L3, L4
        return self.decoder_forward(feats_3d), {"aux3": aux3, "aux4": aux4}