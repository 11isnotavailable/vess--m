# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from monai.networks.nets import DynUNet

# class DynUNetBackbone(DynUNet):
#     """
#     提供分层特征提取接口的 3D 骨架 。
#     """
#     def __init__(self, deep_supervision=False, deep_supr_num=0, **kwargs):
#         super().__init__(deep_supervision=deep_supervision, deep_supr_num=deep_supr_num, **kwargs)
#         if not deep_supervision and hasattr(self, "heads"):
#             self.heads = nn.ModuleList()

#     def encoder_forward(self, x: torch.Tensor):
#         feats = []
#         x = self.input_block(x) # L0
#         feats.append(x)
#         for down_block in self.downsamples: # L1-L4
#             x = down_block(x)
#             feats.append(x)
#         feats.append(self.bottleneck(feats[-1])) # L5 (Bottleneck)
#         return feats

#     def decoder_forward(self, feats):
#         x = feats[5] # Bottleneck
#         skip_feats = [feats[4], feats[3], feats[2], feats[1], feats[0]]
#         for i, upsample in enumerate(self.upsamples):
#             skip = skip_feats[i]
#             if skip.shape[2:] != tuple(s * 2 for s in x.shape[2:]):
#                 skip = F.interpolate(skip, size=[d * 2 for d in x.shape[2:]], mode='trilinear', align_corners=False)
#             x = upsample(x, skip)
#         return self.output_block(x)

#     def forward(self, x: torch.Tensor):
#         feats = self.encoder_forward(x)
#         return self.decoder_forward(feats)
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet

class DynUNetBackbone(DynUNet):
    """
    提供分层特征提取接口的 3D 骨架。
    保持与 MONAI DynUNet 完全兼容，支持加载预训练权重。
    """
    def __init__(self, deep_supervision=False, deep_supr_num=0, **kwargs):
        super().__init__(deep_supervision=deep_supervision, deep_supr_num=deep_supr_num, **kwargs)
        # 移除 head 以免干扰
        if not deep_supervision and hasattr(self, "heads"):
            self.heads = nn.ModuleList()

    def encoder_forward(self, x: torch.Tensor):
        """返回特征列表: [L0, L1, L2, L3, L4, L5(Bottleneck)]"""
        feats = []
        x = self.input_block(x) # L0
        feats.append(x)
        for down_block in self.downsamples: # L1-L4
            x = down_block(x)
            feats.append(x)
        feats.append(self.bottleneck(feats[-1])) # L5
        return feats

    def decoder_forward(self, feats):
        """执行上采样与 Skip Connection，带尺寸自动对齐"""
        x = feats[5] # Bottleneck
        skip_feats = [feats[4], feats[3], feats[2], feats[1], feats[0]]
        
        for i, upsample in enumerate(self.upsamples):
            skip = skip_feats[i]
            # ✅ 关键防御逻辑：如果尺寸不对齐（如 8 vs 7），自动插值对齐
            if skip.shape[2:] != tuple(s * 2 for s in x.shape[2:]):
                skip = F.interpolate(
                    skip, size=[d * 2 for d in x.shape[2:]], 
                    mode='trilinear', align_corners=False
                )
            x = upsample(x, skip)
        
        return self.output_block(x)