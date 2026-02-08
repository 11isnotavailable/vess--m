import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet2D_MIP_Encoder(nn.Module):
    """
    完全对齐 DynUNet 通道数的 2D MIP 编码器 。
    """
    def __init__(self, in_ch=1):
        super().__init__()
        f = [32, 64, 128, 256, 320] # 与 DynUNet filters 对齐 
        self.L0 = DoubleConv(in_ch, f[0])
        self.L1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f[0], f[1]))
        self.L2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f[1], f[2]))
        self.L3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f[2], f[3]))
        self.L4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f[3], f[4]))

    def forward(self, x):
        l0 = self.L0(x)
        l1 = self.L1(l0)
        l2 = self.L2(l1)
        l3 = self.L3(l2)
        l4 = self.L4(l3)
        return l0, l1, l2, l3, l4