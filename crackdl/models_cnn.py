"""CNN-family baselines.

1. PANNs CNN14 (Kong et al., IEEE/ACM TASLP 2020) - from scratch.
2. BC-ResNet (Kim et al., Interspeech 2021) - broadcasted residual learning
   with SubSpectral Normalization (Chang et al., ICASSP 2021).
3. EfficientAT-style MobileNetV3-Large (Schmid et al., ICASSP 2023) -
   ImageNet-initialized via timm, single-channel mel input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- CNN14
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x, pool=(2, 2)):
        x = F.relu_(self.b1(self.c1(x)))
        x = F.relu_(self.b2(self.c2(x)))
        if pool is not None:
            x = F.avg_pool2d(x, pool)
        return x


class CNN14(nn.Module):
    """Faithful CNN14 topology (6 conv blocks 64->2048, mean+max temporal pooling)."""

    def __init__(self, num_classes=3, dropout=0.2):
        super().__init__()
        chs = [64, 128, 256, 512, 1024, 2048]
        self.blocks = nn.ModuleList()
        cin = 1
        for c in chs:
            self.blocks.append(ConvBlock(cin, c))
            cin = c
        self.dropout = dropout
        self.fc1 = nn.Linear(2048, 2048)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, mel):  # (B,1,F,T)
        x = mel
        for i, blk in enumerate(self.blocks):
            pool = (2, 2) if i < len(self.blocks) - 1 else None
            x = blk(x, pool)
            x = F.dropout(x, self.dropout, self.training)
        x = x.mean(dim=2)                      # (B,C,T')
        x = x.max(dim=2).values + x.mean(dim=2)  # (B,C)
        x = F.dropout(x, self.dropout * 2.5, self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, self.dropout * 2.5, self.training)
        return self.head(x)


# ---------------------------------------------------------------- BC-ResNet
class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands=4):
        super().__init__()
        self.S = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        B, C, Fq, T = x.shape
        x = x.view(B, C * self.S, Fq // self.S, T)
        x = self.bn(x)
        return x.view(B, C, Fq, T)


class BCBlock(nn.Module):
    """Broadcasted residual block: freq-depthwise conv (f2) + temporal
    depthwise dilated conv (f1) on frequency-averaged features, broadcast back."""

    def __init__(self, cin, cout, stride_f=1, dilation=1, sub_bands=4):
        super().__init__()
        self.transition = None
        if cin != cout or stride_f != 1:
            self.transition = nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.f2 = nn.Sequential(
            nn.Conv2d(cout, cout, (3, 1), stride=(stride_f, 1), padding=(1, 0),
                      groups=cout, bias=False),
            SubSpectralNorm(cout, sub_bands))
        self.f1 = nn.Sequential(
            nn.Conv2d(cout, cout, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      groups=cout, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False), nn.Dropout2d(0.1))

    def forward(self, x):
        is_transition = self.transition is not None
        if is_transition:
            x = self.transition(x)
        a = self.f2(x)                       # (B,C,F,T)
        b = a.mean(dim=2, keepdim=True)      # freq-avg -> (B,C,1,T)
        b = self.f1(b)
        # per Kim et al. 2021: identity shortcut only in non-transition blocks
        return F.relu(a + b + x) if not is_transition else F.relu(a + b)


class BCResNet(nn.Module):
    """BC-ResNet-style network adapted for 128-mel input (wider stem, extra
    frequency stride in stage 4). Not an exact BC-ResNet-tau configuration;
    report as 'adapted BC-ResNet' with measured parameter count."""

    def __init__(self, num_classes=3, base=16):
        super().__init__()
        c = [base, int(base * 1.5), base * 2, int(base * 2.5), base * 4]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(BCBlock(c[0], c[1]), BCBlock(c[1], c[1]))
        self.stage2 = nn.Sequential(BCBlock(c[1], c[2], stride_f=2, dilation=2),
                                    BCBlock(c[2], c[2], dilation=2))
        self.stage3 = nn.Sequential(BCBlock(c[2], c[3], stride_f=2, dilation=4),
                                    BCBlock(c[3], c[3], dilation=4),
                                    BCBlock(c[3], c[3], dilation=4),
                                    BCBlock(c[3], c[3], dilation=4))
        self.stage4 = nn.Sequential(BCBlock(c[3], c[4], stride_f=2, dilation=8),
                                    BCBlock(c[4], c[4], dilation=8),
                                    BCBlock(c[4], c[4], dilation=8),
                                    BCBlock(c[4], c[4], dilation=8))
        self.post = nn.Sequential(
            nn.Conv2d(c[4], c[4], (8, 1), groups=c[4], bias=False),  # collapse freq (128/16=8)
            nn.Conv2d(c[4], c[4] * 2, 1, bias=False), nn.BatchNorm2d(c[4] * 2),
            nn.ReLU(inplace=True))
        self.head = nn.Linear(c[4] * 2, num_classes)

    def forward(self, mel):
        x = self.stem(mel)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.post(x)          # (B,C,1,T)
        x = x.mean(dim=(2, 3))
        return self.head(x)


# ---------------------------------------------------------------- MobileNetV3 (EfficientAT-style)
def build_mobilenetv3(num_classes=3, pretrained=True):
    import timm
    return timm.create_model("mobilenetv3_large_100", pretrained=pretrained,
                             in_chans=1, num_classes=num_classes)
