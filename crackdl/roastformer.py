"""RoastFormer (OURS): Transient-Aware Broadcast-Residual Conformer for
acoustic crack detection.

Design rationale:
  * Crack events are impulsive broadband transients (50-150 ms) riding on
    quasi-stationary roaster noise. A first-order temporal difference of the
    log-mel spectrogram (delta stream) explicitly enhances onsets while
    suppressing stationary drum/fan noise -> 2-channel input (mel, delta).
  * Broadcasted-residual convolution stages (BC-ResNet style) give a
    parameter-efficient frequency-aware front-end.
  * Frequency attention pooling collapses the frequency axis with learned,
    content-dependent weights (crack energy concentrates at 2-14 kHz but
    varies between first/second crack).
  * Two Conformer blocks model the temporal micro-structure that separates
    the sparser "pop" of first crack from the denser "snap" of second crack.
  * Multi-head attentive temporal pooling focuses on the transient frames
    instead of averaging them away.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_cnn import BCBlock
from models_transformer import ConformerBlock, PositionalEncoding


class FreqAttnPool(nn.Module):
    """Content-dependent attention over the frequency axis, H heads."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.att = nn.Conv2d(channels, heads, 1)
        self.heads = heads

    def forward(self, x):  # (B,C,F,T)
        a = torch.softmax(self.att(x), dim=2)            # (B,H,F,T)
        o = torch.einsum("bhft,bcft->bhct", a, x)        # (B,H,C,T)
        return o.flatten(1, 2)                           # (B,H*C,T)


class AttentiveStatsPool(nn.Module):
    """Multi-head attentive statistics pooling over time: each head owns a
    D/H channel subspace with its own attention distribution; the
    attention-weighted mean and std of every subspace are concatenated,
    giving a (B, 2D) utterance embedding."""

    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.w = nn.Linear(d, heads)

    def forward(self, x):  # (B,T,D)
        B, T, Dm = x.shape
        H = self.heads
        a = torch.softmax(self.w(x), dim=1)                    # (B,T,H)
        xh = x.view(B, T, H, Dm // H)                          # (B,T,H,d)
        mean = torch.einsum("bth,bthd->bhd", a, xh)            # (B,H,d)
        var = torch.einsum("bth,bthd->bhd", a,
                           (xh - mean.unsqueeze(1)) ** 2)
        std = torch.sqrt(var.clamp_min(1e-6))
        return torch.cat([mean.flatten(1), std.flatten(1)], dim=1)  # (B,2D)


class SEBlock(nn.Module):
    """Squeeze-excitation channel attention over (B,C,F,T)."""

    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(4, channels // r)), nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // r), channels), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean(dim=(2, 3)))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class RoastFormer(nn.Module):
    def __init__(self, num_classes=3, width=(40, 56, 80), d=128,
                 conformer_blocks=2, freq_heads=4, pool_heads=4,
                 use_delta=True, use_freq_attn=True, use_attn_pool=True,
                 use_delta2=False, use_se=False):
        super().__init__()
        self.use_delta = use_delta
        self.use_delta2 = use_delta2
        self.use_freq_attn = use_freq_attn
        self.use_attn_pool = use_attn_pool
        c0, c1, c2 = width
        in_ch = 1 + (1 if use_delta else 0) + (1 if use_delta2 else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c0), nn.ReLU(inplace=True))          # (c0,64,T)
        s1 = [BCBlock(c0, c1, stride_f=2, dilation=1),          # (c1,32,T)
              BCBlock(c1, c1, dilation=1)]
        s2 = [BCBlock(c1, c2, stride_f=2, dilation=2),          # (c2,16,T)
              BCBlock(c2, c2, dilation=2)]
        if use_se:
            s1.append(SEBlock(c1))
            s2.append(SEBlock(c2))
        self.stage1 = nn.Sequential(*s1)
        self.stage2 = nn.Sequential(*s2)
        if use_freq_attn:
            self.fpool = FreqAttnPool(c2, heads=freq_heads)
            self.proj = nn.Linear(c2 * freq_heads, d)
        else:
            self.fpool = None
            self.proj = nn.Linear(c2, d)
        self.pos = PositionalEncoding(d)
        self.temporal = nn.ModuleList(
            [ConformerBlock(d, heads=4, conv_kernel=9) for _ in range(conformer_blocks)])
        self.tpool = AttentiveStatsPool(d, heads=pool_heads) if use_attn_pool else None
        hd = 2 * d if use_attn_pool else d
        self.head = nn.Sequential(nn.LayerNorm(hd), nn.Dropout(0.2),
                                  nn.Linear(hd, num_classes))

    @staticmethod
    def _diff(x):
        d = x[..., 1:] - x[..., :-1]
        return F.pad(d, (1, 0))

    def add_delta(self, mel):  # (B,1,F,T) -> (B,{1..3},F,T)
        chans = [mel]
        if self.use_delta:
            chans.append(self._diff(mel))
        if self.use_delta2:
            chans.append(self._diff(self._diff(mel)))
        return torch.cat(chans, dim=1)

    def forward(self, mel):
        x = self.add_delta(mel) if (self.use_delta or self.use_delta2) else mel
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.fpool(x) if self.fpool is not None else x.mean(dim=2)  # (B,C*,T)
        x = self.pos(self.proj(x.transpose(1, 2)))       # (B,T,D)
        for blk in self.temporal:
            x = blk(x)
        pooled = self.tpool(x) if self.tpool is not None else x.mean(dim=1)
        return self.head(pooled)
