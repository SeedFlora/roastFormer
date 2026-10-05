"""Transformer-family baselines.

4. Conformer (Gulati et al., Interspeech 2020) - convolution-augmented
   transformer; this implementation uses absolute sinusoidal positional
   encoding (not the paper's relative-PE MHSA) - state this in the paper.
5. AST-style (Gong et al., Interspeech 2021) - spectrogram transformer,
   ImageNet-pretrained DeiT-Tiny via timm, NON-overlapping 16x16 patches,
   no AudioSet pretraining (controlled same-init comparison).
6. PaSST-style (Koutini et al., Interspeech 2022) - same backbone +
   structured patchout.
7. HTS-AT-style compact Swin Transformer (Chen et al., ICASSP 2022),
   trained from scratch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- Conformer
class ConformerBlock(nn.Module):
    def __init__(self, d, heads=4, conv_kernel=15, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * ff_mult), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(d * ff_mult, d), nn.Dropout(dropout))
        self.norm_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.att_drop = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.LayerNorm(d))
        self.conv_pw1 = nn.Conv1d(d, d * 2, 1)
        self.conv_dw = nn.Conv1d(d, d, conv_kernel, padding=conv_kernel // 2, groups=d)
        self.conv_bn = nn.BatchNorm1d(d)
        self.conv_pw2 = nn.Conv1d(d, d, 1)
        self.conv_drop = nn.Dropout(dropout)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * ff_mult), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(d * ff_mult, d), nn.Dropout(dropout))
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x):  # (B,T,D)
        x = x + 0.5 * self.ff1(x)
        h = self.norm_att(x)
        a, _ = self.att(h, h, h, need_weights=False)
        x = x + self.att_drop(a)
        h = self.conv[0](x).transpose(1, 2)          # (B,D,T)
        h = F.glu(self.conv_pw1(h), dim=1)
        h = F.silu(self.conv_bn(self.conv_dw(h)))
        h = self.conv_drop(self.conv_pw2(h)).transpose(1, 2)
        x = x + h
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class ConformerClassifier(nn.Module):
    def __init__(self, num_classes=3, d=144, blocks=4, heads=4, n_mels=128):
        super().__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=(2, 1), padding=1), nn.ReLU(inplace=True))
        feat = 64 * (n_mels // 4)
        self.proj = nn.Linear(feat, d)
        self.pos = PositionalEncoding(d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads) for _ in range(blocks)])
        self.head = nn.Linear(d, num_classes)

    def forward(self, mel):  # (B,1,F,T)
        x = self.sub(mel)                            # (B,64,F/4,T/2)
        B, C, Fq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)
        x = self.pos(self.proj(x))
        for blk in self.blocks:
            x = blk(x)
        return self.head(x.mean(dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------- AST / PaSST
AST_IMG = (128, 112)  # mel 128 x time padded to 112 -> 8x7 grid of 16x16 patches


def _pad_time(mel, target=AST_IMG[1]):
    T = mel.shape[-1]
    if T < target:
        mel = F.pad(mel, (0, target - T))
    return mel[..., :target]


class ASTClassifier(nn.Module):
    """AST: ViT (DeiT-Tiny, ImageNet pretrained) on mel patches."""

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "deit_tiny_patch16_224", pretrained=pretrained, in_chans=1,
            num_classes=num_classes, img_size=AST_IMG)

    def forward(self, mel):
        return self.vit(_pad_time(mel))


class PaSSTClassifier(nn.Module):
    """PaSST: same ViT backbone, structured patchout on the 8x7 patch grid during training."""

    def __init__(self, num_classes=3, pretrained=True, drop_f=2, drop_t=2):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "deit_tiny_patch16_224", pretrained=pretrained, in_chans=1,
            num_classes=num_classes, img_size=AST_IMG)
        self.grid = (AST_IMG[0] // 16, AST_IMG[1] // 16)  # (8,7) = (freq,time)
        self.drop_f, self.drop_t = drop_f, drop_t

    def forward(self, mel):
        v = self.vit
        x = _pad_time(mel)
        x = v.patch_embed(x)              # (B,N,D) N=56 (row-major: freq rows x time cols)
        x = v._pos_embed(x)               # adds cls token + pos embed
        if self.training and (self.drop_f or self.drop_t):
            B, N, D = x.shape
            Fg, Tg = self.grid
            keep_f = torch.randperm(Fg)[: Fg - self.drop_f].sort().values
            keep_t = torch.randperm(Tg)[: Tg - self.drop_t].sort().values
            grid_idx = (keep_f.unsqueeze(1) * Tg + keep_t.unsqueeze(0)).reshape(-1)
            idx = torch.cat([torch.zeros(1, dtype=torch.long), grid_idx + 1]).to(x.device)
            x = x.index_select(1, idx)
        x = v.norm_pre(x)
        for blk in v.blocks:
            x = blk(x)
        x = v.norm(x)
        return v.head(x[:, 0])


# ---------------------------------------------------------------- HTS-AT-style Swin
def build_htsat(num_classes=3):
    """Compact Swin Transformer on 128x128 padded mel (HTS-AT-style hierarchical
    windowed attention; trained from scratch, no AudioSet pretraining)."""
    from timm.models.swin_transformer import SwinTransformer
    return SwinTransformer(
        img_size=128, patch_size=4, in_chans=1, num_classes=num_classes,
        embed_dim=48, depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 16),
        window_size=4, drop_path_rate=0.1)


class HTSATClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.swin = build_htsat(num_classes)

    def forward(self, mel):  # pad time 101 -> 128
        x = F.pad(mel, (0, 128 - mel.shape[-1])) if mel.shape[-1] < 128 else mel[..., :128]
        return self.swin(x)
