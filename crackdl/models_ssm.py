"""10. Audio Mamba-style bidirectional selective state-space model
(Erol et al., 2024, "Audio Mamba: Bidirectional State Space Model for
Audio Representation Learning"). Pure-PyTorch selective scan (sequences are
short: 8x7=56 mel patches), bidirectional blocks, middle class token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Simplified S6: input-dependent (delta, B, C), diagonal A, sequential scan."""

    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, conv_kernel,
                              padding=conv_kernel - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        # Mamba dt init: softplus(bias) log-uniform in [1e-3, 1e-1], small weight
        with torch.no_grad():
            dt = torch.exp(torch.rand(self.d_inner)
                           * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
                           + torch.log(torch.tensor(1e-3)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inv softplus
            self.dt_proj.weight.mul_(0.01)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):  # (B,L,D)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)                       # (B,L,Di)
        xs = self.conv(xs.transpose(1, 2))[..., :L].transpose(1, 2)
        xs = F.silu(xs)
        dbc = self.x_proj(xs)                             # (B,L,2N+1)
        dt, Bm, Cm = torch.split(dbc, [1, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (B,L,Di)
        A = -torch.exp(self.A_log)                        # (Di,N)
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (B,L,Di,N)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xs.unsqueeze(-1)  # (B,L,Di,N)
        h = torch.zeros(B, self.d_inner, A.shape[1], device=x.device, dtype=dA.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + xs * self.D          # (B,L,Di)
        y = y * F.silu(z)
        return self.out_proj(y)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state)
        self.bwd = SelectiveSSM(d_model, d_state)

    def forward(self, x):
        h = self.norm(x)
        return x + 0.5 * (self.fwd(h) + self.bwd(h.flip(1)).flip(1))


class AudioMamba(nn.Module):
    def __init__(self, num_classes=3, d_model=192, depth=6, patch=16, mel_shape=(128, 112)):
        super().__init__()
        self.mel_shape = mel_shape
        self.patch_embed = nn.Conv2d(1, d_model, patch, stride=patch)
        n_patches = (mel_shape[0] // patch) * (mel_shape[1] // patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([BiMambaBlock(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.mid = n_patches // 2

    def forward(self, mel):
        T = self.mel_shape[1]
        x = F.pad(mel, (0, max(0, T - mel.shape[-1])))[..., :T]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B,N,D)
        # middle cls token (as in Audio Mamba)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([x[:, : self.mid], cls, x[:, self.mid:]], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, self.mid]))
