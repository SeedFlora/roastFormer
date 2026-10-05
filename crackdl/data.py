"""Data loading, GPU mel features, augmentation (SpecAugment, mixup)."""
import os
import numpy as np
import torch
import torchaudio

CACHE = os.path.join(os.path.dirname(__file__), "cache")
NUM_CLASSES = 3
CLASS_NAMES = ["Background", "FirstCrack", "SecondCrack"]

# session-independent protocol: fully trial-disjoint train/val/test,
# one trial per origin in each split (no roasting session shared across splits)
SESSION_TRAIN_TRIALS = {"1yunnan", "8ethiopia", "6kenya"}
SESSION_VAL_TRIALS = {"9yunnan", "3ethiopia", "7kenya"}
SESSION_TEST_TRIALS = {"2yunnan", "4ethiopia", "5kenya"}


def load_split(split):
    z = np.load(os.path.join(CACHE, f"{split}.npz"))
    return {k: z[k] for k in z.files}


def make_splits(protocol="official", seed=0):
    """Returns dict split-> {w32,w16,w48,y,trial,...} numpy arrays."""
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    if protocol == "official":
        return {"train": tr, "val": va, "test": te}
    # session-independent: pool everything, split by whole trials (trial-disjoint)
    pool = {k: np.concatenate([tr[k], va[k], te[k]]) for k in tr}
    out = {}
    for split_name, trials in [("train", SESSION_TRAIN_TRIALS),
                               ("val", SESSION_VAL_TRIALS),
                               ("test", SESSION_TEST_TRIALS)]:
        mask = np.isin(pool["trial"], list(trials))
        out[split_name] = {k: v[mask] for k, v in pool.items()}
    return out


class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, arrays, wave_key="w32"):
        self.w = arrays[wave_key]
        self.y = arrays["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.w[i].astype(np.int16)), int(self.y[i])


def make_loader(arrays, wave_key, batch_size, shuffle, seed=0):
    ds = WaveDataset(arrays, wave_key)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None,
        num_workers=0, pin_memory=True, drop_last=False)


class MelFrontend(torch.nn.Module):
    """int16 wave (B,L)@32k -> normalized log-mel (B,1,n_mels,T). Runs on GPU."""

    def __init__(self, sr=32000, n_fft=1024, hop=320, n_mels=128, fmin=50, fmax=16000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            f_min=fmin, f_max=fmax, power=2.0)
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(self, wave_i16):
        x = wave_i16.float() / 32768.0
        m = self.mel(x)
        m = torch.log(m + 1e-6)
        m = (m - self.mean) / self.std
        return m.unsqueeze(1)

    @torch.no_grad()
    def fit_stats(self, loader, device):
        vals = []
        for w, _ in loader:
            x = w.to(device).float() / 32768.0
            m = torch.log(self.mel(x) + 1e-6)
            vals.append(torch.stack([m.mean(), m.std()]).cpu())
            if len(vals) > 30:
                break
        v = torch.stack(vals)
        self.mean.fill_(v[:, 0].mean().item())
        self.std.fill_(v[:, 1].mean().item())


def augment_wave(w_i16, gain_db=6.0, shift_frac=0.1):
    """Random circular shift + gain. w_i16: (B,L) int16 tensor on device."""
    B, L = w_i16.shape
    x = w_i16.float()
    shifts = torch.randint(-int(L * shift_frac), int(L * shift_frac) + 1, (B,), device=x.device)
    idx = (torch.arange(L, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % L
    x = torch.gather(x, 1, idx)
    g = (torch.rand(B, 1, device=x.device) * 2 - 1) * gain_db
    x = x * (10.0 ** (g / 20.0))
    return x.clamp(-32768, 32767).to(torch.int16)


def spec_augment(mel, n_freq_masks=2, freq_width=16, n_time_masks=2, time_width=20):
    """mel: (B,1,F,T) in-place-ish SpecAugment with zero (post-norm mean) fill."""
    B, _, F, T = mel.shape
    m = mel.clone()
    for _ in range(n_freq_masks):
        w = torch.randint(0, freq_width + 1, (B,), device=mel.device)
        s = torch.randint(0, F, (B,), device=mel.device)
        for b in range(B):
            m[b, :, s[b]:s[b] + w[b], :] = 0.0
    for _ in range(n_time_masks):
        w = torch.randint(0, time_width + 1, (B,), device=mel.device)
        s = torch.randint(0, T, (B,), device=mel.device)
        for b in range(B):
            m[b, :, :, s[b]:s[b] + w[b]] = 0.0
    return m


def mixup(x, y_onehot, alpha=0.2):
    """x: (B,...) features; y_onehot: (B,C). Returns mixed x, y."""
    if alpha <= 0:
        return x, y_onehot
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = lam * x + (1 - lam) * x[perm]
    y2 = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x2, y2
