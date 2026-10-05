"""One-time preprocessing: resample all WAVs to 32k/16k/48k, fixed 1.0s length, cache as npz."""
import os, re, glob
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = os.path.join(os.path.dirname(__file__), "..", "coffee-roasting-acoustic-dataset", "datasets")
OUT = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(OUT, exist_ok=True)

SRS = {"w32": 32000, "w16": 16000, "w48": 48000}
DUR = 1.0  # seconds


def fix_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    out = np.zeros(n, dtype=x.dtype)
    s = (n - len(x)) // 2
    out[s:s + len(x)] = x
    return out


def main():
    for split in ["train", "val", "test"]:
        files = sorted(glob.glob(os.path.join(ROOT, split, "*", "*.wav")))
        assert files, f"no files for {split}"
        data = {k: [] for k in SRS}
        ys, trials, origins, names = [], [], [], []
        for f in files:
            y_lab = int(os.path.basename(os.path.dirname(f)))
            name = os.path.basename(f)
            m = re.match(r"(\d+)([a-z]+)_", name)
            wav, sr = sf.read(f, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            t = torch.from_numpy(wav)
            for key, target_sr in SRS.items():
                r = AF.resample(t, sr, target_sr).numpy()
                r = fix_len(r, int(target_sr * DUR))
                data[key].append(np.clip(r * 32767.0, -32768, 32767).astype(np.int16))
            ys.append(y_lab)
            trials.append(m.group(1) + m.group(2))
            origins.append(m.group(2))
            names.append(name)
        out = {k: np.stack(v) for k, v in data.items()}
        out["y"] = np.array(ys, dtype=np.int64)
        out["trial"] = np.array(trials)
        out["origin"] = np.array(origins)
        out["name"] = np.array(names)
        np.savez(os.path.join(OUT, f"{split}.npz"), **out)
        print(split, out["w32"].shape, "labels:", np.bincount(out["y"]))


if __name__ == "__main__":
    main()
