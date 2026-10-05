"""Check audio preprocessing and RoastFormer gradients with synthetic input."""
import argparse

import torch
from torch.nn import functional as F

from data import MelFrontend
from registry import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use the CUDA image and --gpus all")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    torch.set_num_threads(2)
    wave = torch.randint(-3000, 3001, (2, 32000), dtype=torch.int16)
    labels = torch.tensor([0, 1])
    frontend = MelFrontend().to(device)
    frontend.fit_stats([(wave, labels)], device)
    mel = frontend(wave.to(device))
    assert mel.shape == (2, 1, 128, 101), mel.shape
    assert torch.isfinite(mel).all(), "Non-finite mel features"
    assert frontend.std.item() > 0, "Invalid frontend normalization"

    model, _ = build("roastformer")
    model = model.to(device)
    logits = model(mel)
    assert logits.shape == (2, 3), logits.shape
    assert torch.isfinite(logits).all(), "Non-finite logits"
    loss = F.cross_entropy(logits, labels.to(device))
    assert torch.isfinite(loss), "Non-finite loss"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "Invalid gradients"
    assert any(torch.count_nonzero(g).item() for g in grads), "All gradients are zero"
    print(f"RoastFormer smoke passed on {device}; logits={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
