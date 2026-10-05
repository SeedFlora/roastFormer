"""Measure per-clip inference latency (batch=1) on GPU and CPU for each model."""
import json, os, time
import numpy as np
import torch
from registry import build, count_params, ALL_MODELS

HERE = os.path.dirname(__file__)


def bench(model, x, device, reps=50, warmup=10):
    model = model.to(device).eval()
    xd = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(xd)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(xd)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def main():
    out = {}
    mel = torch.randn(1, 1, 128, 101)
    w16 = (torch.randn(1, 16000) * 3000).to(torch.int16)
    for name in ALL_MODELS:
        m, cfg = build(name)
        x = mel if cfg["input"] == "mel32" else w16
        row = {"params": count_params(m)}
        row["gpu_ms"] = bench(m, x, "cuda") if torch.cuda.is_available() else None
        row["cpu_ms"] = bench(m, x, "cpu", reps=20)
        out[name] = row
        print(name, row, flush=True)
        del m
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "results", "bench.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
