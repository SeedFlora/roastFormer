"""Build every model, run a forward+backward on dummy data, report params."""
import sys, time, traceback
import torch
from registry import build, count_params, ALL_MODELS

device = "cuda" if torch.cuda.is_available() else "cpu"
mel = torch.randn(4, 1, 128, 101, device=device)
w16 = (torch.randn(4, 16000, device=device) * 3000).to(torch.int16)

ok = True
for name in ALL_MODELS:
    try:
        t0 = time.time()
        m, cfg = build(name)
        m = m.to(device)
        x = mel if cfg["input"] == "mel32" else w16
        y = m(x)
        loss = y.float().square().mean()
        loss.backward()
        assert y.shape == (4, 3), y.shape
        print(f"{name:12s} OK  params={count_params(m)/1e6:8.2f}M  out={tuple(y.shape)}  {time.time()-t0:.1f}s")
        del m
        torch.cuda.empty_cache()
    except Exception:
        ok = False
        print(f"{name:12s} FAIL")
        traceback.print_exc()
sys.exit(0 if ok else 1)
