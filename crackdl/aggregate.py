"""Aggregate result JSONs into a summary table (mean +/- std over seeds)."""
import glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(__file__)


def load(protocol="official"):
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "results", protocol, "*.json")):
        with open(p) as f:
            r = json.load(f)
        rows[r["model"]].append(r)
    return rows


def summarize(protocol="official"):
    rows = load(protocol)
    out = {}
    for m, runs in sorted(rows.items()):
        accs = [r["test"]["acc"] for r in runs]
        f1s = [r["test"]["macro_f1"] for r in runs]
        pcf = np.array([r["test"]["per_class_f1"] for r in runs])
        out[m] = dict(
            n_seeds=len(runs),
            acc_mean=float(np.mean(accs)) * 100, acc_std=float(np.std(accs)) * 100,
            f1_mean=float(np.mean(f1s)) * 100, f1_std=float(np.std(f1s)) * 100,
            per_class_f1_mean=(pcf.mean(0) * 100).round(2).tolist(),
            params=runs[0]["params"],
            train_time_s=float(np.mean([r["train_time_s"] for r in runs])),
            train_time_min_s=float(np.min([r["train_time_s"] for r in runs])),
        )
    return out


if __name__ == "__main__":
    import sys
    prot = sys.argv[1] if len(sys.argv) > 1 else "official"
    s = summarize(prot)
    print(f"{'model':14s} {'acc':>14s} {'macroF1':>14s} {'params':>10s} {'bg/fc/sc F1':>24s}")
    for m, r in sorted(s.items(), key=lambda kv: -kv[1]["f1_mean"]):
        print(f"{m:14s} {r['acc_mean']:6.2f}±{r['acc_std']:4.2f} "
              f"{r['f1_mean']:6.2f}±{r['f1_std']:4.2f} {r['params']:>10,d} "
              f"{str(r['per_class_f1_mean']):>24s}  ({r['n_seeds']} seeds)")
