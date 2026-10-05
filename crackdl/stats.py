"""McNemar exact test: CrackFormer vs each baseline on pooled (seed x sample)
test decisions. Prints p-values for the paper."""
import glob, json, os
from math import comb

HERE = os.path.dirname(__file__)


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar. b,c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(model, protocol="official"):
    outs = {}
    for p in glob.glob(os.path.join(HERE, "results", protocol, f"{model}_s*.json")):
        with open(p) as f:
            r = json.load(f)
        if r["model"] != model:          # guard: 'roastformer_s*' also globs 'roastformer_scaleonly_*'
            continue
        outs[r["seed"]] = (r["test"]["y_true"], r["test"]["y_pred"])
    return outs


def main(protocol="official", ours="roastformer"):
    ours_runs = load(ours, protocol)
    models = sorted({json.load(open(p))["model"]
                     for p in glob.glob(os.path.join(HERE, "results", protocol, "*.json"))})
    print(f"McNemar exact test, {ours} vs baselines ({protocol}, pooled over seeds)")
    for m in models:
        if m == ours or m.startswith("roastformer_"):
            continue
        runs = load(m, protocol)
        b = c = both_wrong = both_right = 0
        for s, (yt, yp_ours) in ours_runs.items():
            if s not in runs:
                continue
            yt2, yp_base = runs[s]
            assert yt == yt2
            for t, po, pb in zip(yt, yp_ours, yp_base):
                ok_o, ok_b = po == t, pb == t
                if ok_o and not ok_b:
                    b += 1
                elif ok_b and not ok_o:
                    c += 1
                elif ok_o:
                    both_right += 1
                else:
                    both_wrong += 1
        p = mcnemar_exact(b, c)
        print(f"  vs {m:14s}: ours-only-right={b:4d} base-only-right={c:4d} p={p:.2e}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "official")
