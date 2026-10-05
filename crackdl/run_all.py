"""Run the full experiment grid sequentially (single GPU).

  python run_all.py                 # official protocol, all models, seeds 0..2
  python run_all.py --protocol session --models roastformer,bcresnet,ast,wavlm
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(__file__)
DEFAULT = ["cnn14", "conformer", "ast", "bcresnet", "passt", "htsat",
           "wavlm", "mobilenetv3", "clap", "audiomamba", "roastformer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--protocol", default="official")
    args = ap.parse_args()
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    for m in models:
        for s in seeds:
            out = os.path.join(HERE, "results", args.protocol, f"{m}_s{s}.json")
            if os.path.exists(out):
                print(f"skip {m} s{s} (exists)")
                continue
            cmd = [sys.executable, os.path.join(HERE, "train.py"),
                   "--model", m, "--seed", str(s), "--protocol", args.protocol]
            if m == "roastformer" and s == 0:
                cmd.append("--save_ckpt")
            print(f"=== {m} seed {s} ({args.protocol}) | elapsed {time.time()-t0:.0f}s ===", flush=True)
            r = subprocess.run(cmd, cwd=HERE)
            if r.returncode != 0:
                print(f"!!! {m} s{s} FAILED rc={r.returncode}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
