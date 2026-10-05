"""Unified trainer. Usage:
    python train.py --model roastformer --seed 0 --protocol official
    python train.py --model clap --seed 0            (linear probe path)
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import data as D
from registry import build, count_params

HERE = os.path.dirname(__file__)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def soft_ce(logits, targets, class_weights, smoothing=0.05):
    """Soft-target weighted CE. targets: (B,C) soft labels."""
    C = logits.shape[1]
    t = targets * (1 - smoothing) + smoothing / C
    logp = F.log_softmax(logits, dim=1)
    w = (t * class_weights.unsqueeze(0)).sum(dim=1)
    loss = -(t * logp).sum(dim=1) * w
    return loss.mean()


@torch.no_grad()
def evaluate(model, frontend, loader, device, input_kind):
    model.eval()
    preds, trues = [], []
    for w, y in loader:
        w = w.to(device, non_blocking=True)
        x = frontend(w) if input_kind == "mel32" else w
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            logits = model(x)
        preds.append(logits.argmax(1).cpu()); trues.append(y)
    yp = torch.cat(preds).numpy(); yt = torch.cat(trues).numpy()
    return dict(acc=float(accuracy_score(yt, yp)),
                macro_f1=float(f1_score(yt, yp, average="macro")),
                per_class_f1=[float(v) for v in f1_score(yt, yp, average=None)],
                cm=confusion_matrix(yt, yp).tolist(),
                y_true=yt.tolist(), y_pred=yp.tolist())


def train_standard(args, device):
    set_seed(args.seed)
    model, cfg = build(args.model)
    model = model.to(device)
    if args.epochs:
        cfg["epochs"] = args.epochs
    splits = D.make_splits(args.protocol)
    wave_key = {"mel32": "w32", "wave16": "w16"}[cfg["input"]]
    tr_loader = D.make_loader(splits["train"], wave_key, cfg["batch_size"], True, args.seed)
    va_loader = D.make_loader(splits["val"], wave_key, 128, False)
    te_loader = D.make_loader(splits["test"], wave_key, 128, False)

    frontend = D.MelFrontend().to(device)
    if cfg["input"] == "mel32":
        frontend.fit_stats(tr_loader, device)

    counts = np.bincount(splits["train"]["y"], minlength=D.NUM_CLASSES)
    cw = torch.tensor(counts.sum() / (D.NUM_CLASSES * counts), dtype=torch.float32, device=device)

    pg = cfg.get("param_groups") or model.parameters()
    opt = torch.optim.AdamW(pg, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * max(1, len(tr_loader))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = {"macro_f1": -1}; best_state = None
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        for w, y in tr_loader:
            w = w.to(device, non_blocking=True); y = y.to(device)
            y1 = F.one_hot(y, D.NUM_CLASSES).float()
            w = D.augment_wave(w)
            if cfg["input"] == "mel32":
                x = frontend(w)
                if cfg["specaug"]:
                    x = D.spec_augment(x)
                x, yt = D.mixup(x, y1, cfg["mixup"])
            else:
                x = w.float() / 32768.0
                x, yt = D.mixup(x, y1, cfg["mixup"])
                x = (x * 32768.0).to(torch.int16)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(frontendless(x, cfg))
                loss = soft_ce(logits, yt, cw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        vm = evaluate(model, frontend, va_loader, device, cfg["input"])
        if vm["macro_f1"] > best["macro_f1"]:
            best = vm; best["epoch"] = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.verbose:
            print(f"ep{ep} val acc {vm['acc']:.4f} mf1 {vm['macro_f1']:.4f}")
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    test = evaluate(model, frontend, te_loader, device, cfg["input"])
    out = dict(model=args.model, seed=args.seed, protocol=args.protocol,
               params=count_params(model), epochs=cfg["epochs"],
               train_time_s=train_time, val_best=best, test=test)
    if args.save_ckpt:
        os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
        torch.save(best_state, os.path.join(
            HERE, "ckpt", f"{args.model}_s{args.seed}_{args.protocol}.pt"))
    return out


def frontendless(x, cfg):
    """x already featurized for mel input; wave models take int16 wave."""
    return x


def train_clap(args, device):
    set_seed(args.seed)
    from models_ssl import CLAPEmbedder, LinearProbe
    splits = D.make_splits(args.protocol)
    # fingerprint ties the embedding cache to the exact prep cache contents
    import hashlib
    fp = hashlib.sha1(b"".join(
        np.sort(splits[s]["name"]).tobytes() for s in ["train", "val", "test"])).hexdigest()[:12]
    cache = os.path.join(HERE, "cache", f"clap_emb_{args.protocol}_{fp}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        embs = {s: torch.tensor(z[s]) for s in ["train", "val", "test"]}
        backbone_params = int(z["backbone_params"])
    else:
        emb = CLAPEmbedder(device)
        backbone_params = sum(p.numel() for p in emb.model.parameters())
        embs = {s: emb.embed(splits[s]["w48"]) for s in ["train", "val", "test"]}
        np.savez(cache, backbone_params=backbone_params,
                 **{s: e.numpy() for s, e in embs.items()})
    for s in ["train", "val", "test"]:
        assert len(embs[s]) == len(splits[s]["y"]), f"stale CLAP cache for {s}"
    ys = {s: torch.tensor(splits[s]["y"]) for s in splits}
    probe = LinearProbe(embs["train"].shape[1]).to(device)
    counts = np.bincount(splits["train"]["y"], minlength=D.NUM_CLASSES)
    cw = torch.tensor(counts.sum() / (D.NUM_CLASSES * counts), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr = embs["train"].to(device); Ytr = ys["train"].to(device)
    t0 = time.time()
    best = {"macro_f1": -1}; best_state = None
    for ep in range(300):
        probe.train()
        logits = probe(Xtr)
        loss = F.cross_entropy(logits, Ytr, weight=cw, label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            vp = probe(embs["val"].to(device)).argmax(1).cpu().numpy()
        mf1 = f1_score(ys["val"].numpy(), vp, average="macro")
        if mf1 > best["macro_f1"]:
            best = {"macro_f1": float(mf1), "epoch": ep}
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    with torch.no_grad():
        tp = probe(embs["test"].to(device)).argmax(1).cpu().numpy()
    yt = ys["test"].numpy()
    test = dict(acc=float(accuracy_score(yt, tp)),
                macro_f1=float(f1_score(yt, tp, average="macro")),
                per_class_f1=[float(v) for v in f1_score(yt, tp, average=None)],
                cm=confusion_matrix(yt, tp).tolist(),
                y_true=yt.tolist(), y_pred=tp.tolist())
    return dict(model="clap", seed=args.seed, protocol=args.protocol,
                params=int(sum(p.numel() for p in probe.parameters())),
                backbone_params=int(backbone_params), epochs=300,
                train_time_s=time.time() - t0, val_best=best, test=test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="official", choices=["official", "session"])
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save_ckpt", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = train_clap(args, device) if args.model == "clap" else train_standard(args, device)
    rdir = os.path.join(HERE, "results", args.protocol)
    os.makedirs(rdir, exist_ok=True)
    path = os.path.join(rdir, f"{args.model}_s{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(json.dumps({k: out[k] for k in ["model", "seed", "params", "train_time_s"]}),
          "test:", {k: out["test"][k] for k in ["acc", "macro_f1"]})


if __name__ == "__main__":
    main()
