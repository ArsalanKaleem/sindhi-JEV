"""Phase 3 training (Kaggle T4/P100, fp16, resumable across 9 h sessions).

    python scripts/train.py --config configs/model_v1.json --out /kaggle/working/runs/main
    # next session: attach the previous run's output as an input dataset, then
    python scripts/train.py --config configs/model_v1.json --out /kaggle/working/runs/main \
        --resume /kaggle/input/<prev-output>/runs/main/last.pt

Every "epoch" (= --steps-per-epoch steps, ~20-30 min on a T4) it validates on the fixed synthetic val set,
writes last.pt (+ best.pt by val AUROC on hard negatives) and appends a line to log.jsonl.
GPU time: every session's wall-clock time on CUDA is added to gpu_time.json in the run dir; the script stops
cleanly before --max-hours (session budget) and refuses to start if the run's total would exceed --budget-hours.
CPU smoke test: --device cpu --steps-per-epoch 5 --epochs 1 --batch 4 --val-limit 16 --workers 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.data import Manifest, OnTheFly, Shards, collate, seed_worker  # noqa: E402
from sindhijev.metrics import auroc  # noqa: E402
from sindhijev.model import Verifier, n_params, to_device, warmup_cosine  # noqa: E402
from sindhijev.pairs import PairGenerator  # noqa: E402


def load_cfg(path, data_config=None):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if data_config:  # ablation: other training data mix (validation sets stay the same)
        cfg["data_config"] = data_config
    data_cfg = json.loads((ROOT / cfg["data_config"]).read_text(encoding="utf-8"))
    return cfg, data_cfg


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval()
    recs = []
    for b in loader:
        meta = b["meta"]
        b = to_device(b, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out = model(b)
        seq = torch.sigmoid(out["seq_logit"].float()).cpu().tolist()
        ch = torch.sigmoid(out["char_logit"].float()).cpu() if "char_logit" in out else None
        for i, m in enumerate(meta):
            n = int(b["char_mask"][i].sum())
            recs.append({**m, "seq_y": int(b["seq_y"][i]), "seq_score": seq[i],
                         "char_y": b["char_y"][i, :n].int().cpu().tolist(),
                         "char_score": ch[i, :n].tolist() if ch is not None else [0.0] * n})
    model.train()
    return recs


def val_summary(recs, hard_recs, font_recs=None):
    """val_auroc: v1 val set (all error types). val_auroc_hard: val_hard (correct vs REAL-WORD dot swaps,
    which a dictionary cannot flag) = the decision-gate / model-selection number."""
    a = lambda rs: auroc([r["seq_y"] for r in rs], [r["seq_score"] for r in rs])  # noqa: E731
    out = {"val_auroc": a(recs), "val_auroc_hard": a(hard_recs)}
    if font_recs:  # v3: unseen-font validation (Mirza), correct vs real-word swaps
        out["val_auroc_font_hard"] = a(font_recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model_v1.json"))
    ap.add_argument("--out", default="runs/debug")
    ap.add_argument("--resume", default=None, help="checkpoint path; default: <out>/last.pt if present")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None, help="override total epochs")
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--shards", nargs="*", default=None, help="pre-generated train shards instead of on-the-fly")
    ap.add_argument("--max-hours", type=float, default=8.5, help="stop this session after this many hours")
    ap.add_argument("--budget-hours", type=float, default=None, help="refuse to run past this total for the run")
    ap.add_argument("--set", nargs="*", default=[], help="model config overrides key=value (ablations)")
    ap.add_argument("--data-config", default=None, help="override the training data config (e.g. configs/data_v1.json)")
    a = ap.parse_args()

    t_start = time.time()
    cfg, data_cfg = load_cfg(a.config, a.data_config)
    tr = cfg["train"]
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg["model"][k] = json.loads(v)
    epochs = a.epochs or tr["epochs"]
    spe = a.steps_per_epoch or tr["steps_per_epoch"]
    bs = a.batch or tr["batch"]
    total_steps = epochs * spe
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    amp = dev.type == "cuda" and tr.get("fp16", True)
    gpu_file = out / "gpu_time.json"
    gpu = json.loads(gpu_file.read_text()) if gpu_file.exists() else {"sessions": []}
    used_h = sum(s["hours"] for s in gpu["sessions"])
    budget = a.budget_hours if a.budget_hours is not None else tr.get("budget_hours")
    if dev.type == "cuda" and budget is not None and used_h >= budget:
        sys.exit(f"run already used {used_h:.2f} GPU h >= budget {budget} h: stop and ask before continuing")

    model_cfg = {**cfg["model"], "img_height": data_cfg.get("img_height", 48)}
    model = Verifier(model_cfg).to(dev)
    print(f"params {n_params(model) / 1e6:.2f}M  device {dev}  amp {amp}  steps {total_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best = 0, -1.0
    ck = Path(a.resume) if a.resume else out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step, best = st["step"], st["best"]
        if st.get("gpu") and not gpu["sessions"]:
            gpu = st["gpu"]  # resumed from another session's output dir: carry the GPU-time history
        print(f"resumed from {ck} at step {step} (best {best:.4f})")

    mcfg = model.cfg
    if a.shards:
        train_ds = Shards(a.shards, mcfg)
    else:
        gen = PairGenerator(data_cfg, "train", data_cfg["scan_fonts_train"], root=ROOT)
        train_ds = OnTheFly(gen, mcfg, seed=data_cfg["seed"], length=total_steps * bs + bs)
    val_ds = Manifest(ROOT / "data" / "synth" / "val", mcfg, limit=a.val_limit or tr.get("val_limit"))
    hard_ds = Manifest(ROOT / "data" / "synth" / "val_hard", mcfg, limit=a.val_limit or tr.get("val_limit"))

    class Resume(torch.utils.data.Sampler):  # deterministic order that continues where the last session stopped
        def __init__(self, n, start):
            self.n, self.start = n, start

        def __iter__(self):
            g = torch.Generator().manual_seed(data_cfg["seed"])
            if isinstance(train_ds, OnTheFly):
                return iter(range(self.start, self.n))
            perm_epochs = (self.n // len(train_ds)) + 1
            idx = torch.cat([torch.randperm(len(train_ds), generator=g) for _ in range(perm_epochs)])
            return iter(idx[self.start:self.n].tolist())

        def __len__(self):
            return self.n - self.start

    loader = DataLoader(train_ds, batch_size=bs, sampler=Resume(total_steps * bs, step * bs), collate_fn=collate,
                        num_workers=a.workers, worker_init_fn=seed_worker, pin_memory=dev.type == "cuda",
                        persistent_workers=a.workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate, num_workers=min(a.workers, 2))
    hard_loader = DataLoader(hard_ds, batch_size=bs, collate_fn=collate, num_workers=min(a.workers, 2))
    font_dir = ROOT / "data" / "synth" / "val_font_hard"
    font_loader = (DataLoader(Manifest(font_dir, mcfg, limit=a.val_limit or tr.get("val_limit")), batch_size=bs,
                              collate_fn=collate, num_workers=min(a.workers, 2))
                   if (font_dir / "manifest.jsonl").exists() else None)

    def save(path, extra=None):
        session = {"started": t_start, "hours": (time.time() - t_start) / 3600, "device": str(dev)}
        g = {"sessions": gpu["sessions"] + ([session] if dev.type == "cuda" else [])}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "cfg": cfg, "model_cfg": mcfg, "gpu": g, **(extra or {})}, path)
        if dev.type == "cuda":
            gpu_file.write_text(json.dumps(g, indent=1))

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    model.train()
    t_ep, run = time.time(), {"loss": 0.0, "seq": 0.0, "char": 0.0, "n": 0}
    stop = False
    for b in loader:
        lr = tr["lr"] * warmup_cosine(step, tr["warmup"], total_steps)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        b = to_device(b, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            outp = model(b)
            loss, parts = model.loss(outp, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        run["loss"] += float(loss.detach())
        run["seq"] += float(parts["seq"])
        run["char"] += float(parts.get("char", 0.0))
        run["n"] += 1
        if step % tr.get("log_every", 100) == 0:
            n = run["n"]
            print(f"step {step} lr {lr:.2e} loss {run['loss'] / n:.4f} seq {run['seq'] / n:.4f} "
                  f"char {run['char'] / n:.4f} {(time.time() - t_ep) / max(1, step % spe or spe):.3f}s/step",
                  flush=True)
        hours = (time.time() - t_start) / 3600
        if hours > a.max_hours:
            stop = True
        if step % spe == 0 or step == total_steps or stop:
            v = val_summary(predict(model, val_loader, dev, amp), predict(model, hard_loader, dev, amp),
                            predict(model, font_loader, dev, amp) if font_loader else None)
            n = max(1, run["n"])
            rec = {"step": step, "epoch": step / spe, "lr": lr, "train_loss": run["loss"] / n,
                   "train_seq": run["seq"] / n, "train_char": run["char"] / n, **v,
                   "session_hours": round(hours, 3), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("VAL", json.dumps(rec), flush=True)
            if v["val_auroc_hard"] > best:
                best = v["val_auroc_hard"]
                save(out / "best.pt")
            save(out / "last.pt")
            run = {"loss": 0.0, "seq": 0.0, "char": 0.0, "n": 0}
            t_ep = time.time()
        if stop:
            print(f"session limit {a.max_hours} h reached at step {step}; resume with --resume {out / 'last.pt'}")
            break
    total = sum(s["hours"] for s in json.loads(gpu_file.read_text())["sessions"]) if gpu_file.exists() else 0
    print(f"done at step {step}/{total_steps}; best val AUROC(hard) {best:.4f}; run GPU hours so far {total:.2f}")


if __name__ == "__main__":
    main()
