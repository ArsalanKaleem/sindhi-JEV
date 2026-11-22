"""Pre-generate training pairs on CPU (no GPU hours), for when on-the-fly generation starves the GPU.

    python scripts/make_shards.py --n 1000000 --shard-size 20000 --out /kaggle/working/shards --procs 4

Same generator and seeds as on-the-fly training (example_rng(seed, "train", i)), so a shard run and an
on-the-fly run see the same examples. ~9 KB per pair (PNG in a pickle; 500k pairs ~ 4.7 GB). Run in a Kaggle CPU notebook, save the output,
attach it to the GPU notebook and pass --shards /kaggle/input/<name>/shards/*.pkl to train.py.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sindhijev.data import write_shard  # noqa: E402
from sindhijev.pairs import PairGenerator, example_rng  # noqa: E402

_gen = None


def _init(cfg_path):
    global _gen
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    _gen = PairGenerator(cfg, "train", cfg["scan_fonts_train"], root=ROOT)
    _gen.seed = cfg["seed"]


def _make(job):
    k, start, n, out = job
    path = Path(out) / f"train_{k:04d}.pkl"
    if path.exists():
        return str(path), 0
    write_shard(path, (_gen.sample(example_rng(_gen.seed, "train", i)) for i in range(start, start + n)))
    return str(path), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data_v2.json"))
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--shard-size", type=int, default=20_000)
    ap.add_argument("--out", default="shards")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(k, s, min(a.shard_size, a.n - s), a.out) for k, s in enumerate(range(0, a.n, a.shard_size))]
    t = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.config,)) as pool:
        for path, n in pool.imap_unordered(_make, jobs):
            print(f"{path} {n} pairs, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
