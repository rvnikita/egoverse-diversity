"""Can we get frame-level selection quality at a fraction of the GPU cost?

The experiment:
  ORACLE   embed EVERY frame of each episode on the GPU. This is the answer everyone
           agrees is right and nobody can afford.
  POLICIES pick k frames per episode using CPU-only signals, and embed only those.
  MEASURE  how well the k chosen frames COVER the oracle set, versus what they cost.

Coverage objective (k-center): for every oracle frame, distance to its nearest selected
frame. Low = the k frames represent everything that happened in the episode. This is the
standard objective for "did I keep the important frames", and it is what an expensive
all-frames pipeline would optimise directly.

    AWS_PROFILE=egoverse .venv/bin/python scripts/cascade_experiment.py --episodes 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cheap  # noqa: E402
import egodb  # noqa: E402
import egos3  # noqa: E402
from diversity import vendi_score  # noqa: E402

CLIPS = ROOT / "out" / "mp4"
OUT = ROOT / "out"
ORACLE_CACHE = OUT / "oracle"
ORACLE_CACHE.mkdir(parents=True, exist_ok=True)


def all_frames_jpeg(path: pathlib.Path, width: int = 224) -> list[bytes]:
    """Every frame as JPEG bytes — the oracle's input."""
    import subprocess

    tmp = path.parent / f".all_{path.stem}"
    tmp.mkdir(exist_ok=True)
    for old in tmp.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path),
         "-vf", f"scale={width}:-2", "-q:v", "3", str(tmp / "f_%05d.jpg")],
        capture_output=True,
    )
    frames = [p.read_bytes() for p in sorted(tmp.glob("*.jpg"))]
    for p in tmp.glob("*.jpg"):
        p.unlink()
    tmp.rmdir()
    return frames


def coverage_cost(oracle: np.ndarray, sel_idx: np.ndarray) -> float:
    """Mean cosine distance from each oracle frame to the nearest selected frame."""
    S = oracle[sel_idx]
    sim = oracle @ S.T           # (T, k), both L2-normalised
    return float((1.0 - sim.max(axis=1)).mean())


def greedy_mean_oracle(O: np.ndarray, k: int) -> np.ndarray:
    """Greedy minimisation of MEAN nearest-neighbour distance — the true upper bound.

    Farthest-point sampling is the wrong reference here: it optimises the k-center
    (worst-case) objective and therefore spends budget on outliers, which loses to plain
    uniform sampling on a mean objective. Greedy selection against the metric we actually
    report gives a genuine ceiling to measure policies against.
    """
    n = O.shape[0]
    if k >= n:
        return np.arange(n)
    sim = O @ O.T                      # (n, n) cosine
    best = int(np.argmax(sim.mean(axis=1)))   # medoid: closest to everything
    picked = [best]
    cur = sim[best].copy()             # running max-similarity to the picked set
    for _ in range(k - 1):
        # gain of adding j = reduction in mean (1 - max_sim)
        gains = np.maximum(sim, cur).mean(axis=1) - cur.mean()
        gains[picked] = -np.inf
        nxt = int(np.argmax(gains))
        picked.append(nxt)
        cur = np.maximum(cur, sim[nxt])
    return np.sort(np.array(picked))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--task", default="fold_clothes")
    ap.add_argument("--lab", default="microagi")
    ap.add_argument("--budgets", default="2,4,8,16,32")
    ap.add_argument("--max-frames", type=int, default=900,
                    help="cap per-episode oracle frames so this finishes in a hackathon")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    live = egodb.episodes()
    live = live[~live.is_deleted]
    live = live[(live.lab == args.lab) & (live.task == args.task)]
    live = live[live.zarr_mp4_path.notna() & live.num_frames.between(120, args.max_frames)]
    df = live.sample(min(args.episodes, len(live)), random_state=0)
    print(f"{len(df)} episodes of {args.task} @ {args.lab}, "
          f"{df.num_frames.min():.0f}-{df.num_frames.max():.0f} frames each")

    paths = egos3.fetch_many(df.zarr_mp4_path.tolist(), CLIPS, workers=12)
    print(f"downloaded {len(paths)}")

    import modal

    Embedder = modal.Cls.from_name("egoverse-embed", "Embedder")
    emb = Embedder()

    rows = []
    t_cheap_total = t_gpu_total = 0.0
    total_frames = gpu_frames = 0

    for i, p in enumerate(paths):
        # ---- cheap tier: thumbnails for EVERY frame, CPU only
        t0 = time.monotonic()
        T = cheap.thumbnails(p)
        t_cheap = time.monotonic() - t0
        if T.shape[0] < max(budgets) * 2:
            continue

        # ---- oracle: embed EVERY frame on the GPU
        jpegs = all_frames_jpeg(p)
        n = min(len(jpegs), T.shape[0])
        jpegs, T = jpegs[:n], T[:n]
        cache_f = ORACLE_CACHE / f"{p.stem}.npy"
        t0 = time.monotonic()
        if cache_f.exists():
            O = np.load(cache_f)[:n]
            t_gpu = float("nan")          # cached: not a fresh measurement
        else:
            vecs = []
            for s0 in range(0, n, 256):
                vecs.append(np.array(emb.embed.remote(jpegs[s0 : s0 + 256])["embeddings"],
                                     dtype="float32"))
            O = np.concatenate(vecs, axis=0)
            np.save(cache_f, O)
            t_gpu = time.monotonic() - t0

        t_cheap_total += t_cheap
        if not np.isnan(t_gpu):
            t_gpu_total += t_gpu
            gpu_frames += n
        total_frames += n
        print(f"  [{i+1}/{len(paths)}] {p.stem[:22]} frames={n:4d} "
              f"cheap={t_cheap:.2f}s gpu={t_gpu:.1f}s")

        for k in budgets:
            for name, fn in cheap.POLICIES.items():
                idx = fn(T, k)
                rows.append({
                    "episode": p.stem, "k": k, "policy": name,
                    "frames": n,
                    "coverage": coverage_cost(O, idx),
                    "vendi_sel": vendi_score(O[idx]),
                })
        # References computed ON the oracle embeddings (i.e. after paying full GPU cost):
        #   ORACLE_greedy  optimises the SAME mean objective we report -> true ceiling
        #   ORACLE_fps     optimises k-center -> included to show it is the WRONG ceiling
        from diversity import farthest_point
        for k in budgets:
            for nm, idx in (("ORACLE_greedy", greedy_mean_oracle(O, k)),
                            ("ORACLE_fps", farthest_point(O, k))):
                rows.append({"episode": p.stem, "k": k, "policy": nm,
                             "frames": n, "coverage": coverage_cost(O, idx),
                             "vendi_sel": vendi_score(O[idx])})

    import pandas as pd

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cascade_results.csv", index=False)

    print(f"\n=== cost model (measured over {total_frames:,} frames) ===")
    print(f"  cheap tier (CPU, ALL frames): {t_cheap_total:6.2f}s "
          f"= {t_cheap_total/total_frames*1e6:.1f} us/frame  ({total_frames:,} frames)")
    if gpu_frames:
        cheap_on_gpu_frames = t_cheap_total * gpu_frames / total_frames
        print(f"  oracle    (GPU, ALL frames): {t_gpu_total:6.1f}s "
              f"= {t_gpu_total/gpu_frames*1e3:.1f} ms/frame  ({gpu_frames:,} freshly embedded)")
        print(f"  GPU is {t_gpu_total/max(cheap_on_gpu_frames,1e-9):,.0f}x the cheap tier")
    else:
        print("  oracle embeddings served from cache; no fresh GPU timing this run")

    print("\n=== coverage by policy (lower is better) ===")
    piv = res.pivot_table(index="k", columns="policy", values="coverage", aggfunc="mean")
    order = ["uniform", "motion_peaks", "cheap_fps", "motion_gated_fps",
             "ORACLE_fps", "ORACLE_greedy"]
    piv = piv[[c for c in order if c in piv.columns]]
    print(piv.round(4).to_string())

    print("\n=== % of oracle-quality retained, and GPU saved ===")
    for k in budgets:
        sub = piv.loc[k]
        base, orc = sub["uniform"], sub["ORACLE_greedy"]
        best_name = sub.drop(["ORACLE_fps", "ORACLE_greedy"]).idxmin()
        best = sub[best_name]
        # how much of the uniform->oracle gap the best cheap policy closes
        closed = (base - best) / max(base - orc, 1e-9) * 100
        mean_frames = res[res.k == k].frames.mean()
        print(f"  k={k:3d}  best={best_name:17s} coverage {best:.4f} "
              f"(uniform {base:.4f}, oracle {orc:.4f})  gap closed {closed:5.1f}%  "
              f"GPU used {k/mean_frames*100:4.1f}% of all-frames")

    (OUT / "cascade_cost.json").write_text(json.dumps({
        "frames": total_frames, "cheap_s": t_cheap_total, "gpu_s": t_gpu_total,
        "us_per_frame_cheap": t_cheap_total / total_frames * 1e6,
        "ms_per_frame_gpu": t_gpu_total / total_frames * 1e3,
    }, indent=2))
    print(f"\nwrote {OUT/'cascade_results.csv'} and {OUT/'cascade_cost.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
