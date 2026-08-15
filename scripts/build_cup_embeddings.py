"""Embed the labelled cup_on_saucer pool — the one set with success/failure ground truth.

913 episodes (748 success / 165 failure), outcome recovered from task names because
`eval_success` is the dataclass default on all 439k registry rows (docs/egodb-findings.md).

Samples 32 uniform frames per episode ONCE. Smaller frame budgets (4, 8, 16) are derived
downstream by uniformly subsampling those 32 — a uniform subsample of a uniform sample is
still uniform over the clip, so one GPU pass covers the whole frames-per-episode ablation.

    AWS_PROFILE=egoverse .venv/bin/python scripts/build_cup_embeddings.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import frames as frame_utils  # noqa: E402

LABELS = ROOT / "out" / "cup_labelled.csv"
CLIPS = ROOT / "out" / "cup_mp4"
OUT = ROOT / "out" / "emb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tag", default="cup")
    args = ap.parse_args()

    df = pd.read_csv(LABELS)
    df["mp4"] = df.zarr_mp4_path.map(lambda u: CLIPS / pathlib.Path(str(u)).name)
    df = df[df.mp4.map(lambda p: p.exists() and p.stat().st_size > 0)].reset_index(drop=True)
    print(f"{len(df)} episodes with a local mp4 "
          f"({(df.outcome == 'success').sum()} success / {(df.outcome == 'failure').sum()} failure)")

    # ---- decode: k uniform frames per episode, threaded (ffmpeg releases the GIL)
    print(f"\n== decoding {args.frames} frames/episode ==")
    t0 = time.monotonic()

    def decode(i: int):
        try:
            return i, frame_utils.sample(df.mp4[i], k=args.frames)
        except Exception as exc:  # noqa: BLE001
            print(f"  decode failed {df.mp4[i].name}: {type(exc).__name__}")
            return i, []

    all_frames: list[bytes] = []
    owner: list[int] = []
    slot: list[int] = []  # position of this frame within its episode, 0..k-1
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, (i, fr) in enumerate(pool.map(decode, range(len(df))), 1):
            if len(fr) < args.frames:  # keep the tensor rectangular; drop short clips
                continue
            all_frames.extend(fr[: args.frames])
            owner.extend([i] * args.frames)
            slot.extend(range(args.frames))
            if done % 100 == 0:
                print(f"  {done}/{len(df)} episodes, {len(all_frames)} frames "
                      f"({time.monotonic()-t0:.0f}s)")
    kept = sorted(set(owner))
    print(f"  {len(all_frames)} frames from {len(kept)} episodes in {time.monotonic()-t0:.0f}s")

    # ---- embed on the Modal L4
    print("\n== embedding on Modal GPU ==")
    import modal

    emb = modal.Cls.from_name("egoverse-embed", "Embedder")()
    vecs, t0 = [], time.monotonic()
    for i in range(0, len(all_frames), args.chunk):
        res = emb.embed.remote(all_frames[i : i + args.chunk])
        vecs.append(np.array(res["embeddings"], dtype="float32"))
        n_done = min(i + args.chunk, len(all_frames))
        print(f"  {n_done}/{len(all_frames)}  gpu={res['latency_ms']:.0f}ms  "
              f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
    X = np.concatenate(vecs, axis=0)
    gpu_s = time.monotonic() - t0
    print(f"  {X.shape} in {gpu_s:.0f}s = {gpu_s/len(all_frames)*1e3:.1f} ms/frame")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.tag}.npz"
    np.savez_compressed(
        dest,
        X=X,
        owner=np.array(owner, dtype="int32"),
        slot=np.array(slot, dtype="int16"),
        k_frames=args.frames,
        gpu_seconds=gpu_s,
        episode_hash=df.episode_hash.to_numpy().astype(str),
        outcome=df.outcome.to_numpy().astype(str),
        operator=df.operator.fillna("").to_numpy().astype(str),
        scene=df.scene.fillna("").to_numpy().astype(str),
        task=df.task.to_numpy().astype(str),
        num_frames=df.num_frames.to_numpy(),
        mp4=df.mp4.map(str).to_numpy().astype(str),
    )
    print(f"\nwrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
