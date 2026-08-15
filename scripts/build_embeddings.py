"""Query -> download MP4s -> sample frames -> embed on Modal -> cache to disk.

    AWS_PROFILE=egoverse .venv/bin/python scripts/build_embeddings.py \
        --tasks fold_clothes,fold_laundry,wash_dishes --per-task 30

Writes out/emb/<tag>.npz with per-frame embeddings plus the episode metadata, so every
downstream experiment is numpy over a cached file and the GPU is paid for exactly once.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import egodb  # noqa: E402
import egos3  # noqa: E402
import frames as frame_utils  # noqa: E402

OUT = ROOT / "out" / "emb"
CLIPS = ROOT / "out" / "mp4"


def pick_episodes(tasks: list[str], per_task: int, lab: str | None, seed: int = 0):
    live = egodb.episodes()
    live = live[~live.is_deleted]
    live = live[live.zarr_mp4_path.notna() & (live.num_frames > 30)]
    if lab:
        live = live[live.lab == lab]

    rows = []
    for t in tasks:
        sub = live[live.task == t]
        if sub.empty:
            print(f"  !! no episodes for task={t!r}" + (f" lab={lab}" if lab else ""))
            continue
        take = sub.sample(min(per_task, len(sub)), random_state=seed)
        rows.append(take)
        print(f"  {t:22} {len(take):4d} episodes (of {len(sub)})  labs={sorted(sub.lab.unique())[:4]}")
    if not rows:
        raise SystemExit("no episodes matched")
    import pandas as pd

    return pd.concat(rows).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    ap.add_argument("--per-task", type=int, default=30)
    ap.add_argument("--frames", type=int, default=8, help="frames sampled per episode")
    ap.add_argument("--lab", default=None, help="restrict to one lab (avoids confounding appearance with task)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print("== selecting episodes ==")
    df = pick_episodes(tasks, args.per_task, args.lab)

    print(f"\n== downloading {len(df)} preview mp4s ==")
    t0 = time.monotonic()
    uris = df.zarr_mp4_path.tolist()
    paths = egos3.fetch_many(uris, CLIPS, workers=args.workers)
    got = {p.name: p for p in paths}
    print(f"  {len(paths)}/{len(uris)} downloaded in {time.monotonic()-t0:.0f}s")

    print(f"\n== sampling {args.frames} frames/episode ==")
    all_frames: list[bytes] = []
    owner: list[int] = []  # index into df for each frame
    kept: list[int] = []
    for i, uri in enumerate(uris):
        name = pathlib.Path(uri).name
        p = got.get(name)
        if p is None:
            continue
        fr = frame_utils.sample(p, k=args.frames)
        if len(fr) < 2:
            continue
        all_frames.extend(fr)
        owner.extend([i] * len(fr))
        kept.append(i)
        if len(kept) % 25 == 0:
            print(f"  {len(kept)} episodes, {len(all_frames)} frames")
    print(f"  total {len(all_frames)} frames from {len(kept)} episodes")

    print("\n== embedding on Modal GPU ==")
    import modal

    Embedder = modal.Cls.from_name("egoverse-embed", "Embedder")
    emb = Embedder()
    chunk = 256
    vecs = []
    t0 = time.monotonic()
    for i in range(0, len(all_frames), chunk):
        res = emb.embed.remote(all_frames[i : i + chunk])
        vecs.append(np.array(res["embeddings"], dtype="float32"))
        print(f"  {min(i+chunk, len(all_frames))}/{len(all_frames)} frames "
              f"({res['latency_ms']:.0f}ms gpu)")
    X = np.concatenate(vecs, axis=0)
    print(f"  embedded {X.shape} in {time.monotonic()-t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.tag}.npz"
    meta = df.iloc[kept] if len(kept) != len(df) else df
    np.savez_compressed(
        dest,
        X=X,
        owner=np.array(owner, dtype="int32"),
        episode_hash=df.episode_hash.to_numpy().astype(str),
        task=df.task.to_numpy().astype(str),
        lab=df.lab.to_numpy().astype(str),
        scene=df.scene.fillna("").to_numpy().astype(str),
        operator=df.operator.fillna("").to_numpy().astype(str),
        num_frames=df.num_frames.to_numpy(),
    )
    print(f"\nwrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
