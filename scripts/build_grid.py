"""Build embeddings for the rl2 controlled-diversity grid.

The paper's §III-D subset: one lab collected cup_on_saucer and fold_clothes over a
fixed pool of demonstrators and scenes via a structured assignment matrix, expressly so
scene and demonstrator diversity can be varied independently. That makes it the only
place in EgoVerse where the TRUE diversity of a subset is known by construction — so it
is where a diversity metric can actually be falsified rather than argued about.

    AWS_PROFILE=egoverse .venv/bin/python scripts/build_grid.py --per-cell 8
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
CLIPS = ROOT / "out" / "mp4_grid"
TASKS = ("cup_on_saucer", "fold_clothes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=8, help="episodes per (task, scene)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--tag", default="grid")
    args = ap.parse_args()

    live = egodb.episodes()
    live = live[~live.is_deleted]
    g = live[(live.lab == "rl2") & live.task.isin(TASKS)]
    g = g[g.scene.str.fullmatch(r"\d+", na=False)]
    g = g[g.zarr_mp4_path.notna() & (g.num_frames > 30)]
    g = g[g.scene.astype(int) <= 16]  # 17 sits outside the designed grid

    print("== controlled grid ==")
    print(f"  {len(g)} candidate episodes | scenes {sorted(g.scene.unique(), key=int)}")
    print(f"  operators: {g.operator.nunique()}")

    # Stratified: equal episodes per (task, scene) so scene is not confounded with count.
    picks = []
    for t in TASKS:
        for s in sorted(g.scene.unique(), key=int):
            cell = g[(g.task == t) & (g.scene == s)]
            if cell.empty:
                continue
            picks.append(cell.sample(min(args.per_cell, len(cell)), random_state=0))
    import pandas as pd

    df = pd.concat(picks).reset_index(drop=True)
    print(f"  sampled {len(df)} episodes "
          f"({df.task.value_counts().to_dict()})")

    print(f"\n== downloading {len(df)} mp4s ==")
    t0 = time.monotonic()
    paths = egos3.fetch_many(df.zarr_mp4_path.tolist(), CLIPS, workers=16)
    got = {p.name: p for p in paths}
    print(f"  {len(paths)}/{len(df)} in {time.monotonic()-t0:.0f}s")

    print(f"\n== sampling {args.frames} frames/episode ==")
    all_frames: list[bytes] = []
    owner: list[int] = []
    for i, uri in enumerate(df.zarr_mp4_path):
        p = got.get(pathlib.Path(uri).name)
        if p is None:
            continue
        fr = frame_utils.sample(p, k=args.frames)
        if len(fr) < 2:
            continue
        all_frames.extend(fr)
        owner.extend([i] * len(fr))
    print(f"  {len(all_frames)} frames from {len(set(owner))} episodes")

    print("\n== embedding on Modal ==")
    import modal

    emb = modal.Cls.from_name("egoverse-embed", "Embedder")()
    vecs = []
    t0 = time.monotonic()
    for i in range(0, len(all_frames), 256):
        r = emb.embed.remote(all_frames[i : i + 256])
        vecs.append(np.array(r["embeddings"], dtype="float32"))
        print(f"  {min(i+256, len(all_frames))}/{len(all_frames)}")
    X = np.concatenate(vecs, axis=0)
    print(f"  {X.shape} in {time.monotonic()-t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.tag}.npz"
    np.savez_compressed(
        dest, X=X, owner=np.array(owner, dtype="int32"),
        episode_hash=df.episode_hash.to_numpy().astype(str),
        task=df.task.to_numpy().astype(str),
        scene=df.scene.to_numpy().astype(str),
        operator=df.operator.fillna("").to_numpy().astype(str),
    )
    print(f"\nwrote {dest} ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
