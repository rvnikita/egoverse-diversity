"""Cheap, CPU-only per-frame descriptors — the first tier of the cascade.

The stated industry problem: frame-level selection is known to be what matters, and
known to be unaffordable, so everyone works at episode level instead. The cost is not
decoding — measured at ~0.08 s/episode — it is running a neural net on every frame.

So the cheap tier must avoid the GPU, not the decoder. These descriptors run on every
frame at full coverage, and are used only to *choose* which frames are worth the GPU.

Each returns one vector per frame; all are computed in a single decode pass.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np

GRID = 16  # thumbnails are GRID x GRID grayscale


def thumbnails(path: str | pathlib.Path, grid: int = GRID) -> np.ndarray:
    """Every frame as a tiny grayscale thumbnail, shape (T, grid*grid), values 0..1.

    One ffmpeg pass, raw gray output, no JPEG encode. This is the whole cheap tier's
    input: at 16x16 a frame is 256 bytes, so a 5,000-frame episode is 1.3 MB of RAM.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={grid}:{grid}", "-pix_fmt", "gray",
        "-f", "rawvideo", "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True).stdout
    n = len(out) // (grid * grid)
    if n == 0:
        return np.zeros((0, grid * grid), dtype="float32")
    arr = np.frombuffer(out[: n * grid * grid], dtype=np.uint8).reshape(n, grid * grid)
    return arr.astype("float32") / 255.0


def motion_energy(T: np.ndarray) -> np.ndarray:
    """Per-frame L2 change from the previous frame. High = something is happening."""
    if T.shape[0] < 2:
        return np.zeros(T.shape[0], dtype="float32")
    d = np.linalg.norm(np.diff(T, axis=0), axis=1)
    return np.concatenate([[d[0]], d]).astype("float32")


# ---------------------------------------------------------------- selection policies
#
# Each takes the cheap descriptors and returns the indices of k frames to spend GPU on.


def select_uniform(T: np.ndarray, k: int) -> np.ndarray:
    n = T.shape[0]
    if k >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, k).round().astype(int)


def select_motion_peaks(T: np.ndarray, k: int) -> np.ndarray:
    """The k frames with the most change since the previous frame."""
    e = motion_energy(T)
    if k >= len(e):
        return np.arange(len(e))
    return np.sort(np.argpartition(-e, k)[:k])


def select_cheap_fps(T: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Farthest-point sampling in *thumbnail* space.

    The key idea: run the expensive coverage objective in a 256-dim pixel space that
    costs nothing, then spend GPU only on the frames it picks. If thumbnail-space
    geometry approximates embedding-space geometry well enough, we get frame-level
    selection at episode-level price.
    """
    n = T.shape[0]
    if k >= n:
        return np.arange(n)
    X = T - T.mean(axis=1, keepdims=True)  # remove global brightness
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(nrm < 1e-8, 1.0, nrm)

    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(n))]
    d = 1.0 - X @ X[picked[0]]
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, 1.0 - X @ X[nxt])
    return np.sort(np.array(picked))


def select_motion_gated_fps(T: np.ndarray, k: int, keep: float = 0.5,
                            seed: int = 0) -> np.ndarray:
    """Drop the most static frames, then farthest-point over what remains.

    Motivated by the same source: fake/padded submissions are slow and repetitive, and
    idle filler is a documented failure mode. Gating on motion removes those frames
    before spending any coverage budget on them.
    """
    n = T.shape[0]
    if k >= n:
        return np.arange(n)
    e = motion_energy(T)
    keep_n = max(k, int(n * keep))
    cand = np.sort(np.argpartition(-e, keep_n - 1)[:keep_n])
    sel = select_cheap_fps(T[cand], k, seed=seed)
    return np.sort(cand[sel])


POLICIES = {
    "uniform": select_uniform,
    "motion_peaks": select_motion_peaks,
    "cheap_fps": select_cheap_fps,
    "motion_gated_fps": select_motion_gated_fps,
}
