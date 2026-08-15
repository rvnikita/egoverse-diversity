"""Sample frames out of EgoVerse preview MP4s.

Preview MP4s are ~1 MB against ~120 MB for a single Zarr image chunk, so for anything
that only needs pixels this is ~44x cheaper. ffmpeg does the decoding; no OpenCV.
"""

from __future__ import annotations

import pathlib
import subprocess


def n_frames(path: str | pathlib.Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


def sample(path: str | pathlib.Path, k: int = 8, width: int = 224,
           skip_edges: float = 0.05) -> list[bytes]:
    """k evenly spaced JPEG frames, as bytes.

    `skip_edges` trims the first/last 5% of the clip. Egocentric recordings routinely
    open and close with the camera being put on or taken off, which is not the task and
    would otherwise pollute every episode embedding with the same "hands near face" frame.
    """
    path = pathlib.Path(path)
    total = n_frames(path)
    if total <= 0:
        return []

    lo = int(total * skip_edges)
    hi = max(lo + 1, int(total * (1 - skip_edges)))
    if hi - lo < k:
        lo, hi = 0, total
    idxs = [lo + round(i * (hi - lo - 1) / max(k - 1, 1)) for i in range(k)]

    # One ffmpeg pass: select the wanted frame indices, emit numbered JPEGs.
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    with_tmp = path.parent / f".frames_{path.stem}"
    with_tmp.mkdir(exist_ok=True)
    for old in with_tmp.glob("*.jpg"):
        old.unlink()

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path),
         "-vf", f"select='{expr}',scale={width}:-2",
         "-vsync", "0", "-q:v", "3", str(with_tmp / "f_%03d.jpg")],
        capture_output=True,
    )

    frames = [p.read_bytes() for p in sorted(with_tmp.glob("*.jpg"))]
    for p in with_tmp.glob("*.jpg"):
        p.unlink()
    with_tmp.rmdir()
    return frames
