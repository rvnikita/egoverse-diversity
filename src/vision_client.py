"""Webcam capture + a thin client for the deployed Modal detector."""

from __future__ import annotations

import base64
import io
import os
import pathlib
import time
from dataclasses import dataclass

import requests

DEFAULT_TIMEOUT = 60  # first call may hit a cold container


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]
    center: tuple[float, float]

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return self.width * self.height


class VisionClient:
    def __init__(self, url: str | None = None, threshold: float = 0.25):
        self.url = (url or os.environ.get("VISION_URL", "")).rstrip("/")
        if not self.url:
            raise RuntimeError(
                "No detector URL. Deploy it with `modal deploy src/modal_vision.py`, "
                "then put the printed URL in .env as VISION_URL=..."
            )
        self.threshold = threshold

    def detect(
        self, image_bytes: bytes, labels: list[str] | str, threshold: float | None = None
    ) -> tuple[list[Detection], dict]:
        if isinstance(labels, str):
            labels = [labels]
        payload = {
            "image_b64": base64.b64encode(image_bytes).decode(),
            "labels": labels,
            "threshold": self.threshold if threshold is None else threshold,
        }
        t0 = time.monotonic()
        resp = requests.post(self.url, json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"detector error: {data['error']}")
        dets = [
            Detection(
                label=d["label"],
                score=d["score"],
                box=tuple(d["box"]),
                center=tuple(d["center"]),
            )
            for d in data.get("detections", [])
        ]
        meta = {
            "width": data.get("width"),
            "height": data.get("height"),
            "gpu_ms": data.get("latency_ms"),
            "round_trip_ms": round((time.monotonic() - t0) * 1000, 1),
        }
        return dets, meta


def pick_best(
    dets: list[Detection], modifiers: list[str] | None = None, frame_width: int = 640
) -> Detection | None:
    """Choose among detections. Score wins unless a spatial modifier was spoken."""
    if not dets:
        return None
    mods = modifiers or []
    if "left" in mods:
        return min(dets, key=lambda d: d.center[0])
    if "right" in mods:
        return max(dets, key=lambda d: d.center[0])
    if "near" in mods:
        # Nearer objects sit lower in a forward-facing frame and look bigger.
        return max(dets, key=lambda d: (d.center[1], d.area))
    if "far" in mods:
        return min(dets, key=lambda d: (d.center[1], -d.area))
    return max(dets, key=lambda d: d.score)


class StillCamera:
    """A fixed image pretending to be a camera. For testing the loop with no webcam."""

    def __init__(self, path: str):
        self._bytes = pathlib.Path(path).read_bytes()
        self.path = path

    def grab(self, quality: int = 85) -> bytes:  # noqa: ARG002 - signature parity
        return self._bytes

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class Camera:
    """Webcam frames as JPEG bytes. Falls back to ffmpeg if OpenCV is unavailable."""

    def __init__(self, index: int = -1, warmup_frames: int = 6, probe_limit: int = 4):
        """index=-1 auto-selects the first camera producing a non-black frame.

        macOS routinely exposes virtual cameras (screen-share tools, Continuity) at
        index 0 that open successfully and then hand back pure black. Defaulting to 0
        means your detector silently sees nothing, which is a miserable thing to debug
        at hour six. So probe, and prefer a camera that is actually looking at the room.
        """
        self.warmup_frames = warmup_frames
        self.probe_limit = probe_limit
        self._cap = None
        try:
            import cv2  # noqa: F401

            self._backend = "opencv"
        except ImportError:
            self._backend = "ffmpeg"
        self.index = index if index >= 0 else self._auto_index()

    def _auto_index(self) -> int:
        if self._backend != "opencv":
            return 0
        import cv2

        best, best_brightness = 0, -1.0
        for i in range(self.probe_limit):
            cap = cv2.VideoCapture(i)
            frame = None
            if cap.isOpened():
                for _ in range(self.warmup_frames):
                    ok, f = cap.read()
                    if ok:
                        frame = f
                        break
            cap.release()
            if frame is None:
                continue
            brightness = float(frame.mean())
            if brightness > 5.0:  # a real view of a room is never this dark
                return i
            if brightness > best_brightness:
                best, best_brightness = i, brightness
        return best

    def _open(self):
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"could not open camera {self.index}. On macOS, grant Terminal "
                    "camera access in System Settings > Privacy & Security > Camera."
                )
            # Auto-exposure needs a few frames or the first grab is black.
            for _ in range(self.warmup_frames):
                self._cap.read()
        return self._cap

    def grab(self, quality: int = 85) -> bytes:
        if self._backend == "opencv":
            import cv2

            cap = self._open()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("jpeg encode failed")
            return buf.tobytes()
        return self._grab_ffmpeg()

    def _grab_ffmpeg(self) -> bytes:
        import subprocess

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", "30", "-i", str(self.index),
            "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg", "pipe:1",
        ]
        out = subprocess.run(cmd, capture_output=True, timeout=30)
        if out.returncode != 0 or not out.stdout:
            raise RuntimeError(f"ffmpeg capture failed: {out.stderr.decode()[:400]}")
        return out.stdout

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CpuOwlv2Vision:
    """The real OWLv2 detector, on this laptop's CPU. Same nouns, ~30x slower.

    Middle rung of the fallback ladder: the Modal GPU handles any noun in ~100 ms, this
    handles any noun in ~3-4 s, and LocalColorVision handles only colours but in ~10 ms.
    Too slow for a live voice loop, fine for a deliberate single-shot demo, and it needs
    no network at all — which is the point.

    Drop-in for `VisionClient` — identical `detect()` signature.
    """

    url = "local:owlv2-cpu"

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self._loaded = None

    def _ensure(self):
        if self._loaded is None:
            import owlv2_core

            self._loaded = (owlv2_core, *owlv2_core.load(device="cpu"))
        return self._loaded

    def detect(
        self, image_bytes: bytes, labels: list[str] | str, threshold: float | None = None
    ) -> tuple[list[Detection], dict]:
        from PIL import Image

        if isinstance(labels, str):
            labels = [labels]
        core, processor, model = self._ensure()
        t0 = time.monotonic()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw = core.detect(
            processor, model, img, labels,
            self.threshold if threshold is None else threshold, device="cpu",
        )
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        dets = [
            Detection(label=d["label"], score=d["score"],
                      box=tuple(d["box"]), center=tuple(d["center"]))
            for d in raw
        ]
        return dets, {
            "width": img.width, "height": img.height,
            "gpu_ms": elapsed, "round_trip_ms": elapsed,
        }


COLORS: dict[str, tuple[int, int, int]] = {
    "red": (200, 40, 40),
    "orange": (230, 130, 30),
    "yellow": (225, 210, 60),
    "green": (60, 170, 80),
    "blue": (50, 90, 200),
    "purple": (130, 70, 180),
    "pink": (230, 130, 170),
    "black": (30, 30, 30),
    "white": (235, 235, 235),
}


class LocalColorVision:
    """Insurance policy: colour-blob detection on the laptop, no network, no GPU.

    Only understands labels containing a colour word ("red block", "the blue cube"),
    which covers most hackathon table props. Same trick that worked on the Viam arm:
    threshold in RGB, take the largest blob, return its bounding box.

    Drop-in for `VisionClient` — identical `detect()` signature.
    """

    url = "local:color"

    def __init__(self, threshold: float = 0.25, tolerance: int = 90, min_pixels: int = 300):
        self.threshold = threshold
        self.tolerance = tolerance
        self.min_pixels = min_pixels

    @staticmethod
    def _colour_for(label: str) -> tuple[str, tuple[int, int, int]] | None:
        for name, rgb in COLORS.items():
            if name in label.lower():
                return name, rgb
        return None

    @staticmethod
    def _blobs(mask, max_blobs: int = 8):
        """Connected components of a boolean mask, largest first.

        Uses scipy when available; otherwise falls back to a coarse grid clustering
        that is good enough to separate objects sitting apart on a table.
        """
        import numpy as np

        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None

        if ndimage is not None:
            labelled, n = ndimage.label(mask)
            sizes = ndimage.sum(mask, labelled, range(1, n + 1))
            order = np.argsort(sizes)[::-1][:max_blobs]
            for idx in order:
                ys, xs = np.nonzero(labelled == idx + 1)
                yield ys, xs
            return

        # Fallback: split the mask on columns/rows that contain no pixels at all.
        ys_all, xs_all = np.nonzero(mask)
        if xs_all.size == 0:
            return
        col_has = np.zeros(mask.shape[1], dtype=bool)
        col_has[np.unique(xs_all)] = True
        spans, start = [], None
        for x in range(mask.shape[1] + 1):
            filled = col_has[x] if x < mask.shape[1] else False
            if filled and start is None:
                start = x
            elif not filled and start is not None:
                spans.append((start, x - 1))
                start = None
        spans.sort(key=lambda s: -(s[1] - s[0]))
        for x0, x1 in spans[:max_blobs]:
            sel = (xs_all >= x0) & (xs_all <= x1)
            yield ys_all[sel], xs_all[sel]

    def detect(
        self, image_bytes: bytes, labels: list[str] | str, threshold: float | None = None
    ) -> tuple[list[Detection], dict]:
        import numpy as np
        from PIL import Image

        if isinstance(labels, str):
            labels = [labels]
        t0 = time.monotonic()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # float32, not int16: a squared channel difference reaches 65025 and would
        # silently overflow int16, producing negative sums and NaNs out of sqrt.
        arr = np.asarray(img, dtype=np.float32)

        dets: list[Detection] = []
        for label in labels:
            hit = self._colour_for(label)
            if hit is None:
                continue
            _name, rgb = hit
            dist = np.sqrt(((arr - np.array(rgb, dtype=np.float32)) ** 2).sum(axis=2))
            mask = dist < self.tolerance
            if int(mask.sum()) < self.min_pixels:
                continue
            # One box per blob. Merging every same-coloured pixel into a single box
            # puts the "center" on empty table between two objects, and makes the
            # left/right modifiers meaningless.
            for ys, xs in self._blobs(mask):
                count = xs.size
                if count < self.min_pixels:
                    continue
                score = min(0.99, count / (img.width * img.height) * 8)
                dets.append(
                    Detection(
                        label=label,
                        score=round(float(score), 3),
                        box=(float(xs.min()), float(ys.min()),
                             float(xs.max()), float(ys.max())),
                        center=(round(float(xs.mean()), 1), round(float(ys.mean()), 1)),
                    )
                )
        dets.sort(key=lambda d: -d.score)
        return dets, {
            "width": img.width,
            "height": img.height,
            "gpu_ms": 0.0,
            "round_trip_ms": round((time.monotonic() - t0) * 1000, 1),
        }


def annotate(image_bytes: bytes, dets: list[Detection], chosen: Detection | None = None) -> bytes:
    """Draw boxes on a frame — for sanity-checking coordinates and for the demo screen."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in dets:
        is_chosen = chosen is not None and d is chosen
        color = (0, 220, 0) if is_chosen else (255, 170, 0)
        draw.rectangle(d.box, outline=color, width=4 if is_chosen else 2)
        draw.text((d.box[0] + 4, max(d.box[1] - 12, 2)), f"{d.label} {d.score:.2f}", fill=color)
        cx, cy = d.center
        draw.line([cx - 8, cy, cx + 8, cy], fill=color, width=2)
        draw.line([cx, cy - 8, cx, cy + 8], fill=color, width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
