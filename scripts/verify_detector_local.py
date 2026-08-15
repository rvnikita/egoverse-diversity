"""Run the real OWLv2 detector on the CPU and check the box coordinates.

The padding fix in `owlv2_core.detect` is the riskiest line in the kit: get it wrong
and every box is shifted up and left, so the arm reaches for empty table. This proves
it against a fixture whose object positions are known exactly.

    .venv/bin/python scripts/verify_detector_local.py

Slow (tens of seconds per image on CPU) and downloads ~1.5 GB the first time. That is
fine — it runs once, off the critical path, and needs no GPU and no Modal account.

Doubles as a last-resort fallback at the venue: if Modal is unreachable entirely, the
same `owlv2_core.detect` runs on the laptop, just slowly.
"""

from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import owlv2_core  # noqa: E402

# assets/test-table.jpg is drawn programmatically, so we know exactly where things are.
# 640x480 — deliberately non-square, which is what exposes the padding bug.
FIXTURE = ROOT / "assets" / "test-table.jpg"
GROUND_TRUTH = [
    ("red block", (80, 300, 160, 380)),
    ("blue cube", (460, 120, 560, 220)),
    ("red block", (520, 330, 600, 410)),
]
TOLERANCE_PX = 30  # a detector box need not hug the shape, only find the right object


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def main() -> int:
    from PIL import Image

    img = Image.open(FIXTURE).convert("RGB")
    print(f"fixture {FIXTURE.name}  {img.width}x{img.height} (non-square: padding bug would show here)")

    print("loading owlv2 on cpu (first run downloads ~1.5 GB)...")
    t0 = time.monotonic()
    processor, model = owlv2_core.load(device="cpu")
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    labels = ["red block", "blue cube"]
    t0 = time.monotonic()
    dets = owlv2_core.detect(processor, model, img, labels, threshold=0.12, device="cpu")
    print(f"  inference {time.monotonic() - t0:.1f}s on cpu\n")

    print(f"{len(dets)} detection(s):")
    for d in dets:
        print(f"  {d['label']:11} score={d['score']:.3f} box={d['box']} center={d['center']}")

    print("\nmatching against known object positions:")
    failures = 0
    for label, truth in GROUND_TRUTH:
        tx, ty = centre(truth)
        candidates = [d for d in dets if d["label"] == label]
        best, best_dist = None, 1e9
        for d in candidates:
            dx = abs(d["center"][0] - tx)
            dy = abs(d["center"][1] - ty)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best, best_dist = d, dist
        if best is not None and best_dist <= TOLERANCE_PX:
            print(f"  ok    {label:11} expected centre ({tx:.0f},{ty:.0f})  "
                  f"got ({best['center'][0]:.0f},{best['center'][1]:.0f})  off by {best_dist:.1f}px")
        else:
            got = f"({best['center'][0]:.0f},{best['center'][1]:.0f}) off by {best_dist:.1f}px" \
                  if best else "nothing"
            print(f"  FAIL  {label:11} expected centre ({tx:.0f},{ty:.0f})  got {got}")
            failures += 1

    # The classic OWLv2 padding bug is systematic, not random: every centre lands up and
    # to the left, by a factor of the aspect ratio. Report the signed bias so a failure
    # is diagnosable — a large consistent negative dx/dy means target_sizes handling
    # regressed (e.g. an unpinned transformers), not that the model got worse.
    if dets:
        pairs = []
        for label, truth in GROUND_TRUTH:
            tx, ty = centre(truth)
            cands = [d for d in dets if d["label"] == label]
            if cands:
                nearest = min(cands, key=lambda d: (d["center"][0] - tx) ** 2 + (d["center"][1] - ty) ** 2)
                pairs.append((nearest["center"][0] - tx, nearest["center"][1] - ty))
        if pairs:
            bx = sum(p[0] for p in pairs) / len(pairs)
            by = sum(p[1] for p in pairs) / len(pairs)
            print(f"\nmean offset: dx={bx:+.1f}px dy={by:+.1f}px "
                  f"(a few px is normal; a large negative bias means target_sizes broke)")

    out = ROOT / "out" / "verify-owlv2-cpu.jpg"
    out.parent.mkdir(exist_ok=True)
    from vision_client import Detection, annotate

    annotate_dets = [
        Detection(label=d["label"], score=d["score"], box=tuple(d["box"]), center=tuple(d["center"]))
        for d in dets
    ]
    out.write_bytes(annotate(FIXTURE.read_bytes(), annotate_dets))
    print(f"wrote {out}")

    print("\n" + ("PASS — real OWLv2 boxes land on the objects" if not failures
                  else f"FAIL — {failures} object(s) not localised correctly"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
