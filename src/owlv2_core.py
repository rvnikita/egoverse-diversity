"""OWLv2 detection logic, shared by the Modal GPU app and the local CPU runner.

This module exists so the coordinate maths is written once. It is the riskiest code
in the kit — a subtly wrong box means the arm reaches for empty table — so it lives
somewhere it can be tested without a GPU (see scripts/verify_detector_local.py).

No Modal or CUDA imports here: it takes a loaded model/processor and a PIL image.
"""

from __future__ import annotations

MODEL_ID = "google/owlv2-base-patch16-ensemble"


def load(device: str = "cpu"):
    """Return (processor, model) on the requested device."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model


def detect(processor, model, img, labels: list[str], threshold: float = 0.25,
           device: str = "cpu") -> list[dict]:
    """Open-vocabulary detection. Returns dicts with pixel boxes in `img`'s frame.

    Boxes are [x0, y0, x1, y1] in pixels, `center` is [cx, cy], sorted by score desc.
    """
    import torch

    inputs = processor(text=[labels], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # OWLv2 resizes so the longest side fills a square canvas and pads the remainder
    # at the bottom and right. Historically post-processing ignored that padding, so
    # boxes came out shifted up and left on non-square frames (every webcam frame) and
    # the usual workaround was to pass a pre-squared target_size.
    # https://github.com/huggingface/transformers/issues/27705
    #
    # As of the pinned transformers 4.46.3 that is fixed INSIDE the processor, which
    # does `size = torch.max(img_h, img_w)` itself. So pass the true (height, width) —
    # the documented API. Do NOT pre-square it: that is a no-op here (max(s,s) == s)
    # and it misleads the next reader.
    # Verified end-to-end by scripts/verify_detector_local.py against known positions.
    target_sizes = torch.tensor([(img.height, img.width)], device=device)

    # NB: transformers 4.46 has no `post_process_grounded_object_detection` on this
    # processor — that name belongs to a different model family.
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    dets: list[dict] = []
    for score, label_idx, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x0, y0, x1, y1 = box.tolist()
        # Still clamp: boxes are scaled to the padded square, so one can legitimately
        # extend past the real frame's right/bottom edge into the pad region.
        x0 = round(min(max(x0, 0.0), img.width), 1)
        x1 = round(min(max(x1, 0.0), img.width), 1)
        y0 = round(min(max(y0, 0.0), img.height), 1)
        y1 = round(min(max(y1, 0.0), img.height), 1)
        if x1 <= x0 or y1 <= y0:
            continue  # entirely inside the padding — not a real detection
        dets.append(
            {
                "label": labels[int(label_idx)],
                "score": round(float(score), 3),
                "box": [x0, y0, x1, y1],
                "center": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
            }
        )
    dets.sort(key=lambda d: -d["score"])
    return dets
