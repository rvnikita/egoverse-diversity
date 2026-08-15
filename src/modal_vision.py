"""Open-vocabulary object detection on a Modal GPU.

Deploy once, call from anywhere:

    modal deploy src/modal_vision.py

Then POST base64 JPEG + a list of text labels, get pixel boxes back. No training,
no labelled data — you name an object in English and it points at it.

    curl -X POST $URL -H 'content-type: application/json' \
      -d '{"image_b64":"...", "labels":["red block","gripper"]}'
"""

import base64
import io
import json
import time

import modal

from owlv2_core import MODEL_ID

CACHE_DIR = "/cache"

app = modal.App("robohack-vision")

# Weights live in a Volume so a cold container does not re-download 1.5 GB.
cache = modal.Volume.from_name("robohack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "pillow==11.0.0",
        "fastapi[standard]==0.115.6",
        "scipy==1.14.1",
    )
    .env({"HF_HOME": CACHE_DIR, "TRANSFORMERS_VERBOSITY": "error"})
    # `src/` is not a package, so Modal's automounting ships only this file. Without
    # this line the container dies on `from owlv2_core import MODEL_ID`. Keep it last:
    # add_local_* layers are mounted at runtime and would bust the build cache above.
    .add_local_python_source("owlv2_core")
)


@app.function(image=image, volumes={CACHE_DIR: cache}, timeout=1800)
def download_weights():
    """Run once, ahead of time, so hackathon-day cold starts are seconds not minutes."""
    import owlv2_core

    owlv2_core.load(device="cpu")  # downloads into HF_HOME, which is the Volume
    cache.commit()
    print(f"cached {MODEL_ID} into the volume")


@app.cls(
    image=image,
    gpu="l4",
    volumes={CACHE_DIR: cache},
    # Stay warm for 5 min after the last call — a demo has gaps between takes and
    # you do not want a cold start in front of a judge.
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
class Detector:
    @modal.enter()
    def load(self):
        import owlv2_core

        self.core = owlv2_core
        self.processor, self.model = owlv2_core.load(device="cuda")
        print("detector ready")

    @modal.method()
    def detect(self, image_bytes: bytes, labels: list[str], threshold: float = 0.25):
        from PIL import Image

        t0 = time.monotonic()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Coordinate maths lives in owlv2_core so it is testable off-GPU.
        dets = self.core.detect(
            self.processor, self.model, img, labels, threshold, device="cuda"
        )
        return {
            "detections": dets,
            "width": img.width,
            "height": img.height,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    @modal.fastapi_endpoint(method="POST", label="robohack-detect", docs=True)
    def web(self, payload: dict):
        """POST {"image_b64": str, "labels": [str], "threshold": float}"""
        labels = payload.get("labels") or ["object"]
        if isinstance(labels, str):
            labels = [labels]
        try:
            image_bytes = base64.b64decode(payload["image_b64"])
        except Exception as exc:  # noqa: BLE001 - surface a usable error to the client
            return {"error": f"could not decode image_b64: {exc}"}
        return self.detect.local(
            image_bytes, labels, float(payload.get("threshold", 0.25))
        )


@app.local_entrypoint()
def main(image_path: str = "", labels: str = "red block,cup,hand"):
    """Smoke test against a local image:

        modal run src/modal_vision.py --image-path frame.jpg --labels "red block,cup"
    """
    label_list = [s.strip() for s in labels.split(",") if s.strip()]
    if not image_path:
        # Synthesise a frame so the endpoint can be exercised with zero setup.
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 480), "white")
        ImageDraw.Draw(img).rectangle([260, 200, 380, 300], fill=(200, 30, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        print("no --image-path given; using a synthetic red square on white")
    else:
        with open(image_path, "rb") as fh:
            image_bytes = fh.read()

    result = Detector().detect.remote(image_bytes, label_list)
    print(json.dumps(result, indent=2))
