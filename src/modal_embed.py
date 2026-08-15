"""Frame embedding service on a Modal GPU.

The long pole of the diversity pipeline: turn sampled video frames into vectors.
Everything downstream (Vendi score, frame-selection strategies, dashboards) is numpy
over the cached output, so this runs once per episode set and never again.

    modal run src/modal_embed.py::warm          # cache weights into the Volume
    modal deploy src/modal_embed.py             # stable URL for the client

Design notes:
  * DINOv2 (not CLIP/SigLIP) on purpose — self-supervised visual features, no text
    tower involved anywhere. Track 2 says "aside from text"; using a text-aligned
    encoder would invite exactly the objection we are trying to avoid.
  * Embeddings are L2-normalised here so cosine similarity downstream is a dot
    product, and the Vendi kernel is well-conditioned.
"""

import io
import time

import modal

MODEL_ID = "facebook/dinov2-base"  # 768-d, ungated, ~350MB
CACHE_DIR = "/cache"

app = modal.App("egoverse-embed")
cache = modal.Volume.from_name("egoverse-embed-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "pillow==11.0.0",
        "numpy==2.1.3",
        "fastapi[standard]==0.115.6",
    )
    .env({"HF_HOME": CACHE_DIR, "TRANSFORMERS_VERBOSITY": "error"})
)


@app.function(image=image, volumes={CACHE_DIR: cache}, timeout=1800)
def warm():
    from transformers import AutoImageProcessor, AutoModel

    AutoImageProcessor.from_pretrained(MODEL_ID)
    AutoModel.from_pretrained(MODEL_ID)
    cache.commit()
    print(f"cached {MODEL_ID}")


@app.cls(
    image=image,
    gpu="l4",
    volumes={CACHE_DIR: cache},
    scaledown_window=300,
    timeout=900,
)
@modal.concurrent(max_inputs=2)
class Embedder:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to("cuda").eval().half()
        print("embedder ready")

    @modal.method()
    def embed(self, images: list[bytes], batch_size: int = 64) -> dict:
        """JPEG bytes -> L2-normalised CLS embeddings, shape (N, 768) as a list."""
        import numpy as np
        import torch
        from PIL import Image

        t0 = time.monotonic()
        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]

        out = []
        for i in range(0, len(pil), batch_size):
            chunk = pil[i : i + batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt").to("cuda")
            with torch.no_grad():
                # CLS token: the pooled global descriptor for the frame.
                feats = self.model(**inputs).last_hidden_state[:, 0]
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy())

        arr = np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype="float32")
        return {
            "embeddings": arr.astype("float32").tolist(),
            "dim": int(arr.shape[1]) if arr.size else 0,
            "n": int(arr.shape[0]),
            "model": MODEL_ID,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }


@app.local_entrypoint()
def main():
    """Smoke test with synthetic frames — proves the GPU path end to end."""
    import numpy as np
    from PIL import Image

    frames = []
    for shade in (30, 90, 150, 210):
        img = Image.new("RGB", (224, 224), (shade, shade // 2, 255 - shade))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frames.append(buf.getvalue())

    res = Embedder().embed.remote(frames)
    arr = np.array(res["embeddings"])
    print(f"model={res['model']} n={res['n']} dim={res['dim']} in {res['latency_ms']}ms")
    print("norms (should be ~1):", np.linalg.norm(arr, axis=1).round(4))
    sim = arr @ arr.T
    print("pairwise cosine:\n", sim.round(3))
