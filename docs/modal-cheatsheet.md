# Modal cheatsheet

Verified against **modal client 1.5.4** (the version pinned in this repo). Modal renamed
several decorators in 1.x, so older blog posts will mislead you.

## Setup

```bash
.venv/bin/modal setup            # opens a browser, writes ~/.modal.toml
.venv/bin/modal profile current  # confirm which workspace you are in
```

Free tier includes monthly compute credits — enough for a day of L4 inference. Ask the
Modal table for a hackathon bump anyway.

## The three commands you will actually use

```bash
modal run src/modal_vision.py                    # run the local_entrypoint, tear down after
modal deploy src/modal_vision.py                 # persistent app + a stable URL
modal app logs robohack-vision                   # tail logs
```

`modal run` is for iterating; `modal deploy` is what you want before a demo, because the
URL survives your laptop closing.

Other useful ones:

```bash
modal app list                    # what is deployed
modal app stop robohack-vision    # stop paying for a warm container
modal volume ls robohack-hf-cache # what is in the weights cache
modal shell src/modal_vision.py   # interactive shell in the container image
```

## API shape in 1.x

| Old (blog posts) | Current |
|---|---|
| `@stub.function()` | `@app.function()` |
| `@modal.web_endpoint()` | `@modal.fastapi_endpoint()` |
| `keep_warm=1` | `min_containers=1` |
| `container_idle_timeout=` | `scaledown_window=` |
| `allow_concurrent_inputs=N` | `@modal.concurrent(max_inputs=N)` |
| `.pip_install()` | `.pip_install()` or the faster `.uv_pip_install()` |

`@modal.concurrent` goes **on the class**, under `@app.cls()` — not on individual methods.

## The pattern this repo uses

```python
app = modal.App("robohack-vision")
cache = modal.Volume.from_name("robohack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.5.1", "transformers==4.46.3")
    .env({"HF_HOME": "/cache"})       # so downloads land in the Volume
)

@app.cls(image=image, gpu="l4", volumes={"/cache": cache}, scaledown_window=300)
@modal.concurrent(max_inputs=4)
class Detector:
    @modal.enter()                     # runs once per container, not per request
    def load(self): ...

    @modal.method()
    def detect(self, ...): ...

    @modal.fastapi_endpoint(method="POST", label="robohack-detect")
    def web(self, payload: dict): ...
```

Three things that matter for a hackathon:

- **`@modal.enter()`** loads the model once per container. Put it in the request handler
  by mistake and every call pays the load cost.
- **Volume + `HF_HOME`** means weights download once, ever. Run `download_weights` the
  night before; cold starts then take seconds.
- **`scaledown_window=300`** keeps a container warm for 5 minutes after the last call.
  Demos have gaps. A cold start in front of a judge reads as "broken".
- `label="..."` gives a **stable URL** across redeploys. Without it the URL changes and
  your `.env` goes stale mid-demo.

## Before the demo

```bash
# 1. cache the weights (once, the night before)
modal run src/modal_vision.py::download_weights

# 2. deploy and copy the printed URL into .env as VISION_URL
modal deploy src/modal_vision.py

# 3. right before demoing, pin one container warm so latency is flat
#    (edit scaledown_window / add min_containers=1, redeploy)
```

`min_containers=1` burns credits continuously — turn it on 20 minutes before demos, not
at 10am.

## GPU choice

`t4` < `l4` < `a10g` < `a100` < `h100`. For a ~1.5 GB detector at a few requests per
second, **`l4` is the right answer** — cheap, plentiful, and never queues. Reach for
`a10g`/`a100` only if you are serving an actual VLA policy.

Latency budget for this kit, **measured 2026-08-15 on L4 from NYC**, not estimated:

| | warm | cold |
|---|---|---|
| GPU (`latency_ms`) | **~393 ms** | ~2.8 s (first call after `@modal.enter()`) |
| Network round trip | **~500 ms** | — |
| Total | **~900 ms** | 30 s+ (container pull) |

An earlier draft of this file guessed 60–120 ms on the GPU. That was wrong by 3–4×:
`owlv2-base-patch16-ensemble` runs at 960×960 and re-encodes the label text on every
call. Budget **one second per detection** and design the demo loop around it — one
detection per utterance is fine, per-frame tracking is not.

Cold start is the real risk: the first call after a deploy pulls a ~10 GB CUDA image and
can exceed a 300 s client timeout. Always fire one throwaway request before demoing.

On venue wifi the ~500 ms network half will get worse; that is why `--local-vision` exists.

## Gotchas

- **`include_source`**: functions get your local source by default in 1.x — but only the
  *entrypoint file*. `src/` is not a package, so `from owlv2_core import MODEL_ID` died
  with `ModuleNotFoundError` in the container. Fix, verified 2026-08-15:

  ```python
  image = (
      modal.Image.debian_slim(...)
      .uv_pip_install(...)
      .add_local_python_source("owlv2_core")   # <- last, always
  )
  ```

  Keep it **last** in the chain: `add_local_*` layers mount at runtime, so putting one
  above `uv_pip_install` invalidates the expensive build layer on every source edit.
  Confirm it worked by looking for `🔨 Created mount PythonPackage:owlv2_core` in the
  deploy output.
- Web endpoints return JSON automatically from a dict. Returning bytes needs a
  `fastapi.Response`.
- Volume writes need an explicit `volume.commit()` before the container exits.
- A `modal run` app is torn down when the command ends — your URL dies with it. Use
  `modal deploy` for anything you want to call from another process.
- The first `uv_pip_install` build takes a few minutes; it is cached after that. Build it
  the night before, not at the venue.
