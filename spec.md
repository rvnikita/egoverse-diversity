# Robotics Hackathon kit — architectural map

Prep for the **Robotics Hackathon** (Modal / Lazer / Day by Day Ventures / ElevenLabs),
Sat Aug 15 2026, 9:00–19:00 EDT, 233 Spring St floor 11, NYC.
Event logistics and the plan for the day: `docs/event-brief.md`.

## What this kit is

A **language-to-grasp brain** that works with whatever arm is on the table.
The bet: at a one-day hackathon you will not train a policy, but you *can* be the
person who makes an arm do what someone says out loud. So the kit owns the parts
that are hardware-independent and pre-verified, and leaves a thin adapter for the
arm you actually end up with.

```
  mic ──▶ ElevenLabs STT ──▶ command parser ──▶ label ("red block")
                                                  │
  webcam frame ─────────────────────────────────▶ │
                                                  ▼
                                    Modal GPU endpoint (OWLv2)
                                    open-vocab detection
                                                  │
                                     boxes + centers + scores
                                                  ▼
                                          RobotAdapter.pick()
                                                  │
                                    ElevenLabs TTS ◀── spoken confirmation
```

Open-vocabulary detection is the load-bearing choice: no per-object training, no
labelled data, no fine-tune. You say a noun, you get pixel coordinates. That is
the whole difference between a demo and a day of debugging.

## Modules

| Path | Runs where | Role |
|---|---|---|
| `src/owlv2_core.py` | either | Model load + detection + the coordinate maths. No Modal or CUDA imports, so it is testable off-GPU. |
| `src/modal_vision.py` | Modal (GPU L4) | Wraps `owlv2_core` in a POST endpoint. Weights cached in a Modal Volume. |
| `src/vision_client.py` | laptop | HTTP client, webcam capture, and the two offline detector fallbacks. |
| `src/command.py` | laptop | Utterance → `Command(verb, target_label, modifiers)`. Rule-based, zero deps. |
| `src/voice.py` | laptop | ElevenLabs STT (`scribe_v1`) + TTS (`eleven_flash_v2_5`), plus mic record / speaker playback. |
| `src/robot.py` | laptop | `RobotAdapter` protocol + `MockRobot`, `LeRobotSO101`, `ViamArm` implementations. |
| `src/run_demo.py` | laptop | The loop that wires all of the above together. |
| `scripts/preflight.sh` | laptop | One command that proves the whole chain is alive. |
| `scripts/verify_detector_local.py` | laptop | Runs real OWLv2 on CPU and asserts box centres against known object positions in `assets/test-table.jpg`. |

## Data flow contracts

**Detection request** (`POST /` on the deployed vision endpoint):
```json
{"image_b64": "<jpeg bytes, base64>", "labels": ["red block", "cup"], "threshold": 0.25}
```
**Detection response** — `boxes` are `[x0,y0,x1,y1]` in pixels, `center` is `[cx,cy]`,
sorted by score descending:
```json
{"detections": [{"label":"red block","score":0.41,"box":[..],"center":[320,240]}],
 "width": 640, "height": 480, "latency_ms": 88}
```

Pixel centers only. Turning a pixel into a world coordinate is the arm's job and
lives in the adapter — see `docs/lerobot-cheatsheet.md` for the two ways to do it
(depth lookup, or ray/table-plane intersection).

## Conventions

- Everything the GPU touches is in `src/modal_*.py`; everything else runs on the laptop.
- Secrets via `.env` (see `.env.example`), never committed. Modal-side secrets live in
  a Modal Secret named `elevenlabs`.
- `MockRobot` is the default adapter, so the full pipeline is demoable with no hardware.
- Adapters must be swappable by one flag: `--robot mock|so101|viam`.

## Deep dives

- `docs/important.md` — **the non-negotiables, and the first thing to read**:
  announced rules, the two judging questions, **the three tracks**, the revised
  clock (demos 17:00), prepared defensibility answers. Wins over this file on any
  conflict.
- `docs/event-brief.md` — logistics, hosts, the plan for the day, project shortlist
- `docs/modal-cheatsheet.md` — Modal commands and patterns, verified against client 1.5.4
- `docs/elevenlabs-cheatsheet.md` — STT/TTS/Agents API surface
- `docs/lerobot-cheatsheet.md` — SO-101 bring-up, teleop, pixel→world
- `docs/decisions.md` — why the kit is shaped this way
