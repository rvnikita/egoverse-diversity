# Architectural map

**Track 2 — Quantitative Diversity Measurement.** EgoVerse Data Optimization &
Evaluation Suite, NYC, 15 Aug 2026. Team rvnikita.
The pitch and the numbers live in `README.md`; the day's non-negotiables and the
rehearsed answers live in `docs/important.md`.

## What this is

A diversity score for video subsets, computed from vision embeddings with **no text
encoder anywhere in the pipeline**. It ranks two subsets, returns the same answer every
time, and — because the expensive step happens once — every question after the first is
free.

```
  912 EgoVerse episodes (mp4 previews, R2)
            │
            │  32 uniformly spaced frames each        scripts/build_cup_embeddings.py
            ▼
     Modal L4 · DINOv2-base ──────────────────────────  src/modal_embed.py
     768-d, self-supervised, no text tower
     8.8 ms/frame · 257 s · $0.057 for the whole corpus
            │
            │  mean-pool → one vector per episode
            ▼
     results/episode_vectors.npz  (12 MB, COMMITTED)  ◀── the index. Built once.
            │
   ┌────────┴───────────────────────────────┐
   ▼                                        ▼
 Vendi Score                        cluster-cover selection
 "effectively N distinct"           k-means → nearest real episode
   └────────┬───────────────────────────────┘
            ▼
     scripts/analysis.py   score · coverage · falsification · projection
            ▼
     results/index.html    slide on screen one, evidence below
```

Everything below the index is numpy over a committed file, which is why
`python run_all.py` reproduces every published number in 1–2 minutes on a laptop with no
GPU, no credentials and no network.

## Modules

| Path | Role |
|---|---|
| `run_all.py` | The one command. Regenerates every number and both HTML outputs. |
| `src/diversity.py` | Vendi score, farthest-point, pooling strategies, duplication test. |
| `src/modal_embed.py` | DINOv2 embedding service on a Modal L4. The only GPU code. |
| `src/egodb.py`, `src/egos3.py` | EgoVerse registry (Postgres) and R2 object access. |
| `src/frames.py`, `src/cheap.py` | Frame sampling; CPU-only per-frame descriptors. |
| `scripts/build_cup_embeddings.py` | The GPU pass. Registry → mp4s → frames → vectors. |
| `scripts/analysis.py` | Every measurement the page reports. CPU only. |
| `scripts/slide_parts.py` | The slide markup, shared by both outputs so it exists once. |
| `scripts/build_slide.py` | `results/slide.html` — standalone slide. |
| `scripts/build_dashboard.py` | `results/index.html` — slide hero + evidence. |
| `scripts/llm_judge.py` | The gpt-4o baseline, measured on the same two subsets. |
| `scripts/cascade_experiment.py` | The keyframe-selection negative result. |
| `scripts/select_experiment.py` | Subsets as *training* data — Track 1/3, not submitted. |

## Data contracts

`results/episode_vectors.npz` — the committed index, and the only input the CPU path needs:

| key | shape | meaning |
|---|---|---|
| `V_all32`, `V_8`, `V_4` | (912, 768) | mean-pooled, L2-normalised, at three frame budgets |
| `V_head4`, `V_tail4` | (912, 768) | first / last four frames, for end-state work |
| `episode_hash`, `outcome`, `operator`, `scene` | (912,) | registry metadata |
| `gpu_seconds`, `k_frames` | scalar | what the index cost, so the page can state it |

`results/analysis.json` is the single source of truth for the page — the slide and the
dashboard read numbers from it rather than hard-coding them, so a rebuild cannot silently
disagree with a measurement.

## Conventions

- **No text encoder.** Anywhere. The track asks for diversity measured aside from text.
- **Numbers are read, never typed.** Every figure on the page comes from
  `results/*.json`; a stale claim is therefore a bug, not a typo.
- **The GPU is optional.** Only `build_cup_embeddings.py` and `modal_embed.py` need it.
- **Measured over assumed.** Where a choice was made against an alternative, the losing
  alternative is measured and recorded in `docs/decisions.md`.

## Deep dives

- `docs/important.md` — the two judging questions, the rehearsed answers, the 90 seconds
- `docs/egodb-findings.md` — what the registry actually contains, measured
- `docs/decisions.md` — why each choice, including the ones that failed
- `docs/modal-cheatsheet.md` — Modal commands and patterns

## Legacy: the pre-hackathon arm kit

`src/modal_vision.py`, `src/owlv2_core.py`, `src/vision_client.py`, `src/command.py`,
`src/voice.py`, `src/robot.py`, `src/run_demo.py` and `scripts/preflight.sh` are a
language-to-grasp kit (ElevenLabs STT → OWLv2 open-vocab detection on Modal → robot
adapter) built before the tracks were announced. It is unrelated to this submission and
kept only because it works; `docs/lerobot-cheatsheet.md` and
`docs/elevenlabs-cheatsheet.md` document it.
