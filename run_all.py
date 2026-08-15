#!/usr/bin/env python3
"""Reproduce every number in the dashboard and on the slide, from the committed data.

    python run_all.py

No GPU. No AWS credentials. No network. Runs on a laptop in well under a minute, from
the 12 MB `results/episode_vectors.npz` that ships with this repo.

The GPU stage that produced those vectors is a separate, optional step — it needs Modal
and EgoVerse registry access:

    modal deploy src/modal_embed.py
    AWS_PROFILE=egoverse python scripts/build_cup_embeddings.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("scoring, coverage, falsification sweep, 3D projection, contact sheets",
     [PY, "scripts/analysis.py"]),
    ("dashboard", [PY, "scripts/build_dashboard.py"]),
]

# Not run by default:
#   scripts/llm_judge.py        needs OPENAI_API_KEY and spends ~$0.02; its output is
#                               committed at results/llm_judge.json so the dashboard
#                               renders the comparison without re-running it.
#   scripts/select_experiment.py  measures how each selector's subset performs as TRAINING
#                               data — Track 1/3 territory, not part of this submission.
#                               Its cached CSVs are what the "subsets scored" figure counts.


def main() -> int:
    if not (ROOT / "results" / "episode_vectors.npz").exists():
        print("results/episode_vectors.npz is missing — it should be committed. "
              "Rebuild with scripts/build_cup_embeddings.py (needs Modal + AWS).")
        return 1

    t0 = time.monotonic()
    for i, (name, cmd) in enumerate(STEPS, 1):
        print(f"\n{'='*72}\n[{i}/{len(STEPS)}] {name}\n{'='*72}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"\nFAILED at step {i}: {' '.join(cmd)}")
            return r.returncode

    out = ROOT / "results" / "dashboard.html"
    print(f"\n{'='*72}\nDone in {time.monotonic()-t0:.0f}s.\n\n  open {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
