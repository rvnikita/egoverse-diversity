# Diversity, measured without text

**Track 2 — Quantitative Diversity Measurement.**
EgoVerse Data Optimization & Evaluation Suite, NYC, 15 Aug 2026.

A diversity score for video subsets that uses **no text encoder anywhere**, ranks two
subsets, and — unlike an LLM-as-a-judge — is deterministic, costs 0.28 s of GPU per
episode, and comes with a measured account of what it fails at.

## The finding

> **A diversity score ranks subsets. It does not tell you which subset to train on —
> maximise it blindly and you get the worst one.**

One axis, **atypicality**, drives all three of the questions the organizers posed:

| | measured |
|---|---|
| It inflates the diversity score | farthest-point subset scores **3.31** vs random **1.71** (Vendi, k=32) |
| It concentrates failure demos | **19/32** failures in the top-32 by atypicality vs a **18.1%** base rate — **3.3×**, label-free |
| It makes the subset **worse** to train on | AUROC **0.510** (fps) vs **0.769** (cluster-cover), ρ = **−0.655**, p = 3.2e-09 |

And it is **not** class imbalance doing the work: `matched_random` draws the *same* 17.7
failures at random and scores **0.710** against fps's **0.510**. Same class balance,
different episodes, 0.2 AUROC apart.

## Run it

```bash
pip install -r requirements.txt
python run_all.py            # ~60 s, CPU only, no credentials, no network
open results/dashboard.html
```

That regenerates every number above and the dashboard, from the 12 MB
`results/episode_vectors.npz` committed in this repo.

The GPU stage that produced those vectors is optional and separate:

```bash
modal deploy src/modal_embed.py
AWS_PROFILE=egoverse python scripts/build_cup_embeddings.py   # 29,184 frames, 257 s on an L4
```

## Method, and why each piece

- **DINOv2-base**, not CLIP or SigLIP. Self-supervised visual features, **no text tower**
  in the pipeline at any point — the track asks for diversity measured "aside from text",
  and a text-aligned encoder would invite exactly the objection we are avoiding.
- **Vendi Score** — the exponential of the Shannon entropy of the eigenvalues of the
  normalised cosine kernel. Interpretable units ("effectively 3.3 distinct episodes out of
  32"), needs no reference distribution, and exact duplicates provably cannot raise it.
- **32 uniform frames per episode.** We tried to be cleverer and it did not work: four
  CPU keyframe-selection policies all lost to `np.linspace` (`out/cascade_big.log`), and
  keyframe pooling lost to mean pooling on semantic separation (+0.319 vs +0.352). Uniform
  is not laziness here, it is the measured winner.
- **The audit classifier lives in a different representation** from the score (first 4 +
  last 4 frames, vs mean-of-32). If the audit ran in the space the selector maximises, the
  comparison would be circular by construction.
- **Everything is paired.** All selectors see the same split, the same features, the same
  frozen hyperparameters, 20 seeds each. No choice in the classifier can manufacture a
  difference *between* selectors; it can only move all of them together.

## What it fails at

- **The score is gameable.** Exact duplicates cannot raise it (Δ −0.047), but
  near-duplicates can. Jitter a clone by a cosine distance of 0.0094 — well below the
  0.0424 that separates genuinely distinct real episodes — and the score starts to
  inflate, reaching **+0.661** at 0.125. We measured the threshold instead of claiming
  there wasn't one.
- **AUROC alone would have picked the wrong detector.** `dist_to_centroid` and
  `knn20_dist` score 0.771 and 0.784 — nearly identical — yet their top-32 contain 7 and
  19 failures respectively.
- **One task, one lab, one rig, one scene.** All 912 episodes are `cup_on_saucer` from
  `rl2` on `eva_bimanual` against `white_wall`. That kills the "you're just measuring
  backgrounds" objection by construction, but we cannot show the 3.3× transfers to other
  tasks — there are no outcome labels anywhere else to check against.
- **Session leakage.** Recording day is highly decodable from pixels and failures cluster
  by day. We report a session split (whole days held out) alongside the stratified one;
  absolute AUROC drops and the negative correlation gets *stronger*.

## Where the labels came from

The registry has **no usable success/failure ground truth**: `is_eval` is `False` on all
439,053 rows and `eval_success` is the dataclass default `True` on every non-null row.
The labels used here were recovered from **task-name suffixes** on one lab's episodes —
`cup_on_saucer_success` / `_failure` — giving 747/165. That corner is the only ground
truth in the dataset, and it is what makes the label-free claim checkable at all.
See `docs/egodb-findings.md` for the full audit of what is and isn't populated.

## Repo map

| Path | What |
|---|---|
| `run_all.py` | one command, reproduces everything on CPU |
| `results/dashboard.html` | **the deliverable** — two subsets scored and compared |
| `results/episode_vectors.npz` | 912 × 768 pooled DINOv2 vectors + labels (committed) |
| `src/diversity.py` | Vendi score, farthest-point, pooling, duplication test |
| `src/modal_embed.py` | DINOv2 embedding service on a Modal L4 |
| `scripts/select_experiment.py` | selectors × budgets × seeds → the comparison |
| `scripts/analysis.py` | falsification, label-free ranking, projection, thumbnails |
| `scripts/build_dashboard.py` | renders the dashboard (inline SVG, no JS libraries) |
| `scripts/cascade_experiment.py` | the keyframe-selection negative result |
| `docs/egodb-findings.md` | what the registry actually contains, measured |
| `docs/decisions.md` | why each choice, including the ones that failed |

An earlier pre-hackathon kit (voice → open-vocab detection → arm) also lives in `src/`
(`modal_vision.py`, `owlv2_core.py`, `robot.py`, `voice.py`, `run_demo.py`) and is
described in `spec.md`. It is not part of this submission.
