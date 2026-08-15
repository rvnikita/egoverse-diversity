# Decisions

## 2026-08-14 — Open-vocabulary detection instead of a trained policy
**Decision:** The perception layer is OWLv2 (`google/owlv2-base-patch16-ensemble`) doing
text-prompted detection, not a fine-tuned detector and not a VLA policy.
**Why:** A 10-hour single-day event cannot absorb data collection plus a training run, and
a policy also locks the work to one specific arm. Open-vocab detection needs zero labelled
data, zero training, and generalises to whatever props are on the table — you type a noun
and get pixels. Ruled out: fine-tuning YOLO (needs labels), SmolVLA/π0 (needs teleop
episodes and an arm that exists before you start writing code), CLIP-only (classifies but
does not localise, and we need coordinates).

## 2026-08-14 — Pass the true (h, w) to OWLv2 post-processing, and pin transformers
**Decision:** `post_process_object_detection(target_sizes=(height, width))` — the
documented API — plus a clamp of each box into the real frame. `transformers==4.46.3`
is pinned in the Modal image.
**Why:** OWLv2 pads images to a square, and there is a well-known bug where
post-processing ignored that padding, returning boxes shifted up and left on every
non-square frame (huggingface/transformers#27705). The usual workaround is to pass a
pre-squared `target_sizes`. **That workaround is unnecessary on 4.46.3**: the processor
now does `size = torch.max(img_h, img_w)` internally, with a comment saying why. Reading
the installed source settled it, and a naive-vs-workaround comparison confirmed both
produce byte-identical boxes.
Initially implemented with the pre-squared workaround on the assumption the bug was live;
corrected once measured. Kept the clamp — boxes are scaled against the padded square, so
one can legitimately extend past the real right/bottom edge.
**Consequence:** the version pin matters. On an older transformers the workaround would be
required again, which is why `scripts/verify_detector_local.py` exists and asserts against
known object positions rather than trusting the library.
Also note: this processor has **no** `post_process_grounded_object_detection` — that name
belongs to a different model family.

## 2026-08-14 — Detection maths in a shared module, verified on CPU
**Decision:** `src/owlv2_core.py` holds the load + detect logic; `modal_vision.py` imports
it. `scripts/verify_detector_local.py` runs the same code on the CPU and asserts box
centres against the known positions in `assets/test-table.jpg`.
**Why:** Coordinates are the one thing that must be right — a box off by 40 px sends the
arm at empty table, and it fails *silently*. Testing it should not require a GPU, a Modal
account, or network. Running it locally is what revealed the padding assumption above was
wrong. Real OWLv2 on CPU localises all three fixture objects to within 2 px.
**Trade-off:** CPU inference is ~4.4 s/frame vs ~100 ms on an L4 — useless for a live demo,
fine for a correctness check, and a genuine last-resort fallback if Modal is unreachable.

## 2026-08-14 — A laptop-side colour detector as an offline fallback
**Decision:** `LocalColorVision` duplicates the detection interface using PIL/numpy RGB
thresholding plus connected components, selected by `--local-vision` and used automatically
if the Modal endpoint is unreachable.
**Why:** The GPU detector is a network call, and venue wifi at a 149-person hardware event
is the least reliable component in the whole system. This is the same RGB-thresholding trick
that worked on the Viam arm two weeks ago, so it is known-good. It only understands colour
words, which covers most hackathon table props.
**Trade-off:** Two detection code paths to keep in sync. Accepted — the interface is three
fields wide, and a demo that survives a dead network is worth it.

## 2026-08-14 — One box per blob in the offline detector
**Decision:** Connected-component labelling (scipy, with a column-gap fallback) rather than
one bounding box over all matching pixels.
**Why:** A single box over every red pixel puts its centre on empty table *between* two red
objects, and makes "the one on the left" meaningless. Caught in testing: two red squares at
x≈120 and x≈560 produced one box spanning 83→597, centre (340, 355) — empty table.

## 2026-08-14 — float32 for the colour-distance computation
**Decision:** Cast the image to float32 before squared differences.
**Why:** With int16, a squared channel difference reaches 65025 and overflows, giving
negative sums and NaNs out of `sqrt`. Symptom was a white background matching "blue".

## 2026-08-14 — Camera index auto-selection by frame brightness
**Decision:** `Camera(index=-1)` (the default) probes indices 0..3 and picks the first
returning a frame with mean brightness > 5.
**Why:** On this machine index 0 is a virtual camera that opens successfully and returns
pure black. A detector fed black frames finds nothing and looks like a model bug — an
expensive thing to debug at hour six. Explicit `--camera N` still overrides.

## 2026-08-14 — Rule-based command parsing, no LLM
**Decision:** `src/command.py` is regex/keyword based.
**Why:** The only genuinely hard part is extracting the noun phrase, and the phrase is then
passed verbatim to an open-vocab detector that already handles "small red block". An LLM
call would add latency, an API key, and a failure mode to the most latency-sensitive part
of the loop, in exchange for handling compound commands we are unlikely to demo. The
`parse()` signature leaves room to add one later.

## 2026-08-14 — Adapters, with mock as the default
**Decision:** `RobotAdapter` protocol; `MockRobot` is the default and `--robot` switches
implementation. `LeRobotSO101.pick_at()` raises `NotImplementedError` rather than guessing.
**Why:** Which arm is on the table is unknown until arrival, so the hardware-specific part
must be the thin, swappable part. Mock-by-default means the full pipeline is demoable and
testable tonight with no hardware. Raising on `pick_at` is deliberate: SO-101 inverse
kinematics is build-specific, and a plausible-looking wrong implementation would drive a
gripper into a table. `point_at()` is safe and implemented.

## 2026-08-15 — Explicit `add_local_python_source` on the Modal image
**Decision:** `src/modal_vision.py`'s image chain ends with
`.add_local_python_source("owlv2_core")`.
**Why:** `src/` is not a package, so Modal 1.x automounting ships only the entrypoint file
and the container died at import time on `from owlv2_core import MODEL_ID` — after paying
the full ~3 GB torch/CUDA image build. Naming the sibling module explicitly is the fix.
It goes last in the chain because `add_local_*` layers are mounted at runtime; putting it
earlier would invalidate the expensive `uv_pip_install` layer above it on every source edit.

## 2026-08-15 — Budget one second per detection, not 100 ms
**Decision:** The demo loop assumes ~900 ms per detection round trip (GPU ~393 ms +
network ~500 ms from NYC), and does one detection per utterance rather than per frame.
**Why:** Measured against the deployed L4 endpoint, five consecutive warm calls. The
cheatsheet's original 60-120 ms figure was an unverified guess and is 3-4x optimistic:
`owlv2-base-patch16-ensemble` runs at 960x960 and re-encodes label text every call.
Not worth optimising today — one detection per spoken command is well inside budget, and
the obvious win (caching text embeddings for repeated labels) is a change to the riskiest
file in the kit on demo day. Revisit only if per-frame tracking becomes the demo.

## 2026-08-15 — Pivot to Track 2, and DINOv2 rather than any text-aligned encoder
**Decision:** The submission is Track 2 (Quantitative Diversity Measurement). The encoder
is `facebook/dinov2-base`; no text tower appears anywhere in the pipeline.
**Why:** The track asks to "score or represent diversity in a manner aside from text".
CLIP/SigLIP would technically work but invites the exact objection the track is written
against. DINOv2 is self-supervised, so the features cannot be accused of smuggling in a
language prior. Organizers' slide 4 says "Pick one track… narrow and working, not broad
and broken", so the Track 1 and Track 3 results we also have are presented as *validation
of the score*, not as separate submissions.

## 2026-08-15 — Uniform frame sampling, after trying to beat it and failing twice
**Decision:** 32 uniformly spaced frames per episode, mean-pooled.
**Why:** Two independent experiments say the clever alternatives lose.
(1) `scripts/cascade_experiment.py` over 45 episodes / 14,658 frames: cheap CPU
keyframe policies (thumbnail farthest-point, motion peaks, motion-gated FPS) vs
`np.linspace`, scored on coverage of a full-GPU oracle — uniform wins at k>=4, gap closed
0.0%. (2) `scripts/semantic_test.py`: keyframe pooling separation gap +0.319 vs mean
pooling +0.352; delta pooling -0.000, i.e. total failure.
**Consequence:** the cost claim ("0.28 s of GPU per episode") rests on a measured negative
result rather than on an assumption, which is the stronger position under questioning.

## 2026-08-15 — The audit classifier uses a different representation from the score
**Decision:** Selection and Vendi run on mean-of-32 vectors; the downstream classifier runs
on first-4 + last-4 frames concatenated.
**Why:** If the audit lived in the same space the selector maximises, "diverse subsets
train better/worse" would be circular by construction. Separating them makes the
downstream measurement external to the score. The specific choice is task-motivated —
"did the cup end up on the saucer" is a claim about the end state — and measured on the
full pool with 5-fold CV: mean-of-32 AUROC 0.789, last-4 0.899, first4+last4 0.906.
Held identical across all selectors, so it cannot manufacture a difference between them.

## 2026-08-15 — class_weight='balanced' is mandatory, not a tuning knob
**Decision:** `LogisticRegression(class_weight="balanced")` throughout.
**Why:** The pool is 82/18. Without it, logistic regression predicts the majority class on
small training sets and balanced accuracy comes out at exactly 0.500 — by construction,
not by measurement. The first run of the experiment reported chance performance for every
selector for precisely this reason, which looked like "the task is unlearnable" and was
actually a misconfigured baseline. Full-pool CV reaches AUROC 0.906, so the task is very
learnable.

## 2026-08-15 — `matched_random`, not stratified-random, as the class-balance control
**Decision:** The control draws the *same number of failures that farthest-point actually
realised* at each budget and seed, uniformly at random from the pool.
**Why:** Stratifying a random draw to the pool's own 18% ratio equals plain random in
expectation and controls for nothing. Since fps pulls 17.7/32 failures against random's
6.6, "your gain is just class rebalancing" is the first thing a judge will say. The
matched control answers it with a number: same failure count, AUROC 0.710 vs fps's 0.510,
so the effect is the identity of the episodes, not their labels.

## 2026-08-15 — Report the duplication threshold, not a pass/fail
**Decision:** `scripts/analysis.py` sweeps injected perturbation and reports where the
Vendi score stops being duplication-proof, next to the distance between genuinely distinct
real episodes.
**Why:** Exact clones provably cannot raise the score, and a binary "PASSES" hides that
near-duplicates *can*: inflation starts at an injected cosine distance of ~0.0094, well
below the 0.0424 separating real distinct episodes, and reaches +0.661 at 0.125. The score
is therefore gameable by submitting jittered copies. Naming the threshold is a stronger
answer than asserting robustness, and it is the honest one.

## 2026-08-15 — Ship pooled episode vectors, not the per-frame tensor
**Decision:** `results/episode_vectors.npz` (12 MB, 912 x 768 x 5 poolings) is committed;
`out/emb/cup.npz` (78 MB of per-frame embeddings) is gitignored.
**Why:** The deliverable is a repo link and judging question 1 is "does it run". A judge
must be able to clone and reproduce with no GPU, no AWS credentials and no network;
`python run_all.py` does that in 56 s. The per-frame tensor is regenerable from
`scripts/build_cup_embeddings.py` and too large to ship comfortably.

## 2026-08-15 — Drop the success/failure axis from the submission
**Decision:** The dashboard and README present diversity only. The success/failure
classifier, the failure-enrichment result and the diversity-vs-downstream-utility
correlation are removed from the pitch; `scripts/select_experiment.py` stays in the repo
as supporting evidence and is labelled as out of scope.
**Why:** Track 3 ("The Human Reward Model") is explicitly about separating success from
failure demos. Building the Track 2 pitch on a failure result meant arguing someone else's
track, and the organizers' slide 4 is unambiguous: "Pick one track… narrow and working,
not broad and broken." It also made the pitch defensive — leading with what the score gets
wrong instead of what it is for.
**Consequence:** external validation of the score had to be re-grounded on something purely
diversity-shaped. `objects` (12 prop combos) and the recording date (10 sessions) are
registry metadata the encoder never saw, so subset coverage of them is genuine outside
evidence. At k=16 the diverse subset covers 6.6/10 sessions vs random's 4.7.

## 2026-08-15 — Measure the LLM judge rather than assert it is worse
**Decision:** `scripts/llm_judge.py` runs gpt-4o on the identical contact sheets, 5x at
default temperature, 5x at temperature 0, plus a duplication test on the judge itself.
**Why:** The track names LLM-as-a-judge as the problem, so the comparison had to be
measured or dropped — asserting it would have been the weakest thing on the page. Results:
a 20-point spread across five identical calls at temperature 0, a ranking that flipped
between two full runs, and a diversity rating that went UP (45 → 67) when 30% of the clips
were replaced with exact copies of other clips in the same grid.
**Fairness:** the panel states plainly that the judge is doing a harder task (semantics
with no reference set, plus an explanation we cannot produce). The claim is narrowed to
what was measured: as a subset-ranking instrument it is not reproducible and not
falsifiable.

## 2026-08-15 — Frame the economics as index-vs-query, not as a unit-price win
**Decision:** The cost argument is "an embedding is a one-off index, a judge is a
per-query cost", with the measured pair: $0.057 to embed all 912 episodes, then 1,560
subset scorings at zero marginal cost, versus ~$13 and ~3 hours for the same sweep.
**Why:** Per-call the judge is cheap ($0.0016), so a unit-price comparison is unconvincing
and easy to attack. The real difference is structural and shows up the moment curation
becomes a loop rather than a single question — which is the actual use case.
