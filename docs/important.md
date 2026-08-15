# IMPORTANT — the non-negotiables

Live sheet for the day. Everything here is a **must**, not a nice-to-have.
Logistics and the project shortlist live in `docs/event-brief.md`; this file is
only the things that lose the day if you get them wrong.

---

## 1. The rules, as announced

| | |
|---|---|
| **Team size** | max **4** |
| **Stack** | anything |
| **Deliverable** | **repo link + ONE summary slide** |
| **Submission due** | **16:45** |
| **Demos** | **17:00** |
| **Awards** | **18:50** |

Judging is **two questions only**:

> **1. Does it run?** — "Working code is the baseline. If it doesn't execute, it doesn't count."
>
> **2. Is the method defensible?** — "Can you justify every design decision under questioning?"

Read that twice. There is **no** points for ambition, polish, or market size.
A small thing that executes and that you can defend line-by-line beats a big
thing that half-works. Scope to that, from hour one.

---

## 2. The three tracks

Pick one and say which one, out loud, in the first sentence of the demo. Judges
score against the track's **Deliver** row — that row is the acceptance criterion,
not a suggestion.

### 1 — The Curation Engine

| | |
|---|---|
| **Problem** | Dumping everything into training is inefficient and can hurt performance. |
| **Build** | A filtering pipeline or heuristic scoring pipeline that picks an optimal subset from the egoverse dataset. |
| **Deliver** | Keep/drop recommendations plus a validation report using proxy metrics. |

### 2 — Quantitative Diversity Measurement

| | |
|---|---|
| **Problem** | Teams rely on expensive, subjective LLM-as-a-judge pipelines to assess diversity. |
| **Build** | Showcase a way to score or represent diversity in a manner aside from text. |
| **Deliver** | A score that ranks two subsets, and a dashboard comparing them. |

### 3 — The Human Reward Model

| | |
|---|---|
| **Problem** | Success and failure demos are mixed together in human data. |
| **Build** | A classifier that flags success vs. failure/drop from video and annotations alone. |
| **Deliver** | Tagged episodes, a prevalence audit of failure demos, or a confidence meter over a video segment. |

**Note:** all three end in an *artifact you hand over* — a report, a score + a
dashboard, tagged episodes. None of them end in "a model that works". Build the
deliverable first and improve the method after; a great method with no deliverable
scores nothing against question 2.

---

## 3. The revised clock

`docs/event-brief.md` assumed demos at 18:00. They are at **17:00**. You have
**one hour less** than the plan says. Cut it from the build block, not from the
robustness block.

| Time | What | Hard rule |
|---|---|---|
| 09:00–10:00 | Recon the tables | Laptop stays closed. Find out what arms exist. |
| 10:00–10:30 | Team + scope | One sentence ending in a physical motion. |
| 10:30–13:30 | Ugly end-to-end | One object, one grasp, one command. No parallelising before this exists. |
| 13:30–15:00 | Improve | Only things that survive question 2. |
| **15:00–16:00** | **Robustness** | Run it **10 times**. Fix what fails. **Record a video of a successful run.** |
| **16:00–16:30** | **FREEZE** | No new code. None. Not "one small thing". |
| 16:30–16:45 | Submit | Repo link + slide. **Submit at 16:30**, not 16:44. |
| 16:45–17:00 | Rehearse | 90 seconds, out loud, twice. |
| 17:00 | Demos | |
| 17:00–18:50 | Stay in the room | Long gap before awards = judging + investors (Day by Day). Talk to Modal, ElevenLabs, Lazer. |

---

## 4. "Does it run?" — what that actually requires

- [ ] **A recorded video of a successful run**, made by 16:00, on the laptop AND
      on the phone. Robotics demos fail on the 4th take. The video is not
      cheating — it is what you show while the arm re-homes.
- [ ] **A path that works with no wifi.** The detector is a network call.
      `--local-vision` is the fallback — test it for real, on battery, once.
- [ ] **A path that works with no hardware.** `--robot mock` must still run the
      full loop. If the arm dies at 16:58, you still have a demo.
- [ ] **One command in the README that a judge can copy-paste** and have it run.
      If they can't run it in 60 seconds, "does it run" is their call, not yours.
- [ ] `./scripts/preflight.sh` green **before leaving the house** and again at 15:00.
- [ ] Pinned `requirements.txt`, `.env.example` present, **`.env` never committed**.
- [ ] Repo public (or the judges have access) — check the link in an incognito window.

**The rule:** at every point after 13:30 there must exist a committed state that
demos. Never be mid-refactor. Branch for anything risky.

---

## 5. "Is the method defensible?" — prepared answers

Defensible ≠ impressive. Defensible = *you know why, you know the tradeoff, and
you know what it fails at.* Have a number for each. Rehearse these:

| They ask | The answer |
|---|---|
| Why open-vocab detection (OWLv2) instead of training a detector? | No labelled data and no fine-tune fits in 10 hours. Text-conditioned queries turn a spoken noun straight into a box. Tradeoff: lower precision than a YOLO fine-tuned on a fixed set — we trade precision for an unbounded vocabulary. |
| Why Modal? | Serverless GPU (L4), weights cached in a Volume so no cold reload, scale-to-zero, nothing to operate. Tradeoff: it is a network dependency — hence the local CPU fallback. |
| Why is the threshold 0.25? | Chosen against `assets/test-table.jpg`; `scripts/verify_detector_local.py` asserts box centres against known positions. Below it we get furniture, above it we drop dim objects. |
| Why hand back pixel centres, not world coordinates? | That is the hardware-independence boundary. Pixel→world needs the arm's calibration and depth; it lives in the adapter (`docs/lerobot-cheatsheet.md` has both methods). |
| Why a rule-based parser, not an LLM? | Deterministic, no latency, no API in the hot path, legible failure mode. An LLM goes in when the grammar stops being closed. |
| What does it fail at? | Lighting, occlusion, and ambiguous labels — two red blocks means it picks one; that's why it says the confidence out loud. |
| What's next? | Calibration, closed-loop visual servoing, and teleop-recorded data if a leader arm exists. |

**Rules for answering:**
- Give the **number**, not the adjective. "88 ms p50 over 40 calls", not "fast".
- Volunteer what you did **not** do and why. Naming your own limits is what
  reads as defensible.
- **"We didn't measure that"** is a full, correct, winning answer. Bluffing is
  the only way to actually fail question 2.
- Never say "it just works."

---

## 6. The one slide

One slide. Design it at 15:30, not 16:40.

- [ ] Project name + **one sentence** of what it does, ending in a physical motion
- [ ] The pipeline diagram (steal the ASCII one from `spec.md`)
- [ ] **What is real vs mocked, stated plainly** — honesty here *is* question 2
- [ ] **One number** (latency, or success rate over 10 runs)
- [ ] Repo link, large, plus a QR code
- [ ] Team names (max 4)

---

## 7. The 90-second demo script

Rehearsed out loud, twice, before 17:00:

1. **The problem** — one sentence.
2. **The thing moves.** Show it in the first 20 seconds. Talk over the motion.
3. **Why it is hard** — the one technical claim you can defend under questioning.
4. **What's next** — one sentence, honest.

If the live run fails: keep talking, cut to the video, say *"that's the failure
mode I mentioned — lighting"*. A named, expected failure is evidence you
understand the system. Silence and a re-plug is not.

---

## 8. Hard don'ts

- ✗ Don't write code after 16:00.
- ✗ Don't add a dependency after 15:00.
- ✗ Don't demo anything that requires venue wifi to work.
- ✗ Don't start anything needing a fine-tune, a dataset, or a 3D print.
- ✗ Don't join a team by idea. Join by **hardware that is on the table**.
- ✗ Don't leave submission to the last 15 minutes.
- ✗ Don't commit `.env`.

---

## 9. Pack

Laptop charged · charger · **USB-C hub + USB-A adapter** (arms eat USB-A) ·
headphones · phone hotspot tested · phone charged (it's the backup camera).
