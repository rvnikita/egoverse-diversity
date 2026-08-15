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

- [x] **The deliverable is a static page** (`results/index.html`) — it cannot fail to render. `python run_all.py` in a visible terminal is the live proof, and it is theatre, not the demo.
      on the phone. Robotics demos fail on the 4th take. The video is not
      cheating — it is what you show while the arm re-homes.
- [x] **No wifi needed.** `results/index.html` is self-contained — inline SVG,
      inline base64 thumbnails, no CDN, no fonts, no network. Open it from disk.
- [x] **No GPU and no credentials needed.** `python run_all.py` regenerates every
      number from the committed 12 MB vector cache. Verified from a clean clone
      into a **fresh venv** (the earlier test reused a venv that already had
      pandas/sklearn and hid a missing-dependency bug).
- [ ] **One command in the README that a judge can copy-paste** and have it run.
      If they can't run it in 60 seconds, "does it run" is their call, not yours.
- [x] Fresh-venv clone test green at 15:20. Re-run it after the last push.
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
| **You select with k-means on DINOv2 and score with Vendi on the same DINOv2 space — isn't your subset winning by construction? And is 1.97 vs 1.71 even outside the noise?** | "We measured that. Over 300 random draws Vendi is 1.93 ± 0.14, so on Vendi alone our 1.97 is inside the noise — it is printed on the page. The separation is coverage: our 32 represent 59% of the corpus and the best random draw out of 300 reached 45%. And the score ranks subsets where no selector is involved at all: 32 clips from one operator on one day score 1.53, a day-spanning 32 scores 2.69, zero overlap in 50 paired trials." |
| Why DINOv2, not CLIP or SigLIP? | Self-supervised, no text tower anywhere. The track asks for diversity "aside from text"; a text-aligned encoder invites exactly the objection the track is written against. |
| Why the Vendi Score? | Interpretable units — "effectively N distinct episodes out of 32" — no reference distribution required, and exact duplicates provably cannot raise it. That last part makes it falsifiable, so we run the test and publish where it stops holding. |
| Why 32 uniform frames — isn't uniform lazy? | Measured, not lazy. Four CPU keyframe policies all lost to `np.linspace` (0.0% of the gap to a full-GPU oracle closed) and keyframe pooling lost to mean pooling (+0.319 vs +0.352). Round two is content-aware: contact events, not timestamps. |
| Why cluster cover and not farthest-point? | FPS scores far higher (3.31 vs 1.97) and is the worse subset — it collects outliers and represents 10% of the corpus against 59%. The highest score is not the best subset; that is on the page. |
| You gave gpt-4o one frame per episode and DINOv2 32 — rigged. | It is doing a harder job and the page says so. But its failures are on **identical** inputs: same image, temperature 0, five calls, 20-point spread; the rating rose 45→67 when 30% of clips were exact copies. Richer input cannot fix non-determinism on a fixed input. |
| Why not just stratify by recording day instead of embeddings? | Here you could — day-stratified scores 2.69. But operator and scene are empty for ~81% of the live registry (`docs/egodb-findings.md`); pixels are the only field always populated. On this pool the metadata *does* exist, which is what let us use it as ground truth to validate the pixel score. |
| Isn't the cost argument thin? gpt-4o is $0.0016 a call. | Yes, and the page says so — 10,000 scorings is $16. Dollars are the weakest form of it. What does not scale is 10,000 × 1.5 s of serial latency, and that the answers are not reproducible. |
| What does it fail at? | Near-duplicates: jitter a copy past cosine distance 0.0094 — below the 0.0424 separating genuinely distinct episodes — and the score inflates, up to +0.661. We publish the threshold. Also one task, one lab, one scene: constant background kills the "you are measuring rooms" objection by construction, but we have not shown transfer. |
| What's next? | The two limits we named: the same index across tasks and scenes, and content-aware frame selection. |

**Rules for answering:**
- Give the **number**, not the adjective.
- Volunteer what you did **not** do and why. Naming your own limits is what
  reads as defensible.
- **"We didn't measure that"** is a full, correct, winning answer. Bluffing is
  the only way to actually fail question 2.
- Never say "it just works."

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

Start `python run_all.py` in a visible terminal *before* speaking.

**Open, verbatim:**
> "We took Track 2, quantitative diversity — and while I talk, this terminal is
> re-deriving every number on that slide from scratch, on CPU, no network."

1. **The problem.** Today you ask an LLM. Same image, temperature zero, five
   calls: a 20-point spread. Replace 30% of the clips with exact copies and its
   diversity rating goes *up* — 45 to 67.
2. **What we built.** Embed every episode once; diversity becomes a lookup.
   $0.057 for all 912 episodes, every question after that free and identical.
3. **The result.** Point at the cloud. Orange is what our 32 reach that random
   misses — 194 episodes against 59. Coverage 59% vs 44%, and no random draw
   out of 300 got past 45%.
4. **The honest part.** On the Vendi score alone we are inside random's noise.
   It is printed on the page. That is exactly why coverage and the metadata
   checks are there.

**Close, verbatim:**
> "The index cost six cents, once. Every question after that is free, gives the
> same answer every time, and we published the exact distance at which someone
> could fool it — that's the difference between an opinion and a measurement."

If the live run fails: keep talking. The page is static and already open; the
terminal is theatre, not the deliverable.

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
