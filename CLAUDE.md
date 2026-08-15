# CLAUDE.md

Robotics hackathon kit (Modal / Lazer / Day by Day Ventures / ElevenLabs),
Sat Aug 15 2026. Architecture lives in `spec.md`; read it before touching code.

## Read `docs/important.md` first

**`docs/important.md` is the highest-priority document in this repo.** Read it at
the start of every session, before `spec.md` and before any code. It is the live
sheet for the day and everything in it is a **must**, not a nice-to-have. Where it
contradicts any other doc — including `spec.md`, `README.md`, or
`docs/event-brief.md` — **`docs/important.md` wins.**

What it holds:

| § | Contents |
|---|---|
| 1 | The announced rules and the **two judging questions**: *does it run?* and *is the method defensible?* |
| 2 | **The three tracks** — Curation Engine, Quantitative Diversity Measurement, Human Reward Model — with each one's Problem / Build / **Deliver** |
| 3 | The revised clock (demos at **17:00**, not 18:00) with hard per-block rules |
| 4 | The concrete checklist for "does it run?" — video, offline path, mock-hardware path, one copy-pasteable command |
| 5 | Prepared, rehearsed answers for "is the method defensible?" |
| 6–7 | The one slide, and the 90-second demo script |
| 8–9 | Hard don'ts, and the pack list |

How it changes what you do:

- **Every change is scored against the two judging questions in §1.** If a change
  doesn't help something run, or doesn't make the method more defensible, it is
  out of scope — say so rather than building it.
- **The track's *Deliver* row in §2 is the acceptance criterion.** Build the
  handed-over artifact (report, score + dashboard, tagged episodes) first; improve
  the method after.
- **Respect the freeze in §3.** No new code after 16:00, no new dependency after
  15:00. After 13:30 there must always be a committed state that demos — branch
  for anything risky, never leave the tree mid-refactor.
- **Never break the no-wifi and no-hardware paths** (§4). `--local-vision` and
  `--robot mock` must keep working through every change.
- **Keep it current.** When a rule, a time, or a track detail changes during the
  day, update `docs/important.md` in the same edit — a stale entry there is worse
  than no entry.

## Doc layout

Root is the map, `docs/` is the territory (see the global rule in
`~/.claude/CLAUDE.md`): `README.md` for humans, `spec.md` as the architectural
map, everything else under `docs/`, with non-obvious choices appended to
`docs/decisions.md` as they are made.
