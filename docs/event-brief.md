# Robotics Hackathon — brief

## Logistics

| | |
|---|---|
| **What** | Robotics Hackathon (the hosts also call it "Day Shift") |
| **When** | Saturday 15 August 2026, **09:00 – 19:00 EDT** (10 hours, single day) |
| **Where** | **233 Spring St, floor 11**, New York, NY 10013 (Hudson Square, W of 6th Ave) |
| **Status** | Registration **approved** 14 Aug 17:49 UTC |
| **Ticket** | https://luma.com/e/ticket/evt-vsmrsPGFYTi7lBC?pk=g-CbPGw1IgoP3ipS0 |
| **Event page** | https://luma.com/caxc4ftw |
| **Size** | ~149 registered |
| **Format** | In person only. No livestream, no remote track. Hardware on the tables. |

Nearest subway: Spring St (C/E) or Houston St (1). It is a ~5 min walk from either.

## Who is running it

| Person | Role | Note |
|---|---|---|
| **Mark Grinev** | primary host, "Building Mecka" | [@mark_grinev](https://x.com/mark_grinev) · the Luma owner |
| **Jonathan Chang** | ElevenLabs | **approved your registration ~11 min after you emailed him** — thank him in person |
| **Matt Zebert** | GTM @ Modal | |
| **Monishee Matin** | Modal | |
| **Garrett Skrovina** | Growth @ Lazer | ex-CoinDesk, Roc Nation |

Partners: **Modal** (serverless GPU), **ElevenLabs** (voice), **Lazer**, **Day by Day Ventures** (VC — so there are likely investors in the room).

Saying hello to Jonathan early is the single highest-value social move of the day: he
already did you a favour, and ElevenLabs credits usually come from that table.

## What the hosts said

> "Robotics is where software was in the '90s. The tools are just starting to work.
> The hardware is finally cheap enough. The problems are wide open."

> "You don't need a robotics background. You don't need a team. You don't need a
> polished idea. Come hang out. Meet people. Break things."

No published agenda, tracks, judging criteria, or prizes. Expect that to be announced
in the room. **Plan for a demo, not a deck** — a one-day in-person hardware event
almost always ends in live demos, and anything requiring a training run will not finish.

## Your angle

You are going solo and offered to be an extra pair of hands. That is a strong position
if you arrive with a *capability*, not a project. The capability you have:

- Two weeks ago at the Viam hackathon you got a UFACTORY Lite 6 to do
  camera → colour detection → depth → world transform → motion-planned grasp.
  **You have already debugged a real perception-to-grasp pipeline.** Most people in
  that room will not have.
- This kit generalises that into something arm-agnostic: say a noun out loud, get
  pixel coordinates of that object, hand them to whatever adapter fits the hardware.

So the pitch when you join a team is concrete: *"I can give us open-vocabulary object
detection on a GPU and a voice interface, both already deployed and tested. I need
whoever knows the arm to tell me how to turn a pixel into a motion."*

## The plan for the day

> **Superseded by `docs/important.md`.** The hosts announced submission at 16:45,
> demos at **17:00**, awards 18:50 — an hour earlier than the schedule below.
> Use the clock in `important.md`; the notes here still hold for *what* to do in
> each block.

**Before you leave the house**
- `./scripts/preflight.sh` → all green (see README)
- Charge laptop, pack **USB-C hub, USB-A adapter, charger, headphones**. Robot arms
  eat USB-A ports and venue tables never have enough.
- Phone hotspot ready. Venue wifi at a 149-person hardware event will be bad, and your
  detector is a network call. `--local-vision` is the offline fallback; test it once.

**09:00–10:00 — arrive, do not open the laptop yet**
Find out what hardware is actually on the tables. That decides everything. Ask:
what arms, how many, is there a depth camera, is there a leader arm for teleop.
Say hi to Jonathan (ElevenLabs) and the Modal table; ask both for credits.

**10:00–11:00 — team + scope**
Pick a team by hardware, not by idea. Then cut the idea down to one sentence that
ends in a physical motion. Anything that needs a trained policy is out of scope for
10 hours unless the team already has a dataset.

**11:00–15:00 — build**
Get the boring thing working first: one object, one grasp, one voice command,
end-to-end, ugly. Then improve. Do not parallelise before that exists.

**15:00–17:00 — make it robust**
Run it 10 times. Fix what fails. Robotics demos fail on the 4th take, in front of
judges, because of lighting, a moved table, or a dead USB cable.

**17:00–18:00 — freeze and rehearse**
Stop building. Record a video of a successful run as insurance. Rehearse the 90-second
story: problem → the one thing we made move → why it is hard → what is next.

**18:00–19:00 — demos**

## Project shortlist

Ordered by "can this actually be finished in 10 hours".

1. **Voice-commanded pick-and-place** (what this kit does). "Pick up the red block on
   the left." Hits both Modal and ElevenLabs, and the failure modes are known.
   Demo-safe because pointing still looks good even if grasping fails.
2. **The arm that explains itself.** Same pipeline, but ElevenLabs narrates its
   reasoning out loud — "I see two red blocks, taking the left one, confidence 41%."
   Judges consistently reward legibility, and it turns a detection log into theatre.
   Cheap to add on top of #1.
3. **Teach-by-showing.** Point the camera at an object, say "this is a widget", and it
   remembers the phrase → uses open-vocab detection to find more of them. Feels like
   learning without any training run.
4. **Sorting by natural language.** "Put all the metal ones in the left bin." One
   detection pass, N grasps. Reads as autonomy, is really a for-loop.
5. **Remote hands.** Voice command from a phone → Modal → arm in the room. Only worth
   it if the team already has network control working.

Avoid: anything needing a fine-tune, anything needing a new 3D print, anything whose
demo requires the wifi to work.

## Questions to ask in the first hour

- What arms are here, and is anything already calibrated?
- Is there a depth camera, or only RGB? (Changes pixel→world completely.)
- Are there Modal / ElevenLabs credits, and where do I get them?
- What is the actual judging criterion, and how long is a demo slot?
- Is there a leader arm (teleop) — i.e. is recording a dataset even possible today?
