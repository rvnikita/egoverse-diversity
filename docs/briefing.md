# Robotics Hackathon — background briefing

Source document for the audio briefing (`briefing/day-shift-briefing.mp3`). Also
suitable as a NotebookLM upload if you want a more conversational rendering.

Figures are from public reporting current to **August 2026**. Where something could not
be verified it says so — do not let a summariser smooth those gaps over.

---

## The connective thesis (the most useful thing here)

The primary host, **Mark Grinev**, gives his bio as "Building Mecka". **Mecka AI** is his
company, and it does not build robots — it builds **robot training data**.

So the line in the invite — *"Robotics is where software was in the '90s. The tools are
just starting to work. The hardware is finally cheap enough."* — is not atmosphere. It is
a company thesis stated as a slogan: **the bottleneck in robotics is data, not algorithms
and not hardware.**

That is also the field's consensus. Data scarcity is repeatedly identified as the primary
constraint holding robotics back from the success language models had, because the
physical world cannot be scraped the way the internet was.

**Consequence for the day:** a project that is clever about data — collecting it, or
sidestepping the need for it — speaks directly to the person who organised the event.
Open-vocabulary detection is a legitimate answer rather than a dodge: it delivers
capability with zero training data, because you name an object in English and get pixels.

---

## Modal — serverless GPU compute (name on the door)

- Write ordinary Python, decorate a function to request a GPU (e.g. `gpu="h100"`); Modal
  handles containerisation, scaling and provisioning. No YAML, no Kubernetes.
- Founded by **Erik Bernhardsson** (ex-Spotify; author of the Luigi and Annoy open-source
  projects) and **Akshat Bubna**.
- **$355M Series C, May 2026**, led by Redpoint Ventures and General Catalyst, at a
  **$4.65B valuation** — roughly quadrupling its prior mark. Reached unicorn status
  Sept 2025.
- Revenue **~$60M → ~$300M ARR in roughly eight months**.
- **More than a third of revenue comes from Sandboxes**: isolated environments where
  AI-generated code executes before touching production. Every coding agent needs
  somewhere safe to run what it just wrote, and Modal became that place.

### The cold-start engineering
- Runs the **gVisor** runtime (`runsc`) — a userspace kernel, so guest containers never
  use the host kernel directly.
- **Memory snapshots**, descended from **CRIU** (checkpoint/restore in userspace, a Linux
  technique first presented in 2011 for live-migrating containers). Rather than re-running
  initialisation, Modal restores the memory state captured *after* the model loaded.
- Reported benchmark: a Stable Diffusion function that cold-starts in **13 s restores in
  3.5 s** — about **2.5× faster**.
- Practical upshot: cache weights into a Volume ahead of time, and keep a container warm
  during demos.

---

## ElevenLabs — voice AI

- London, founded 2022. Contact at this event: **Jonathan Chang**.
- **$500M Series D, Feb 2026**, led by Sequoia at an **$11B valuation**; BlackRock, NVIDIA
  and Salesforce also participating. Tripled the $3.3B mark set after its Jan 2025
  Series C.
- Subsequently reported to be in early talks for a secondary tender at roughly **$22B**.
- **~$500M ARR as of April 2026**, up from ~$350M at end-2025.
- Used by **41% of the Fortune 500**. Customers include the Washington Post, TIME,
  HarperCollins, Paradox Interactive, Deutsche Telekom, Square and Revolut.
- Products: text-to-speech, speech-to-text (**Scribe**), dubbing, sound effects, music, and
  an **agents platform** where a voice agent can call your own functions.

### Practical notes
- For live interaction use **`eleven_flash_v2_5`** (~75 ms model inference). When a robot
  answers a person, latency beats fidelity — a 400 ms pause reads as broken, while a
  marginally less warm voice goes unnoticed.
- STT model is **`scribe_v1`**.
- The agents platform is the tempting robotics demo (agent tool = "move the arm"), but it
  needs a public webhook for tool calls, so treat it as a stretch goal.

---

## Mecka AI — robotics data (the host's company)

Not on the sponsor list, but arguably the most important company present.

- Collects **human motion data at scale**: ships iPhones and custom cameras to hundreds of
  thousands of contributors across **12 countries**, recording how people move, walk and
  manipulate objects by hand.
- Packages that first-person footage and sensor data into training corpora sold to robot
  builders. Positions itself as **"the data and deployment layer for physical AI"**.
- **~$60M Series A, June 2026**, led by Framework Ventures; **~$68M total** raised since
  founding. Targeting a **$100M annual run rate**.
- Acquired **Docula** to process motion data.
- Note the rhyme with NVIDIA's GR00T approach (20,000+ hours of human egocentric video):
  human video at scale is the bet the whole field is making.

---

## Lazer Technologies

A product and engineering studio with a crypto practice. **Garrett Skrovina** is Head of
Crypto; previously CoinDesk, Roc Nation and Paradigm Talent Agency. The robotics
connection is not evident from outside the company — worth simply asking.

## Day by Day Ventures — UNVERIFIED

Listed as a partner on the event page but has essentially no public footprint. Searches
surface **Day One Ventures** (Masha Bucher's firm) instead. **These may not be the same
entity and should not be treated as such.** If investors are in the room they are likely
from here. Ask rather than assume.

---

## The tech people will name-drop

The category is **VLA — Vision-Language-Action**: camera images + a language instruction +
the robot's current state go in; actions come out.

| Model | Origin | The idea | Relevance here |
|---|---|---|---|
| **π₀ (pi-zero)** | Physical Intelligence | Trains on the Open-X Embodiment dataset, then fine-tunes on carefully curated data | Argues post-training data *quality* is critical: low-quality teleoperation data contains human mistakes the robot faithfully imitates |
| **GR00T N1** | NVIDIA | 2B-param LLM. System 1 = diffusion policy at ~10 ms latency for low-level control; System 2 = LLM planner for task decomposition | Trained on 20,000+ hrs human egocentric video plus Isaac simulator synthetic data |
| **SmolVLA** | Hugging Face / LeRobot | 16× fewer parameters than OpenVLA, within ~5% on LIBERO (82–90% success) | Runs **15–30 Hz on a single RTX 4090** — the only one plausibly usable on hackathon hardware |

### Hardware
- Most likely the **SO-101** (or predecessor SO-100): 6-DOF, 3D-printed, from Hugging Face
  and TheRobotStudio. Roughly **$100–500** in parts. Feetech STS3215 servos.
- Driven through **LeRobot**, Hugging Face's open-source robotics library.
- Defining feature: **leader–follower teleoperation**. Physically move a leader arm, the
  follower copies, and that records a demonstration in `LeRobotDataset` format.

### The arithmetic that kills most training plans
Training ACT on ~50 episodes is a few hours on an A100 — *and the episodes must be
recorded first*. In a ten-hour day this only works if a team is recording **before noon**,
training on Modal while building everything else, and holding a scripted fallback for the
demo. Never bet the demo on a training run finishing.

---

## What it means for the day

1. **Data is the host's religion.** Be ready to say something intelligent about where your
   project's capability comes from.
2. **Infrastructure companies run hackathons to see unanticipated usage.** Modal used for
   something other than a chatbot, and ElevenLabs for something other than a narrator, is
   inherently interesting to both.
3. **There are investors in the room.** The 90-second story carries as much weight as the
   code: problem → the one thing you made move → why it's hard → what's next.
4. **Robotics demos fail on the fourth take**, in front of judges, from changed lighting or
   a bumped table. Run it ten times first; record a successful run as insurance.

---

## Sources

- Modal funding and revenue: [TechStartups](https://techstartups.com/2026/05/21/modal-labs-raises-355m-quadrupling-valuation-to-4-65b-as-ai-infrastructure-demand-surges/), [SiliconANGLE](https://siliconangle.com/2026/05/21/serverless-ai-infrastructure-startup-modal-labs-seals-355m-funding-round/), [General Catalyst](https://www.generalcatalyst.com/stories/our-investment-in-modal)
- Modal memory snapshots: [Modal blog](https://modal.com/blog/mem-snapshots), [Modal docs](https://modal.com/docs/guide/memory-snapshots)
- ElevenLabs: [TechCrunch](https://techcrunch.com/2026/02/04/elevenlabs-raises-500m-from-sequioia-at-a-11-billion-valuation/), [CNBC](https://www.cnbc.com/2026/02/04/nvidia-backed-ai-startup-elevenlabs-11-billion-valuation.html), [Sacra](https://sacra.com/c/elevenlabs/)
- Mecka AI: [Fortune](https://fortune.com/2026/06/01/mecka-ai-series-a-60-million-robotics-data-training/), [BetaKit](https://betakit.com/mecka-ai-acquires-docula-as-it-builds-the-data-layer-for-robotics/), [Upstarts Media](https://www.upstartsmedia.com/p/mecka-ai-robotics-data-startup)
- VLA models: [GR00T N1 paper](https://arxiv.org/pdf/2503.14734), [VLA architecture comparison](https://blog.pebblous.ai/report/vla-architecture-comparison/en/), [SmolVLA / LearnOpenCV](https://learnopencv.com/smolvla-lerobot-vision-language-action-model/)
- SO-101 and LeRobot: [Hugging Face SO-101 docs](https://huggingface.co/docs/lerobot/en/so101), [LeRobot GitHub](https://github.com/huggingface/lerobot)
