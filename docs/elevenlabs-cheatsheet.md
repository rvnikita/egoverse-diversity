# ElevenLabs cheatsheet

Verified against **elevenlabs-python 2.64.0** (pinned in `requirements.txt`).
Jonathan Chang at the event is ElevenLabs — ask him for credits.

## Models worth knowing

| Model id | Use | Note |
|---|---|---|
| `eleven_flash_v2_5` | **TTS for a robot talking to a person** | ~75 ms model latency. Default in `src/voice.py`. |
| `eleven_multilingual_v2` | TTS, higher quality | Slower; fine for a pre-rendered narration track. |
| `eleven_v3` | TTS, most expressive | Best for a recorded demo video, not a live loop. |
| `scribe_v1` | **STT** | What `voice.transcribe()` uses. |

For a live robot, latency beats fidelity: a 400 ms pause before the arm acknowledges you
feels broken, and no judge notices the voice is slightly less warm.

## The two calls you need

```python
from elevenlabs.client import ElevenLabs
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

# TTS -> generator of audio chunks
audio = b"".join(client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",   # "Rachel", present on every account
    text="Picking up the red block.",
    model_id="eleven_flash_v2_5",
    output_format="mp3_44100_128",
))

# STT -> object with .text
buf = io.BytesIO(wav_bytes)
buf.name = "audio.wav"                 # the SDK infers mime type from the filename
result = client.speech_to_text.convert(file=buf, model_id="scribe_v1",
                                       language_code="eng", tag_audio_events=False)
print(result.text)
```

`buf.name` is the non-obvious one — without it the upload is rejected.

## In this repo

```bash
.venv/bin/python src/voice.py tts "Robot online."   # speak something
.venv/bin/python src/voice.py stt                   # record until silence, transcribe
.venv/bin/python src/voice.py loop                  # echo loop, to test the mic
```

`voice.listen()` records until you stop talking (`silence_rms=500` by default). A
hackathon room is loud — **raise `silence_rms` if recording never stops, lower it if it
cuts you off**. This is the single most likely thing to need tuning on site.

Playback prefers macOS `afplay` (no Python audio deps, never fights the device). The
`elevenlabs.play()` helper needs `mpv` installed, which is why it is only the fallback.

## Voices

```python
for v in client.voices.search().voices:
    print(v.voice_id, v.name)
```

Stock voice `21m00Tcm4TlvDq8ikWAM` ("Rachel") works on a brand-new key, which is why the
kit hardcodes it — a demo should not depend on a voice you cloned in your own account.

## Checking your quota

```python
sub = client.user.subscription.get()
print(sub.tier, sub.character_count, "/", sub.character_limit)
```

`scripts/preflight.sh` prints this. Watch it: a chatty demo loop burns characters faster
than you expect, and running dry mid-demo is silent — TTS just fails.

## Agents Platform (only if you have spare time)

`client.conversational_ai.*` gives you a full duplex voice agent with tool calling —
the agent can call *your* function, which could be `pick_at()`. That is a genuinely
impressive demo, but it needs a public webhook for tools, so it is a stretch goal after
the arm already moves. The STT→parse→TTS loop in this kit is the safe version of it.

## Gotchas

- `text_to_speech.convert()` returns a **generator**, not bytes. Join it.
- Errors surface as exceptions from the SDK, not error fields — wrap TTS in try/except so
  a quota failure never kills the demo loop (`run_demo.py` already does).
- STT wants a real audio container (wav/mp3), not raw PCM.
- Long text is chunked and billed by character; keep spoken confirmations short. This is
  also better UX — "Got it." beats a sentence.
