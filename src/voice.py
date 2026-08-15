"""ElevenLabs speech in / speech out, plus mic capture and playback.

Verified against elevenlabs-python 2.64.0.

STT: `scribe_v1`         — send a wav/mp3 file object, get text back.
TTS: `eleven_flash_v2_5` — ~75 ms model latency, the right choice when a robot is
                           talking back to a person in real time.

Recording uses `sounddevice` if present, else macOS `ffmpeg -f avfoundation`, so this
works on a fresh machine with only ffmpeg installed. Playback prefers `afplay` on
macOS because it has no Python dependency and never fights the audio device.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import wave

STT_MODEL = "scribe_v1"
TTS_MODEL = "eleven_flash_v2_5"
# "Rachel" — a stock voice present on every account, so the demo works with a
# brand-new key. Swap for anything from `client.voices.search()`.
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"

SAMPLE_RATE = 16000


def _client():
    from elevenlabs.client import ElevenLabs

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (see .env.example)")
    return ElevenLabs(api_key=key)


# --------------------------------------------------------------------------- record


def record_wav(seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Fixed-length recording -> wav bytes."""
    try:
        import sounddevice as sd
    except ImportError:
        return _record_ffmpeg(seconds, sample_rate)

    frames = sd.rec(
        int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16"
    )
    sd.wait()
    return _wav_bytes(frames.tobytes(), sample_rate)


def record_until_silence(
    max_seconds: float = 12.0,
    silence_rms: int = 500,
    silence_tail: float = 0.9,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Record until the speaker stops. Needs `sounddevice`; falls back to fixed length.

    A hackathon room is loud, so `silence_rms` is deliberately generous — raise it if
    recording never terminates, lower it if it cuts you off mid-sentence.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        return record_wav(5.0, sample_rate)

    block = int(sample_rate * 0.1)
    chunks: list[bytes] = []
    quiet_for = 0.0
    spoke = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        for _ in range(int(max_seconds / 0.1)):
            data, _overflow = stream.read(block)
            samples = np.asarray(data, dtype=np.int16).reshape(-1)
            chunks.append(samples.tobytes())
            # `audioop` is gone in Python 3.13, so compute RMS directly.
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
            if rms > silence_rms:
                spoke, quiet_for = True, 0.0
            else:
                quiet_for += 0.1
                if spoke and quiet_for >= silence_tail:
                    break
    return _wav_bytes(b"".join(chunks), sample_rate)


def _record_ffmpeg(seconds: float, sample_rate: int) -> bytes:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("need either the `sounddevice` package or ffmpeg to record")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-i", ":0",
        "-t", str(seconds), "-ac", "1", "-ar", str(sample_rate), path,
    ]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg record failed: {out.stderr.decode()[:400]}")
    with open(path, "rb") as fh:
        data = fh.read()
    os.unlink(path)
    return data


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ------------------------------------------------------------------------------ stt


def transcribe(wav_bytes: bytes, language: str | None = "eng") -> str:
    buf = io.BytesIO(wav_bytes)
    buf.name = "audio.wav"  # the SDK infers the mime type from the filename
    result = _client().speech_to_text.convert(
        file=buf,
        model_id=STT_MODEL,
        language_code=language,
        tag_audio_events=False,
    )
    return (getattr(result, "text", "") or "").strip()


def listen(**kwargs) -> str:
    """Record until silence, then transcribe. Returns '' if nothing was said."""
    return transcribe(record_until_silence(**kwargs))


# ------------------------------------------------------------------------------ tts


def speak(text: str, voice_id: str = DEFAULT_VOICE, play_it: bool = True) -> bytes:
    """Synthesise and (by default) play. Returns the mp3 bytes."""
    chunks = _client().text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=TTS_MODEL,
        output_format="mp3_44100_128",
    )
    audio = b"".join(chunks) if not isinstance(chunks, bytes) else chunks
    if play_it:
        play(audio)
    return audio


def play(audio: bytes, suffix: str = ".mp3") -> None:
    player = shutil.which("afplay") or shutil.which("ffplay")
    if player:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            path = tmp.name
        cmd = [player, path] if player.endswith("afplay") else [
            player, "-nodisp", "-autoexit", "-loglevel", "quiet", path
        ]
        subprocess.run(cmd)
        os.unlink(path)
        return
    from elevenlabs import play as el_play  # needs mpv installed

    el_play(audio)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    mode = sys.argv[1] if len(sys.argv) > 1 else "tts"
    if mode == "tts":
        speak(" ".join(sys.argv[2:]) or "Robot online. Tell me what to pick up.")
    elif mode == "stt":
        print("speak now...")
        print("heard:", listen() or "(nothing)")
    elif mode == "loop":
        speak("Ready.")
        while True:
            said = listen()
            if not said:
                continue
            print("heard:", said)
            if said.lower().startswith(("quit", "exit", "stop listening")):
                speak("Bye.")
                break
            speak(f"You said: {said}")
