"""The loop: hear a command, find the thing, move, say what happened.

    # no hardware, no mic, no keys — proves the wiring
    python src/run_demo.py --robot mock --text "pick up the red block" --no-voice

    # full voice loop against a real arm
    python src/run_demo.py --robot so101 --voice

Flags exist so you can degrade gracefully: any of voice, vision, or robot can be
stubbed out independently when one of them is broken at 4pm.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import command as command_mod  # noqa: E402
import robot as robot_mod  # noqa: E402
from vision_client import (  # noqa: E402
    Camera,
    CpuOwlv2Vision,
    LocalColorVision,
    StillCamera,
    VisionClient,
    annotate,
    pick_best,
)

SNAP_DIR = pathlib.Path("out")


def say(text: str, use_voice: bool) -> None:
    print(f"ROBOT: {text}")
    if not use_voice:
        return
    try:
        import voice

        voice.speak(text)
    except Exception as exc:  # noqa: BLE001 - never let TTS kill the demo
        print(f"  (tts failed, continuing silently: {exc})")


def hear(use_voice: bool, fallback: str = "") -> str:
    if not use_voice:
        return fallback or input("YOU: ")
    import voice

    print("(listening...)")
    heard = voice.listen()
    print(f"YOU: {heard or '(nothing)'}")
    return heard


def handle(cmd, cam, vision, bot, args, state) -> None:
    if cmd.verb == "stop":
        bot.stop()
        say("Stopped.", args.voice)
        return
    if cmd.verb == "home":
        say(cmd.describe(), args.voice)
        bot.home()
        return
    if cmd.verb == "place":
        say(cmd.describe(), args.voice)
        bot.release()
        return

    target = cmd.target or state.get("last_target", "")
    if not target:
        say("What should I look for?", args.voice)
        return
    state["last_target"] = target

    if vision is None or cam is None:
        say(f"Vision is off, so I cannot find the {target}.", args.voice)
        return

    say(cmd.describe(), args.voice)

    frame = cam.grab()
    labels = [target] + [l for l in args.extra_labels.split(",") if l.strip()]
    dets, meta = vision.detect(frame, labels)
    print(
        f"  {len(dets)} detection(s) in {meta['round_trip_ms']:.0f} ms "
        f"(gpu {meta['gpu_ms']:.0f} ms) frame {meta['width']}x{meta['height']}"
    )

    matching = [d for d in dets if d.label == target]
    best = pick_best(matching or dets, cmd.modifiers, meta["width"] or 640)

    if best is None:
        say(f"I do not see a {target}.", args.voice)
        return

    print(f"  chose {best.label} score={best.score:.2f} center={best.center}")

    if args.save_frames:
        SNAP_DIR.mkdir(exist_ok=True)
        # Millisecond precision: whole-second names collide during rapid takes and
        # you lose the frame that showed the failure you are trying to debug.
        path = SNAP_DIR / f"frame-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.jpg"
        path.write_bytes(annotate(frame, dets, best))
        print(f"  wrote {path}")

    frame_size = (meta["width"] or 640, meta["height"] or 480)
    try:
        if cmd.verb == "point":
            bot.point_at(best.center, frame_size)
            say(f"There it is. Confidence {int(best.score * 100)} percent.", args.voice)
        else:
            ok = bot.pick_at(best.center, frame_size)
            say("Got it." if ok else "I could not grab it.", args.voice)
    except NotImplementedError as exc:
        print(f"  adapter cannot do that: {exc}")
        say("I can see it, but I cannot reach it yet.", args.voice)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="mock", choices=["mock", "viam", "so101"])
    ap.add_argument("--voice", dest="voice", action="store_true", help="mic + speaker")
    ap.add_argument("--no-voice", dest="voice", action="store_false", help="type instead")
    ap.set_defaults(voice=False)
    ap.add_argument("--text", default="", help="run one typed command and exit")
    ap.add_argument("--no-vision", action="store_true", help="skip the detector")
    ap.add_argument(
        "--local-vision",
        action="store_true",
        help="colour-blob detection on this laptop instead of the Modal GPU "
        "(no network, ~10ms, only understands colour words)",
    )
    ap.add_argument(
        "--cpu-vision",
        action="store_true",
        help="real OWLv2 on this laptop's CPU (no network, any noun, ~3-4s per frame)",
    )
    ap.add_argument("--camera", type=int, default=-1,
                    help="-1 = auto-pick a camera that is not returning black frames")
    ap.add_argument("--image", default="", help="use a still image instead of the webcam")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--extra-labels", default="", help="comma-separated distractor labels")
    ap.add_argument("--save-frames", action="store_true", default=True)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    vision = None
    cam = None
    if not args.no_vision:
        try:
            if args.local_vision:
                vision = LocalColorVision(threshold=args.threshold)
            elif args.cpu_vision:
                vision = CpuOwlv2Vision(threshold=args.threshold)
            else:
                vision = VisionClient(threshold=args.threshold)
            cam = StillCamera(args.image) if args.image else Camera(args.camera)
            print(f"vision: {vision.url}")
        except Exception as exc:  # noqa: BLE001
            print(f"vision unavailable ({exc}); falling back to local colour detection")
            try:
                vision = LocalColorVision(threshold=args.threshold)
                cam = StillCamera(args.image) if args.image else Camera(args.camera)
                print(f"vision: {vision.url}")
            except Exception as exc2:  # noqa: BLE001
                print(f"no vision at all ({exc2}); running blind")
                vision, cam = None, None

    bot = robot_mod.build(args.robot)
    bot.connect()

    state: dict[str, str] = {}
    say("Ready.", args.voice)

    try:
        if args.text:
            handle(command_mod.parse(args.text), cam, vision, bot, args, state)
            return
        while True:
            heard = hear(args.voice, "")
            if not heard:
                continue
            if heard.strip().lower() in {"quit", "exit", "q"}:
                break
            cmd = command_mod.parse(heard)
            if not cmd:
                say("I did not catch a command.", args.voice)
                continue
            handle(cmd, cam, vision, bot, args, state)
    except KeyboardInterrupt:
        print()
    finally:
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    main()
