#!/usr/bin/env bash
# Run this the moment you sit down at the venue. It tells you which of the four
# things you depend on are alive, in the order that matters.
#
#   ./scripts/preflight.sh
#
# Exit code is the number of failed checks, so it is usable in a loop.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
FAILS=0
pass() { printf "  \033[32mok\033[0m    %s\n" "$1"; }
fail() { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAILS=$((FAILS+1)); }
warn() { printf "  \033[33mwarn\033[0m  %s\n" "$1"; }

[ -f .env ] && set -a && . ./.env && set +a

echo "== 1. python + deps =="
if [ -x "$PY" ]; then pass "venv at .venv ($($PY --version 2>&1))"; else fail "no .venv — run: uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt"; fi
for mod in modal requests PIL numpy; do
  if $PY -c "import $mod" 2>/dev/null; then pass "import $mod"; else fail "import $mod"; fi
done
for mod in elevenlabs cv2 sounddevice scipy; do
  if $PY -c "import $mod" 2>/dev/null; then pass "import $mod"; else warn "import $mod (optional, degrades gracefully)"; fi
done

echo
echo "== 2. offline pipeline (must pass even with no wifi) =="
if $PY src/run_demo.py --robot mock --no-voice --local-vision \
     --image assets/test-table.jpg --text "pick up the red block on the left" \
     >/tmp/preflight_demo.log 2>&1 && grep -q "Got it" /tmp/preflight_demo.log; then
  pass "parse -> local detect -> mock grasp"
else
  fail "offline pipeline broken; see /tmp/preflight_demo.log"
fi

echo
echo "== 3. modal =="
if [ -f "$HOME/.modal.toml" ]; then
  pass "modal credentials present"
  if $PY -m modal profile current >/dev/null 2>&1; then
    pass "modal profile: $($PY -m modal profile current 2>/dev/null)"
  else
    fail "modal token rejected — rerun: .venv/bin/modal setup"
  fi
else
  fail "no ~/.modal.toml — run: .venv/bin/modal setup"
fi

if [ -n "${VISION_URL:-}" ]; then
  CODE=$(curl -s -o /tmp/preflight_vision.json -w '%{http_code}' -m 90 \
    -X POST "$VISION_URL" -H 'content-type: application/json' \
    --data-binary @<($PY - <<'EOF'
import base64, json, pathlib
print(json.dumps({"image_b64": base64.b64encode(pathlib.Path("assets/test-table.jpg").read_bytes()).decode(),
                  "labels": ["red block", "blue cube"], "threshold": 0.2}))
EOF
) 2>/dev/null)
  if [ "$CODE" = "200" ] && grep -q '"detections"' /tmp/preflight_vision.json; then
    N=$($PY -c "import json;print(len(json.load(open('/tmp/preflight_vision.json'))['detections']))" 2>/dev/null)
    pass "GPU detector responded (${N} detections)"
  else
    fail "detector at VISION_URL returned HTTP $CODE — see /tmp/preflight_vision.json"
  fi
else
  warn "VISION_URL unset — deploy with: .venv/bin/modal deploy src/modal_vision.py"
fi

echo
echo "== 4. elevenlabs =="
if [ -n "${ELEVENLABS_API_KEY:-}" ]; then
  if $PY - <<'EOF' 2>/dev/null
import os, sys
from elevenlabs.client import ElevenLabs
c = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
sub = c.user.subscription.get()
print(f"tier={getattr(sub,'tier','?')} chars_left="
      f"{getattr(sub,'character_limit',0) - getattr(sub,'character_count',0)}")
EOF
  then pass "elevenlabs key valid"; else fail "elevenlabs key rejected"; fi
else
  warn "ELEVENLABS_API_KEY unset — voice will be skipped (ask the ElevenLabs table for credits)"
fi

echo
echo "== 5. hardware =="
# Report every index, with brightness: a "working" camera that returns black frames
# looks identical to a broken one until you check. Warmup reads matter — the first
# grab after a cold open often fails.
CAMOUT=$($PY - 2>/dev/null <<'EOF'
import cv2
found = []
for i in range(3):
    cap = cv2.VideoCapture(i)
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(6):
            ok, frame = cap.read()
            if ok:
                break
    cap.release()
    if ok and frame is not None:
        found.append(f"{i}:{frame.shape[1]}x{frame.shape[0]}@{frame.mean():.0f}")
print(" ".join(found))
EOF
)
if [ -n "$CAMOUT" ]; then
  pass "cameras (index:WxH@brightness) $CAMOUT"
  if printf '%s ' "$CAMOUT" | grep -qE '@[0-4] '; then
    warn "one camera returns near-black frames (a virtual camera or a covered lens) — the kit auto-skips it; force with --camera N"
  fi
else
  warn "no readable camera (grant Terminal access in System Settings > Privacy & Security > Camera, or use --image)"
fi
ARMS=$(ls /dev/tty.usb* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')
if [ -n "$ARMS" ]; then pass "serial devices: $ARMS"; else warn "no USB serial device (no arm plugged in yet)"; fi

echo
if [ "$FAILS" -eq 0 ]; then
  printf "\033[32mall good\033[0m — %s\n" "$(date '+%H:%M:%S')"
else
  printf "\033[31m%d check(s) failed\033[0m\n" "$FAILS"
fi
exit "$FAILS"
