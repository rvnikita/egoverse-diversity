"""Robot adapters — the only part of the kit that depends on which arm is on the table.

Everything upstream produces a pixel target. An adapter's job is to turn a pixel into
a motion. Three are provided:

  MockRobot   — verified. Prints what it would do. The demo runs end-to-end without hardware.
  ViamArm     — pattern lifted from a working Viam setup (UFACTORY Lite 6). Needs creds.
  LeRobotSO101 — UNVERIFIED without hardware. Structure and API calls are right, but the
                 calibration and the pixel->world mapping must be checked on the real arm.

Add a new arm by implementing `RobotAdapter`. Keep it to five methods.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class RobotAdapter(Protocol):
    name: str

    def connect(self) -> None: ...
    def home(self) -> None: ...
    def point_at(self, pixel: tuple[float, float], frame_size: tuple[int, int]) -> None: ...
    def pick_at(self, pixel: tuple[float, float], frame_size: tuple[int, int]) -> bool: ...
    def release(self) -> None: ...
    def stop(self) -> None: ...


# --------------------------------------------------------------------------- pixel->world


def pixel_to_table_xy(
    pixel: tuple[float, float],
    frame_size: tuple[int, int],
    *,
    table_span_mm: tuple[float, float],
    table_center_mm: tuple[float, float] = (0.0, 0.0),
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Cheapest mapping that works: assume a fixed overhead camera and a flat table.

    Calibrate by putting an object at two known positions and adjusting `table_span_mm`
    and `table_center_mm` until predictions match. This takes about five minutes and is
    far faster than a real intrinsic calibration — and for a top-down camera over a flat
    table it is nearly as good.

    Returns (x_mm, y_mm) in the arm's base frame.
    """
    px, py = pixel
    w, h = frame_size
    if not w or not h:
        raise ValueError("frame_size must be non-zero")

    u = (px / w) - 0.5  # -0.5 .. +0.5
    v = (py / h) - 0.5
    if flip_x:
        u = -u
    if flip_y:
        v = -v

    x = table_center_mm[0] + u * table_span_mm[0]
    y = table_center_mm[1] + v * table_span_mm[1]
    return x, y


def pixel_to_ray_table_intersection(
    pixel: tuple[float, float],
    frame_size: tuple[int, int],
    *,
    fov_deg: float = 60.0,
    camera_height_mm: float = 400.0,
) -> tuple[float, float]:
    """Alternative when the camera looks straight down from a known height.

    Uses a pinhole model with a horizontal field of view — no calibration file needed,
    just a tape measure and the camera's spec sheet.
    """
    px, py = pixel
    w, h = frame_size
    f_px = (w / 2) / math.tan(math.radians(fov_deg) / 2)
    x = (px - w / 2) * camera_height_mm / f_px
    y = (py - h / 2) * camera_height_mm / f_px
    return x, y


# ------------------------------------------------------------------------------- mock


class MockRobot:
    """Does nothing but say what it would do. Default adapter, always works."""

    name = "mock"

    def __init__(self, table_span_mm=(400.0, 300.0), verbose: bool = True):
        self.table_span_mm = table_span_mm
        self.verbose = verbose
        self.holding = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [mock] {msg}")

    def connect(self):
        self._log("connected")

    def home(self):
        self._log("moving to home pose")

    def _world(self, pixel, frame_size):
        return pixel_to_table_xy(pixel, frame_size, table_span_mm=self.table_span_mm)

    def point_at(self, pixel, frame_size):
        x, y = self._world(pixel, frame_size)
        self._log(f"pointing at pixel {pixel} -> table ({x:.0f}, {y:.0f}) mm")

    def pick_at(self, pixel, frame_size) -> bool:
        x, y = self._world(pixel, frame_size)
        self._log(f"hover over ({x:.0f}, {y:.0f}) at z=120mm")
        self._log("descend to z=30mm, close gripper, lift to z=150mm")
        self.holding = True
        return True

    def release(self):
        self._log("opening gripper")
        self.holding = False

    def stop(self):
        self._log("STOP — cancelling motion")


# -------------------------------------------------------------------------------- viam


class ViamArm:
    """Talks to a Viam machine that exposes a generic service handling DoCommand.

    Mirrors a setup that has been driven successfully before: the perception+motion
    logic lives on the robot as a generic service, and this is a thin client. Set
    VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID and SERVICE_NAME in .env.
    """

    name = "viam"

    def __init__(self, service_name: str = "rv_code", address: str | None = None):
        import os

        self.service_name = os.environ.get("VIAM_SERVICE_NAME", service_name)
        self.address = address or os.environ.get("VIAM_ADDRESS", "")
        self.api_key = os.environ.get("VIAM_API_KEY", "")
        self.api_key_id = os.environ.get("VIAM_API_KEY_ID", "")
        self._machine = None
        self._svc = None

    def connect(self):
        import asyncio

        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._connect_async())

    async def _connect_async(self):
        from viam.robot.client import RobotClient
        from viam.services.generic import Generic

        opts = RobotClient.Options.with_api_key(
            api_key=self.api_key, api_key_id=self.api_key_id
        )
        self._machine = await RobotClient.at_address(self.address, opts)
        self._svc = Generic.from_robot(self._machine, self.service_name)
        print(f"  [viam] connected to {self.address}, service {self.service_name}")

    def _do(self, cmd: dict) -> dict:
        return self._loop.run_until_complete(self._svc.do_command(cmd))

    def home(self):
        self._do({"cmd": "scan"})

    def point_at(self, pixel, frame_size):
        # The on-robot service does its own detection; pixel is advisory.
        print(f"  [viam] hover (service-side detection); client saw pixel {pixel}")
        self._do({"cmd": "hover"})

    def pick_at(self, pixel, frame_size) -> bool:
        result = self._do({"cmd": "grab"})
        return bool(result.get("ok", True))

    def release(self):
        self._do({"cmd": "release"})

    def stop(self):
        self._do({"cmd": "status"})


# ----------------------------------------------------------------------------- lerobot


class LeRobotSO101:
    """SO-100 / SO-101 follower arm over the LeRobot motor bus.

    UNVERIFIED: written from the LeRobot API but never run against hardware. Before
    trusting it, on the real arm:
      1. Find the port:   lerobot-find-port
      2. Calibrate:       lerobot-calibrate --robot.type=so101_follower \
                              --robot.port=<port> --robot.id=hack
      3. Confirm the joint names printed by `describe()` match your build.
      4. Calibrate `table_span_mm` with `pixel_to_table_xy` before any descent.

    Keep MIN_Z as a hard floor so a bad pixel cannot drive the gripper into the table.
    """

    name = "so101"
    MIN_Z_MM = 25.0

    def __init__(self, port: str | None = None, table_span_mm=(400.0, 300.0)):
        import os

        self.port = port or os.environ.get("SO101_PORT", "")
        self.table_span_mm = table_span_mm
        self.robot = None

    def connect(self):
        if not self.port:
            raise RuntimeError(
                "SO101_PORT not set. Run `lerobot-find-port` and put the device path "
                "(e.g. /dev/tty.usbmodem58760431541) in .env"
            )
        # Module path per current LeRobot docs: `so_follower`, not `so101_follower`.
        # If this import fails, run `lerobot-info` and check the installed layout.
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(SO101FollowerConfig(port=self.port, id="hack"))
        self.robot.connect()
        print(f"  [so101] connected on {self.port}")

    def describe(self) -> dict:
        return self.robot.get_observation() if self.robot else {}

    def home(self):
        self._send({f"{j}.pos": 0.0 for j in
                    ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")})

    def _send(self, action: dict):
        if self.robot is None:
            raise RuntimeError("call connect() first")
        self.robot.send_action(action)

    def point_at(self, pixel, frame_size):
        x, y = pixel_to_table_xy(pixel, frame_size, table_span_mm=self.table_span_mm)
        # Pan only — safest possible "point": no descent, no reach.
        pan = max(-100.0, min(100.0, x / (self.table_span_mm[0] / 2) * 45.0))
        print(f"  [so101] pan to {pan:.1f} for table ({x:.0f}, {y:.0f}) mm")
        self._send({"shoulder_pan.pos": pan})

    def pick_at(self, pixel, frame_size) -> bool:
        raise NotImplementedError(
            "Inverse kinematics for the SO-101 is build-specific. Either drive joints "
            "directly after measuring your arm, or record a teleop dataset and train a "
            "policy — see docs/lerobot-cheatsheet.md. Use point_at() for a safe demo."
        )

    def release(self):
        self._send({"gripper.pos": 40.0})

    def stop(self):
        if self.robot is not None:
            self.robot.disconnect()


def build(kind: str) -> RobotAdapter:
    kinds = {"mock": MockRobot, "viam": ViamArm, "so101": LeRobotSO101}
    if kind not in kinds:
        raise SystemExit(f"unknown robot {kind!r}; pick one of {sorted(kinds)}")
    return kinds[kind]()
