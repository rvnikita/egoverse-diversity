# LeRobot / SO-101 cheatsheet

The SO-100/SO-101 (Hugging Face + TheRobotStudio, ~$100–500 in parts) is the most likely
arm on a cheap-hardware hackathon table. Commands below are from the current LeRobot docs.
**None of this has been run against hardware by us** — treat it as a map, verify on site.

## Install

```bash
pip install lerobot
pip install -e ".[feetech]"    # SO-101 uses Feetech STS3215 servos
lerobot-info                   # confirm the install and see the module layout
```

## Bring-up, in order

```bash
# 1. which USB port is which arm (unplug when prompted)
lerobot-find-port

# 2. only for brand-new motors: assign ids/baudrates, one motor at a time
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX

# 3. calibrate — REQUIRED, and the step people skip and then wonder why nothing works.
#    Centre all joints, press enter, then sweep each joint through its full range.
lerobot-calibrate --robot.type=so101_follower \
                  --robot.port=/dev/tty.usbmodemXXXX --robot.id=hack

# leader arm (teleop), if there is one
lerobot-calibrate --teleop.type=so101_leader \
                  --teleop.port=/dev/tty.usbmodemYYYY --teleop.id=hack_leader
```

If an arm on the table is already calibrated, **do not recalibrate it** — you will burn
20 minutes and possibly break a teammate's setup.

## Python control

```python
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

robot = SO101Follower(SO101FollowerConfig(port="/dev/tty.usbmodemXXXX", id="hack"))
robot.connect()
print(robot.get_observation())        # joint positions — check the joint NAMES here
robot.send_action({"shoulder_pan.pos": 10.0, "gripper.pos": 40.0})
robot.disconnect()
```

Joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
`gripper`. Actions are a dict of `"<joint>.pos"`. Confirm the names from
`get_observation()` rather than trusting this list — builds vary.

## Teleop and datasets (only if there is a leader arm)

```bash
lerobot-teleoperate --robot.type=so101_follower --robot.port=... \
                    --teleop.type=so101_leader --teleop.port=...

lerobot-record --robot.type=so101_follower --robot.port=... \
               --teleop.type=so101_leader --teleop.port=... \
               --dataset.repo_id=$HF_USER/hack-pick --dataset.num_episodes=50
```

**Reality check on training in one day.** ACT on ~50 episodes is a few hours on an A100,
and you still need the episodes recorded first. SmolVLA fine-tuning is lighter and runs
15–30 Hz on a 4090 at inference. If the team wants a policy, the *only* realistic path is:
record early (before noon), train on Modal while you build everything else, and keep a
scripted fallback for the demo. Do not bet the demo on the training run.

```bash
lerobot-train --policy.type=act --dataset.repo_id=$HF_USER/hack-pick
lerobot-eval  --policy.path=<checkpoint> --env.type=... --eval.n_episodes=10
```

## Pixel → world: the actual hard part

The detector gives you a pixel. The arm needs millimetres. Three options, cheapest first
(`src/robot.py` implements the first two):

**1. Two-point affine calibration** — `pixel_to_table_xy()`. Assume a fixed camera and a
flat table. Put an object at two known positions, adjust `table_span_mm` and
`table_center_mm` until predictions match. Five minutes, no camera intrinsics, and for a
top-down camera over a flat table it is nearly as good as anything else. **Do this.**

**2. Pinhole ray → table plane** — `pixel_to_ray_table_intersection()`. Needs the camera's
horizontal FOV and its height above the table. A tape measure and a spec sheet.

**3. Depth camera.** If there is one, read depth at the pixel and back-project. Beware
0 and 65535 sentinel values meaning "no reading" — sample an expanding window around the
pixel and reject sentinels. (This is what worked on the Viam Lite 6 setup.)

Whichever you use: **keep a hard `MIN_Z` floor** so a bad pixel cannot drive the gripper
into the table. `LeRobotSO101.MIN_Z_MM` exists for this.

## Safety, which is also demo-preservation

- Cap joint speed while testing. The SO-101 is small but it will happily slam itself.
- Keep a hand on the power supply.
- `point_at()` before `pick_at()`. Pointing proves the perception is right without
  risking a collision, and it is a perfectly good demo on its own.
- Run the full sequence 10 times before demoing. Failures cluster around lighting changes
  and someone bumping the table.

## Fallback if there is no arm at all

`--robot mock` runs the whole pipeline and prints the motion it would command. Combined
with `--save-frames`, you get annotated detection images — enough to demo the perception
and voice layers honestly while saying "the motion layer is one adapter away".
