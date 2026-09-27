# Assetto Corsa capture log

The evidence behind the capture setup: how the Windows capture box is
configured, what was measured on it, and what has been recorded. The steps
themselves are in [`../README.md`](../README.md) §2–4; this file says why they
are what they are.

## Capture machine

| | |
|---|---|
| GPU | GTX 970 (Maxwell), driver 560.94 |
| CPU | i5-6600, 4 cores, no hyper-threading |
| OBS | 32.2.2 — x264, CRF 18, `veryfast`, Hybrid MP4, 1280x720 at 60 fps |
| Game | CSP 0.2.11 (build 3465), Pure 0.190 renderer, `pure` post-processing filter |
| Python | 3.13, `torch 2.14.0+cpu` — capture, packing and CPU eval only; training runs on the Mac |

**No NVENC.** OBS 32 probes the driver at startup and hides every NVENC encoder
if it is too old. `obs-nvenc-test.exe` in OBS's `bin\64bit` prints the verdict:
`reason=outdated_driver` for 560.94. The 580 branch is the last with Maxwell
support and would clear it, but x264 is enough: the grid decodes at 100%, and
frames are downscaled 8x before training, which erases encoder differences.

**OBS drops nothing.** Measured on a 279 s session: 60.00 fps output, MP4 frame
spacing exactly 16.667 ms on every frame, AC rendering at 59.80 fps, worst AC
hitch 33 ms. The early worry that x264 would starve AC on four cores did not
materialise.

## Game setup, and why

- **Bumper camera** (drivable mode 3). No car body in frame. It sits about
  1.15 m above the road, close to driver-eye height (1.02 m on the MX-5, 1.19 m
  on the Abarth), so the height gap to head-mounted glasses is small.
- **Damage 0%.** AC's damage display draws a white car schematic at a fixed
  screen position whenever a tyre locks. It was in **28.8%** of the first
  session's frames.
- **Ghost car off**, racing line off, every HUD app off except Timecode.
- **One weather system: Pure.** Sol's renderer under Pure's filter flattens the
  sky to a single teal and makes time of day invisible — it looks identical at
  noon and dusk. One session has this look and is kept as a style variant.
- **Time of day works under Pure.** Noon against 18:30 at matched track
  positions: mean brightness 142–159 against 77–96. Evening is about half as
  bright, which is *below* the 0.75 gain floor of `photometric_jitter`, so real
  cross-time footage covers something the augmentation does not.
- **Save the replay after every session.** The camera rig re-renders it from new
  viewpoints later, with exact labels. `cfg/replay.ini` now records at
  `LEVEL=4` (15 ms, ~67 Hz, measured in the header of the first replay saved
  with it; it was 3, 30 ms, ~33 Hz, which every earlier replay has) with
  `MAX_SIZE_MB=500` (it was 26). The size cap is a ring buffer: at 26 MB,
  multi-car replays had already wrapped and kept only their last minutes, and a
  single car at 60 Hz would have hit it within a session. The original is kept
  as `replay.ini.before-primal`.

## Labels

**The timecode encodes the camera's own track position**,
`ac.worldCoordinateToTrack(ac.getSim().cameraPosition).z`, since `6cbef50`.

Before that it encoded `ac.getCar(0).splinePosition`, the car's *centre*, while
the image comes from a camera ahead of it. Measured on the bumper camera:

| Car | Camera ahead of car centre | Camera height | Image ahead of the old label |
|---|---|---|---|
| `ks_abarth500_assetto_corse` | 0.98 m | 1.137 m | **0.889 m** |
| `ks_mazda_mx5_cup` | 1.87 m | 1.160 m | **1.738 m** |

Constant to a few centimetres within a car, so it cancelled for same-car pairs,
but an Abarth reference against an MX-5 live lap carried a **0.85 m** target
error — more than half the 1.5 m budget. Sessions recorded before the change
carry `label_offset_m` in `run.json`, and `pack.py` shifts their labels by it.
Verified on the MX-5 session: packed labels sit exactly 1.738 m from the decoded
ones at identical frames. The track coordinate agrees with `splinePosition` to
within 0.13–0.18 m, so the change does not alter how `s` is parametrised.

## Per-frame telemetry

The timecode app also logs, for every rendered frame, AC's speed, G-forces
(`car.acceleration`), local velocity and local angular velocity (what a phone
gyroscope would measure), with two of AC's own times: `sim_ms` for the frame and
`phys_ms` for the physics state it shows, so sensor latency can be simulated
from ground truth. Rows carry the barcode's counter; `capture/frame_log.py`
joins them onto the decoded labels, and its `check` command verifies the join by
requiring the logged camera position to match the barcode to within one
quantisation step. The app writes one folder per launch under its own
`frame_log/`; `session import` copies the newest into the session.

Since the 2026-09-27 recording session each row also has:

| Columns | What |
|---|---|
| `cam_x/y/z`, `cam_fwd_*`, `cam_up_*`, `cam_fov_deg` | the rendering camera's full pose in world coordinates (metres, Y up) and vertical FOV |
| `cam_trk_x`, `cam_trk_h` | the camera's track x (−1 left edge, +1 right edge) and height above the track |
| `side_l_m`, `side_r_m` | AI spline to each track edge at the camera; lateral metres = `cam_trk_x * (side_l_m + side_r_m) / 2` |
| `cam_yaw_deg` | the camera's heading relative to the car's, positive right: the wander's and the head's turn as applied |
| `car_x/y/z`, `car_fwd_*`, `car_up_*`, `car_s`, `car_trk_x` | the car's pose, spline position and track x |
| `steer_deg`, `gas`, `brake`, `gear`, `rpm` | driver inputs |
| `lap_count`, `lap_ms`, `game_time_s` | AC's completed laps, current lap time, and in-game time of day in seconds |

If logging fails, the error goes to `frame_log_error.txt` next to the app and
`capture.preflight` reports it.

Logging runs after the barcode is drawn and inside `pcall`: a failure switches
logging off without touching the label. Verified under LuaJIT with a stub AC
before installing: the barcode is identical to the previous version's on every
frame, and a mid-run telemetry failure leaves it drawing. Sessions recorded
before this have no telemetry.

## Two clocks, and why duplicates are normal

AC renders at ~60 Hz and OBS samples at exactly 60 Hz, on independent clocks. As
they slide past each other, OBS sometimes catches one rendered frame twice (a
duplicate counter) and sometimes misses one (a gap). The two counts come out
nearly equal — 1,494 and 1,438 over 16,742 frames — and arrive in bursts when the
phases align. This is not frame loss.

`pack.py` originally split a lap at *any* counter step other than 1, so a
three-minute session packed as 19 fragments, the longest 13.9 s. It now drops
duplicates and keeps gaps up to `MAX_COUNTER_GAP` (4, the sampler's widest
stride): the same footage packs as 6 laps, one complete. Only the checksum is a
pass/fail gate.

## HUD contamination check

The damage schematic is a fixed-position HUD element, so a shape match catches
it: the IoU of the frame's near-white mask against the known schematic's mask.
Real schematic frames score **0.99**; the worst kerb or white building in Pure
footage scores **0.17**. A brightness-only check gave ~12% false positives on
Pure footage, because kerbs, the pit building and painted grid boxes are all
bright white. The detector is a scratch script and is not in the repo yet.

## Probe (`capture/ac_probe`)

A CSP app that logs, ten times a second, where the active camera sits relative
to the car and the track, plus replay state. It only logs while its window is
open, so it can never end up in capture footage. Each launch writes
`probe_<car>_<n>.csv` at a free index; an earlier version wrote one file per car
and a later launch overwrote the Abarth measurement.

MX-5 Cup, every camera mode:

| Camera | Ahead of car centre | Height | Pitch |
|---|---|---|---|
| Cockpit (driver eyes) | −0.49 m | 1.02 m | −1.6° |
| Dash | −0.26 m | 1.01 m | 0° |
| Bonnet | +0.53 m | 1.01 m | 0° |
| **Bumper** | **+1.87 m** | **1.16 m** | 0° |
| Chase / Chase 2 | −4.2 / −5.1 m | 1.74 / 2.24 m | −9° / −14° |

All use a 60° vertical FOV. The bot drives close to the AI line (normalised
lateral −0.10 to +0.18), and the track is 9.6–14.6 m wide, which leaves several
metres of tarmac on each side.

**Replays** record every 30 ms (~33 Hz; the header stores `30.0`, and playback
measured 34.2 frames/s) and render at 60 fps by interpolating, so fine body
shake above ~15 Hz is smoothed away. Spline position follows the replay and
freezes on pause while the frame counter keeps running.

## Camera rig (`capture/ac_rig`)

Re-renders a replay from a camera rigidly mounted to the car at a chosen
sideways offset, forward offset, height and FOV. One replay becomes several
virtual lines with exactly known separation, which is what the `lines` gate
needs and what bot footage otherwise lacks.

Settings live in `rig.txt` next to the script, re-read twice a second:
`enabled`, `replay_only`, `lateral_m` (**positive is right**), `wander_m`,
`wander_len_m`, `wander_yaw`, `wander_seed`, `forward_m`, `height_m`, `fov_deg`,
`edge_limit`, and `weather` / `rain` to override the
recorded conditions (`-1` keeps them; values are `ac.WeatherType`, e.g. 15 clear,
17 scattered clouds, 19 overcast, 7 rain), and the `look*` / `glance*` keys
below. Offsets that would leave the tarmac are shrunk by bisection in track
coordinates.

**Looking into corners.** With `look = 1` the camera also turns about the car's
up axis toward the point `look_ahead_*_m` (15-20 m) further along the track on
the camera's own line, by a share `look_gain_*` (0.3-0.8) of that angle, capped
at `look_max_deg` (35°) and smoothed with a 0.25 s time constant. Distance and
gain are drawn afresh for every render, so each re-render of a replay gets a
different head. On a straight the point lies dead ahead wherever the camera
sits, so the head only turns where the road does. On top come occasional short
glances aside, `glance_deg` (8°) at most, 0.6-1.4 s out and back, every
`glance_every_s` (10 s) on average. The rig log's `yaw_deg` is now the whole turn
and `head_yaw_deg` the head's part; the timecode log's `cam_yaw_deg` has the
applied turn for every frame. Verified with a stub AC: on a 40 m circle the head
turns gain × the geometric angle to within 0.01°, on a straight 0.000° with the
camera 2 m off the line, and with the option off the camera matches the previous
rig exactly from the first second on (the render now starts before the first
placement, which removes a one-frame smoothing reset).

Each render logs to its own `rig_<settings>_<n>.csv` — a new file whenever the
settings change or the replay is rewound, never overwriting — with the car's and
the camera's lateral position in metres, the track width, and the offset
actually applied. Import a render with `session import --rig`: the log is copied
into the session as `rig_log.csv`, and packing derives `line_mean_m` and
`line_std_m` from it.

**Wander, for training data.** `wander_m` adds a smooth random sideways drift on
top of `lateral_m`: three sines at incommensurate wavelengths around
`wander_len_m` (default 200 m) with seeded random phases, driven by distance
driven rather than track position, so every lap takes a different line through
the same corner instead of the model being able to learn line-by-place. With
`wander_yaw = 1` the camera also turns along the drift, as a car changing line
does. At 2.5 m and 200 m, over 20 km: offset within ±1.8 m for 90% of the
distance, heading turn 4.5° at the 95th percentile and 5.3° at most, and 26 cm
of sideways drift inside a median 12-frame clip at 35 m/s (65 cm at most) —
continuous, a few centimetres per frame. Doubling `wander_len_m` halves both the
drift and the turn. A lap's mean offset is then close to zero and says nothing
about which lines it covered, so evaluating wander laps needs per-frame lateral
position, not `line_mean_m`. Keep fixed offsets for evaluation renders.

**Live or replay.** With `replay_only = 0` the rig runs while the bot drives, and
the car is unaffected — only the viewpoint moves. Live is preferred for training
data: physics at 60 fps, where a replay is recorded at ~33 Hz and interpolated,
and no second pass. Save the replay anyway, for fixed-offset renders later.

Time of day probably cannot be changed in a replay. CSP's replay override covers
weather, rain, wetness, wind and temperature but has no time field, and its time
setters are documented as offline-races-only. Time-of-day variety needs separate
drives.

Verified in the MX-5 replay at 2.5 m: median applied offset 2.50 m, guardrail
active on 14% of samples and never below 0.96 m, camera track x never past the
0.85 limit, camera 1.15 m above the road. That test ran before the sign was
flipped, so it moved left.

Not yet verified: recording *during* a replay. The replay control bar may land
in frame or over the barcode, which the first render's checksum and a frame
check will show.

## Sessions

All Brands Hatch Indy (1,915.8 m spline), all `split=train`, all checksum
100.00%.

| Session | Car | Time | Look | Complete laps | Use |
|---|---|---|---|---|---|
| `20260920T202343Z` | Abarth | — | Sol + Pure mix | 1 | **excluded**: damage schematic in 29% of frames |
| `20260921T171903Z` | Abarth | ~13:00 | Sol + Pure mix, teal sky | 5 | style variant |
| `20260921T181357Z` | Abarth | 12:00 | Pure, light clouds | 4 | |
| `20260921T215154Z` | Abarth | 18:30 | Pure, light clouds | 4 | |
| `20260921T221531Z` | MX-5 Cup | 12:00 | Pure, overcast | 3 | replay saved |

Laps within a session agree to about half a second (Abarth 55.3–56.3 s, MX-5
55.2–55.4 s): the bot drives the same lap every time. The variation that matters
comes from *across* sessions.

`data/packed_ac` holds the four usable sessions at 128x80 with 2 m bins, the
resolution and spacing the synthetic checkpoints were trained at.

### The 2026-09-27 recording session

The first sessions with the full telemetry log, recorded unattended: Claude
drives Content Manager, AC and OBS, and `capture.autostop` ends each recording.

**Highlands Short, MX-5 Cup, 11:57, scattered clouds, AI 100, rig wander 2.5 m**
(`ks_highlands__layout_short__20260927T212549Z`). 20 laps of ~51 s, 98.04%
checksum-valid: the invalid 2% is one 22 s run at the end, the pause menu,
which hides the barcode. `trim --last-lap` cuts that and the partial lap before
it. Pre-flight and full-session checks: label speed / car speed 1.002, every
lap's `s` wrap within 4.4 m of AC's lap counter, telemetry joined on 99.9%.

- **Render hitches.** 478 render intervals over 50 ms in 1106 s, the longest
  1.2 s; the camera moves as far as the label across each one, so labels stay
  exact and only the picture skips (up to 44 m in one frame). About 15 a minute
  is the baseline on this track; the worst minutes are the ones where analysis
  or a second program started during the recording. Nothing heavy runs on this
  PC while recording.
- **AI spline kink** about 38 m past the line (`s` 0.021-0.024): the label runs
  up to 1.5 m ahead of the camera for a few metres and then agrees again (over
  50 frames, label 55.66 m against camera path 55.62 m).
- **Camera at the edge.** `|track x|` over 1.02 on 0.11% of frames, at most
  1.078, in five corners where the AI spline puts the edge only 1.4 m out. The
  frames show road beside the barrier, not the inside of a wall.
- **`phys_ms` equals `sim_ms`** on every frame so far: AC reports the physics
  state as of the frame, so the log gives no sensor latency to simulate.

## Open items

1. **Zero-shot sim-to-real.** Evaluate the synthetic checkpoints on
   `data/packed_ac`. Never run; the most important number outstanding.
2. **Rig renders** of the MX-5 replay at +3, +1.5, 0, −1.5 and −3 m: the first
   real footage with known line separation.
3. **More tracks, and a held-out one** (Silverstone National planned). Needed
   for G2 and the leakage control on real footage.
4. **Time-gap gate:** error against the time-of-day gap between live and
   reference, to size how long a reference stays valid. Needs time of day
   recorded per session; `race.ini` holds the sun angle at import.
5. **Height and FOV sweeps** through the rig, with FOV matched to the glasses.
6. **Compression augmentation:** glasses stream at low bitrate, while CRF 18 plus
   an 8x downscale is effectively clean.
