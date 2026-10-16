# Model releases

A release is what the app (or any device) needs to run PRIMAL: `releases/<name>/` with the
Core ML encoder and head, `release.json` (the runtime contract: preprocessing, clip timing,
reference bins, tracker, confidence, session settings, and the check against PyTorch) and
`golden/` test cases. `train/release.py` builds one from a run and refuses to write it if
Core ML disagrees with PyTorch beyond its tolerance (descriptor cosine 0.995 on every golden
frame, the head's best bin the same). `releases/` is not in git: the folders live on this
Mac; this page is their record.

```bash
PYTHONPATH=. ./.venv/bin/python -m train.release RUN --name primal-<what>-<yyyymmdd> [--encoder-precision float32]
```

## The releases

| Release | Run | Encoder | Runs on (iPhone) | Check (cosine min) | In the app |
|---|---|---|---|---|---|
| `primal-v3-mobilenet-lr1-s0-20261010` | `v3_mobilenet_lr1_s0` | float16 | Neural Engine | 0.9995 | replaced |
| `primal-v3-mobilenet-lr1-s0-20261012` | `v3_mobilenet_lr1_s0` | float16 | Neural Engine | 0.9995 | replaced |
| `primal-newtracks-pitch-s0-20261014` | `newtracks_pitch_s0` | **float32** | GPU (not the Neural Engine) | 1.0000 | replaced |
| `primal-qat8-20261014` | `newtracks_pitch_qat8_s0` | float16 | Neural Engine | 0.9997 | replaced: its reference never closed |
| `primal-qat8-lb075-20261016` | `newtracks_pitch_qat8_s0` | float16 | Neural Engine | 0.9997 | **current** (app branch `model-qat8-lb075`) |

The 20261012 rebuild of the release model carries every tracker and session setting in
`release.json`; the two 20261014 releases use the same contract (plus
`check.encoder_precision`). `primal-qat8-lb075-20261016` is `primal-qat8-20261014` with one
session setting changed (`release.json` `derived`), made by `experiments/release_retune.py`.

### The loop closure's threshold (2026-10-16)

In the app primal-qat8-20261014 stopped closing the reference by itself (a night video filmed
off a screen, then the same video through the fake GoPro in the Simulator), where the v3 release
had. Not the app: the Swift engine, the Core ML encoder and head and the preprocessing all match
Python on that footage (`experiments/release_on_footage.py`, `experiments/golden_replay.py` +
the app's golden replay test). The closure waits for the last 12 frames to look like the first
12 again, mean cosine over `loop_back` (0.85, tuned on v3). The newer models score the same
place on another pass lower (`experiments/loop_margin.py`: on the night video qat8 peaks at 0.874
at the line, v3 at 0.943; on Pavle's night session qat8 is under 0.85 on 8 of 12 laps), so qat8
passed it on clean frames by a hair and missed on a camera's. Whole sessions replayed at lower
thresholds (`experiments/loop_back_sweep.py`, START 3 s before a line, no hand close):

| | night1 | indoor | BHGP | RaceChrono | Dragy | adventure | rotax-night |
|---|---|---|---|---|---|---|---|
| qat8, 0.85 | never | right | never | 94.4 s* | right | 2 laps | right |
| qat8, **0.75** | **right** | right | 1.8 s off, laps right | 93.6 s* | right | **right** | right |
| v3, 0.85 | right | right | never | 94.5 s* | right | 2 laps | right |
| v3, 0.75 | right | 43.7 s, false | 1.8 s off | **15.3 s, false** | right | right | **36.5 s, false** |

(* RaceChrono's first lap starts slowly: the START-to-START lap differs from the line's; the
laps after it are right.) qat8 separates places more (its mid-lap score ~0.3-0.5 against v3's
0.4-0.7), so a lower bar admits no look-alike for it and several for v3: 0.75 is the release's
setting for qat8, and v3 keeps 0.85. The app no longer has a button to close the reference by
hand (`closeReferenceHere` is unused since "Line here" went), so a missed closure is a session
without a delta.

### What each model is good at

Tracker, share of ticks off by more than 100 ms (lower is better); details in
`docs/ml-pivot.md` ("The ten new circuits, and pitch"; "On the target hardware").

| Test | v3_mobilenet_lr1_s0 | newtracks_pitch_s0 | newtracks_pitch_qat8_s0 |
|---|---|---|---|
| unseen new tracks, level | 18.5% | **15.0%** | 17.4% |
| unseen new tracks, camera pitched ±8 deg | 46-50% | **23-26%** | 26-32% |
| unseen original tracks (G2) | 24.1% | **22.6%** | 27.9% |
| real: indoor kart / BHGP / RaceChrono | 16.5 / 29.4 / 15.8% | 16.2 / **28.2** / 16.8% | **13.7** / 29.5 / **13.6%** |
| stress demo (shake, nodding, braking, glances) | 11.0% | 5.1% | **4.1%** |
| Core ML float16 on real frames | passes | **fails** (0.932): hence float32 | passes |

- **`primal-newtracks-pitch-s0-20261014`:** the most robust to camera pitch (braking,
  looking at the wheel) and the best on unseen tracks. Its encoder had to be float32: in
  float16 Core ML disagrees with PyTorch on a third of real frames. Float32 is exact and runs
  on the GPU; expect a little more power than the Neural Engine. Measure its speed on the
  phone with `-modelcheck YES`.
- **`primal-qat8-20261014`:** the pitch model fine-tuned under simulated int8; float16 now
  exact, so it runs on the Neural Engine. Best on two of the three real videos, a little
  weaker on pitch and on the simulated tests.
- **In the app for the next track test: `primal-qat8-lb075-20261016`** (Neural Engine, closes
  the reference on every real video primal has, the night video end to end in the Simulator).
  `primal-newtracks-pitch-s0-20261014` (GPU) would need its own `loop_back` measured first.

Neither runs on Halo's NPU in int8 yet (`docs/ml-pivot.md`, "MobileNet on Halo's NPU").

## Putting a release in the app

The app (`~/code/hotlapp_V1.0`, its own repo) holds exactly one release in
`hotlapp_V1.0/Models/<release>/`; Xcode compiles the `.mlpackage`s into the bundle, where two
releases would collide (same file names). So:

1. Copy `releases/<name>/` from this repo to `~/code/hotlapp_V1.0/hotlapp_V1.0/Models/<name>/`.
2. Remove the old release's folder from `Models/`.
3. Build; then on the phone launch with `-modelcheck YES`: the encoder's cosine to PyTorch
   on the golden frames must be at least 0.995 and the head's best bin the same, and it
   reports the time per frame. (The Simulator cannot run the Neural Engine; check there on
   the CPU only.)
4. The app reads every number from `release.json` (`ModelRelease.swift`); nothing in the app
   code needs changing for a release with the same contract.
5. Before it goes to a track: the reference must close by itself on real footage at the
   release's session settings (`experiments/loop_margin.py RUN` for the room at the line,
   `experiments/loop_back_sweep.py RUN` for whole sessions), and end to end in the Simulator:
   `fake-gopro` streaming a real video, the app launched with `-cameraAddress 127.0.0.1
   -source gopro -autoconnect YES -autostart YES`, its console showing the reference closing
   and the laps (`hotlapp_V1.0/CLAUDE.md`). The golden tests alone did not catch
   primal-qat8-20261014's missed closures.
