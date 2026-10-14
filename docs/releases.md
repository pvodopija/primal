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
| `primal-v3-mobilenet-lr1-s0-20261012` | `v3_mobilenet_lr1_s0` | float16 | Neural Engine | 0.9995 | **current** |
| `primal-newtracks-pitch-s0-20261014` | `newtracks_pitch_s0` | **float32** | GPU (not the Neural Engine) | 1.0000 | — |
| `primal-qat8-20261014` | `newtracks_pitch_qat8_s0` | float16 | Neural Engine | 0.9997 | — |

The 20261012 rebuild of the release model carries every tracker and session setting in
`release.json`; the two 20261014 releases use the same contract (plus
`check.encoder_precision`).

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
- **Recommended for the next track test: `primal-newtracks-pitch-s0-20261014`** (pitch was
  the problem in real sessions); `primal-qat8-20261014` as the alternative to compare.

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
