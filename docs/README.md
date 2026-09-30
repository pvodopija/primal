# Read this first

This repository is **PRIMAL** (PRogress estIMation on repeAting signaL):
given one reference lap and a live POV clip, estimate progress `s` along
that lap. Everything the product shows — time delta, delta rate, sector
attribution — is a lookup on top of `s`.

The parent checkout is `hotlapp`. This folder is a **submodule** at
`primal/`, from the private repo `pvodopija/primal`. It is not part of the
`locamotif` Rust crate.

## Reading order

1. **This file** — scope, closed paths, product constraints.
2. [`ml-pivot.md`](ml-pivot.md) — the design: what is built, what it measures,
   what is not built yet, and what has been ruled out.
   [`training.md`](training.md) — how a training step is built: data, sampling,
   augmentation, loss, gates and controls, and what is not done yet.
3. [`../README.md`](../README.md) — how to capture, pack, train, and run gates.
   [`capture-log.md`](capture-log.md) holds the measurements behind the capture
   setup and the inventory of recorded sessions.
4. Code, in this order: `train/dataset.py`, `train/model.py`, `train/eval.py`,
   `train/synthetic.py`. Capture only if the task is the AC overlay / labels.

Do not start from the iOS app, the Rust DSP, or `vprdemo`. Those are other
trees with other (mostly closed) plans.

## What this repo is

| Piece | Role |
|-------|------|
| `capture/` | Assetto Corsa + OBS pipeline. Timecode overlay burned into pixels is the ground truth. Windows-only. |
| `train/` | Shared encoder, live↔reference correlation, dilated 1-D conv head, gates, synthetic renderer. |
| `tests/` | Overlay wire format, fake-recording e2e, sampler invariants + a learning smoke test. |
| `docs/ml-pivot.md` | Active system design. |

The network is given a live clip of K frames **and** one reference lap. It
predicts which point of the reference the live clip is at. Track identity
reaches the output only through a dot product, so the weights structurally
cannot memorize a circuit.

## Where the project stands

Trained on **real Assetto Corsa footage** (`data/packed_ac_v3`: 380 laps on 12
circuits, 9 for training and 3 never seen), with both negative controls passing.
The product metric is the share of 15 Hz ticks whose delta is more than 100 ms
off, on whole laps of the unseen circuits, from the camera alone:

| | unseen circuits | same car as the reference | median |
|---|---|---|---|
| tracker in metres (the original design) | 30.8% | 38.5% | 2.3 m / 62 ms |
| **tracker in reference-lap time** | **12.2%** | **5.3%** | **1.6 m / 43 ms** |
| with a true speed signal, for comparison | 2.3% | 2.4% | 1.0 m / 28 ms |

What got it there, all measured in [`ml-pivot.md`](ml-pivot.md):
- **Tracking in reference-lap time.** The reference lap already knows where the
  driver brakes, so in its time the kart moves at almost one second per second.
  This replaced most of what a speed sensor would give.
- **Runtime stride 2.** At stride 4, runtime sat at the fast edge of what
  training saw, and the estimate trailed the kart by ~60 ms.
- **Head-pose augmentation with mirroring,** then a clean fine-tune.

Synthetic pretraining **does not transfer** (a synthetic checkpoint scores at
chance on real footage), so synthetic data is for plumbing and architecture
only.

Open, in order of value:
- a measured speed from the camera (road flow on Halo at 90-120 fps), now a
  refinement rather than a requirement;
- a lap aligner, which would unlock voting across reference laps and a
  reference that improves every lap;
- more circuits;
- an abstain signal;
- the port to Halo's NPU. It fits (27-34% of the NPU by Vela) once GroupNorm is
  replaced, which needs a retrain.

Against classical SeqSLAM on identical real clips, PRIMAL is 29x more accurate
on trained tracks and 9x on the unseen one (v2 data). How this compares with GPS
lap timers, and the prior art the matcher builds on, is under *Landscape* in
[`ml-pivot.md`](ml-pivot.md).

## Product constraints

These bound every design choice. Do not relax them to make training easier.

- **Runtime:** ~15 Hz, ~25 min session. **The delta is shown on the glasses**
  (decided). Brilliant Labs Halo is the first prototype. By Vela's estimate the
  whole pipeline fits on its NPU, standalone. A phone is optional for setup and
  analysis. See *Hardware target* in [`ml-pivot.md`](ml-pivot.md).
- **Precision:** ≤ 100 ms time delta ≈ 1.5 m at kart speed. Place-recognition
  benchmarks that score "correct within 25 m" are the wrong metric.
- **Camera only** (decided). The product meets its targets from the glasses
  camera alone: no GNSS, no compass, no phone motion sensors required. They may
  assist outdoors, never be needed. Indoors there is no GNSS and a compass is
  unreliable, and indoor tracks (about 30% of tracks by the user's estimate;
  no public count confirms or refutes it) are where GNSS-based timers cannot follow. RTK/GNSS is fine for
  training and evaluation labels only.
- **The map is context, not parameters.** One set of weights for every track.
  A new circuit costs a reference lap, not a GPU day.
- **Head movement:** tolerate normal hot-lapping head motion — principally
  looking into the corner. Beyond that, **abstain**: freeze or grey the delta
  and recover when the driver resumes normal habits. A driver looking 90° to
  the side is not reading a delta. Lateral line invariance matters more than
  yaw invariance, because the line varies every lap while the driver is still
  driving.
- **IMU** is a filter input at runtime (motion model, bridging dropped frames),
  not a learned descriptor and not view canonicalization.

Core ML export and on-device wiring live in the `hotlapp` iOS app. They are
out of scope until the offline loop is honest.

## Closed paths — do not revive

Recorded so a fresh agent does not re-implement them.

| Path | Why it is closed |
|------|------------------|
| IMU-only descriptor / motif / `Feature` matching (`locamotif` crate) | No longitudinal observability on a straight. Turn shape is not discriminative enough between similar corners. |
| Single-frame absolute `s` regressor | Bakes the map into the weights. One model per track. Absolute position on an unseen circuit is not a transferable function of appearance. Retained only as the leakage control. |
| Off-the-shelf visual place recognition (e.g. EigenPlaces retrieval) | Trained with 10–25 m positives. Similarity is a plateau, not a metre-level peak. |
| Direct pair-of-videos → delta | Sidesteps `s`, so no line offset, no sector attribution, no debuggable bottleneck. |
| Three heads (R / A / M) on a pretrained trunk, with distillation | Never built. A 0.67M from-scratch encoder with one head hit the precision target, so the complexity was not needed. Revisit only if real footage proves capacity-limited. |
| Cancelling head pose by differencing camera yaw rate against phone gyro | Rate is not pose; gyro bias integrates to 6–60° over a lap; yaw is one DOF of six. Replaced by trained invariance plus abstention. |

The `locamotif` submodule still contains the IMU DSP, motif lock, and a Bayes
filter over turn features. Treat that crate as a **legacy sibling**, not as
something to import or extend. `vprdemo/` in the parent is a retrieval
experiment; it is not this training codebase.

## Gates

| Gate | Question |
|------|----------|
| G0 | Can the architecture fit one lap pair at all? |
| G1 | Unseen laps of a seen track, full augmentation. |
| G2 | Unseen track — the actual product question. |
| `lines --axis line` | Error must not grow with racing-line separation. |
| `lines --axis yaw` | Error must not grow with camera-yaw separation. |
| Wrong-reference | Pair live clips with another track's map → chance. |
| Leakage | No-reference absolute regressor on a held-out track must fail. If it succeeds, something on screen is leaking the answer (usually a HUD map). |

Do not add heads or Core ML export until the gates pass **on real footage** and
both negative controls hold there. Details in [`ml-pivot.md`](ml-pivot.md).

## Synthetic previews

What `train.preview` renders from packed synthetic laps. Sanity checks of
resolution and correspondence, not performance claims.

| Image | What it shows |
|-------|----------------|
| [synth_clip.png](img/synth_clip.png) | One training sample: live clip, photometric jitter, target bin. |
| [synth_sheet.png](img/synth_sheet.png) | Contact sheet along a lap. |
| [synth_lines.png](img/synth_lines.png) | Cross-line pairing — same place, different racing line. |
