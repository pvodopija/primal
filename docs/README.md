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
3. [`../README.md`](../README.md) — how to capture, pack, train, and run gates.
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

The synthetic feasibility loop is **closed and passing**. On 18 procedural
tracks, G2 — a track the model has never seen — reaches **1.41 m / 57 ms**,
under the 1.5 m product target, with both negative controls passing and error
flat against both racing-line and camera-yaw separation.

Two things that result does *not* settle, and they are the next work:

- **Sim-to-real is entirely unmeasured.** The renderer is flat-shaded polygons.
  The architecture is sound and data-hungry; whether it survives real footage
  is unknown and is the largest open risk.
- **The aliasing tail is unfixed.** ~1–2% of ticks land 175–300 m out, and no
  amount of data changed it. The particle filter is the answer and it does not
  exist yet.

Numbers, method, and the full gate table are in [`ml-pivot.md`](ml-pivot.md).

## Product constraints

These bound every design choice. Do not relax them to make training easier.

- **Runtime:** iPhone, Core ML / Apple Neural Engine, ~15 Hz, ~25 min session.
- **Precision:** ≤ 100 ms time delta ≈ 1.5 m at kart speed. Place-recognition
  benchmarks that score "correct within 25 m" are the wrong metric.
- **No GNSS at runtime.** RTK/GNSS is fine for training and eval labels only.
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
