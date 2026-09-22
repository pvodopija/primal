# PRIMAL design — progress estimation against a reference lap

> **Status: rewritten after the synthetic feasibility loop closed.** An earlier
> version of this document described a three-head network (Head R retrieval,
> Head A alignment, Head M ego-motion) on a 15–20M pretrained trunk at 256×256,
> reaching precision by distillation from a VGGT-class teacher. That design was
> never built. A 0.67M from-scratch encoder with a **single** head reached the
> precision target on synthetic data instead, so the three-head plan is closed
> and has been removed rather than archived — a dead architecture in a live
> document pollutes every decision made near it.
>
> Also closed, from the version before that: IMU-only descriptors, motif
> matching, and the `Feature` type in the `locamotif` crate. An IMU descriptor
> cannot resolve position along a straight.

---

## Goal

Given **one reference lap as context**, estimate progress `s` along it from a
live POV clip. Everything the product shows is a lookup on `s`:

| Output | Derivation |
|--------|------------|
| Time delta | `t_live − t_ref(s)` |
| Delta rate ("gaining / losing now") | `d/dt` of the above |
| Sector / corner attribution | integrate delta rate between fixed `s` bounds |
| Lateral line offset | not currently produced — see *Not built yet* |

No GNSS at runtime. GNSS/RTK is free at training and evaluation time; the
constraint is on the product, not the dataset.

---

## Precision budget

One number drives the design. A kart at 15 m/s covers 1 m in **67 ms**, so
"metre-level position" and "sub-100 ms delta" are the same requirement.

| Quantity | Karting value |
|----------|---------------|
| Speed range | 11–22 m/s (40–80 km/h) |
| Lap time | 45–60 s |
| Track length | 1.0–1.2 km |
| **Target delta error** | **≤ 100 ms (≈ 1.5 m)** |
| Target delta jitter | ≤ 50 ms std over a 1 s window |
| Live inference rate | 15 Hz |
| Runtime | iPhone, Core ML / ANE, ~25 min session |

Published place-recognition benchmarks score "correct within 25 m." That is
~1.7 **seconds** of delta at kart speed. Retrieval models are trained to be
*invariant* to the distinction this product must *measure*. This remains the
single most important fact about the problem.

---

## Core principle

**The map is context, not parameters.**

| | Weights | Map |
|---|---------|-----|
| Content | how to compare two views of a place | what this track looks like |
| Produced by | offline training | one reference lap |
| Cost to add a track | — | one lap |

Track identity reaches the output **only** through a dot product between live
and reference descriptors, so the weights structurally cannot hold a circuit.
This is not an aspiration; it is measured. Swapping in another track's
reference moves error from 0.90 m to 172.7 m against a chance level of 175.0 m
(*wrong-reference control*). The model is genuinely reading the reference.

---

## What is actually built

```
live clip  [B, K, 3, H, W] --\
                              encoder (shared)  ->  L [B, K, D]
reference  [N, 3, H, W] ----/                       R [N, D]

C = L R^T                                           [B, K, N]
dilated 1-D conv head over the reference axis       logits [B, N]
```

0.67M parameters total. The target is a continuous index into the reference
lap; `s = index / N`.

**The head is convolutional along the reference axis with circular padding.**
That makes it translation-equivariant on a loop and independent of lap length,
so one set of weights serves tracks of any size.

**The encoder is resolution-agnostic.** Its stem is pooled to a fixed 3×5 grid
before projection, so weight shapes do not depend on input size and one
checkpoint runs on 128×80 synthetic frames and 160×96 AC frames alike. The grid
is not 1×1 on purpose: *where* a landmark sits in frame is evidence for
sub-bin alignment. Note this makes the weights *loadable* across resolutions,
not *invariant* to field of view — see *Cameras and field of view*.

### Why the output is a distribution

The prediction is continuous, read out by soft-argmax over reference bins, but
it is *parametrised* as a distribution. Under aliasing the truthful answer is
multi-modal, and a scalar head trained with L2 is obliged to emit the
conditional mean: on a track where turn 2 resembles turn 6, that is a piece of
tarmac the kart has never visited, at low loss and undetectably. Sub-bin
precision comes from the windowed expectation, the same mechanism that gets
sub-pixel disparity out of a stereo cost volume.

This is also what makes "I don't know" expressible, which the product needs
(see *Head movement*).

### Sampler properties that carry the design

- **One reference per batch.** A 1 m-spaced reference is ~1000 frames;
  per-sample references would mean 16000 encoder passes for a batch of 16.
- **The reference is rolled** every batch and the target rolls with it, so any
  absolute notion of track position is wrong on every sample. The shortcut is
  unlearnable rather than merely discouraged.
- **Live and reference are jittered independently**, so matched exposure cannot
  serve as a matching cue.
- Speed invariance is free: a clip at stride `d` is that stretch at `d` times
  the speed, with exact labels.

---

## Measured, on synthetic data

All numbers from 18 procedurally generated tracks (15 train, 3 holdout),
8 laps each, 128×80, 350 reference bins at 2 m.

| Gate | Question | Result |
|------|----------|--------|
| Seen laps, seen tracks | sanity | 1.04 m · 75.3% within 1 bin |
| **G1** | unseen laps of a seen track | **1.35 m / 60 ms** · 66.6% |
| **G2** | **unseen track — the product question** | **1.41 m / 57 ms** · 65.9% |
| `lines` | error vs racing-line separation | **ratio 1.06×** — flat |
| `lines --axis yaw` | error vs camera yaw separation | **ratio 1.16×** — flat |
| Wrong-reference control | is it reading the reference? | 0.90 m → 172.7 m (chance 175.0) ✓ |
| Leakage control | is the answer on screen? | holdout 71.9 bins (chance 64.0) ✓ |

### What these established

**Precision was data-limited, not capacity-limited.** Going from 12 usable laps
across 3 tracks to 105 laps across 15 dropped G2 from 2.09 m to 1.41 m with the
model byte-identical. The seen-track number got *worse* (0.40 → 1.04 m), which
is the point: the model could no longer memorise the training laps and learned
the task instead. The generalization gap fell from 5.0× to 1.3×.

This matters for deployment: the target was reached **without** growing the
encoder, which still has to run on the ANE at 15 Hz.

**It matches place, not viewpoint.** Error is flat across racing-line
separation out to 5 m (1.14 / 1.11 / 1.06 / 1.13 / 1.29 m across bands, 840
pairs). An earlier 12-lap model showed 3.30× here; that was the same
overfitting, not a different failure.

**Yaw invariance must be trained in, and is cheap once it is.** A
yaw-naive model measured on yawed data collapses: flat to 5°, 15 m of error at
10–20°, and **139 m beyond 20°** — a ratio of 10.78×. The same model trained
with yaw variation holds 3.06 m at 20°+, a ratio of 1.16×, while keeping
lateral invariance at 1.08×. On matched data the yaw-trained model is better at
*every* separation band including the easiest.

**The aliasing tail is untouched by data.** Worst-case error stayed around
175–300 m across every scale. ~1–2% of ticks land outside five bins, and when
wrong, the correct answer is typically still the second or third peak of the
distribution. This is the failure a single-shot estimate cannot fix and is the
whole argument for the estimator below.

### Against a classical baseline

`train/seqslam.py` implements SeqSLAM (Milford & Wyeth, 2012) behind the same
`forward(live, reference) -> logits` interface, so `train.eval lines --matcher
seqslam` measures it through the identical path: same pairs, same rolled
reference, same soft-argmax readout. It was sanity-checked first by localising a
lap against itself: 0.53–0.58 bins median. Chance is 175 m.

Each checkpoint was evaluated on its own training set, so these PRIMAL numbers
are seen-lap numbers; the honest figure for an unseen track is G2 above.
SeqSLAM has no weights, so the split does not matter to it.

| Line separation | Pairs | PRIMAL | SeqSLAM |
|---|---|---|---|
| 0.0–0.5 m | 78 | 1.14 m | 119.95 m |
| 0.5–1.0 m | 134 | 1.11 m | 118.25 m |
| 1.0–2.0 m | 246 | 1.07 m | 137.23 m |
| 2.0–3.0 m | 186 | 1.17 m | 154.69 m |
| 3.0 m and up | 196 | 1.35 m | 175.66 m |

| Yaw separation | Pairs | PRIMAL | SeqSLAM |
|---|---|---|---|
| 0–2° | 40 | 2.29 m | 158.02 m |
| 2–5° | 114 | 2.33 m | 166.46 m |
| 5–10° | 190 | 2.39 m | 170.74 m |
| 10–20° | 320 | 2.59 m | 184.20 m |
| 20° and up | 176 | 3.01 m | 188.51 m |

(This re-run of the line sweep gave a 1.18× ratio for PRIMAL where the table
above gives 1.06×; sampling differs between runs, and both are flat.)

**The near/far ratio cannot rank matchers.** On the yaw axis SeqSLAM scores a
*better* ratio than PRIMAL, 1.19× against 1.32×, while sitting at or above
chance in every band. A matcher that fails everywhere is perfectly
"invariant". Always read the ratio next to the absolute error per band.

**SeqSLAM fails on lap identity, not on separation.** It localises a lap against
itself to about 1.1 m, and a different lap of the same track under 0.5 m away at
120 m. The cliff comes before the first band; the rise across bands after it is
drift toward chance. Whole-image pixel difference does not survive a second lap,
which is the case a learned descriptor exists for.

### Multi-peak reference probe

Concatenate several laps into one reference and check where the belief puts its
mass. Because the repeat structure is *constructed*, the correct answer is known
exactly, which makes this the only test here that probes the shape of the
distribution rather than the error of its argmax.

Measured on 18-track synthetic data with `runs/g1_scale`, 64 live clips per row,
mass counted within +-5 bins of each copy's correct position:

| reference | live | shares | mass captured |
|---|---|---|---|
| `[lap00, lap00]` identical | middle lap | **50.0 / 50.0** | 98.7% |
| `[lap00, lap07]` left+right | middle lap | 67.4 / 32.6 | 97.0% |
| `[lap00, lap07]` left+right | lap00 | 94.8 / 5.2 | 99.0% |
| `[lap00, lap07]` left+right | lap07 | 6.1 / 93.9 | 98.8% |

**The identical case is a leakage test and it passes exactly.** Two byte-identical
copies make the logits at `p` and `p+N` equal by construction, so anything other
than 50/50 would mean absolute position had leaked past the rolled-reference
sampler. It reads 50.0/50.0. That property was asserted from the start and had
never been checked.

**The two lopsided rows are the positive control**, and they are what make the
middle row trustworthy: the metric demonstrably *can* resolve a strong
preference, so an even split elsewhere is a measurement rather than a blind
instrument. This is exactly the sensitivity the leakage gate still lacks.

**Order does not matter.** Swapping the reference to `[lap07, lap00]` reproduces
every share to the decimal, so the preference follows content, not position.

### What the probe found that the `lines` gate cannot see

Sweeping the live lap across the ladder against a fixed `[lap00, lap07]`
reference:

| live | line_mean_m | dist to L / R | inverse-distance | measured share of L |
|---|---|---|---|---|
| lap01 | -0.98 | 1.41 / 3.68 | 72.3% | 81.9% |
| lap03 | -0.34 | 2.05 / 3.04 | 59.7% | 67.4% |
| lap02 | -0.25 | 2.14 / 2.95 | 58.0% | 60.3% |
| lap04 | +0.37 | 2.77 / 2.33 | 45.7% | **63.0%** |
| lap05 | +1.51 | 3.90 / 1.19 | 23.4% | 39.4% |
| lap06 | +2.33 | 4.72 / 0.37 | 7.3% | 10.6% |

The trend is monotonic and in the right direction, but **every row sits above the
inverse-distance prediction**, and `lap04` prefers the *farther* reference
outright. The empirical 50/50 crossing sits near +0.8 m where the geometric
midpoint is +0.16 m -- a bias worth about 0.65 m of line separation.

Two conclusions, the second more important than the first:

- The model's *confidence* is graded by viewpoint similarity even where its
  *answer* is not. The `lines` gate reports a flat 1.06-1.18x error ratio and
  concludes "matches place, not viewpoint"; both hold at once, because that gate
  measures only error magnitude and is structurally blind to how mass is
  allocated.
- **`line_mean_m` is an incomplete descriptor of which line was driven.** It
  cannot explain `lap04`, and the laps differ in `line_std_m` (0.10 to 1.27) and
  `apex_gain` (0.01 to 0.57) as well as in mean. The `lines` gate sweeps error
  against exactly this quantity, so its x-axis carries unmodelled error. A
  per-frame lateral separation would be sharper than a lap mean.

The rig (`capture/ac_rig`) re-renders one replay from several fixed offsets, so
every other variable is held byte-identical. That would make this probe a clean
isolation of line preference instead of a suggestive one, and is the reason the
rig renders matter beyond just populating `line_mean_m`.

### What these do not establish

The renderer is flat-shaded polygons with no textures, weather, motion blur, or
elevation. These results say the architecture is sound, data-hungry, and
learns the invariances asked of it. **They say nothing about real footage.**
The sim-to-real gap is entirely unmeasured and is the largest open risk.

---

## Head movement

**Product decision: tolerate normal hot-lapping head movement; abstain beyond
it.** A driver hot-lapping looks at predictable places — principally *into* the
corner they are entering. When they turn 90° to watch a kart alongside, they
are not reading a delta, and a frozen or explicitly-unavailable display is the
correct output. Recovery should be immediate once they resume normal habits.

Consequences:

| | |
|---|---|
| Priority | **Lateral line invariance matters more than yaw invariance.** Line varies every lap and the driver is still driving. |
| Yaw range to support | ordinary corner-seeking head turn, not arbitrary pose |
| Beyond that range | abstain, do not guess |
| Mechanism | the distribution is already the abstain signal — a flat or multi-modal belief is a refusal |

The renderer models the two parts separately because they are different things.
`--look-ahead-gain` turns the camera toward the curvature 20 m ahead, which is
the natural hot-lapping behaviour and the part worth being invariant to.
`--yaw-spread` applies a constant per-lap yaw offset, which is really a
*mounting* error rather than head movement; it should stay small.

A previous design proposed cancelling head pose by differencing camera yaw rate
against phone gyro. That is closed. Rate is not pose, gyro bias integrates to
6–60° over a lap, and yaw is one of six degrees of freedom. The replacement is
not cancellation but **invariance where it is cheap and abstention where it is
not**.

The phone IMU still has real jobs — a motion model for the estimator, and
bridging dropped frames over a wireless link — but view canonicalization is not
one of them.

---

## Not built yet

### Estimator — the missing half

Classical recursive Bayesian estimation over `(s, v)`. No learning.

- **Particle filter, not Kalman.** The belief is genuinely multi-modal during
  acquisition and after an excursion, which is exactly what the measured
  aliasing tail looks like.
- **Widen, don't drift.** Where the observation is uncertain, the belief must
  grow rather than become confidently wrong.
- **Emits `s` with covariance**, and the display suppresses or greys out above
  a threshold. This is the mechanism that implements the abstain decision
  above. Showing nothing beats showing a wrong delta.

The measured tail is the case for this: the network usually *has* the right
answer as a secondary peak and merely picks the wrong one. A filter carrying
"I was at 190 m a moment ago at 27 m/s" discards the impostor instantly. Every
number in this document is the network judged with this entire half missing.

It has no ML in it and can be built and tested today against synthetic `s`
trajectories.

### Lateral line offset

A product output in the goal table with no current source. The head predicts
progress only. Revisit once the estimator exists.

For *training* line invariance on real footage, the camera rig in
`capture/ac_rig` re-renders a replay from known sideways offsets, which gives AC
data the line separation the `lines` gate needs.

### Cameras and field of view

The encoder loads at any resolution, but a different **field of view** moves
the scene across the frame and is not handled. Measured on synthetic data:
resizing a 100° view into the model's input gives 102 m of error, and portrait
3:4 gives 164 m. Cropping to a matched FOV first recovers both to ~4.2 m.

So: treat every camera as a pinhole with known intrinsics, crop to a canonical
FOV at pack time, then resize. The canonical FOV can be no wider than the
narrowest camera in the system. Store intrinsics per session. This matters at
two joins — training on AC and deploying on glasses, and sharing reference laps
between users. It does not arise when reference and live come from the same
device.

---

## Data

### Synthetic renderer

`train/synthetic.py` writes the same packed format as the AC pipeline, so
`train.dataset` cannot tell them apart. It exists to check plumbing and
architecture, and it is deliberately *easier* than reality: landmark colours
are distinct per track, so cross-track aliasing is mild. Within a track it is
*harder* than assumed — the scene is a periodic kerb pattern plus ~39 coloured
boxes over 700 m, and 23% of reference bins have a look-alike elsewhere above
0.8 cosine.

A dataset is fully described by its command line, so regenerating on another
machine beats copying frames. Only real footage is worth transferring.

Viewpoint variation is a deterministic ladder across laps rather than random
draws, so every dataset is guaranteed to contain widely separated pairs. Lateral
and yaw ladders are **permuted independently per track**: correlated axes could
be satisfied by one cue, and each would contaminate the separation the other's
gate sweeps (measured correlation +0.15).

### Assetto Corsa capture

Screen capture returns a frame presented an unknown number of milliseconds ago,
so reading shared memory alongside it gives `s` at *grab* time, not *render*
time — metres of label noise at speed, and jittery, so it cannot be calibrated
away.

The fix is to make the game render its own ground truth. A CSP Lua app draws a
2×27 grid of black and white cells encoding a frame counter, the rendering
camera's track position as 20-bit fixed point, and parity. It encodes the camera
rather than the car because the two sit up to 1.7 m apart along the track, by
an amount that differs per car. `calibrate` locks the geometry,
`decode` reads cell centres and verifies the checksum, and **`pack.py` crops
the band away before resizing**. That crop is load-bearing: without it the
network would read the answer off the screen, which is exactly what the leakage
control exists to catch.

Acceptance before a real session: checksum pass rate > 99.9%. Duplicate
counters and gaps are normal — AC and OBS run on independent clocks — and
packing absorbs them. Measurements, setup and the session inventory are in
[`capture-log.md`](capture-log.md).

**HUD completely off, especially the track map with the position dot.**

### Augmentation policy

Two kinds of style need **opposite** treatment.

| | Capture-side | Scene-side |
|---|---|---|
| Examples | exposure, white balance, motion blur, rolling shutter, lens distortion, noise, compression | textures, barriers, banners, foliage, weather, time of day |
| Want | **maximal invariance** | **moderate invariance** |
| Cost | near zero | high |
| Priority | **first** | targeted only |

**The over-invariance trap.** A kart circuit is a small piece of tarmac where
every corner resembles every other. The banners, the fence panels, the one tree
— those *are* the localization signal. Train too much scene-side invariance and
aliasing gets **worse**. The measured aliasing tail already shows there is no
margin here. Keep aliasing a first-class metric so this is visible if it
happens.

Appearance augmentation cannot corrupt an `s` label, because `s` is invariant
to appearance.

---

## Evaluation

| Rule | Reason |
|------|--------|
| Hold out **entire tracks**, never frames | frame-level splits leak the map and report fantasy numbers |
| Hold out **entire laps**, never frames, within a track | same reason, one level down |
| Report delta error in **milliseconds** as well as metres | ms is the product |
| Report **worst case**, not just median | a driver remembers the lap that read 0.8 s wrong |
| Sweep error against **viewpoint separation**, don't aggregate | a flat average hides "matches viewpoint, not place" |
| Track the **sim → real** drop as a standing metric | the health of the entire sim investment |
| Both negative controls, every time | a good score without them may be leakage |

Gates are implemented in `train/train.py` (`--gate g0|g1|g2`) and
`train/eval.py` (`gates`, `lines --axis line|yaw`, `leakage`).

### Negative controls

**Wrong-reference.** Pair live clips with another track's reference. Error must
collapse to chance (`N/4` bins). If it does not, the model is answering from
the live clip alone and the reference is decoration.

**Leakage.** Train a single-frame, no-reference absolute regressor. On seen
tracks succeeding is *expected* — memorising one circuit from pixels is easy,
which is why an absolute regressor is the wrong factorisation rather than a
bug. The diagnostic is the **held-out track**: absolute position on an unseen
circuit is not a learnable function of appearance, so a good score there means
something on screen is giving the answer away. Currently a weak instrument — it
also scores at chance on seen tracks, so it rules out a blatant indicator
without demonstrating its own sensitivity. It matters far more on real footage,
where a HUD is a live risk rather than a theoretical one.

### Kill criteria

| Condition | Conclusion |
|-----------|------------|
| Can't beat 1.5 m median on held-out **sim** tracks | architecture is wrong — **passed, 1.41 m** |
| Sim-pretrained + real footage can't beat 150 ms on the home circuit in good light | sim investment isn't paying; go real-data-first |
| Aliasing unfixable at indoor venues | narrow the product to outdoor circuits |
| Sustained 15 Hz impossible in the thermal envelope | reduce rate, lean harder on the filter |

---

## Rejected factorizations

Recorded so they are not relitigated.

| Approach | Why not |
|----------|---------|
| **Single frame → absolute `s` regressor** | Bakes the map into weights: one model per track, and absolute position on an unseen circuit is not a transferable function of appearance. Retained only as the leakage control. |
| **Retrieval only (nearest neighbour on embeddings)** | Trained with 10–25 m positives; the similarity surface is a plateau, not a metre-level peak. |
| **Direct pair-of-videos → delta** | Sidesteps `s`, so no line offset, no sector attribution, no debuggable bottleneck. |
| **Three heads (R / A / M) on a pretrained trunk, with distillation** | Never built. A 0.67M from-scratch encoder with one head reached the precision target, so the complexity was not needed. Revisit only if real footage proves capacity-limited. |
| **Cancelling head pose via phone gyro** | Rate is not pose; bias integrates to 6–60° per lap; yaw is one DOF of six. Replaced by trained invariance plus abstention. |
| **Sequence transformer over map frames as context tokens** | Data-hungry, quadratic in map length, no calibrated uncertainty, hard to deploy. |
| **3D Gaussian-splat map + render-and-compare** | Genuinely metric, too heavy for a phone at 15 Hz today. |
| **IMU-only descriptors / motif matching** | No longitudinal observability on a straight; turn shape not discriminative between similar corners. |

---

## Risks

| Risk | Mitigation |
|------|------------|
| **Sim2real gap** — the largest open risk, entirely unmeasured | capture-side randomization first; real footage next; track the drop as a standing metric |
| **Perceptual aliasing** — measured, ~1–2% of ticks, unfixed by data | particle filter; same-circuit hard negatives; cap scene-side augmentation; keep the aliasing metric |
| **Capacity on real footage** — unknown; real scenes carry far more texture than the renderer | do not size the model on synthetic; measure on real |
| **Field of view / aspect mismatch** across glasses, AC, and shared maps | canonical-FOV crop at pack time; store intrinsics per session |
| **HUD leakage** on real captures | HUD off; crop the timecode band; leakage control on every real dataset |
| **Glasses do not fit inside a full-face karting helmet** | non-algorithmic and unresolved; test with hardware before further engineering |
| **Glasses ↔ phone clock offset** | delta accuracy is bounded by timestamp accuracy; calibrate explicitly, target ≤ 10 ms |
| **Kart vibration** (no suspension) → blur | capture-side augmentation must include it; measure on real footage early |
| **Stale map after a model update** | store a weight hash with the map; refuse to load a mismatch |
| **Thermals** over a 25 min session | 15 Hz, small encoder, fp16 |

---

## Build order

1. **AC capture — validation first.** One track, two or three laps, pushed all
   the way through `decode` before driving a real session. The failure to avoid
   is 30 minutes of undecodable footage.
2. **Full AC session**, then pack, then `train.preview pair` on real frames
   before any training. If a human cannot tell which reference frame matches,
   the labels are misaligned and no training will fix it.
3. **Re-run every gate on real footage**, both negative controls included. This
   is the actual feasibility answer.
4. **Particle filter** against synthetic `s` trajectories. No ML, testable
   today, and the only thing that addresses the aliasing tail.
5. **Capture-side augmentation**, measured as a sim→real drop.
6. Core ML port, on-device latency and thermals.

Steps 1–3 need a Windows box and a person; step 4 does not. They are
independent and should run in parallel.

---

## File map

```
docs/
  README.md           scope, closed paths, reading order
  ml-pivot.md         this document — active design
  img/                synthetic-data previews

capture/              AC + OBS + timecode overlay (Windows-only)
train/                encoder, correlation head, gates, synthetic renderer
tests/                overlay wire format, fake-recording e2e, sampler invariants
```

Everything from packing onward is portable; only capture is Windows-only.
