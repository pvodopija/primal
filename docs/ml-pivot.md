# ML pivot — learned track-progress localization from POV video

> **Supersedes the IMU-descriptor plan.** The previous version of this document
> described learned encoders over Core Motion signals. That path is closed: an
> IMU-only descriptor cannot resolve position on a straight, and turn shape is
> not discriminative enough between similar corners to carry a live time delta.
>
> This document describes a different system: **visual** track-progress
> localization against a reference lap, trained mostly in simulation, running
> on-device. It does not reuse the DSP pipeline, the `Feature` type, or the
> motif matcher.

---

## Goal

Given **one reference lap as context**, estimate progress along it from live
POV video plus IMU, with metre-level precision and calibrated uncertainty.

Everything the product shows is a lookup on top of that estimate:

| Output | Derivation |
|--------|------------|
| **Time delta** | `t_live − t_ref(s)` |
| **Delta rate** ("gaining / losing now") | `d/dt` of the above |
| **Lateral line offset** | Head A's `Δlateral` |
| **Sector / corner attribution** | integrate delta rate between fixed `s` bounds |

No GNSS at runtime. GNSS/RTK is used freely at **training and evaluation**
time — the constraint is on the product, not the dataset.

---

## Precision budget

The whole design is driven by one number. A kart at 15 m/s covers 1 m in
**67 ms**. So "metre-level position" and "sub-100 ms delta" are the same
requirement.

| Quantity | Karting value |
|----------|---------------|
| Speed range | 11–22 m/s (40–80 km/h) |
| Lap time | 45–60 s |
| Track length | 1.0–1.2 km |
| Reference map rate | 30 Hz → 1350–1800 frames/lap |
| Map spacing at speed | 0.37–0.73 m per frame |
| **Target delta error** | **≤ 100 ms** (≈ 1.5 m) |
| **Target delta jitter** | ≤ 50 ms std over a 1 s window |
| Live inference rate | 15 Hz |

Note the consequence: published place-recognition benchmarks score "correct if
within 25 m." That is ~1.7 **seconds** of delta at kart speed. Off-the-shelf
retrieval models are trained to be *invariant* to the distinction we need to
*measure*. This is the single most important fact about the problem.

---

## Vocabulary

Terms used throughout, including standard ML jargon.

### Network structure

| Term | Meaning |
|------|---------|
| **Parameters / weights** | The learned numbers inside a network. Fixed after training, shipped in the app, identical for every user and track. |
| **Trunk** | The large shared front of the network that turns pixels into a numerical description. Synonyms in the literature: **backbone**, **encoder**. Holds ~90% of parameters and compute. |
| **Head** | A small network on the trunk's output that answers one specific question. Several heads share one trunk so the expensive part runs once per frame. |
| **Feature grid** / **patch tokens** | The trunk's spatial output: a coarse grid (e.g. 16×16) where each cell is a vector describing that patch of the image. Retains *where*. |
| **Pooling** | Collapsing the grid into a single vector. Discards *where*, keeps *what*. |
| **Embedding** / **descriptor** | A fixed-width vector summarising an image. Two images of the same place should have nearby embeddings. |
| **Frozen / fine-tune** | A frozen block's weights are not updated during training; fine-tuning updates them slowly from a pretrained starting point. |
| **Distillation** | Training a small **student** network to imitate a large **teacher**. How a phone-sized model inherits a research model's accuracy. |

### Geometry and state

| Term | Meaning |
|------|---------|
| **s** | Arc length along the track centreline, in metres. "Track progress." The system's primary state. |
| **Δs** | Longitudinal offset between two views, in metres. Head A's main output. |
| **Δlateral** | Sideways offset between two views — the racing-line difference. |
| **Δyaw** | Heading difference between two views, in degrees. |
| **Ego-motion** | Self-motion between consecutive frames. Frame-to-frame **visual odometry** (VO). |
| **Loop closure** | Recognising a return to a previously visited place. Here it is free: the track is a loop, every lap closes it. |
| **Metric scale** | Whether outputs are in real metres. Monocular vision is scale-ambiguous in general; recoverable here because camera height above a flat surface is fixed and measurable. |

### Training

| Term | Meaning |
|------|---------|
| **Contrastive learning** | Training embeddings by pulling matching pairs together and pushing non-matching pairs apart. **InfoNCE** and **triplet loss** are the common objectives. |
| **Hard negative mining** | Deliberately training against the *most confusable* non-matches. Here: other corners of the same circuit. |
| **Aleatoric / heteroscedastic uncertainty** | Input-dependent noise the network predicts alongside its answer. Trained with a **Gaussian NLL** (negative log-likelihood) loss instead of L2, so the network can say "I don't know." |
| **Correlation volume** | Explicit all-pairs similarity between two feature grids. The standard mechanism for learned correspondence (optical flow, stereo, VO). |
| **Cross-attention** | Transformer mechanism letting one set of tokens read from another. Alternative to a correlation volume for the same job. |
| **Domain randomization** | Randomising nuisance factors in simulation so the model doesn't overfit to the renderer's look. |
| **Sim2real gap** | The accuracy drop when a sim-trained model meets real footage. Our standing health metric. |

### System and deployment

| Term | Meaning |
|------|---------|
| **Bayes filter** | Recursive estimator maintaining a probability distribution over state, updated by a motion model and sensor observations. **Particle filter** is the sample-based variant we need (handles multi-modal beliefs). |
| **Perceptual aliasing** | Two different places looking the same. The dominant failure mode at kart circuits. |
| **Map** | The stored per-track data produced by one reference lap. Not weights. |
| **ANE** | Apple Neural Engine — the phone's NPU. Reached via **Core ML**. |
| **fp16** | 16-bit floats. Standard on-device weight/activation format. |
| **RTK** | Real-Time Kinematic GNSS — centimetre-accurate positioning. Training/eval labels only. |

---

## Core principle

**The map is context, not parameters. A prompt, not a weight.**

Two kinds of numbers, kept strictly separate:

| | Weights | Map |
|---|---------|-----|
| Content | *how to compare two views of a place* | *what this track looks like* |
| Produced by | offline training | one reference lap |
| Varies with | model version | track |
| Size | ~20–40 MB | ~40 MB |
| Cost to add a track | — | one lap |

Everything downstream follows from this. It is why a new circuit costs a lap
instead of a GPU day, why the system has calibrated uncertainty, and why it is
debuggable.

---

## Rejected factorizations

Recorded so we don't relitigate them.

| Approach | Why not |
|----------|---------|
| **Single frame → absolute `s` regressor** | Bakes the map into weights. One model per track; new track needs data collection + retraining. Discards the user's reference lap, which is the best-matched data that will ever exist for that circuit. No uncertainty. Undebuggable. And absolute position on an unseen track is not a transferable function, so sim training buys nothing. |
| **Retrieval only (nearest-neighbour on embeddings)** | Ceiling is the map's frame spacing, and in practice much worse: retrieval models are trained with 10–25 m positives, so the similarity surface is a plateau, not a peak. Cannot reach 1.5 m. |
| **Direct pair-of-videos → delta** | Sidesteps `s` entirely, so no lateral offset, no sector attribution, no interpretable bottleneck, no way to diagnose a bad reading. |
| **Sequence transformer with map frames as context tokens** | The literal reading of "map as prompt," and the trendiest option. Data-hungry, no calibrated uncertainty, quadratic in map length, hard to deploy. Revisit at scale, not now. |
| **3D Gaussian-splat map + render-and-compare** | Genuinely metric 6-DoF and increasingly practical. Map build and per-frame rendering are both too heavy for a phone at 15 Hz today. Park it. |
| **IMU-only (previous plan)** | No longitudinal observability on straights; corner shapes not discriminative enough. See supersedes note. |

---

## System overview

```
                     live frame (256x256x3)
                               |
                    +----------v-----------+
                    |        TRUNK         |   ~15-20M params
                    |  pretrained backbone |   ~10 ms fp16 on ANE
                    +----------+-----------+
                               |
              16x16 grid of 128-d vectors  +  512-d pooled vector
                               |
        +----------------------+----------------------+
        |                      |                      |
   +----v----+           +-----v-----+          +-----v-----+
   | HEAD R  |           |  HEAD A   |          |  HEAD M   |
   | coarse  |           |  precise  |          | ego-motion|
   +----+----+           +-----+-----+          +-----+-----+
        |                      |                      |
   likelihood over        dS, dLat, dYaw          dS, dYaw
   all map frames         + sigma, vs K            + sigma,
   (dot product)          candidates               vs previous frame
        |                      |                      |
        +----------------------+----------------------+
                               |            +---- IMU (100 Hz)
                    +----------v-----------+
                    |      ESTIMATOR       |   particle filter
                    |  state: (s, v)       |   no learning
                    +----------+-----------+
                               |
                     s = 412.3 m  +/- 0.8
                               |
                    +----------v-----------+
                    |   PRODUCT OUTPUTS    |   t_ref(s) lookup
                    +----------------------+
                               |
                 delta -0.24 s | rate -0.03 s/s | line +0.4 m
```

The trunk runs **once per live frame**. Reference frames are encoded once, at
map-build time, and never again — the map stores trunk *outputs*, not images.

---

## Blocks

### Trunk

| | |
|---|---|
| **In** | 256×256×3 image |
| **Out** | 16×16×128 feature grid, plus 512-d pooled vector |
| **Size** | ~15–20M params, ~10 ms fp16 on ANE |
| **Init** | pretrained vision backbone, frozen initially |

Must emit **spatial** features, not only a pooled vector — Head A needs
correspondence, which requires knowing *where*. Pool for R, keep the grid
for A and M.

Start from a pretrained backbone with the trunk frozen and only heads trained.
Unfreeze the last stages later if head-only training saturates. Training a
trunk from scratch on our data volume is strictly worse and should not be
attempted.

### Head R — coarse retrieval

| | |
|---|---|
| **In** | 512-d pooled vector |
| **Out** | projected descriptor; compared to all map descriptors by dot product |
| **Result** | likelihood over map positions, ±15 m |
| **Trained on** | real footage primarily (appearance-dependent) |

The head is only the projection. The comparison is a dot product — arithmetic,
not a network.

Two non-standard choices, both deliberate:

**Distance-calibrated targets.** Instead of binary positive/negative, regress
the similarity toward `exp(−|Δs| / τ)` with `τ ≈ 5–8 m`. Binary targets with a
25 m threshold produce a plateau; a calibrated target produces a peak with
usable gradient.

**Same-circuit hard negatives.** The hardest negatives are not other tracks —
they are the *other corners of this track*. Mine them explicitly every epoch.
Aliasing is the dominant failure mode and this is the main lever against it.

**Multi-yaw map entries.** A pooled descriptor is the one place head yaw
genuinely hurts: a query 40° off-axis will not match a reference recorded
head-on. Fix it in the map, not in the query — crop several virtual yaw views
from each wide-FOV reference frame and store a descriptor per crop. Five crops
takes the descriptor table from 1.8 MB to 9 MB and the dot product from
< 0.5 ms to ~2 ms. Both remain negligible. Train R with matching yaw
augmentation so the crops are usable.

Output must be a distribution, not an argmax. The estimator needs the full
shape, including secondary peaks.

### Head A — precise alignment

The block the product depends on.

| | |
|---|---|
| **In** | live feature grid + stored feature grid of one candidate map frame |
| **Out** | `Δs`, `Δlateral`, `Δyaw`, each with predicted σ |
| **Range** | trained for `Δs ∈ [−15, +15] m` |
| **Runs on** | K = 3–5 candidates per frame |
| **Trained on** | simulation (perfect labels), fine-tuned on real |

Mechanism: correlation volume or cross-attention between the two grids, then a
small regression head. This is the same architecture family as learned optical
flow and stereo — study RAFT to understand it.

Why it transfers: this is a **relative geometry** function. It reads parallax,
scale change, and structure, not track identity. It never needs to know which
circuit it is looking at, so training it on sim tracks produces weights valid
on a kart circuit it has never seen. This is where "metre-level precision
encoded in the model" genuinely lives, and it is the correct answer to the
question the rejected `s`-regressor was trying to answer.

**Build order note:** evaluate a pretrained feed-forward geometry model
(VGGT-class; see *Prior art*) as Head A before training one. If accuracy holds,
distil it to phone size. Training this head from scratch is the fallback, not
the plan.

### Head M — ego-motion

| | |
|---|---|
| **In** | feature grids of the last two live frames |
| **Out** | `Δs`, `Δyaw` since previous frame, with σ |
| **Trained on** | simulation (perfect labels) |

Never touches the map. Two jobs:

1. **Runtime:** dead reckoning through featureless straights, occlusion by
   other karts, and glare — the regimes where R and A have nothing to grip.
   Looming (scale expansion of approaching structure) is real longitudinal
   information, unlike double-integrated IMU.
2. **Map build:** integrating Head M over the reference lap produces **metric
   arc length with no GNSS**. Loop closure at the lap boundary redistributes
   scale drift around the circuit.

Metric scale is recoverable because camera height above a flat surface is
fixed and measurable. Verify this assumption early — it is load-bearing.

### Estimator

Not a neural network. Classical recursive Bayesian estimation over state
`(s, v)`.

| Input | Role |
|-------|------|
| Head R likelihood | coarse, multi-modal observation |
| Head A output | precise local correction, weighted by its σ |
| Head M + IMU | motion model |

Requirements:

- **Particle filter**, not Kalman — the belief is genuinely multi-modal during
  acquisition and after an excursion.
- **Multi-hypothesis** at session start, on pit exit, and whenever confidence
  collapses.
- **Widen, don't drift.** On a straight where Head A's σ is large, uncertainty
  must grow rather than the estimate becoming confidently wrong.
- Emits `s` with covariance. Downstream suppresses or greys the display when
  covariance exceeds a threshold. Showing nothing beats showing a wrong delta.

This is the half of the system with no machine learning in it, and it is
roughly half the work. It can be built and tested today against synthetic `s`
trajectories with no network at all.

### Map builder

One reference lap in, one map file out. Runs live during the lap at 30 Hz.

Contents:

| Field | Size (60 s lap) |
|-------|-----------------|
| Pooled descriptors, 1800 frames × 5 yaw crops × 512, fp16 | 9 MB |
| Feature grids, every 3rd frame, 600 × 16×16×128, fp16 | 39 MB |
| `s(i)` from Head M integration + loop closure | negligible |
| `t_ref(s)` | negligible |
| Thumbnails for debug UI | ~5 MB |
| Trunk weight hash | — |

**Maps are only valid for the trunk that produced them.** Store the trunk
hash and refuse to load a mismatch. Silent degradation from a stale map is
otherwise a very expensive class of bug.

`t_ref(s)` need not come from a single lap — a synthetic reference composed of
best sectors is a straightforward product feature once the estimator is
trustworthy.

---

## Head pose and the phone IMU

**Decided: POV camera on the glasses, phone in pocket.** No chassis-mounted
camera. This section records how head motion is handled given that decision.

### The trap: cancellation

The tempting plan is to recover head yaw by differencing two rates — camera
yaw rate from the image, vehicle yaw rate from the phone — and then
canonicalize the view before embedding. The phone half of that is sound:
`ω · û` (gyro projected onto the gravity direction) is invariant to how the
phone sits in the pocket, and in a kart the driver's hip is effectively pinned
to the seat with no suspension in between, so torso yaw tracks chassis yaw
well.

The plan still doesn't hold, for three reasons:

| Problem | Detail |
|---------|--------|
| **Rate ≠ pose** | Head yaw offset is `∫(ω_cam − ω_veh) dt`. Phone gyro bias is 0.1–1 °/s after warm-up, so blind integration drifts 6–60° over a 60 s lap. |
| **One DOF of six** | Head pose is 3 rotations + 3 translations. Gravity-referenced yaw rate addresses one differential. Pitch moves the horizon across the whole feature grid; roll is significant under helmet mass; head translation is 10–30 cm — the same order as the `Δlateral` we are trying to measure. |
| **Sync** | Differencing requires aligning glasses frames to phone IMU within ~10 ms. Glasses frames arrive over a wireless link with variable latency. |

If cancellation is attempted anyway, **high-pass, never integrate**. Head yaw
is genuinely zero-mean over a few seconds while vehicle yaw is not, so a
complementary filter with a 2–5 s time constant is stable where pure
integration is not.

### The approach: estimation, not cancellation

Head A already outputs `Δyaw`. Head pose is therefore not an error term to be
removed — it is a quantity the network reports.

| Requirement | Consequence |
|-------------|-------------|
| Head A trained across the real pose range | sim renders the same `s` at arbitrary head yaw / pitch / roll / offset, with perfect labels |
| Head R robust to yaw | multi-yaw crops in the map (see Head R) + yaw augmentation in training |
| Filter aware of pose uncertainty | large predicted σ on extreme head angles widens the belief rather than corrupting it |

### What the phone IMU is actually for

Not view canonicalization. Its jobs:

| Job | Value |
|-----|-------|
| Motion model for the estimator | vehicle yaw rate + longitudinal acceleration at 100 Hz, independent of vision |
| Cross-check on Head M | detects visual ego-motion failure during glare, blur, or occlusion |
| Bridging dropped frames | wireless link degradation is expected; IMU carries the gap |

This is genuinely valuable and worth the integration cost. It is simply a
different job from the one it was originally proposed for.

### Upside: head pose is a product feature

Where the driver looks is a coaching signal, not only a nuisance. "Eyes up,
look through the corner" is standard karting instruction, and `Δyaw` versus a
reference lap makes it measurable — late apex fixation, missed corner exit,
head still pointed at the previous turn. A chassis-mounted camera cannot
produce this. It is the strongest argument for the POV decision beyond the
eyes-up HUD.

### Open non-algorithmic risk

Physical fit of camera glasses inside a full-face karting helmet is unresolved
and independent of everything above. Test it with hardware before committing
further engineering.

---

## Runtime and memory budget

The "won't passing the whole map be slow?" question, answered numerically.
Reference frames are encoded once; runtime only reads precomputed tensors.

| Operation | Cost per live frame |
|-----------|--------------------|
| Trunk (1 image) | ~1–5 GFLOP, ~10 ms |
| Head R vs **entire** map (1800 frames × 5 yaw crops) | 9000 × 512 = 4.6M MAC ≈ 9.2 MFLOP, **~2 ms** |
| Head A × K=4 candidates | ~100 MFLOP, ~2–3 ms |
| Head M | ~50 MFLOP, ~1 ms |
| Estimator | negligible |
| **Total** | **trunk + ~50%**, ≈ 16 ms |

At 15 Hz that is roughly 25% duty cycle on a modern phone. Cost does **not**
grow with lap length in any way that matters — scoring the full map is under
1% of one trunk pass, so always score all of it rather than a window. The full
similarity vector is what makes aliasing visible and recovery possible.

Map build: 1800 frames × ~10 ms ≈ 18 s of compute, absorbed live during the
reference lap. The map is ready at the line.

The same amortization applies in training: encode a batch of frames once
through the shared trunk, then form all pairs and windows *within* the batch.
Never re-encode a reference frame per query.

---

## Training

### What is trained where

| Block | Primary data | Why |
|-------|-------------|-----|
| Trunk | pretrained; fine-tuned last | volume we can't match |
| Head A | **sim**, fine-tuned on real | needs dense geometric labels; transfers across tracks |
| Head M | **sim** | same |
| Head R | **real** | appearance-dependent; sim tracks are the wrong domain |
| Estimator | not trained | — |

Sim is used precisely where it is uniquely valuable: perfect dense geometry on
tracks whose identity doesn't matter. Real data is used where it is uniquely
valuable: appearance of the actual circuits, and all evaluation.

### Losses

| Head | Loss |
|------|------|
| R | distance-calibrated similarity regression + InfoNCE with same-circuit hard negatives |
| A | Gaussian NLL on `(Δs, Δlateral, Δyaw)` — predicts σ, not just the value |
| M | Gaussian NLL on `(Δs, Δyaw)`; optional photometric consistency as auxiliary |

Every regression head predicts uncertainty. This is not a refinement — the
estimator is unusable without it, and it is what makes "I don't know on this
straight" an expressible output rather than a silent error.

### Distillation

Standing strategy: **train or evaluate big offline, ship small.**

Keep a large teacher (a VGGT-class geometry model for A, a large retrieval
model for R) purely as an accuracy ceiling and a distillation target. Any
accuracy gap between teacher and on-device student is a known, quantified,
recoverable quantity. Without a teacher you cannot tell a capacity problem
from a data problem.

### Staging

Each stage produces a component that is independently testable.

| Stage | Work | Gate |
|-------|------|------|
| 0 | Eval harness + RTK-labelled real lap | delta error measurable in ms |
| 1 | Head M, sim only | median `Δs` error on held-out sim tracks |
| 2 | Head M + σ + capture augmentation | sim→real drop quantified |
| 3 | Estimator on synthetic `s`, no network | recovers from injected kidnapping |
| 4 | Head R on real footage | ±15 m recall + aliasing metric |
| 5 | Head A — evaluate pretrained first, then distil | ≤ 1.5 m median on held-out sim tracks |
| 6 | Full loop offline on recorded real laps | ≤ 100 ms delta error |
| 7 | Core ML port, on-device latency + thermals | 15 Hz sustained for a 25 min session |

---

## Data

### Simulation

| | |
|---|---|
| **Prototype** | Assetto Corsa via shared memory |
| **Label source** | `normalizedSplinePosition` — arc length projected onto the track spline, [0,1]; × track length for metres |
| **Lateral label** | world position minus the corresponding spline point |
| **Rate** | 60+ Hz |

Effectively free, exactly the label we want. Two limits to plan around: AC
cannot render faster than real time, so an hour of data costs an hour of wall
clock on a Windows box; and its EULA is not a foundation for commercial
training data.

A third limit, added by the POV decision: training Head A over the real head-
pose range needs **programmatic per-frame camera pose relative to the vehicle**
— arbitrary yaw, pitch, roll, and translation offsets with labels. AC's camera
system does not expose this cleanly, which makes head-pose randomization the
strongest single argument for owning the renderer.

If this scales, own the renderer — UE5 / CARLA / Isaac, headless, faster than
real time, programmatic camera and scene randomization, no licence question.
Also note that **no sim has kart circuits**, which is exactly why only the
track-agnostic heads (A, M) are trained here.

### Real footage

**GPS-free is a runtime constraint, not a dataset constraint.** A dual-band
RTK receiver on the kart gives centimetre-accurate position for a few hundred
euros; project onto a fitted centreline for `s`.

An afternoon at the home circuit produces perfectly-labelled data in the exact
target domain — cheaper than building one sim track, and of far higher value
for Head R. Real+RTK is also the only acceptable source for evaluation.

### Augmentation policy

Two kinds of "style" exist and they need **opposite** treatment.

| | Capture-side | Scene-side |
|---|---|---|
| Examples | exposure, white balance, motion blur, rolling shutter, lens distortion, vignetting, sensor noise, compression, resolution tier changes | textures, barriers, banners, foliage, weather, time of day |
| Want | **maximal invariance** | **moderate invariance** |
| Cost | near zero — an image-degradation pipeline over the renderer | high — generative augmentation |
| Priority | **do this first** | targeted only |

Capture-side randomization buys most of the sim2real win for approximately
zero GPU cost. Do it before anything expensive.

Scene-side is where generative augmentation (Cosmos Transfer-class: 2B params,
~65 GB VRAM, 93-frame chunks, roughly 8–20 GPU-hours per video-hour with the
distilled variant) belongs. Viable as targeted augmentation at 10–50 hours;
absurd as the whole pipeline.

**The over-invariance trap.** A kart circuit is a small piece of tarmac where
every corner resembles every other corner. The banners, the specific fence
panels, the one tree — those *are* the localization signal. Train too much
scene-side invariance and Head R learns to ignore precisely the cues that
separate turn 2 from turn 6. Aggressive appearance randomization makes
aliasing **worse**. Measure aliasing as a first-class metric so this is
visible when it happens.

One useful property: appearance augmentation cannot corrupt an `s` label,
because `s` is invariant to appearance. Geometry-preserving conditioning keeps
labels valid for free.

---

## Evaluation

Define this before writing training code.

| Rule | Reason |
|------|--------|
| Hold out **entire tracks**, never frames | frame-level splits leak the map and report fantasy numbers |
| Report **delta error in milliseconds** vs RTK | recall@N is not the product |
| Break out by regime: corner / braking / straight | straights are the weak regime and will hide in an average |
| Report **worst case**, not just median | a driver remembers the one lap that read 0.8 s wrong |
| Track **sim-only → real-only** drop as a standing metric | the health of the entire sim investment |
| Measure **aliasing** directly | for each map position, similarity to its most confusable other position on the same circuit |
| Measure **recovery time** after injected kidnapping | pit exit, spin, off-track |

### Kill criteria

| Condition | Conclusion |
|-----------|------------|
| Head A can't beat 1.5 m median on held-out **sim** tracks | architecture is wrong; stop and rethink |
| Sim-pretrained + 20 h real can't beat 150 ms on home circuit in good light | sim investment isn't paying; go real-data-first |
| Aliasing unfixable at indoor venues | narrow the product to outdoor circuits |
| Sustained 15 Hz impossible within thermal envelope | reduce rate and lean harder on Head M |

---

## Prior art

| Idea | Source | Relevance |
|------|--------|-----------|
| **Hierarchical localization** | coarse retrieval → fine local matching, ~2019 onward | our overall decomposition; the mainstream paradigm |
| **Correlation-volume correspondence** | RAFT and successors | Head A's mechanism |
| **Learned VO with uncertainty** | DPVO, DROID-SLAM lineage | Head M |
| **Feed-forward geometry transformers** | DUSt3R → MASt3R → **VGGT** (CVPR 2025) → VGGT-Ω, OmniVGGT | candidate pretrained Head A; evaluate before training |
| **Relocalization on those models** | Reloc3R, Reloc-VGGT | closest published work to Head A's exact job |
| **Distance-calibrated retrieval** | CosPlace / EigenPlaces lineage, with our threshold tightened to metres | Head R |
| **Aleatoric uncertainty** | Kendall & Gal | σ heads |
| **Sequence-based place recognition** | SeqSLAM | conceptual ancestor of using the temporal prior; superseded here by a proper filter |

The genuinely modern move relative to a naive reading of this document: write
less of it. Start from pretrained weights for the trunk, evaluate a pretrained
geometry model as Head A, and distil rather than train from scratch.

---

## Risks

| Risk | Mitigation |
|------|------------|
| **Perceptual aliasing**, especially indoor venues | same-circuit hard negatives; cap scene-side augmentation; multi-modal filter; measure aliasing explicitly |
| **Sim2real gap** | capture-side randomization first; real+RTK fine-tuning; track the drop as a metric |
| **Metric scale assumption** (fixed camera height) breaks | validate in stage 1; fall back to scale from loop closure only |
| **Head pose** decorrelated from vehicle motion (POV mount is decided) | estimate, don't cancel — Head A reports `Δyaw`, trained over the real pose range in sim; multi-yaw crops in the map for Head R |
| **Glasses do not fit inside a full-face karting helmet** | non-algorithmic and unresolved; test with hardware before further engineering |
| **Glasses ↔ phone clock offset** | delta accuracy is bounded by timestamp accuracy; calibrate the offset explicitly, target ≤ 10 ms |
| **Kart vibration** (no suspension) → blur | measure early on real footage; capture-side augmentation should include it |
| **Occlusion** by other karts in a full field | Head M carries through; filter widens rather than jumps |
| **Indoor light flicker** beating against frame rate | fixed exposure if the capture API allows; measure at the worst venue early |
| **Thermals** over a 25 min session | 15 Hz, 256 px, fp16; vented mount, not a pocket |
| **Stale map after model update** | trunk hash in the map file; hard refusal on mismatch |
| Building the sim before the eval harness | stage 0 is the gate; nothing else starts first |

---

## Build order

1. **Eval harness + one RTK-labelled real lap.** Offline replay, delta error in
   milliseconds, regime breakdown. Nothing else starts before this.
2. **Estimator on synthetic trajectories.** Pure math, no network, immediately
   testable. Builds the half of the system that has no ML in it.
3. **AC capture harness + Head M.** Easiest head to validate, most transferable
   result. Overfit 100 samples to zero loss before anything else.
4. **Capture-side augmentation + σ heads.** Measure the sim→real drop. This
   number decides how much more sim work is worth doing.
5. **Head R on real footage.** Aliasing metric from day one.
6. **Head A** — evaluate pretrained VGGT-class first, distil second, train
   from scratch only as fallback.
7. **Full offline loop**, then Core ML port, then on-device.

Stage 1 and 2 are independent and can proceed in parallel. Stage 3's real-data
result is the highest-information cheap experiment available and should not be
deferred.

---

## File map

This document lives in the `locamotif-ml` repository (`ml/` submodule of
hotlapp). It does not share code with the `locamotif` IMU crate. Start at
[`README.md`](README.md) in this folder.

```
docs/
  README.md           scope, closed paths, reading order
  ml-pivot.md         this document — active design
  img/                synthetic-data previews

capture/              AC + OBS + timecode overlay
train/                encoder, correlation head, gates, synthetic renderer
tests/
```
