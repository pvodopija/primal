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

## Landscape

### What drivers use today

GPS lap timers are the comparison a driver will make. Their accuracy claims
mix best cases with specifications:

- The 25 Hz consumer GNSS modules this class of timer is built on are specified
  at **1.5 m CEP**: half of all fixes within 1.5 m (u-blox M10). Racelogic's
  VBOX Sport states **±2 m at 95%**. RaceBox advertises "precision as fine as
  10cm" and gives no accuracy figure in its technical specifications.
- **Phones are worse:** about 4.9 m typical under open sky (GPS.gov), usually at
  1 Hz, and smoothed or snapped for navigation. Fixes off the track and sudden
  jumps are normal on a phone.
- **Centimetres need RTK**: a base station sending live corrections.

A live delta needs consistency more than accuracy. Most GNSS error drifts over
tens of minutes, so two laps a few minutes apart share it. What remains is
jitter. Against a reference lap from another day, the whole drift remains, and
because it is a shift in space, its effect along the track changes sign with
the track's heading. Lap *times* cancel constant errors at the line and are far
better than any live delta.

Estimated, not measured — live delta error at kart speed (11-22 m/s):

| Situation | Along-track error | Delta error |
|---|---|---|
| Dedicated 25 Hz timer, open sky, reference from the same session | 0.2-0.5 m | ~10-45 ms |
| Dedicated timer near trees, grandstands, buildings | 0.5-2 m | ~25-180 ms |
| Dedicated timer, reference from another day | 1-3 m | ~45-270 ms, varying through the lap |
| Phone GPS | ~5 m, with jumps | several hundred ms |

Measured by Racelogic: aligning two laps at Silverstone National by distance
instead of position gave 0.3 s of error. Timers that predict from wheel speed or
distance drift like that through a lap.

Against the precision budget's 1.5 m, PRIMAL holds 1.6 m median on trained
tracks and 4.3 m on an unseen one (*Estimator*). Its millisecond figures are
measured at AC car speeds, faster than a kart, so metres are the fair
comparison. A video reference does not drift between days, the one row where
GPS is structurally weak.

Garmin Catalyst (about $1000, dash-mounted, for cars) combines a camera with
10-25 Hz GNSS and inertial sensors, marketed as "True Track Positioning". How
much the camera contributes to position is not published.

### Prior art for the matcher

The matcher is built from standard parts: a shared (Siamese) encoder and an
all-pairs similarity grid, as in SiamFC tracking and FlowNet/RAFT optical flow;
a map given as input rather than memorised, as in OrienterNet; and SeqSLAM's
sequence matching, which it beats on real footage by 9x on an unseen track and
29x on trained ones, and by 40-150x across a change of car and weather
(*Against a classical baseline*). The code and weights here are written and trained from
scratch. What is specific is the combination: progress against the driver's own
reference lap, time-spaced bins that read out as delta, no GNSS, a glasses
camera, and a phone. No literature or patent search has been done.

---

## Hardware target

### Decided

**The delta is shown on the glasses.** No phone mount: the phone is compute and
storage, the glasses are camera and display. Where the network runs is open:
on the phone, split between glasses and phone, or entirely on the glasses.

### What the network costs

Counted on `runs/ac2_time_allframes` (667 k parameters, 1.3 MB at fp16). Only
the newest frame is encoded each tick; the other clip frames are cached.

| Reference length | Encoder (per frame) | Head over all bins | Correlation | Per tick | At 15 Hz |
|---|---|---|---|---|---|
| Vallelunga, 860 bins | 34.7 M MACs | 73.8 M | 1.3 M | 110 M | 1.65 G MAC/s |
| Silverstone, 1300 bins | 34.7 M | 111.6 M | 2.0 M | 148 M | 2.22 G MAC/s |
| Black Cat, 3227 bins | 34.7 M | 277.0 M | 5.0 M | 317 M | 4.75 G MAC/s |

One tick takes 2.3 ms on a single Mac CPU core in fp32, against a 67 ms budget.
On a phone's neural engine (tens of TOPS) the arithmetic is negligible; the
power is waking the accelerator, estimated at tens of mW and not measured. The
head scans every bin, so it grows with track length; restricting it to a window
around the tracker's position would cut it if that ever matters.

The encoder accepts any resolution. Its cost at other inputs, as a share of a
46 GOPS Ethos-U55 at theoretical peak (real utilisation is typically 30-50%, so
busy time is 2-3x these):

| Encoder input | M MACs/frame | 15 fps | 30 fps | 60 fps |
|---|---|---|---|---|
| 148x80 (today) | 34.7 | 2% | 5% | 9% |
| 160x120 (4:3) | 54.9 | 4% | 7% | 14% |
| 224x168 | 107.5 | 7% | 14% | 28% |
| 320x240 | 216.6 | 14% | 28% | 57% |
| 640x480 | 862.9 | 56% | 113% | 225% |

### Meta Ray-Ban glasses, through the Wearables Device Access Toolkit

- Video is **streamed over Bluetooth** to the phone: at most 720p at 30 fps,
  lowered automatically when bandwidth is short. A third-party SDK guide lists
  presets 360x640, 504x896 and 720x1280 (portrait) at 2-30 fps. No third-party
  code runs on the glasses.
- The display exists only on Ray-Ban Display ($799), with the EMG wrist band
  for gestures.
- Ray-Ban Meta Gen 1 has a 154 mAh battery; Meta estimates about 30 minutes of
  livestreaming, at 5 °C or warmer, cut short by heat. Gen 2 claims twice the
  battery life for moderate use.
- **Apps built with the toolkit cannot be published** outside restricted test
  channels (developer preview, as of 23 September 2026).

With the stream received on the phone, the phone side is estimated, not
measured, at roughly 0.2-0.6 W with the screen off: Bluetooth receive,
hardware decode, the network, app overhead. That is 1-2% of an iPhone battery
per 25-minute session. **The glasses are the constraint**, not the phone.

### Brilliant Labs Halo

- **Camera:** PixArt PAG7982J1, 640x480, **global shutter**, 81.2° horizontal
  (about 65° vertical, 4:3), 40 mW at full frame rate. Brilliant does not state
  the rate; a module vendor lists the sensor at **120 fps at VGA** over
  MIPI/DVP. Whether Halo's microcontroller can take it in is untested: full VGA
  at 120 fps is about 37 MB/s and 8.3 ms per frame, so a high rate would need a
  small window or binning on the sensor.
- **Compute:** Alif Balletto B1, Cortex-M55 at 160 MHz with an **Ethos-U55 NPU**
  (128 MACs per cycle, about 46 GOPS), 2 MB SRAM, 1.8 MB MRAM.
- **Radio:** Bluetooth LE 5.3.
- **Display:** colour micro-OLED, 640x480 panel; the Lua API draws a 256x256
  round area.
- **Other:** 300 mAh battery, accelerometer and magnetometer (**no gyroscope**),
  bone-conduction speakers, about 40 g, $399.
- **Software:** open firmware (Zephyr RTOS with a Lua runtime, Apache-2.0; the
  Alif SDK parts are proprietary to Alif silicon), flashed over the air. The
  stock SDK only takes single JPEG photos: no continuous capture and no NPU
  API. Both would be our own firmware, in C with TensorFlow Lite Micro and
  Arm's Vela compiler.

**Two ways to use it.** The encoder on the glasses, sending embeddings to the
phone, which runs the head and the tracker and returns the delta; or everything
on the glasses, with the phone only storing reference laps and, optionally,
sending a speed reading. By the tables above both fit the NPU: the encoder at
160x120 and 30 fps is about 7% of peak; encoder plus head for Silverstone at
30 Hz about 22%. Memory is estimated to fit in 2 MB: weights about 0.7 MB at
8 bits, one reference lap 170-410 KB, activations a few hundred KB.

**Embeddings instead of video** remove the link as a bottleneck. One embedding is
128 numbers, about 136 bytes at 8 bits with a timestamp:

| Over Bluetooth, 30 fps | Per second | Fits in Bluetooth LE (est. 50-150 KB/s to a phone)? |
|---|---|---|
| 480p JPEG video | ~0.75-1.2 MB (est.) | no |
| 160x120 JPEG video | ~90-180 KB (est.) | borderline |
| **embeddings** | **~4 KB** (8 KB at 60 fps) | **a few % of it** |

What that buys: no video stream, which is the power drain Meta limits to
about 30 minutes; frames timestamped at capture, so link delay changes when
the number appears (est. 40-80 ms) but not its accuracy, and camera, tracker
and display can share one clock; a global shutter against kart vibration; and
control of exposure against motion blur. The whole pipeline is estimated at
150-250 mW, several hours on 1.1 Wh. Unmeasured.

Higher frame rates barely help the matcher: consecutive ticks' errors are
already 84-96% correlated at 15 Hz and decorrelate only over about a second,
so more frames within that second do not average them out, and between ticks
the tracker's dead reckoning is accurate once speed is known. They help a lot
with **measuring motion**. At 120 fps and 30 m/s the car moves 0.25 m per frame,
so the road 3 m ahead grows about 8% between frames (17% at 60 fps) and shifts
half as many pixels, with no rolling-shutter distortion. That is the regime
where frame-to-frame flow is easy, and it is exactly what defeated the
encoder's motion vectors. The global shutter is not what fixes that: AC renders
each frame at one instant, so our footage is already global-shutter, and the
motion vectors failed on it. Frame rate and dense flow are the fix; the global
shutter keeps it valid on hardware, where a rolling shutter read out over tens
of milliseconds would turn head turns and kart vibration into shear and
wobble, an error that grows with motion. Hence a **two-rate design**: the place encoder at
15-30 fps sending embeddings, and motion measured at up to 120 fps on a small
road window on the glasses, reduced to one speed per tick for the tracker.
Speed is the largest measured lever, so this is where a fast camera pays.
Whether the chip can match or flow a road window at 120 fps is a hardware
question. The model was trained with frames 1/15 s apart, so other rates mean
sampling or retraining.

**The speed lane, as planned.** A band of road 3-10 m ahead at full
resolution, every frame. About 100 textured points tracked frame to frame
(pyramidal Lucas-Kanade), each also tracked back: a point that does not return
to where it started is a mismatch, an occlusion or another kart, and is
dropped. The survivors go through the ground-plane fit above (forward, sideways,
yaw, pitch, roll), and the 4-8 frames of a tick reduce to one timestamped
speed with a quality score: the share of points that survived the round trip,
and how well the frame predicted from that motion matches the next real one.
The tracker's scale state absorbs the camera height, which is not known per
driver. No network outputs speed: both learned-speed and block-matching
attempts produced errors that grow with speed. A small learned flow model is
the fallback if tracking fails on bland tarmac or in low light, and would
still leave the metres to the geometry. Estimated cost of sparse tracking at
120 fps is 5-10% of the M55.

`capture/flow_speed.py` implements it for recordings: frames scaled to Halo's
focal length, the 3-10 m band, forward-backward tracking, and the ground-plane
fit with exact translation terms (the first-order ones read 5-10% fast at
0.25-0.5 m per frame). On a rendered flat road at 30 m/s it is exact at 120 fps
(87% of points survive the round trip) and within 0.6% at 60 fps (47%), and it
breaks at 30 fps (9%). That is synthetic, without noise or blur; the first real
test is the AC 720p/60 recordings, then `--skip 2` for 30 fps, then the tracker
on the Silverstone streams.

Resolution is not the constraint. The matcher uses 148x80, 1/26 of VGA. For
motion, one pixel covers about 1.3 cm of road 5 m ahead on Halo (focal length
about 373 px), against 6.9 cm in the packed frames where classical flow found
no texture, and 0.8 cm in the AC recordings. So a flow test on the 720p footage
should first be scaled down to Halo's focal length, or it measures a sharper
camera than the product will have. Halo's pixels are also read uncompressed on
the glasses, where Meta's stream is compressed video that smears fine texture.
Low-light noise, dynamic range and exposure are the camera questions left, and
need the hardware.

**What the model must change for the Ethos-U55:** GroupNorm and GELU are not
native operators. GroupNorm was chosen because live and reference frames pass
through the encoder in very different batch sizes, so replacing it (BatchNorm
with care over its statistics, or another supported normalisation) needs
retraining and re-verifying against today's gates, as does 8-bit quantisation.
The head's dilations up to 8 and its circular padding must compile under Vela.
The field of view differs from the AC footage (81°x65° against 91°x58°), so AC
has to be captured or cropped to Halo's (*Cameras and field of view*).

**Unknowns, in order:** whether the camera streams continuously under our own
firmware, and at what rate for a small window (up to 120 fps would enable
on-glasses speed); the encoder's real latency and power on
the NPU; Bluetooth throughput to an iPhone; 25-minute battery and temperature;
fit inside a full-face helmet. Taps on the frame will not work under a helmet,
and voice competes with the engine, so starting a reference lap needs another
input: detecting the start line when the lap closes on itself, setting up on
the phone, or a hand gesture seen by the camera in the pits.

### Other glasses considered

| Glasses | Why not first |
|---|---|
| Rokid Glasses (Snapdragon AR1, 49 g, 210 mAh, micro-LED) | strong compute, small battery for a phone-class chip; unclear whether custom models can run through its SDK |
| RayNeo X3 Pro (AR1, Android, 76 g) | reviewers report about 30 minutes with the camera on |
| INMO Air3 (Android, 119 g), Snap Specs (132 g, $2,195) | too heavy for a helmet |
| Mentra Live (open source, 12 MP, 119°, landscape) | no display |
| Even Realities G1/G2, Vuzix Z100 | no camera |
| Meta Ray-Ban, Ray-Ban Display, Oakley Meta | closed; video streamed to the phone; cannot publish yet |

Halo is the only light, open glasses with a camera, a display and an NPU. It is
also early (firmware at 72 commits when checked) and just shipping. The
network stays hardware-neutral: the map is context, so the same weights serve
either platform, and Meta remains the distribution target once its toolkit can
publish.

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

**The reference is a grid built from the lap's own labels.** Each reference lap
is resampled onto N bins. Bin k sits at axis coordinate k/N and holds the frame
nearest it, found around the loop so the finish line is not an edge. A live
frame's target is its own axis coordinate times N, so a live frame at the same
place as bin k's frame targets exactly k. Every bin also records its track
position and the reference lap's elapsed time there, so errors in metres and in
milliseconds are read straight off the grid. The milliseconds are the difference
in reference time between the predicted and the true place, which is exactly
the delta error.

**Bins are spaced in reference time, not distance** (`--reference-axis`, default
`time`). A time bin is the same slice of delta everywhere: about 1 m apart in
slow corners and 2.5 m on straights, at the same bin count. A 2 m distance bin
instead spans anything from 28 ms on a fast straight to 191 ms in a slow
corner. Either way a bin is a *place*: the reference lap is a finished
recording, so standing still keeps you in one bin while your own clock runs.

A reference lap with a hole wider than one bin is not used as a reference: a bin
whose picture shows somewhere else is a wrong answer baked into the map. It
still serves as a live lap, where a hole does no harm.

**Every clip frame is supervised**, not just the last (`--aux-all-frames`,
default on): each frame's correlation row is trained against its own position.

**An estimator tracks progress over time** — see *Estimator*.

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

**On real footage** the same holds. Measured with the current model
(`ac2_time_allframes`) against SeqSLAM on identical forward-only clips, 960 per
gate, median error:

| pair type | G1 PRIMAL | G1 SeqSLAM | G2 PRIMAL | G2 SeqSLAM |
|---|---|---|---|---|
| same session | 1.68 m | 3.89 m | 2.14 m | 4.86 m |
| cross time of day | 1.47 m | 579 m | 4.36 m | 43.9 m |
| cross car and weather | 1.58 m | 233 m | 4.29 m | 189 m |
| **all** | **1.59 m / 52 ms** | 46.6 m / 1139 ms | **3.56 m / 94 ms** | 31.1 m / 783 ms |

SeqSLAM is competitive only when nothing changes between the two laps; any
change in light, car or weather sends it toward chance.

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

## Measured, on real footage

19 Assetto Corsa sessions over six tracks, 143 laps, 148x80 square pixels,
Silverstone National held out entirely. Trained from scratch on this data; the
synthetic checkpoints were not used as initialisation. Current model: time bins,
every frame supervised (`runs/ac2_time_allframes`).

| gate | median | p90 | within 5 | entropy |
|---|---|---|---|---|
| seen laps, seen tracks | 1.33 m / 41.8 ms | 4.49 m / 145 ms | 98.8% | 2.259 |
| **G1** held-out laps | **1.45 m / 46.1 ms** | 4.55 m / 147 ms | 99.7% | 2.273 |
| **G2** unseen track | **3.58 m / 95.9 ms** | 87.3 m / 1860 ms | 78.1% | 3.126 |
| wrong-reference control | 1.28 m -> 577.19 m | | | PASS |
| leakage control | seen 4.15 bins, holdout 58.47 (chance 64) | | | PASS |

The leakage control measures the data rather than the model, and the data has
not changed since it passed. Milliseconds are exact reference-time differences;
earlier figures divided metres by local speed, which misstates slow corners.

### What this established

**Synthetic pretraining does not transfer, and the architecture was never the
problem.** Evaluated zero-shot on real footage, a synthetic checkpoint read
464 m against a 479 m chance level — indistinguishable from guessing. The same
architecture trained on real footage reaches 1.45 m on held-out laps. G0 on a
single real lap pair reaches 0.17 m. Nothing was wrong with the model, the
packing or the labels; the sim-to-real gap is simply total.

**The binding constraint is number of tracks.** Lap generalisation costs 1.09x
(1.33 -> 1.45 m). Track generalisation costs 2.7x (1.33 -> 3.58 m). Six
circuits is what limits G2, not laps per circuit and not capacity. Nothing that
has improved held-out laps has moved the unseen track.

**The leakage control is now a working instrument.** On synthetic it scored at
chance on seen tracks as well as held-out ones, so a null result could not
distinguish "no leak" from "broken detector". On real footage it clearly
succeeds where memorisation is possible (4.15 bins against chance 64) and fails
where it must (58.47 bins). Its PASS now carries evidential weight.

**The aliasing tail survived the move to real data, and it is what blocks the
product.** G2's median sits at the 100 ms budget; its p90 is 1860 ms and 22% of
ticks land outside five bins against 0.3% on G1. On an unseen circuit the model
is usually right and occasionally catastrophically wrong. No amount of data has
moved this, on synthetic or real. See *Estimator* for what does.

### Where the model looks

Masking thirds of the frame, equal area, measured on the trained model:

| masked | holdout | train |
|---|---|---|
| top third (sky) | 2.26x | 4.95x |
| bottom third (tarmac) | 1.10x | 1.50x |
| **middle third (horizon)** | **187x** | **341x** |

Localisation happens in the horizon band — barriers, trackside furniture,
distant track geometry. Tarmac contributes almost nothing. Two things follow:
the model has a single narrow dependency with no redundancy, which matters when
framing changes; and sky reliance is twice as strong on seen tracks as unseen,
which looks like a session-specific memorisation shortcut rather than genuine
signal, and is an argument for random sky masking during training.

### Robustness already present

Degrading only the live clip, since the reference is a stored map: held frames
1.08x, irregular arrival 1.05x, blur 1.01x, all combined 1.06x on the held-out
track. The existing sampler already covers this — `p_static` is a wireless
stall and the stride ladder is irregular timing. The blur figure is optimistic;
a box filter is not low-bitrate H.264.

### What moved the numbers, and what did not

| change | measured effect | verdict |
|---|---|---|
| Grid fix (frames at bin centres against targets at bin starts; finish-line search) | accuracy unchanged | the model had learned the half-bin offset; the bug corrupted the demo videos, the SeqSLAM check and the metrics |
| Loss weighted by 1/speed | slow corners 55% -> 45% over budget, fast straights 17% -> 20%; overall 24% -> 25% | no net gain: the model is precision-limited, not allocation-limited. Removed |
| Time bins | slow sections up to ~20% less delta error on whole-lap streams, neutral at speed | modest gain exactly where the budget is tightest. Default |
| Every clip frame supervised | G1 26% -> 21% of ticks over 100 ms; G2 unchanged | best model on trained tracks. Default |
| Clip span 0.37 s -> 0.73 s at inference | Silverstone single-shot p90 2.3 s -> 0.8 s | more temporal context is cheap robustness against look-alikes |

One training seed each; the per-speed comparisons were checked on both random
clips and whole-lap streams before being believed, and the first read of time
bins overstated them.

## Estimator

`train/estimator.py` is a particle filter over (track position, speed) that folds
in one belief per tick, as the phone would receive them. No learning.

- **Predict**: advance each particle by its speed, with room to brake and accelerate.
- **Update**: weight it by the belief at its position — tempered, because ticks
  are correlated and the softmax is overconfident, and floored, so a confidently
  wrong tick only dents the true place.
- **Recover**: redraw a small share of particles from the belief every tick at
  a tiny prior weight. One wrong tick barely registers; a few ticks of
  consistent contradicting evidence take over. That is "recover as soon as the
  driver looks forward again".
- **Output**: the densest particle cluster's position, and its share of the
  weight as confidence.

Measured with `python -m train.eval stream`: whole cross-session laps at 15 Hz,
clip frames 1/15 s apart (a 0.73 s span), the first 2 s of each lap left out as
acquisition. Streams are harsher than the random-clip gates: every pair crosses
sessions, and every Silverstone live lap is an MX-5 against Abarth references.

| | median | p90 | >100 ms | >10 m | worst |
|---|---|---|---|---|---|
| G1 single-shot | 1.66 m / 59 ms | 151 ms | 25.2% | 0.8% | 110 m |
| G1 filter | 1.66 m / 58 ms | 158 ms | 26.2-26.6% | 0.7% | 37 m |
| G2 single-shot | 4.60 m / 125 ms | 1558 ms | 59.5% | 22.1% | 1272 m |
| G2 filter | 4.34 m / 117 ms | 313 ms | 56.9-57.4% | 15.9% | 1178 m |

**It removes the tail and leaves the median alone.** The worst case on trained
tracks falls from 110 m to 37 m, and the unseen track's p90 from 1.6 s to 0.3 s.

What a driver would see, as the share of filter ticks off by more than a
distance:

| | > 2 m | > 4 m | > 6 m | > 10 m |
|---|---|---|---|---|
| G1, trained tracks | 42% | 14% | 4% | 0.7% |
| G2, Silverstone | 75% | 53% | 37% | 16% |

The demo videos show the same: on Silverstone, misses of several metres are
routine, not occasional.

### Why the median does not move

Per-tick errors are correlated noise rather than a fixed bias. Consecutive
ticks share most of their clip frames (lag-1 autocorrelation 0.84-0.96), but
errors decorrelate within about a second, so they *could* be averaged — by
carrying position accurately across more than a second, which needs the speed.
The filter can only infer speed from the same noisy position stream, so it
cannot average without lagging through braking zones. On the unseen track the
remaining failures are sticky: the aligner stays confidently wrong for 1-3 s,
and after a few ticks the filter follows it. A constant per-pair bias of up to
0.9 m sits underneath, consistent with the inter-session label offsets.

### Speed is the missing input

The same streams, with the filter also given the true speed from the labels
(a 0.1 s central difference) plus controlled error. Ranges are over three
filter seeds.

| speed given to the filter | G1 >100 ms | G2 >100 ms | G2 >10 m | G2 median |
|---|---|---|---|---|
| none | 26.2-26.6% | 56.9-57.4% | 15.9% | 117 ms |
| **true, stated to +-2 m/s** | **6.1-6.8%** | **12.4-16.5%** | **0.6%** | **52-55 ms** |
| true, +3% random per tick | 9.9% | 20.6% | 2.7% | 60 ms |
| true, 5% slowly drifting error | 50.3% | 71.6% | 17.1% | 152 ms |
| true, 10% slowly drifting error | 68.8% | 71.0% | 27.0% | 171 ms |

**An unbiased speed signal is worth more on the unseen track than everything
else measured put together.** It takes Silverstone from 57% of ticks over budget
to 12-17%, and catastrophic errors from 16% to under 1%, worst case 14.7 m.
With it the filter wants to trust vision far less (likelihood power 0.2) and
dead-reckon between fixes.

**A slowly drifting speed is worse than none.** The filter believes it and
dead-reckons away from the truth. A speed source must be unbiased, or the filter
must carry its bias as a state — the standard way to fuse a drifting inertial
sensor with position fixes, which vision provides here.

The filter now carries it (`speed_scale_walk`): each particle also guesses the
sensor's current scale, and the vision fixes decide which guesses survive. On a
synthetic stream whose sensor reads 0.85x true speed and later 1.10x, position
error falls from 16 m to 1.3 m, and the learned scale follows to 0.84 and 1.09.

The filter must also be told a realistic uncertainty. Given the exact speed but
told it was good to +-0.3 m/s, it collapsed its speed spread and did worse
(21% over budget on G1) than when told +-2 m/s (6.5%).

### How often, how clean, how late

Measured on the six Silverstone streams with true speed fed at a rate, with
noise, or late; likelihood power tuned on G1 for each row.

| Speed signal | G2 median | >100 ms | >10 m |
|---|---|---|---|
| none | 4.1 m / 109 ms | 54% | 13% |
| every tick (15 Hz) | 1.9 m / 52 ms | **12%** | 0% |
| 5 Hz | 2.0 m / 56 ms | 14% | 0% |
| 2 Hz | 2.2 m / 59 ms | 19% | 0% |
| **1 Hz** | 2.5 m / 70 ms | **29%** | 0.2% |
| every 2 s | 3.0 m / 82 ms | 41% | 4% |
| every 5 s | 3.8 m / 102 ms | 51% | 12% |
| 15 Hz, +-0.5 m/s noise | 2.0 m / 54 ms | 13% | 0% |
| 15 Hz, 0.4 s late | 2.5 m / 68 ms | 21% | 0% |
| 1 Hz, +-0.5 m/s noise | 2.6 m / 71 ms | 30% | 0.3% |
| 1 Hz, 0.4 s late | 2.8 m / 75 ms | 34% | 0.5% |

**5 Hz is nearly as good as every tick; 1 Hz gives about half the gain and
removes the catastrophes; every 5 s does almost nothing**, because braking
changes speed by 10 m/s in a second. **Noise of +-0.5 m/s is harmless; lateness
is not.** A delay-aware update, comparing a timestamped reading with each
particle's speed at that time, should recover most of the latency loss. Not
built.

`python -m train.preview infer ... --speed-sigma 2.0 --speed-every 15` films a
lap with a second tracker given the labelled speed (`--speed-every` in ticks).
On the cross-car Silverstone demo lap it goes from 3.87 m / 106 ms median and
52% over budget without speed to 1.59 m / 42 ms and 6% with speed every tick.

### Where the speed can come from

- **Not from the reference match.** Reading speed off the slope of the clip's
  path through the reference is a finite difference of per-frame positions, so
  it cannot beat per-frame precision divided by the clip span: 11% error on
  trained tracks and 33% on Silverstone even with every frame supervised, and
  it makes the filter worse. A line search over candidate slopes (SeqSLAM's
  velocity search on learned rows) is immune to look-alike frames — 0.03 against
  1-2 bins per step for a per-frame fit on synthetic rows — but cannot beat that
  bound either. Measured, and removed.
- **Frame-to-frame visual motion**, independent of the reference. At the packed
  148x80 resolution it is not good enough:
  - Classical flow (DIS, Lucas-Kanade) fitted to the road plane reads near zero.
    Tarmac that small has no texture to track, and rotation dominates what does
    move.
  - A small network trained on frame pairs (reversed pairs must read negative;
    random gaps; static pairs) reaches 7% median error on trained tracks and 13%
    on Silverstone, but hedges toward the average: it reads fast stretches low
    and slow ones high. With the bias state it helps trained tracks (26.5% ->
    22.2% of ticks over budget, worst 37 m -> 11 m) and not Silverstone (56.6%
    -> 55.9%).
  - The filter tolerates 15% random speed error (Silverstone 21% over budget)
    and a slowly varying 13% (28.5%). What it cannot use is error that follows
    the speed itself, which is what the network's hedge is.

- **The video encoder's own motion vectors**, at full resolution. Measured on
  eight sessions at 720p and one at 360p (`capture/motion_vectors.py`: x264
  `veryfast`, P-frames only, 60 fps) and fitted per frame to a ground plane
  with sideways drift and three rotations. **Not usable as recorded:**
  - The encoder stops tracking the road near the car as speed rises. It codes
    those blocks from scratch instead: on Black Cat the lower half is 82%
    tracked at 10-20 m/s and 25% at 40-50 m/s; on Silverstone 21-29% at every
    speed, and barely better at 360p (22-34%), where the motion in pixels
    halves. Near the
    car the road zooms (about 17% per frame 3 m ahead at 30 m/s), which a
    block that can only shift cannot follow.
  - What remains is the far road, where motion is sub-pixel. Fitted speed
    divided by true speed falls from 0.85 at low speed to 0.33 above 45 m/s on
    Black Cat, and sits at 0.1-0.35 on Silverstone; correlation with true
    speed is -0.50 to +0.16.
  - Fed to the filter with the bias state on the Silverstone streams it is
    much worse than no speed: 14.5 m median and 90% of ticks over budget,
    against 4.3 m and 57%. An error that grows with speed is the one kind
    the bias state cannot follow.

  Open: whether a thorough search (`--x264-params me=umh:merange=64:subme=7`)
  recovers the middle distance. Even then, in the product the vectors would
  come from the glasses' encoder, whose settings are not ours, and phones
  decode with hardware that does not normally expose them.
- **The phone IMU**, already planned as a filter input. Integrated acceleration
  drifts; the filter's bias state and the vision fixes make the drift
  observable. Unbuilt. Halo has no gyroscope, so on that hardware an inertial
  speed would come from the phone.
- **GNSS Doppler, as a speedometer only.** Receivers measure velocity from the
  satellites' Doppler shifts, separately from position and far more cleanly:
  0.1-0.4 m/s horizontal on Android phones in motion (ISPRS 2022). iPhones
  report speed about once a second, which is the 1 Hz row above: Silverstone
  from 54% of ticks over budget to about 29%, more once latency is handled.
  It costs the phone 15-185 mW and the glasses nothing; an external 10-25 Hz
  GNSS unit over Bluetooth would be close to the oracle. Positioning stays
  visual. It would turn the product constraint "no GNSS at runtime" into "no
  GNSS positioning at runtime", which is undecided.
- **Pretrained visual odometry** (DPVO, DROID-SLAM, TartanVO, Monodepth2's pose
  network; the comma.ai speed challenge is the same task on dashcams). One
  camera gives speed only up to an unknown scale, and the filter's scale state
  can learn that scale from the vision fixes. Needs sharper frames than 148x80,
  which costs glasses power if it means sending more pixels. Untested; the
  cheapest first test is RAFT optical flow with the ground-plane fit above.
- **Not radio timing between phone and glasses.** They move together, so any
  Doppler shift between them measures only their relative motion. Ground speed
  needs a stationary partner: satellites, or a trackside beacon.

### Confidence

On an earlier model, filter confidence flagged catastrophic errors with AUC 0.77
but ordinary misses not at all (0.49 on G1): it reads 1.0 while locked onto the
wrong place. Hiding the least confident 30% of Silverstone ticks cut those more
than 10 m out from 16% to 9%. An "unavailable" readout needs a sharper signal,
most likely the disagreement between the filter's prediction and the incoming
belief.

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

### An independent speed signal

The largest measured lever on the unseen track, and unbuilt. It must be unbiased
or have its bias estimated; the filter can now estimate it. See *Estimator*.

### An abstain signal

Filter confidence does not yet separate ordinary misses from good ticks. See
*Estimator*.

### Lateral line offset

A product output in the goal table with no current source. The head predicts
progress only.

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

Halo's camera sees about 81°x65° at 4:3 against the AC footage's 91°x58°, so
targeting it means AC captured or cropped to Halo's field of view. Meta's
toolkit presets are portrait, which needs the same check.

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
`train/eval.py` (`gates`, `lines --axis line|yaw`, `leakage`). `stream` runs whole
laps through the estimator at 15 Hz, the way the product does, and with
`--speed-sigma` measures what an independent speed signal would be worth.

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
| **Sim2real gap** — measured and total: synthetic checkpoints score at chance on real footage | closed as a strategy. Train on real footage directly; synthetic is for plumbing and architecture only |
| **Perceptual aliasing** — the estimator removes the catastrophic tail, but 1-3 s sticky locks survive on unseen tracks | an independent speed signal (it takes those locks from 16% of ticks to under 1%); more circuits; cap scene-side augmentation |
| **Biased speed signal** — a 5% drift is worse than no speed at all | only fuse an unbiased source, or carry its bias as a filter state |
| **Capacity on real footage** — unknown; real scenes carry far more texture than the renderer | do not size the model on synthetic; measure on real |
| **Field of view / aspect mismatch** across glasses, AC, and shared maps | canonical-FOV crop at pack time; store intrinsics per session |
| **HUD leakage** on real captures | HUD off; crop the timecode band; leakage control on every real dataset |
| **Glasses do not fit inside a full-face karting helmet** | non-algorithmic and unresolved; test with hardware before further engineering |
| **Glasses ↔ phone clock offset** | delta accuracy is bounded by timestamp accuracy; timestamp frames at capture on the glasses; on Halo, camera, tracker and display can share one clock; otherwise calibrate explicitly, target ≤ 10 ms |
| **Glasses battery and heat** — Meta estimates about 30 minutes of livestreaming, cut short by heat | send as little as possible from the glasses: the lowest video preset, or embeddings instead of video (Halo); measure 25 minutes before building on either |
| **Halo's continuous capture is unverified** — the stock SDK takes single photos | a hardware test of capture rate at a small size before any port work beyond the model changes |
| **Input inside a full-face helmet** — frame taps impossible, voice against engine noise | detect the start line when the lap closes on itself; set up on the phone; a hand gesture seen by the camera in the pits; the wrist band on Meta Display |
| **Meta toolkit apps cannot be published yet** | prototype on Halo; keep the network hardware-neutral |
| **Kart vibration** (no suspension) → blur | capture-side augmentation must include it; measure on real footage early |
| **Stale map after a model update** | store a weight hash with the map; refuse to load a mismatch |
| **Lighting-dependent model bias** — the same car at dusk against noon reads about 1 m apart on the unseen track and 0.5 m on a trained one; a model-free match (ORB + RANSAC) shows the labels themselves agree within 0.2 m | more lighting variety in capture and augmentation; re-measure the dusk/noon asymmetry after each change |
| **Label accuracy** — 0.3 m is 20 ms at kart speed | measured model-free at 0.10-0.18 m between sessions, inside that; recheck when capture changes |
| **Patents on matching or localisation methods** — the matcher's parts are published and standard, but no search has been done | a freedom-to-operate check by a patent attorney before any commercial launch |
| **Thermals** over a 25 min session | 15 Hz, small encoder, 8-bit on an NPU; on the glasses the radio dominates, so embeddings over video |

---

## Build order

1. **An independent speed signal.** The largest measured lever on the unseen
   track. The filter side, estimating its bias, is built; a delay-aware update
   for timestamped readings is not. Encoder motion vectors failed as recorded
   (one thorough-search check remains). Candidates: GNSS Doppler speed (1 Hz
   on a phone; a product decision), the IMU (capture should log AC's own
   acceleration so a drifting IMU can be simulated with ground truth), and
   pretrained dense flow or visual odometry on sharper frames.
2. **Hardware feasibility on Halo.** Chip-friendly layers and 8-bit weights,
   verified against today's gates; a Vela estimate of cycles and memory; then
   on hardware: continuous capture rate, encoder latency on the NPU, Bluetooth
   throughput, 25-minute power and temperature, helmet fit. AC footage at
   Halo's field of view once it is chosen.
3. **More circuits.** Track generalisation costs 2.7x and nothing else has moved
   it.
4. **An abstain signal** sharp enough to grey out a wrong delta.
5. On-device port: Core ML on the phone and/or the Ethos-U55 on the glasses.

---

## File map

```
docs/
  README.md           scope, closed paths, reading order
  ml-pivot.md         this document — active design
  img/                synthetic-data previews

capture/              AC + OBS + timecode overlay (Windows-only), encoder motion vectors, flow speed
train/                encoder, correlation head, reference grid, estimator, gates, synthetic renderer
tests/                overlay wire format, fake-recording e2e, sampler and grid invariants, estimator, motion vectors, flow speed
```

Everything from packing onward is portable; only capture is Windows-only.
