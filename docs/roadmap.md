# Roadmap

The working plan: where PRIMAL stands, its strengths and weak spots, and the order
of the next steps. The measurements behind every number are in
[`ml-pivot.md`](ml-pivot.md); the requests to the Windows machine in
[`handoff.md`](handoff.md). Update the status column as steps move; rewrite the
plan when a result changes it.

**Goal:** a lap delta within 100 ms, from the glasses' camera, that teaches the
right lessons: a driver must be able to trust "I gained in that corner".

**Last reviewed:** 2026-10-06.

---

## Where we are

- **In the sim it works.** On Assetto Corsa (AC) tracks it never trained on, the
  Halo-sized MobileNetV3-Small is more than 100 ms off 7.6% of the time, median
  about 33 ms (ResNet-18: 7%, too big for Halo).
- **On real footage it works, never trained on real footage.** A kart GoPro clip:
  median about 58 ms, 26% over 100 ms. A historic F1 helmet camera at Brands Hatch GP
  (racing speed, traffic, head motion), cropped to the training field of view: median
  58 ms, 29% over 100 ms; every lap's final delta within 0.15 s; SeqSLAM, the
  classical baseline, 218 ms.
- **Driven live on AC** by Pavle with the overlay (Brands 40-57 ms, Noja 87 ms median).
- **Ahead or behind is answered right ~99% of the time** where the truth is clear, in
  the sim and on the F1 footage. **Gaining or losing over a few seconds is not yet
  reliable on real footage** (65% over 2 s, 75% over 5 s, balanced; 86-91% in the sim),
  and a second of hindsight does not fix it there: the real error is slow, either the
  matcher's or the reconstructed truth's. An independent truth (a GPS logger) decides.
- **Everything left is the step from a sim and internet clips to a human in a real
  kart with our own camera.**

---

## Two products, one engine

The same matcher and tracker can ship two ways:

1. **Video analysis, first:** a web or desktop tool. Upload a session video from any
   camera, get the delta drawn over it and the time gained or lost in every corner.
   No hardware to build, no real-time limit (hindsight is free), the field of view is
   measured from the laps themselves (`experiments/real_fov.py`), and it can earn money
   and, with permission, collect real footage before the glasses product exists.
2. **Live in the glasses, second:** the delta in the driver's view, Halo first. The
   premium product once the hardware is proven.

Hardware independence comes from a short camera contract (frame rate, field of view
wide enough to crop to the training view, a steady mount, timestamps; field of view
calibrated from the first laps) and one 8-bit model exported to common runtimes.

**The business side** (user conversations, competitors and how they started, licensing,
costs, funding, the business plan) runs in a separate chat; its results come back here
as decisions.

---

## SWOT

### Strengths: what we can rely on

| | |
|---|---|
| S1 | The method works on unseen AC tracks: 7.6% over 100 ms with a Halo-sized model |
| S2 | It carries over to real footage it never saw (median ~55-60 ms) |
| S3 | Head movement is understood and mostly fixable without retraining: the head-angle search takes a held 7° turn from 13.5% to 1.1%; knowing the head angle at every moment takes corner glances to 0.9%. Roll is the cheap one: a held 10° lean costs about 4-5 points, against about 26 for a 7° turn |
| S4 | Tools: the AC rig, the virtual camera, the imperfect-driver test, consistency metrics, the live loop (2.8 ms a frame on the Mac's GPU), the wrong-reference and leakage controls on every run |
| S5 | Camera only, nothing to install, works where GPS cannot (indoor tracks) |

### Weaknesses, ranked by how much each threatens the product

| | Weakness | Evidence |
|---|---|---|
| W1 | **Real-world proof is thin** | One internet clip, one session (the easy case), a reconstructed truth. No footage of our own, no measured truth, no kart driver's view (wheel, hands), no indoor track |
| W2 | **Never tested on a human driver** | A human-like varying pace about doubles the error (7.6% -> 17.6%) |
| W3 | **The delta wobbles** | ±30-50 ms over 1-2 s: one corner's gain is about the size of the noise |
| W4 | **Head turns on Halo** | Its 81° is narrow; looking into corners costs 20-22% even with the search |
| W5 | **Hardware untested** | MobileNet never compiled for Halo's NPU (Vela figures are the old encoder's), memory looks tight; continuous capture, power, heat, helmet fit unknown |
| W6 | **Different lines unmeasured on real footage** | Training covers only the rig's ~2.5 m sideways wander |
| W7 | **Missing pieces** | No abstain signal, no lap aligner (self-improving reference), no camera speed on Halo |

### Opportunities

| | |
|---|---|
| O1 | The sim rig with the live overlay: unlimited human laps with exact truth, and the "does it feel useful?" test |
| O2 | Fix it in the reference, not the model: a wide or stitched reference, the angle search, voting across laps. Cheap, no retraining |
| O3 | A "visual gyro" for the fast part of head movement: Halo has no gyroscope, but the camera measures its own rotation from frame to frame; take out the turn the reference made at that place, and keep it honest with the search |
| O4 | A display built for teaching: per corner or sector rather than a jittery number. Over 5 s the true change (158 ms) is twice the error (71 ms); over 2 s they are about equal |
| O5 | Pretraining was the biggest single lever; more general or real driving video may push it further |
| O6 | Indoor karting: GPS timers cannot work there |
| O7 | Train on how people actually look: both laps following the road ahead, so the common case is the trained case (see *Decisions in progress*) |
| O8 | Motorcycles: a delta in the helmet, where a dash is hard to read leaned over. Needs large roll tolerance (40-55° lean); GPS timers with predictive delta already exist there, so the edge is the display, not the sensor |
| O9 | **Head coaching, which no GPS timer can do:** the camera sits where the driver looks, so it can measure how early and how far the head turns into each corner and tie that to the corner's time gain from the delta. Head only: telling tangent point from future path needs an eye tracker |

### Threats

| | |
|---|---|
| T1 | Real karts are harsher: vibration with no suspension, sun glare, visors, rain, other karts |
| T2 | Halo may not stream continuously or fit under a helmet, forcing a hardware change |
| T3 | **Trust:** phantom corner gains (wobble, looking into corners) teach the wrong lessons; one bad session can end a coaching tool |
| T4 | GPS lap timers (AiM, MyChron, RaceBox) are cheap and good outdoors; the edge is indoor tracks, no installation, and the delta in the glasses |
| T5 | Limited time, compute and budget; the pull to keep polishing sim numbers |
| T6 | Patents unchecked; internet footage cannot be used commercially |

---

## How to prioritise

**First find out, cheaply, what could kill the product; polish only after that.**
Sim precision is no longer the bottleneck. The big unknowns sit on the real-world side:
a human driving (W2), real footage (W1), the glasses (W5).

---

## The plan

| # | When | Step | Fixes | Effort | Status | What would change the plan |
|---|---|---|---|---|---|---|
| 1 | Now | **Live AC overlay, driven by Pavle** (Windows builds it, [`handoff.md`](handoff.md) 2026-10-03) | W2, W3, T3 | Driving + log analysis | Done 2026-10-04: stops, pace and confidence fixed; Noja lean open, its drive tests sent to Windows | If it feels useless even when accurate, step 2 matters more than accuracy |
| 2 | Now | **A delta for teaching:** per corner / sector gained or lost (live ahead/behind is ~99% right; gaining/losing over 2-5 s is not), a 1 s hindsight option in the live loop, scored on existing laps, then on the live logs | W3, T3 | Hours, Mac | Not started | |
| 2b | Now | **Video analysis prototype (product 1):** a session video in, the delta drawn over it and a per-corner gained/lost table out; any camera, its field of view measured from the laps; shown on the F1 and kart footage | Products | 1-2 days, Mac | Proposed | If per-corner calls are not reliable on real footage, it waits for the GPS-truth footage |
| 2c | Now | **Camera contract and export:** the few things any camera must provide, and the model in 8-bit ONNX and TensorFlow Lite | Products, W5 | ~1 day, Mac | Proposed | |
| 2d | Now | **GoPro to phone, live:** the GoPro livestreams RTMP (480p/720p) to a local server while saving full quality to its SD card; measure the delay, then run the live loop on it. Needs a GoPro (HERO10 or newer) and an RTMP server on the Mac first | Products | An evening | Proposed | If the delay is several seconds, live use falls back to per-corner voice calls after each corner |
| 3 | Now | **Compile MobileNet for Halo with Vela:** size, NPU time, unsupported layers | W5 | ~1 hour, Mac | Not started | If it does not fit: a smaller backbone or distillation, before anything else |
| 4 | Now | **Wider wander from Windows:** wide-FOV re-renders of existing replays, wander about ±4 m, look-into-corner off, `lateral_m.npy`; fixed ±3 m offset laps on the holdouts. Then fine-tune the current model on it | W6 | Windows rendering, ~1 h Mac | Requested | If lines far apart cost precision on the usual line, keep it to a share of clips |
| 5 | Now | **Measure how much heads really move:** on the real kart clip, the yaw, pitch and roll between laps at the same place (from the ORB matches); at the track day, a helmet GoPro's own gyro log | W4 | ~1 hour Mac; then the track day | Proposed | Sets the ranges for step 5b and how much head-turn work is needed at all |
| 5b | Now | **Natural head on both laps:** both live and reference follow the road ahead with the head doing part of what the eyes do, and lean (roll) into corners, each lap with its own habit, made exactly from the wide renders on the Mac; fine-tune and compare | W4, O7 | Half a day + ~1 h fine-tune | Proposed | If both-laps-looking (5.6% today) does not drop, the search carries it alone |
| 6 | Next | **Real footage of our own with truth:** a helmet camera at eye height plus a GPS logger, several sessions and days, an indoor track if possible; evaluate, then fine-tune on it | W1, T1 | A track day or two | Waiting on a track day | If still over ~40% after fine-tuning, rethink before hardware work |
| 7 | Next | **Halo on hardware:** continuous capture, NPU speed, Bluetooth, 25 minutes of power and heat, helmet fit | W5, T2 | Needs the glasses | Waiting on the glasses | If capture or fit fails, change the hardware path |
| 8 | Meanwhile | **Head turns:** a reference stitched from several laps (simulated on the wide renders), and the visual gyro (the camera's own frame-to-frame rotation, the kart's turn taken out) with the search | W4, O2, O3 | Days, Mac | Not started | |
| 9 | Meanwhile | **Tracker for human pace,** tuned on the live AC laps from step 1 | W2 | Days, Mac | Waiting on step 1 | |
| 10 | Later | Lap aligner -> voting across laps, a self-improving reference; an abstain signal | W7 | Weeks | | |
| 11 | Later | Wider data: the kart driver's view (wheel, hands pasted in), kart tracks, indoor AC mods | W1 | Ongoing | | |
| 12 | Later | Patent freedom-to-operate check before anything commercial | T6 | A patent attorney | | |
| 13 | Later | Motorcycles: roll tolerance to 40-55° (de-rotate by a measured roll, or a roll search), bike footage, helmet fit | O8 | Weeks | | |
| 14 | Later | Head coaching: head angle against the direction of travel (the point the scene flows out of), per corner, related to the corner's time across laps. Needs steps 2, 5 and 8 first | O9 | Weeks | Idea | If head habits do not relate to corner times in real laps, drop it |

**Paused:** more AC precision tuning (new backbones, more seeds) and the FOV sweep
(Halo's view is fixed; the sweep only informs later glasses). Either comes back if
a real-world test points there.

### How new data reaches the model

New data does not need a full retrain. Every model is trained in two stages: about
2 hours from ImageNet (6000 steps), then a clean fine-tune of 1500 steps (~35
minutes). New data is added by continuing from the current checkpoint for 1500-3000
steps with the new laps **mixed into** the old ones (new data alone makes it forget),
against a control that continues the same number of steps without them. A full
retrain only when the architecture changes (for example the NPU-friendly
normalisation) or after several fine-tunes stack up.

---

## Decisions in progress

- **Train on natural gaze (step 5).** What hurts is the head angle *between* the
  live lap and the reference at the same place, not the head angle itself. A driver
  with a steady habit looks the same way at the same corner every lap, so if both
  laps follow the road ahead, that difference stays small. Today's models were
  trained mostly on views fixed to the car plus random held poses. Measured: both
  laps looking into corners costs 5.6% against 0.9% straight ahead.
- **The camera follows the head, not the eyes.** For small shifts of gaze the eyes do
  most of the work, so the head turns less than the gaze. The test head that turns
  into corners (median 4°, 90th percentile 11°) aims at where the kart will be in
  a second, which is where the eyes look; the head probably turns less. Step 5
  measures it.
- **Roll is handled in part:** training includes ±8° of roll, and a held 10° roll
  costs about 4-5 points (MobileNet 9.2% -> 13.9%, cross-session) where a 7° turn
  costs about 26. Roll leaves the middle of the picture in place and loses only
  its corners, and a lean that repeats every lap cancels against the reference.
- **Halo has no gyroscope** (accelerometer and compass only, [`ml-pivot.md`](ml-pivot.md),
  *Motion sensors*): the fast head angle has to come from the camera itself.
- **Pending answers from Pavle:** when the Halo glasses are in hand; when a kart
  track day is possible. They set the order of steps 6 and 7.
