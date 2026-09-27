# How the matcher is trained

What happens between a recorded Assetto Corsa session and a checkpoint, as the
code does it today. Defaults are the ones the current model
(`runs/ac2_time_allframes`) was trained with. Design reasons and results live
in [`ml-pivot.md`](ml-pivot.md); this document is the mechanics.

## 1. From recording to packed laps

`capture/pack.py`, run on the Windows box, turns a session (an OBS recording plus
the labels decoded from the on-screen barcode) into laps:

1. **Labels per video frame.** `labels.parquet` holds, for every frame of the
   mp4, AC's render counter and the camera's track position `s` (0-1 around the
   lap), decoded from the barcode AC draws into the corner of the picture.
2. **Duplicates dropped.** AC renders at about 60 Hz and OBS samples at exactly
   60 Hz on its own clock, so OBS sometimes catches one render twice. A frame
   whose counter did not change is dropped.
3. **Laps split** where the counter jumps by more than 4 renders (a stall), runs
   backwards, or `s` wraps from ~1 to ~0 (the finish line). Pieces shorter than
   120 frames are discarded, so partial laps at session start and end survive
   only if long enough; they are never used as references (below).
4. **Pixels.** The barcode band is cut off (with a 4 px margin), and the rest is
   resized with area averaging to the packed size, keeping pixels square.
   `packed_ac_v2` is **148x80, BGR, uint8**: 91.5° horizontal by 57.8° vertical.
5. **Files per lap:** `frames.npy` [N, 80, 148, 3], `s.npy` [N], `t.npy` [N]
   (seconds since the lap's first frame, from the video frame index). A lap id is
   `<session>__lapNN`.
6. **`index.json`** records per lap: track (circuit plus layout), session, car,
   `split` (`train` or `holdout`), frame count, track length, `ref_bins` (track
   length / 2 m, so ~1000 bins per lap), field of view, and the camera rig's
   sideways offset for rig renders.

`packed_ac_v2`: 143 laps, 528,728 frames, 17 GB, six circuits, 19 sessions
across two cars (Mazda MX-5 Cup, Abarth 500) and several times of day. Silverstone
National is `split=holdout` and never trained on. Most of the sessions checked
so far were recorded with the rig's *wander*: the camera drifts sideways up to
about 3 m/s, so the line varies within and between laps.

## 2. The reference grid

A reference lap is not used frame by frame. `train/dataset.py`
`build_reference_grid` resamples it onto **N bins spaced evenly in the reference
lap's own elapsed time** (`--reference-axis time`, the default): each bin is the
same slice of delta everywhere, so bins are dense in slow corners (~0.8 m) and
sparse on straights (~3.3 m), about 2 m on average. Each bin holds the recorded
frame nearest to it, searched around the loop so the finish line is not an
edge, and carries its exact position in metres and its reference time.

A live frame's **target** is its own position mapped onto that grid, a
continuous bin number. A lap with a gap wider than one bin (a recording hole)
is never used as a reference; it can still be a live lap.

## 3. One training step

`train/dataset.py` `AlignmentBatches` builds every batch from scratch. There
are no epochs and no shuffled list: step *k* draws everything from its own
random generator seeded with `(seed, k)`, so any run is exactly reproducible and
any step can be regenerated alone.

Each step:

1. **Pick a track**, uniformly among tracks that have a usable reference and at
   least one other lap. Tracks are equally likely whatever their lap count.
2. **Pick a reference lap** on it, uniformly among laps usable as references.
3. **Roll the reference.** The whole reference is rotated by a random number of
   bins (`roll_reference`), and every target shifted to match. Bin 412 is a
   different place every step, so the network cannot learn "bin 412 is the
   hairpin"; it can only find the answer by comparing live frames with the
   reference. This is what makes the map context rather than memory.
4. **Build 8 live clips** (`batch_size`), each from a lap other than the
   reference, drawn independently:
   - **12 frames** (`clip_len`). The **last frame is the one being localised**:
     the network sees only the past, as it will at runtime.
   - **Stride** 1, 2, 3 or 4 frames between clip frames, uniformly. At 60 fps
     that is a 0.18-0.73 s span. A stride-4 clip of a slow corner looks like a
     faster car, so this is the speed augmentation: live and reference driven at
     different speeds. Runtime uses frames 1/15 s apart, stride 4.
   - **Reversed** with probability 0.25: the same frames backwards in time, the
     target still the last frame shown. The network cannot assume motion only
     runs forward, so it has to match each frame rather than extrapolate a slope.
   - **Static** with probability 0.03: twelve copies of one frame, as when
     standing still.
5. **Photometric jitter** (`photometric_jitter`), drawn separately for the live
   clips and the reference, so matching exposure can never be a cue:
   - one gamma (0.8-1.3), gain (0.75-1.3), brightness offset (±0.08) and
     per-channel colour gain (0.9-1.1) per clip, and one set for the reference;
   - per-frame flicker (±3%) and pixel noise (σ 0.012 on a 0-1 scale).
6. **Targets:** the last frame's continuous bin, a **soft target** (a Gaussian of
   σ = 2 bins around it, wrapped at the lap end), and each clip frame's own bin
   for per-frame supervision.

## 4. What the network computes

`train/model.py`:

- **Encoder**, shared by live and reference frames: five 3x3 convolutions with
  stride 2 (3 → 32 → 64 → 96 → 128 → 128 channels), each with GroupNorm and GELU,
  pooled to a 3x5 grid, projected to 128 numbers and scaled to unit length. It
  accepts any resolution; at 148x80 it is 34.7 M multiply-adds per frame.
- **Correlation:** every live frame's 128 numbers dotted with every reference
  bin's, times a learned sharpness, giving a 12 x N grid of scores.
- **Head:** that grid read along the reference axis, with the 12 clip frames as
  channels: a 1-D convolution to 64 channels, four residual blocks with dilation
  1, 2, 4, 8 and circular padding (the lap is a loop), and one output logit per
  bin. It looks for the diagonal stripe of matches.
- **Readout:** the peak bin, refined by averaging over ±8 bins around it
  (`soft_argmax_circular`), not over the whole lap, so a second peak elsewhere
  cannot drag the answer between two places.

667 k parameters in all.

## 5. The loss

`alignment_loss`, three terms:

| Term | Weight | What it does |
|---|---|---|
| Cross-entropy of the head's output against the soft target | 1 | the main objective: put the probability on the right bins |
| Per-frame correlation rows against each frame's own soft target (`--aux-all-frames`) | 0.5 | every frame must localise on its own through the encoder alone; gets training started before the head knows anything and sharpens what the head combines |
| Distance from the refined readout to the target, only when the peak is already within 8 bins | 0.5 | polishes precision; limited to near misses so it cannot sharpen a confidently wrong peak |

## 6. Optimisation

`train/train.py`:

- AdamW, learning rate 3e-4 with a one-cycle schedule (15% warm-up), weight
  decay 1e-4, gradient norm clipped at 1.0.
- 3000 steps for G1, each one reference lap plus 8 clips. On the M4 Pro (Metal)
  a run takes about 40 minutes.
- The ~1000-frame reference is encoded in chunks with gradient checkpointing,
  so gradients flow through the reference too without holding it all in memory.
- Every 200 steps: 24 evaluation batches without jitter, from a separate random
  stream, so they are never training clips. The checkpoint with the lowest
  median error in metres is kept as `best.pt`.

## 7. Gates: what "held out" means

| Gate | Trains on | Evaluated on | Purpose |
|---|---|---|---|
| G0 | one pair of laps, no roll, no jitter, 400 steps | the same pair | can it learn at all |
| **G1** | train-split laps, minus one lap per track | the held-out laps, against training laps as references | new laps of known tracks |
| **G2** | train-split laps | the `holdout` split (Silverstone) | a track never seen |

Laps are held out whole, never frames: neighbouring frames of a lap are nearly
identical, and a frame-level split would report a fantasy number. In G1 the
held-out lap is the last one per track by id, and only on tracks that keep at
least two laps for training.

## 8. Controls, before believing any number

Required for every training (`AGENTS.md`); `python -m train.eval gates` runs the
first, `python -m train.eval leakage` the second:

- **Wrong reference.** Live clips paired with *another track's* reference. The
  error must collapse to chance (a quarter of the lap). If it does not, the
  network is answering without reading the reference, from memorised
  appearance.
- **Leakage.** A network trained to predict position from single frames with
  no reference at all must fail on the held-out track. If it succeeds,
  something on screen gives the answer away, a HUD map for example. On seen
  tracks it succeeds, as it can memorise them; that shows the detector works.

Beyond the gates, `train.eval stream` runs whole laps at 15 Hz through the
particle filter as the phone would, and `train.eval lines` measures error
against racing-line separation.

## 9. What we do not do yet

The augmentation policy in `ml-pivot.md` calls camera-side effects near-free
and first priority. Implemented today is the photometric jitter above and
nothing else. Not done:

| Missing | Why it matters |
|---|---|
| Geometric jitter: small crops, scale, rotation (head roll), shifts | the glasses sit differently every session; the head rolls in corners |
| Motion blur and vibration | a kart has no suspension; AC renders none |
| Rolling-shutter shear | irrelevant for Halo's global shutter, not for most cameras |
| Lens distortion, compression artefacts, sensor noise at low light | real cameras; AC is clean |
| **Occluders: kart nose, steering wheel, hands, visor edges and tint** | the product camera sees them; the model never has |
| Camera height | AC's rig sits at a car driver's eye height (~1.15 m); a kart driver's is lower |
| Field of view | every frame is AC's 91.5° x 57.8°; Halo is about 81° x 65° |
| **Mirroring live and reference together** | a mirrored circuit is a plausible new circuit: a cheap way to add track variety, the lever the unseen track needs |
| Local lighting: moving shadows, sun glare | the dusk-against-noon bias suggests lighting is not yet learned |

## 10. Input resolution

Every real-footage result so far is at **148x80**. That size came from the
synthetic phase (128x80 and 160x96), chosen for a phone neural engine at 15 Hz,
and the real data was packed at the same scale so results stayed comparable. A
sharper input has never been tried on real footage. What it would cost:

| Encoder input | Multiply-adds per frame | Packed data | Training time | On Halo's NPU at 30 fps |
|---|---|---|---|---|
| 148x80 (today) | 34.7 M | 17 GB | ~40 min | ~5% of peak |
| 296x160 (2x) | ~140 M | ~68 GB | ~4x | ~20% of peak |
| full 480p | ~860 M | ~300 GB, beyond this Mac's free disk | ~25x | over 100%, not possible |

Whether it helps is open. More pixels give finer landmarks, which may sharpen
precision on known tracks, but also more track-specific texture to memorise;
the encoder pools everything to a 3x5 grid before comparing. The deployable range
is up to about 320x240 on Halo, so a 2x pack is the experiment worth running:
Windows repacks at `--height 160`, and the same training runs on it.

## 11. Running it

```bash
./.venv/bin/python -m train.train --data data/packed_ac_v2 --gate g1 --name <run>
./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/<run>/best.pt
./.venv/bin/python -m train.eval leakage --data data/packed_ac_v2
./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/<run>/best.pt
```
