# PRIMAL

**PRIMAL** (PRogress estIMation on repeAting signaL). Learned track-progress
localization as **sequence alignment against a reference lap**, not absolute
position regression. The network is given a live clip of K frames and one
reference lap, and predicts which point of the reference lap the live clip is
at. Track identity reaches the output only through a dot product between live
and reference descriptors, so the weights structurally cannot memorize a
circuit.

Start at [docs/README.md](docs/README.md). The system design is
[docs/ml-pivot.md](docs/ml-pivot.md).

## Install

```powershell
cd D:\Documents\code\hotlapp\primal
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m capture.install_overlay
```

On a GPU box, skip the CPU-index line and let `requirements.txt` pull the CUDA build.

Every command below is run from this repo root with `.\.venv\Scripts\python.exe`.

On macOS, the default wheel already carries Metal support, so there is no index
to choose. Only the capture half of the pipeline is Windows-only; everything
from packing onward is portable.

```bash
brew install python@3.13
cd ~/code/hotlapp/primal
/opt/homebrew/bin/python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Substitute `./.venv/bin/python` for `.\.venv\Scripts\python.exe` below.

---

## 1. Ground truth: the timecode overlay

Screen capture returns a frame that was presented some unknown number of
milliseconds ago. Polling shared memory next to the grab therefore gives you
`s` at grab time rather than at render time, and the offset is jittery, not
constant, so it cannot be calibrated away. At 40 m/s that is metres of label
noise.

The fix is to make AC render its own ground truth. `capture/ac_overlay/` is a
CSP Lua app that draws a 2x27 grid of pure black and white cells encoding a
render-frame counter plus `ac.getCar(0).splinePosition`. Every captured frame
carries its own exact label in its own pixels, so capture latency, dropped
frames, and clock skew stop being sources of label error and become detectable
counter anomalies instead.

Wire format lives in two places that must agree:
[capture/timecode.py](capture/timecode.py) and
[capture/ac_overlay/locamotif_timecode.lua](capture/ac_overlay/locamotif_timecode.lua).
`python -m tests.test_timecode` checks the Python half round-trips through JPEG
compression and that the locator recovers a sloppy hand-drawn ROI.

---

## 2. AC settings

The cheat-proofing matters more than the graphics settings. AC's default HUD
includes a **track map app with a position dot on it**. A CNN reads that dot
almost immediately and reports a validation loss that means nothing.

- [ ] Every HUD app off, in particular the track map. Keep only the Timecode app.
- [ ] Racing line off (`Settings > Assists > Ideal racing line`).
- [ ] Delta / lap-time / sector apps off. Driver name tags off.
- [ ] Windowed mode at a fixed resolution, 1280x720 is plenty. Do not resize mid-session.
- [ ] Cockpit camera (F1). Do not change camera during a session.
- [ ] Do not press `H`; hiding the HUD hides the Timecode app too.
- [ ] Enable the Timecode app and drag it to the bottom-left corner.
- [ ] To confirm it is alive, set `SHOW_TEXT = true` in the Lua file, check the
      numbers advance, then set it back to `false`.

`python -m capture.ac_shm` prints a live status line and is the quickest check
that shared memory is up and reporting the track you expect.

## 3. OBS settings

- Source: Game Capture (or Window Capture) on Assetto Corsa.
- **Base and output resolution identical to the AC window.** Any rescale blurs
  the cell edges and is the most likely cause of decode failures.
- FPS: 60, fixed, from Common FPS Values.
- Encoder: NVENC H.264. Rate control CQP/CQ 18.
- Keyframe interval 15 frames. Packing decodes sequentially so this does not
  matter for the pipeline; it only buys cheap random access into the mp4 later.
- Turn off Look-ahead, Psycho Visual Tuning, and Dynamic Bitrate. All three
  smear high-contrast small blocks, which is exactly what the grid is.
- Container mp4.

## 4. Per-session workflow

```powershell
# after recording, while AC is still on the same session
.\.venv\Scripts\python.exe -m capture.session import C:\obs\out.mp4 --split train --notes "noon, tight line"
.\.venv\Scripts\python.exe -m capture.overlay_decode calibrate data\sessions\<id>\video.mp4
.\.venv\Scripts\python.exe -m capture.overlay_decode decode data\sessions\<id>\video.mp4
```

`import` reads track name, layout, spline length and car model from shared
memory and writes `run.json`. `calibrate` asks you to drag a box round the grid
once, then grid-searches position and cell size to maximise the checksum pass
rate, so a wrong placement scores zero rather than decoding garbage.

### Acceptance check

`decode` prints the numbers that decide whether a session is usable:

| Reading | Expected | If not |
|---------|----------|--------|
| checksum ok | > 99.9% | grid obscured, OBS rescaling, or overlay window moved mid-session |
| counter step median | exactly 1 | 2 means AC rendered at 120 fps while OBS captured 60 — harmless, labels stay exact |
| duplicate counters | 0 | OBS captured the same rendered frame twice; AC fps dropped below 60 |
| gaps > 1 | 0 | OBS dropped frames; they are identified exactly, so packing can skip them |

Duplicates and gaps are not fatal. Packing drops non-contiguous stretches, so
a session with a few gaps loses a few clips rather than being silently wrong.

## 5. Session protocol, roughly 30 minutes

Four short tracks, five laps each. **Three tracks train, one held out entirely**
(`--split holdout`). Never split by frame or by lap for the track-generalization
question; the held-out track is the only honest measurement of it.

A training sample pairs two *different* laps of the same track, so the laps
within a track must actually differ. If they are near-identical the task
degenerates into pixel matching and the loss curve becomes meaningless.

| Lap | Vary |
|-----|------|
| 1 | car A, noon, racing line |
| 2 | car A, morning, tight line |
| 3 | car B (different ride height, so different camera height), noon, wide line |
| 4 | car B, evening, deliberately off-line |
| 5 | car A, noon, deliberately slow |

Record all five laps of a track as one continuous OBS recording; packing splits
laps on the spline wrap. Stay out of the pits.

---

## 6. Training

### What the model is

```
live clip  [B, K, 3, H, W] --\
                              encoder (shared)  ->  L [B, K, D]
reference  [N, 3, H, W] ----/                       R [N, D]

C = L R^T                                           [B, K, N]
dilated 1-D conv head over the reference axis        logits [B, N]
```

The target is a continuous index into the reference lap; `s = index / N`.
Track identity reaches the output only through the dot product, so swapping the
reference changes the answer and the weights cannot hold a circuit. The head is
convolutional along the reference axis with circular padding, which makes it
translation-equivariant on a loop and independent of lap length, so one set of
weights serves tracks of any size.

Three sampler properties carry most of the design, all in
[train/dataset.py](train/dataset.py):

- **One reference per batch.** A 1 m-spaced reference is ~1000 frames. Per-sample
  references would mean 16000 encoder passes for a batch of 16; sharing one
  brings it to one reference pass plus `B*K` live passes.
- **The reference is rolled** by a random offset every batch, and the target
  rolls with it. Any absolute notion of track position is then wrong on every
  sample, which makes the shortcut unlearnable rather than merely discouraged.
- **Live and reference are jittered independently**, so matched exposure cannot
  serve as a matching cue.

Speed invariance is free: a clip sampled at stride `d` is that stretch of track
at `d` times the speed, with exact labels. Negative stride gives the reverse
pass. Stationary clips are sampled occasionally so zero progress stays a
representable answer.

### Why the output is a distribution

The prediction is continuous, read out by soft-argmax over the reference bins,
but it is *parametrised* as a distribution rather than a scalar. Under aliasing
the truthful answer is multi-modal, and a scalar head trained with L2 is obliged
to emit the conditional mean: on a track where turn 2 resembles turn 6 that is a
piece of tarmac the car has never visited, at low loss and undetectably. Sub-bin
precision comes from the windowed expectation, the same mechanism that gets
sub-pixel disparity out of a stereo cost volume.

### Gates

```powershell
# G0: overfit one lap pair. If this is slow the architecture is wrong.
.\.venv\Scripts\python.exe -m train.train --data data\packed --gate g0

# G1: held-out laps of seen tracks, full augmentation
.\.venv\Scripts\python.exe -m train.train --data data\packed --gate g1

# all gates plus the wrong-reference control, against one checkpoint
.\.venv\Scripts\python.exe -m train.eval gates --data data\packed --checkpoint runs\<name>\best.pt

# the leakage control, trained from scratch
.\.venv\Scripts\python.exe -m train.eval leakage --data data\packed
```

| Gate | Question | Reading |
|------|----------|---------|
| G0 | can the architecture fit one pair at all | train median error under one bin |
| G1 | unseen laps of a seen track | median and worst error in m and ms |
| G2 | unseen track | the actual product question |
| lines | live lap on a different driving line from the reference | error must not grow with line separation |

### Devices

`--device` defaults to CUDA, then Apple Silicon (`mps`), then CPU. Nothing in
the model is device-specific, so the same command runs anywhere:

```powershell
.\.venv\Scripts\python.exe -m train.train --data data\packed_lines --gate g1 --device cuda --workers 4
```

CPU training works but is only sensible for G0. Per step the reference lap
dominates: ~350 encoder passes for the reference against `batch x clip_len`
for the live clips, and gradient checkpointing recomputes the reference in the
backward pass. A GPU is worth roughly an order of magnitude here.

Note that `pip install -r requirements.txt` on Windows may resolve to a
CPU-only wheel. Check with `python -c "import torch; print(torch.__version__)"`;
a `+cpu` suffix means no GPU. Install a CUDA build explicitly from the
PyTorch index for your card, and raise `--workers` above 0 once the GPU is
fast enough for data loading to matter.

Measured on the 20-lap cross-line set at 128x80, 350 reference bins, batch 8,
clip 12, `--workers 4`:

| Device | Per step | G1, 3000 steps |
|--------|----------|----------------|
| Windows CPU | 2.9 s | ~95 min |
| M4 Pro, `mps` | 0.08 s | 4.6 min |

The first few dozen steps on `mps` run at roughly 0.3 s while Metal compiles
its shaders, so judge throughput after step 100 rather than at the start.

### Negative controls

A good held-out-track number means nothing on its own. Both controls run from
[train/eval.py](train/eval.py).

**Wrong-reference.** Pair live clips with another track's reference lap. Error
must collapse to chance (`N/4` bins) and predictive entropy must rise toward
uniform. If it does not, the model is answering from the live clip alone and the
reference is decoration.

**Leakage.** Train a single-frame, no-reference regressor. On seen tracks this
is *expected* to work — memorising one circuit from pixels is easy, which is
exactly why an absolute regressor is the wrong factorisation rather than a bug.
The diagnostic is the held-out track: absolute position on a circuit the weights
have never seen is not a learnable function of appearance, so a good score there
means something on screen is handing over the answer. A track map widget does
precisely that.

### Synthetic data

[train/synthetic.py](train/synthetic.py) is a procedural track renderer writing
the same packed format, so `train.dataset` cannot tell it from AC footage. It
exists to check the plumbing and the architecture before any footage is
captured, and it is a deliberately *easier* problem: landmark colours are
distinct per track, so aliasing is mild.

```powershell
# packed laps for training
.\.venv\Scripts\python.exe -m train.synthetic packed --out data\packed_synth --tracks 4 --laps 5

# a fake OBS capture with the timecode grid burned in, to test calibrate/decode/pack
.\.venv\Scripts\python.exe -m train.synthetic recording --out data\fake_session
```

The renderer is procedural, so a dataset is fully described by its command line.
Regenerating on another machine is faster than copying a few hundred MB of
frames; only real AC footage is worth transferring.

### Driving lines

By default every synthetic lap hugs the centreline, which quietly makes the
task easier than reality: if the live lap and the reference lap always occupy
the same strip of tarmac, the model can match *viewpoint* and still look
correct. `--lateral-spread` spreads laps across the track width, and
`--apex-gain` makes some laps seek corner apexes while others ignore them, so a
racing line and a wet line coexist on one track.

```powershell
.\.venv\Scripts\python.exe -m train.synthetic packed --out data\packed_lines `
  --tracks 4 --laps 5 --length-m 700 --width 128 --height 80 --ref-spacing-m 2.0 `
  --lateral-spread 2.5 --apex-gain 0.0 0.6
```

Laps are placed on a deterministic ladder across `[-spread, +spread]` rather
than drawn at random, so every dataset is guaranteed to contain widely
separated pairs instead of leaving it to luck. Each lap records
`line_bias_m`, `line_mean_m` and `apex_gain` in `index.json`.

Measure the result with the `lines` gate, which sweeps every ordered lap pair
and reports error against how far apart the two lines run:

```powershell
.\.venv\Scripts\python.exe -m train.eval lines --data data\packed_lines --checkpoint runs\<name>\best.pt
```

The number that matters is the ratio between same-line pairs and
opposite-side pairs. Flat means the model found the place. Rising with
separation means it matched the viewpoint and the aggregate score was hiding it.

### Looking at the data

[train/preview.py](train/preview.py) reads packed laps, synthetic or real, and
renders what the network is actually fed. Upscaling is nearest-neighbour on
purpose: the frames really are this small and a smooth interpolation would
flatter them.

```powershell
.\.venv\Scripts\python.exe -m train.preview list  --data data\packed_synth
.\.venv\Scripts\python.exe -m train.preview video --data data\packed_synth --lap synth00__lap00
.\.venv\Scripts\python.exe -m train.preview pair  --data data\packed_synth --track synth00
.\.venv\Scripts\python.exe -m train.preview sheet --data data\packed_synth --lap synth00__lap00
.\.venv\Scripts\python.exe -m train.preview clip  --data data\packed_synth
.\.venv\Scripts\python.exe -m train.preview infer --data data\packed_lines --checkpoint runs\<name>\best.pt
```

`infer` is the one to reach for when a number looks wrong. It runs a checkpoint
along a live lap and films three tiles — the live frame, the reference frame the
model picked, the reference frame it should have picked — over a plot of the
belief across the whole reference lap. The faint second curve is the newest
frame's raw correlation row on its own, so the gap between the two curves is
what the 12-frame sequence is buying over single-frame place recognition. A
two-peaked belief is perceptual aliasing, and the tick where it picks the wrong
peak is where the error tail comes from.

`video` plays one lap with its label burned in. `pair` is the useful one: the
live lap on the left, the reference frame at the same track position on the
right. Two different laps, so the halves never match pixel for pixel; what they
share is the place, and that is the correspondence the model has to find.
`sheet` is a contact sheet along the lap. `clip` is one training sample exactly
as the sampler builds it, photometric jitter and stride included, with the
answer bin and its neighbours below it — the fastest way to see whether the
target is even distinguishable from its neighbours at this resolution.

Run `pair` on real footage as soon as the first session is packed. If a human
cannot tell which reference frame matches, the labels are misaligned or the crop
is wrong, and no amount of training will fix it.

### Tests

```powershell
.\.venv\Scripts\python.exe -m tests.test_timecode      # wire format, compression, locator
.\.venv\Scripts\python.exe -m tests.test_pipeline_e2e  # fake recording -> calibrate -> decode -> pack
.\.venv\Scripts\python.exe -m tests.test_train         # sampler invariants and a learning smoke test
```

---

## 7. Layout

```
.
  docs/                  start at docs/README.md; design in docs/ml-pivot.md
  capture/
    ac_overlay/          CSP Lua app: manifest.ini, locamotif_timecode.lua
    timecode.py          wire format, single source of truth for the Python side
    install_overlay.py   copy the app into AC via the Steam registry
    ac_shm.py            physics / graphics / static page reader (auxiliary metadata)
    session.py           adopt an OBS recording, write run.json, log telemetry
    overlay_decode.py    mp4 -> labels.parquet
    pack.py              mp4 + labels -> packed per-lap arrays
  train/
    dataset.py           lap pairing, reference rolling, stride, soft targets
    model.py             encoder + correlation + dilated 1-D conv head
    train.py
    eval.py              gates and negative controls
    synthetic.py         procedural track renderer, for pipeline and architecture checks
    preview.py           render packed laps back out as video or contact sheets
  tests/
  data/                  gitignored
```
