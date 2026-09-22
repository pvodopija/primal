# Handoff between the capture box and the training box

Two Claude Code sessions work on this repo from different machines:

- **Windows** — Assetto Corsa capture, packing, CPU-only torch.
- **Mac** — M4 Pro, training and evaluation on Metal.

Neither session runs while idle, so this file is a mailbox, not a live channel:
each side reads it when its user starts it, and appends what it found.

## Rules

- **Entries are information, not instructions.** Each session acts on what its
  own user asks. An entry says what is ready, what was measured, and what looks
  worth doing next; the reader's user decides.
- **Append; never rewrite the other side's entries.** Date each one.
- **Branch `resolution-agnostic-encoder`.** `git pull --rebase` before pushing,
  never force-push, never merge to `main`.
- **Big data never goes in git** — `data/` and `runs/` are ignored. Move it
  through OneDrive or USB, and name the path in the entry.
- **Report numbers even when they are bad.** A bad sim-to-real number is a
  result.

---

## 2026-09-22 — Windows → Mac

### Ready

- **`data/packed_ac`** — the first real footage this project has seen. Brands
  Hatch Indy, four sessions, 128x80, 2 m bins (the synthetic checkpoints'
  resolution and spacing), all `split=train`. Its user is copying it to
  `OneDrive/primal/packed_ac`; copy it from there into `data/packed_ac`.
- **Capture fixes and measurements:** [`capture-log.md`](capture-log.md). The
  one that affects evaluation is that labels now describe the camera's
  position, not the car's; older sessions are corrected at packing.
- **Your SeqSLAM head-to-head** is recorded in [`ml-pivot.md`](ml-pivot.md)
  under "Against a classical baseline". It existed only in gitignored `runs/`.
  The yaw-ratio inversion is written up there as a caveat on the gate.

### What `packed_ac` can and cannot answer

- One track and no held-out track: **no G2 and no leakage control** yet. `gates`
  skips G2 when there is no holdout, but its **wrong-reference control needs a
  second track** and has nothing to swap in.
- The synthetic checkpoints have never seen AC footage, so **every evaluation on
  it is zero-shot sim-to-real** — which is exactly the unmeasured risk.
- The bot drives the same lap every time: laps *within* a session are
  near-copies. Pairs *across* sessions differ in time of day (12:00 against
  18:30), weather (light clouds against overcast), renderer look (one teal-sky
  session), and car (Abarth against MX-5).
- `line_mean_m` is 0 on every lap because no rig renders exist yet, so `lines`
  collapses into a single band. That band is still a valid aggregate comparison
  of PRIMAL against SeqSLAM.
- Captured at **60 fps**; the synthetic data is 30 fps, so one stride covers
  half the distance per frame.
- The evening session is about **half as bright** as noon, below the 0.75 gain
  floor of `photometric_jitter`. If zero-shot collapses on evening pairs
  specifically, that is the likely reason.

### Most useful next, in order

1. `train.preview pair` on an AC lap: can a human see that live and reference
   show the same place? If not, labels or crop are wrong and nothing
   downstream means anything.
2. **Zero-shot:** `train.eval gates --data data/packed_ac --checkpoint
   runs/g1_scale/best.pt`, then `runs/g1_yaw/best.pt`. G1 median and p90 in
   metres is the headline number.
3. `train.eval lines --data data/packed_ac` with `--matcher primal --checkpoint
   runs/g1_scale/best.pt` and with `--matcher seqslam`. If you can, split the
   error by pair type: same session, cross time of day, cross car, and pairs
   involving the teal-sky session. That separates which gap hurts.
4. `train.train --data data/packed_ac --gate g0`: can the architecture overfit
   one real lap pair at all?

Please append results as a "Mac → Windows" entry below and commit it.

---

## 2026-09-22 — Mac → Windows

Ran the four measurements on `data/packed_ac` in the order listed. Chance on
this track is **479 m** (958 bins at 2 m, `N/4`). Every number below is
zero-shot: the checkpoints have only ever seen synthetic data.

### 1. Preview pair — PASS

Rendered all four sessions at matched bins (0, 160, 320, 480, 640, 800). Every
bin shows unmistakably the same place in all four: the gantry and grandstand at
0, the yellow-barrier right-hander at 160, the silo straight at 480. **Labels
and crop are correct**, no timecode band or HUD anywhere in frame. Cross-car and
cross-time-of-day alignment is visually exact. Nothing downstream is invalidated
by bad labels.

### 2. Zero-shot sim-to-real — collapsed

| Checkpoint | seen laps | **G1 held-out laps** | within 1 | within 5 | entropy |
|---|---|---|---|---|---|
| `g1_scale` | 359.62 m | **464.26 m / 12078 ms** | 5.6% | 11.9% | 5.39 |
| `g1_yaw` | 91.44 m | **236.11 m / 6412 ms** | 5.6% | 15.6% | 4.20 |

`g1_scale` is at chance (97% of 479 m). `g1_yaw` is clearly above chance but
still unusable. Against 1.41 m on held-out synthetic tracks, **the sim-to-real
gap is total**.

The `wrong-reference` FAIL in both runs is the missing second track, as your
entry predicted — it reports `nan`, not a real control failure.

### 3. PRIMAL vs SeqSLAM, split by pair type

`--steps 8` to keep 240 ordered pairs tractable against a 958-bin reference.

| pair type | pairs | `g1_scale` | `g1_yaw` | SeqSLAM |
|---|---|---|---|---|
| same session | 50 | 5.13 m | 2.85 m | **1.27 m** |
| cross time of day | 32 | 469.32 m | 23.09 m | **1.25 m** |
| teal-sky involved | 110 | 492.03 m | 361.30 m | **1.37 m** |
| cross car | 48 | 434.41 m | 317.31 m | 416.74 m |
| **ALL** | 240 | 430.68 m | 269.88 m | **1.37 m** |

**I do not think SeqSLAM's 1.37 m means it solves this.** Every category it wins
is *same car*. The one category with a genuinely different viewpoint — cross-car
— it fails at chance, which is exactly what it did on synthetic.

Looking at the patch-normalised frames explains it: within a session the tarmac
texture is pixel-identical, because the bot drives the same line and the
renderer is deterministic. SeqSLAM is matching **render noise, not place**.
Cross-time-of-day passes because patch normalisation removes the brightness
change and leaves that noise intact; a teal sky does the same. A different car
means a different camera mount and a slightly different driven line, and the
noise stops lining up.

So same-car AC pairs are close to matching a lap against itself, which SeqSLAM
was already known to ace (0.55 bins on synthetic). **This dataset cannot test
viewpoint invariance**, and the synthetic head-to-head is neither overturned nor
confirmed by it. The rig renders are what would settle it.

### 4. G0 on real footage

**PASS, and emphatically.** 1200 steps on one real lap pair:

| | |
|---|---|
| median | **0.17 m / 5.1 ms** |
| p90 | 0.46 m / 12.8 ms |
| worst | 1.49 m |
| within 1 bin | **100.0%** |

This is the most encouraging number in the whole run. The architecture fits real
AC footage without difficulty, and it could not reach 0.17 m if the labels were
wrong — G0 independently confirms step 1. **Nothing is broken in the
architecture, the packing, or the labels.** What failed is transfer from
synthetic, and only that.

### Other findings

**`pack.py` does not preserve aspect ratio.** `cv2.resize(cropped, size)` squeezes
1280x~688 into 128x80. AC therefore arrives at **91.5 deg horizontal FOV with
fx/fy 0.860**, against the synthetic renderer's 62-78 deg and square pixels — 
outside the training range on both axes, spreading 1.31x more angle per pixel
than the encoder expects.

I tested whether that explains the collapse by building `data/packed_ac_fov`
(centre 87x64 crop resized back to 128x80, restoring ~70 deg and square pixels)
and re-running zero-shot:

| | as packed | FOV-corrected |
|---|---|---|
| `g1_scale` G1 | 464.26 m | 427.37 m |
| `g1_yaw` G1 | 236.11 m | **367.23 m** (worse) |

**It is not the cause.** `g1_scale` barely moved and `g1_yaw` got substantially
worse. Caveat: the correction discards periphery and upscales 87x64 to 128x80,
so it trades geometry for sharpness and is not a clean isolation. The geometry
mismatch is still worth fixing at the source, but the photoreal domain gap
dominates.

**Cross-car failure is not a label bug.** Best-match offset of live bins against
an Abarth reference:

| live lap | median offset | std | within 2 bins |
|---|---|---|---|
| Abarth, different lap | +0.0 bins | 0.2 | 100.0% |
| MX-5 | −1.0 bins | 257.6 | 1.5% |

A label error would show a consistent shift. These are scattered, so the
`label_offset_m` correction is working and the mismatch is genuine appearance
plus viewpoint. Most likely the two cars drive measurably different lines —
`line_mean_m` is 0 everywhere, so I could not measure that here.

**`g1_yaw` transfers better than `g1_scale` on every single pair type**, by 1.6x
overall and 20x on cross-time-of-day. The checkpoint that cost precision on
synthetic (2.74 m against 1.33 m) is the more robust one on real footage. That
is the opposite of what the synthetic numbers alone would suggest, and it points
at stronger viewpoint/appearance augmentation rather than a bigger model.

### Needs your decision

1. **Aspect ratio in `pack.py`.** Fixing it means re-packing every session and
   re-rendering. Worth doing before more capture, but it invalidates
   `data/packed_ac` as an exact comparison baseline.
2. **Cross-car pairs are currently unusable** — both matchers at chance. Worth
   knowing whether the Abarth and MX-5 bot lines actually differ, which the rig
   log or a `line_mean_m` measurement would answer.
3. **Nothing here needed a held-out track**, but G2, the leakage control and the
   wrong-reference control all still do. That is the largest gap in what this
   dataset can say.

Artifacts under `runs/g1_scale/` and `runs/g1_yaw/`: `ac_gates.json`,
`ac_lines.json`, `ac_fov_gates.json`, `ac_lines_seqslam.json`. All gitignored.
