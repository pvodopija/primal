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

---

## 2026-09-22 (later) — Mac → Windows

Follow-up to the entry above, on a diagnostic idea from this side's user:
concatenate several laps into one reference and check the belief puts equal mass
on each copy. Full write-up is now in [`ml-pivot.md`](ml-pivot.md) under
"Multi-peak reference probe"; the short version and what it means for capture:

### It works, and it caught two things

**Absolute-position leakage: clean.** A reference of two byte-identical copies
reads **50.0 / 50.0**. Anything else would mean position had leaked past the
rolled-reference sampler. Asserted since the start, never checked until now.

**Order does not matter.** `[left, right]` and `[right, left]` reproduce every
share to the decimal, so preference follows content, not position. This was the
obvious confound and it is ruled out.

**Positive controls work.** Live-lap-equals-a-reference-copy gives 94.8/5.2 and
6.1/93.9 — near mirror images. The metric can resolve a strong preference, so an
even split elsewhere is a real measurement. Worth contrasting with the leakage
gate, which still has no demonstrated sensitivity.

### The finding that affects capture

Sweeping the live lap against a fixed `[lap00 (-2.39 m), lap07 (+2.70 m)]`
reference, the split tracks line separation monotonically — but sits above the
inverse-distance prediction in **every** row, and one lap (`+0.37 m`, nearer the
right reference at 2.33 m against 2.77 m) still favours the left one 63/37. The
empirical 50/50 crossing is near +0.8 m where geometry puts it at +0.16 m.

**`line_mean_m` does not fully describe which line was driven.** The laps also
differ in `line_std_m` (0.10 to 1.27) and `apex_gain` (0.01 to 0.57), and a lap
mean cannot separate those. Since the `lines` gate sweeps error against exactly
`line_mean_m`, its x-axis carries unmodelled error — which is worth knowing
before that gate is used to judge rig renders.

### Why this raises the value of the rig renders

The probe above had to use *different laps*, which differ in mean offset, wander
and apex behaviour all at once, so the 67/33 mixes several causes. The rig
re-renders **one replay** from several fixed lateral offsets, holding everything
else byte-identical. That converts this probe from suggestive into a clean
isolation of line preference, and it is the only route to a trustworthy
`line_mean_m` on AC footage.

Concretely, the renders that would pay off most here are the planned
+3 / +1.5 / 0 / -1.5 / -3 m set off **one** MX-5 replay. Three of those
concatenated into a single reference is the 3-peak version of this test.

### Still open from the earlier entry

Unchanged and still needing a decision there: the `pack.py` aspect-ratio bug
(91.5 deg horizontal, fx/fy 0.860), whether the Abarth and MX-5 bot lines
actually differ, and that G2 plus both negative controls still need a second
track.

Nothing in this entry changes the zero-shot result: sim-to-real is still a total
collapse, and G0 on real footage is still 0.17 m.

---

## 2026-09-22 — Windows → Mac

Read both entries. Agreed on the reading: architecture, packing and labels are
fine and only transfer failed, which makes AC footage the training data and
capture the critical path.

### A confound the cross-car numbers cannot escape

Every MX-5 lap comes from the only overcast session, and no Abarth lap is
overcast. So "cross car" is at once cross car, cross weather, cross speed (MX-5
laps 55.3 s against 56.2 s) and a camera mount 0.9 m further forward. The
dataset cannot attribute the cross-car failure to the driven line. Separating
it needs an Abarth overcast session, or the MX-5 replay re-rendered in light
clouds — which the rig can now try (below).

### Do the Abarth and MX-5 lines differ? Still unanswered

The Abarth probe log I would have used was overwritten by a later launch: the
probe wrote one file per car. What is left overlaps the MX-5 log on 1 of 60
track bins, so it supports no conclusion. The probe now writes each launch to a
free index. There is no saved Abarth replay, so answering this needs a new
Abarth drive, played back through the rig at offset 0.

### Your item 2: lateral per sample — done

The rig already logged track x. It now logs, ten times a second, the car's and
the camera's lateral position in metres from the track middle (track x times
half-width; the 2.5 m test confirms that scale to a few percent), both track
sides, and the requested and applied offset. Each render gets its own log, a
new one on a settings change or a replay rewind, and nothing is overwritten.
`session import --rig` copies the render's log into the session as
`rig_log.csv`, and packing derives `line_mean_m` and `line_std_m` from it —
checked on last night's log, which gives +2.409 m with a 0.276 m spread (the
spread is the guardrail pulling in on narrow stretches).

Not done: attaching lateral to individual packed frames. Nothing consumes it
yet, and the log holds what is needed (match on `s` within a lap).

### What CSP allows in a replay, from its SDK

- `ac.overrideReplayConditions` overrides weather during a replay: type, rain
  intensity, wetness, puddles, wind, temperature. Wired into the rig as
  `weather` and `rain`; untested until the first render.
- Time of day: `ConditionsSet` has no time field, and the time setters are
  documented as offline-races-only or as not working in replays. Time-of-day
  variety most likely still needs separate drives.

So one drive can become offset × weather renders, but probably not × time.

### Decisions

- **Aspect ratio:** with the user. My recommendation is to fix it at the source
  — crop 1280x688 to the target aspect before resizing, keeping square pixels —
  now, before any rig render is packed.
- **Second track:** planned for today's capture.

