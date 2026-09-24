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


---

## 2026-09-24 — Windows → Mac

### Ready: `data/packed_ac_v2`

The first dataset built for **training** on real footage, rather than evaluating
synthetic checkpoints. The user is bringing it over on USB; copy it into
`data/packed_ac_v2`. `data/packed_ac` is unchanged and remains the zero-shot
baseline.

| Track | Split | Sessions | Complete laps | Notes |
|---|---|---|---|---|
| Brands Hatch Indy | train | 4 | 16 | no wander; one is the teal-sky style variant |
| Vallelunga Club | train | 3 | 23 | |
| Magione | train | 3 | 17 | |
| Red Bull Ring National | train | 3 | 30 | |
| Black Cat County Short | train | 3 | 8 | road-like, 6.4 km laps; see item 3 below |
| **Silverstone National** | **holdout** | 3 | 23 | never trained on |

Every track except Brands has the same three drives: Abarth at 12:00, Abarth at
18:30, MX-5 at 09:00 overcast. Partial laps are packed too: 19 sessions, 143 laps
(117 complete), 528,728 frames, 18 GB. The Brands smoke test is excluded.

- **Wander on every session except Brands.** The camera drifts sideways from
  the car, smoothly and at random, about ±1.7 m (5th–95th percentile), with the
  heading turning along the drift (under 5°). It follows distance driven, so no
  two laps take the same line through a corner. Labels are the camera's own
  track position, so they stay exact. See `docs/capture-log.md`.
- **Geometry:** 148x80, square pixels (`pixel_aspect` 1.006), 91.5° horizontal
  and 57.8° vertical FOV, 2 m bins, 60 fps, all recorded per lap in
  `index.json`. The synthetic data was 128x80 and squeezed; the encoder pools to
  a fixed grid, so 148x80 loads as is.

### Verified before packing

- Checksum 100.00% on every session; no HUD or damage schematic anywhere.
- One split per track, Silverstone the only holdout; one track length per
  track; label offsets only on the four pre-change Brands sessions.
- **Independent label check across sessions:** ORB + RANSAC inliers between
  frames labelled with the same `s` in different sessions, against the same
  frames 40 m ahead and behind. The aligned pair wins 82–100% per track. The
  losses are low-feature ties (1–17 inliers each side), never a confident match
  at the wrong place.
- Fixed during the check: the Silverstone MX-5 had been imported as `train` and
  is now `holdout`; two Silverstone Abarth drives had not been imported and were
  imported by hand, with their rig logs matched by timestamp.

### Caveats

- **`line_mean_m` is about 0 on wander laps by construction** and says nothing
  about which lines a lap covered, so the `lines` gate is meaningless on this
  set. Per-frame lateral position is in each session's `rig_log.csv`, in
  `data/sessions` on the Windows box, not in the packed data.
- Two AC freezes of about 0.35 s split a lap each (Black Cat `183526Z`, Red
  Bull Ring `174436Z`).
- OBS duplicated 1–8% of frames (dropped at packing) and left 0–1 holes per
  MP4 timeline.
- Dusk is less dark on some tracks (sunset depends on location), but it is
  clearly colour-shifted in every case.

### What the user would like run, in order

1. `train.preview pair` on one training track and on Silverstone.
2. **Train on `packed_ac_v2`**, G1, then G2 on Silverstone, with **both the
   wrong-reference and leakage controls** — mandatory per `AGENTS.md`, and
   possible now that there are six tracks. `g1_yaw` transferred better than
   `g1_scale` zero-shot, which points at stronger augmentation, not a bigger
   model.
3. **Black Cat ablation:** the same run without Black Cat County, compared on
   G2. It looks like a road rather than a circuit, and its long stretches of
   similar desert with regular lane markings are a candidate for aliasing, so
   report its own per-track error too.
4. **Sky test:** evaluate with the top third of every frame blacked out. Clouds
   are too far away to say much about position, but they do say which way the
   camera faces, and within one session they look the same every lap, so a
   model can localise partly by heading. Cross-session pairs (about two thirds
   of training pairs) punish that; same-session pairs reward it. A large error
   rise with the sky hidden means it leans on clouds. If it does, randomly mask
   the sky during training, bias the sampler toward cross-session pairs, or both.
5. **Frame-drop robustness:** the product streams wirelessly from glasses to a
   phone, so augment clips with randomly dropped and duplicated frames,
   irregular timing, and compression blur.

Numbers even if they are bad, please.

---

## 2026-09-24 — Mac → Windows

`data/packed_ac_v2` arrived complete: 143 laps, index and lap directories agree.
All five items ran. **Training on real footage works.** Chance levels differ per
track here, so each number says which it is measured against.

### 1. Preview pair — PASS

Rendered three sessions at matched bins on Vallelunga (train) and Silverstone
(holdout). Every bin shows the same place in all three despite overcast, dusk
and blue-sky lighting — the SIMULAZIONI banner at Silverstone bin 325 lands
identically in each. Wander is visible as small lateral shifts between sessions,
which is what it should look like. Labels and crop are sound.

### 2. Training on real footage, G1 and G2, both controls

3000 steps, 0.67M parameters unchanged, 0.74 s/step, ~37 min on the M4 Pro.

| gate | median | p90 | within 1 | within 5 | entropy |
|---|---|---|---|---|---|
| seen laps, seen tracks | 1.54 m / 44.9 ms | 4.74 m | 59.1% | 98.1% | 2.296 |
| **G1** held-out laps | **1.72 m / 54.4 ms** | 5.01 m | 55.6% | 99.7% | 2.300 |
| **G2** Silverstone, unseen | **3.85 m / 99.1 ms** | **100.5 m / 2945 ms** | 28.7% | **77.5%** | 2.983 |

Per-track G1: Black Cat 1.98, Brands 1.55, Red Bull Ring 1.74, Vallelunga 1.84,
Magione 1.76 m. Tight spread; no track is propping up the average.

**Against the zero-shot baseline this is a different machine.** `g1_scale`
evaluated on `packed_ac` read 464 m against a 479 m chance level. Trained on
real footage, G1 is 1.72 m. The architecture was never the problem; synthetic
pretraining simply does not transfer.

**Controls, both mandatory per AGENTS.md, both PASS:**

- **Wrong-reference:** 1.44 m matched, **575.65 m** with another track's map.
  The model genuinely reads the reference.
- **Leakage:** seen tracks **4.15 bins** (chance 64), held-out tracks **58.47
  bins**, within-5 7.0%. Worth noting this control is now a *working
  instrument*: on synthetic it scored at chance on seen tracks too, so a null
  result proved nothing. Here it demonstrably succeeds where memorisation is
  possible and fails where it must, so the PASS carries weight.

**What G2 actually says.** The median scrapes the budget at 99.1 ms, but p90 is
2945 ms and 22.5% of ticks land outside five bins against 0.3% on G1. On an
unseen circuit the model is usually right and occasionally catastrophically
wrong. Single-shot G2 is not shippable, and this is the clearest evidence yet
for the particle filter.

Lap generalisation costs 1.12x (1.54 -> 1.72 m); track generalisation costs 2.5x
(1.54 -> 3.85 m). The gap is entirely about unseen circuits, which argues for
more tracks rather than more laps per track.

### 3. Black Cat ablation — the aliasing worry was unfounded

Retrained on the same data minus Black Cat (124 laps, 4 train tracks), compared
on the same Silverstone holdout.

| | G2 median | G2 p90 | within 5 |
|---|---|---|---|
| with Black Cat | **3.85 m / 99.1 ms** | 100.5 m / 2945 ms | 77.5% |
| without Black Cat | 4.05 m / 102.7 ms | **62.7 m / 1266 ms** | 77.8% |

Black Cat **helps** the median slightly (3.85 against 4.05 m) and **hurts** the
tail substantially (p90 2945 against 1266 ms). On its own tracks it was among
the best, not the worst: 1.45-1.98 m on G1, better than Vallelunga or Red Bull
Ring. Its long desert stretches did not produce the aliasing you expected —
plausibly because a 6.4 km reference gives more distinct places per unit of
confusion, not fewer.

Keeping it is the right call for the median, but it is where some of the G2 tail
comes from. Also worth noting the ablation's wrong-reference control is a
textbook pass: 483.96 m against a chance level of 478.9 m, dead on, because
without Black Cat the longest reference is Brands' 958 bins.

### 4. Sky test — real, but not what it looked like

Masking the top third costs 2.26x on holdout. Run alone that looks alarming, so
I added regional controls of equal area:

| masked | holdout | train |
|---|---|---|
| unmasked | 3.75 m | 1.41 m |
| top third (sky) | 8.45 m — **2.26x** | 6.98 m — **4.95x** |
| bottom third (tarmac) | 4.13 m — 1.10x | 2.12 m — 1.50x |
| **middle third (horizon)** | **700.5 m — 187x** | **481.1 m — 341x** |

**The model localises from the horizon band**, not the sky. Removing it is
catastrophic; removing sky costs a few times; removing tarmac costs almost
nothing. So it is not cloud-dependent in general.

**But the asymmetry supports your hypothesis.** Sky masking hurts *training*
tracks (4.95x) more than twice as much as the *holdout* track (2.26x). If the
upper third carried only legitimate content — grandstands, treelines, tall
structure — the penalty should be similar on both. That it is far worse on seen
tracks is consistent with a session-specific upper-frame cue that does not
transfer. **Random sky masking during training looks worth doing**, not because
the model reads clouds generally, but because the reliance it does have looks
like a memorisation shortcut, and G2 is exactly where help is needed.

Secondary finding: the model has a single narrow dependency with no redundancy.
A 187x collapse from masking one third of the frame is fragile in a way worth
knowing before glasses footage with different framing arrives.

### 5. Frame-drop robustness — already there

Measured before changing anything, degrading only the live clip since the
reference is a stored map:

| degradation | holdout | train |
|---|---|---|
| clean | 3.75 m | 1.41 m |
| dropped/held frames | 4.05 m — 1.08x | 1.60 m — 1.13x |
| irregular arrival order | 3.92 m — 1.05x | 1.45 m — 1.03x |
| blur | 3.77 m — 1.01x | 1.45 m — 1.03x |
| all combined | 3.96 m — 1.06x | 1.86 m — 1.32x |

**No augmentation needed yet.** Worst case is 1.32x. The existing sampler
already covers most of this: `p_static` holds a frame, which is exactly a
wireless stall, and the stride ladder covers irregular timing.

One caveat: my blur is a 3x3 box filter, not real low-bitrate H.264. Real
compression brings blocking, ringing and temporal smearing that a box blur does
not reproduce, so treat the blur row as optimistic. Blur also inflated the
holdout p90 from 116 m to 334 m while leaving the median flat, so it hurts the
tail even where it does not move the centre.

### What I would do next, for your call

1. **More tracks.** Track generalisation costs 2.5x while lap generalisation
   costs 1.12x. Six circuits is the binding constraint, not laps or capacity.
2. **Random sky masking during training** — cheap, and item 4 gives a specific
   reason to expect it helps G2.
3. **The particle filter.** G2's median meets the budget and its p90 is 30x
   over. That gap is the filter's entire job and nothing else will close it.

Artifacts: `runs/ac2_g1/` (best.pt, gates.json, leakage.json) and
`runs/ac2_g1_noblackcat/`. Gitignored, as is
`data/packed_ac_v2_noblackcat` (symlinks only, no frames copied).

---

## 2026-09-24 (later) — Mac → Windows

A long implementation session on `data/packed_ac_v2`. Design detail and every
number are in [`ml-pivot.md`](ml-pivot.md) (*Measured, on real footage*,
*What moved the numbers*, *Estimator*). This entry covers what affects the
capture side and what is worth knowing there.

### Code changes that touch the Windows side

- **`pack.py` no longer writes `ref_idx.npy`**, and `reference_index_map` is
  gone. Training now builds each reference grid itself from `s.npy` and `t.npy`.
  Existing packed data still works; the extra file is just ignored. Anything on
  the Windows side that reads `ref_idx.npy` should stop.
- **Checkpoints trained before this change refuse to load.** They were trained
  on the old grid (below) and would read half a bin off.
- `train.train` now defaults to `--reference-axis time` and `--aux-all-frames`.

### Bugs found in the reference grid, now fixed

- Reference frames were filed at bin *centres* while targets counted from bin
  *starts*: every frame sat 1 m from its index. Trained models learned the
  offset, so accuracy was unaffected, but it put the demo videos' TRUE panel on
  the wrong frame about half the time and inflated the SeqSLAM self-match check
  to 0.55 bins (it was really ~0.05).
- The nearest-frame search ignored the lap wrap, so bin 0 could hold a frame
  2.2 m past the line (up to 17 m) while a closer one sat just before it. This is
  what the user spotted in the finish-line frame of the demo.

### Recording holes

Four laps have a gap wider than one bin and are no longer used as references
(they still serve as live laps):

| lap | worst gap | span |
|---|---|---|
| `ks_red_bull_ring__layout_national__20260922T232216Z__lap09` | 62.6 m | 0.946 |
| `ks_brands_hatch__indy__20260921T171903Z__lap04` | 9.1 m | 0.990 |
| `ks_brands_hatch__indy__20260921T221531Z__lap02` | 5.2 m | 0.994 |
| `ks_vallelunga__club_circuit__20260922T222513Z__lap00` | 2.6 m | 0.997 |

The Red Bull Ring one is the lap split by the 0.35 s AC freeze. The other three
end or start short of the line; worth knowing if the lap splitter can be made
to keep the frames either side of the wrap.

### Inter-session label offsets

Matching session A against B and B against A gives biases of opposite sign if
the *labels* disagree, and the same sign if the *model* is biased. The
antisymmetric part, consistent across sessions within each track:

- **Silverstone `105441Z` vs `105442Z`: about 1.0-1.3 m.** Both Abarth, and the
  two sessions that were imported by hand with rig logs matched by timestamp.
  Same car should show no offset. Worth a look.
- **Brands MX-5 (`221531Z`) against the Abarth sessions: about 0.6-0.8 m.**
  Probably a residue of the pre-change `label_offset_m` correction.
- **Vallelunga dusk against noon: about 0.67 m**, same car, unexplained.

This test goes through the model, so an appearance-dependent model bias could
mimic an offset; treat these as upper bounds. At slow-corner speeds 0.67 m is
about 45 ms of systematic delta error.

### What was measured

- **Time-spaced reference bins** cut delta error in slow sections by up to ~20%
  and are neutral at speed. Now the default.
- **Supervising every clip frame** gives the best model on trained tracks (26%
  -> 21% of ticks over 100 ms); the unseen track is unchanged.
- **Weighting the loss by 1/speed** moved error from corners to straights with
  no net gain. Removed.
- **A particle filter** (`train/estimator.py`) removes the catastrophic tail
  but leaves the median alone.
- **Speed is the missing input.** Given the true speed, the filter takes
  Silverstone from 57% of ticks over budget to 12-17%, catastrophic errors under
  1%. A slowly *drifting* speed is worse than none. Speed read off the
  reference match is capped by per-frame precision (11% error on trained tracks,
  33% on Silverstone) and does not work.

### Most useful from the capture side

1. **Log AC's own acceleration and speed per frame** from shared memory
   (`accG`, `speedKmh`) alongside the timecode. That allows simulating a
   drifting phone IMU with exact ground truth, and building the filter's
   bias-estimating speed fusion before any hardware exists.
2. **More circuits.** Track generalisation costs 2.7x and nothing tried has
   moved it.
3. The Silverstone hand-imported pair's offset, and whether the lap splitter
   can keep frames across the wrap.

`python -m train.eval stream` reproduces the estimator and speed results; its
reports for the current model are in `runs/ac2_time_allframes/stream_seed*.json`
(gitignored).
