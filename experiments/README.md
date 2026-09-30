# Experiments

The scripts behind the findings in [`docs/ml-pivot.md`](../docs/ml-pivot.md), kept so a
result can be checked or extended instead of rebuilt. They are research code: run from
the repository root with `PYTHONPATH=.`, paths are the ones used at the time
(`data/packed_ac_v2` or `v3`), and results land in `runs/<run>/`. The measured outputs
are copied into [`results/`](../results/INDEX.md) by `python -m experiments.index_runs`.

The shell chains in `chains/` ran unattended on the Mac. They refer to these scripts by
the session scratch folder they lived in at the time; point them at `experiments/` to
re-run them.

## Which script backs which finding

| Finding (section of `ml-pivot.md`) | Script | Results |
|---|---|---|
| **Most of the unseen-track error was a lag** | `bias.py` (signed errors), `lag.py` (per-lap lag, backward readings, corrections; stride argument), `belief_shape.py` (the belief skews backward, no second bump) | `lag*.json`, `bias.json` |
| the stride test and the centred-strides fine-tune | `chains/strides.sh`, `chains/lagfix.sh` (the weak-match augmentation that did not help) | `strides26`, `lagfix_*` |
| every model at stride 2 and 4, lag-corrected | `chains/after.sh`, `chains/after2.sh` | `stream_lag_*.json` |
| **The reference lap as the speed prior** | `reftime_predictability.py` (labels only), `reftime_filter.py`, `reftime_filter_wide.py` (tuning on G1, testing on the unseen tracks) | `reftime_filter*.json` |
| **Voting across reference laps** | `multiref.py` (aligned by labels), `multiref2.py` (by the live matcher), `multiref3.py` (by the lap aligner) | `multiref*.json` |
| **A reference that improves every lap** | `refine_map.py` (v1: drift), `refine_v2.py` (placed against the anchor only), `refine_v3.py` (midpoint), `refine_v4.py` (how exact placement must be), `refine_v5.py` / `refine_v5s.py` (per-lap lag removed; stride 4 / 2), `refine_v6.py` (speed-fed placement), `refine_v7.py` (per-frame placement), `refine_v8.py` (the lap aligner) | `refine*.json` |
| **What unblocks both: a lap aligner** | `aligner_test.py` (placement precision; `STRIDE` env), `chains/aligner2.sh` | `aligner_mid*` |
| What moved the numbers (A/B, ablation, recipe, seeds) | `chains/ab_batch.sh`, `chains/ablation.sh`, `chains/recipe.sh`, `chains/overnight.sh`, `chains/followup.sh`, `chains/chain.sh` | `ab_*`, `abl_*`, `recipe_*` |
| the learning curve | `chains/learning_curve.sh` (v2, 1-5 tracks), `chains/v3_stage1.sh` (v3, 5-9 tracks) | `lc*`, `v3_curve_*` |
| v3 recipe, baseline, head-turn re-renders, seeds | `chains/v3_lib.sh`, `chains/v3_stage1.sh`, `chains/v3_stage2.sh` | `v3_*` |
| input resolution | `downsample_pack.py` (74x40, 111x60 copies of a packed set) | `ab_res*` |
| 8-bit costs nothing | `int8_check.py` (simulated on the Mac); the real int8 export is `npu/` | |
| v3 data checks | `check_v3.py` | |
| figures sent in conversation | `show_aug.py` (what the encoder gets after augmentation), `norms_figure.py` (normalisations and activations) | |
| tables across runs | `summarise.py` | |

`tools/crlf_edit.py` edits files that use Windows line endings (`train/*.py`,
`capture/session.py`) without converting them.
