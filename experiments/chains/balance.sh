#!/bin/zsh
# Channel balancing for int8 (Halo's NPU): the pitch model (newtracks_pitch_s0) trained on for
# 2500 steps under simulated int8 with TFLite's min/max ranges and the channel-balancing term
# (train/quantize.py RangeBalance: a layer's loudest channel over its median beyond 4x
# penalised; the stem spanned 28-34x). Then: the spread, Halo's real int8 (TFLite on the
# exported model, agreement with PyTorch on sim and real frames; Vela), accuracy (new held-out
# tracks level and pitched, the night session against the transponder, the real-footage
# scoreboard) and the standard evaluations, against the pitch model.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
COMMON=(--with-look --aug-yaw 10 --encoder mobilenet --backbone-lr-scale 1.0
        --wide data/packed_ac_v3_wide,data/packed_ac_b1007_new_live --wide-derotate --max-lateral-gap 3 --head 10 8 8 --shake 1)
P=newtracks_pitch_s0
B=newtracks_pitch_bal8_s0
NPU=runs/lc_logs/npu
mkdir -p $NPU
train $B --seed 0 --init runs/$P/best.pt --steps 2500 --lr 1e-4 $COMMON --qat int8 --qat-ranges minmax --balance 1.0
echo "=== spread  $(date +%H:%M)"
PYTHONPATH=. ./.venv/bin/python experiments/channel_spread.py $P $B 2>&1 | grep "x;"
echo "=== Halo int8 (TFLite)  $(date +%H:%M)"
for r in $P $B; do
  PYTHONPATH=. ./.venv/bin/python -m npu.export_mobilenet $r $NPU/$r.npz > /dev/null 2>&1
  .venv-npu/bin/python npu/build_mobilenet.py $NPU/$r.npz $NPU/$r 2>&1 | grep -E "Keras float32|int8 TFLite|Vela Shared" | sed "s/^/  $r: /"
done
echo "=== accuracy  $(date +%H:%M)"
evaluate $B look
PYTHONPATH=. ./.venv/bin/python experiments/newtracks_eval.py $B > runs/$B/newtracks_eval.log 2>&1
grep -E "^  (level|pitch [-+]8|pitch [-+]12|shake 2) " runs/$B/newtracks_eval.log
REAL_FOOTAGE=night1 PYTHONPATH=. ./.venv/bin/python experiments/session_bars.py $B 2>&1 | grep -E "lap times|at the line against"
for spec in "indoor-pov c80" "BHGP full" "race-chrono "; do
  f=${spec%% *}; c=${spec#* }
  REAL_FOOTAGE=$f REAL_CROP=$c PYTHONPATH=. ./.venv/bin/python experiments/real_truth_orb.py $P $B 2>&1 | grep "single:" | sed "s/^/  $f $c: /"
done
./.venv/bin/python -m experiments.index_runs
echo "balance done $(date +%H:%M)"
