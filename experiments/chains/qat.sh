#!/bin/zsh
# Quantization-aware fine-tune: newtracks_pitch_s0 (the pitch-emphasis model) trained on for
# 2000 steps with its encoder under simulated int8 (train/quantize.py: int8 weights per
# channel, every layer's output int8 over a tracked 99.99th-percentile range, rounding passed
# straight through), the same data and pose settings. Then: Core ML agreement in float16
# and int8 weights + activations (sim and real frames; the release and the pitch model too),
# accuracy in float and as on the NPU (PRIMAL_SIMULATE=int8) on the new held-out tracks and
# the real footage, the standard evaluations, and a float16 release if it passes its check.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
COMMON=(--with-look --aug-yaw 10 --encoder mobilenet --backbone-lr-scale 1.0
        --wide data/packed_ac_v3_wide,data/packed_ac_b1007_new_live --wide-derotate --max-lateral-gap 3 --head 10 8 8 --shake 1)
Q=newtracks_pitch_qat8_s0
P=newtracks_pitch_s0
newtracks() {  # run [int8]
  local sim=$2 out=runs/$1/newtracks_eval${2:+_$2}.json
  [[ -f $out ]] || PRIMAL_SIMULATE=$sim PYTHONPATH=. ./.venv/bin/python experiments/newtracks_eval.py $1 > runs/$1/newtracks_eval${2:+_$2}.log 2>&1
  echo "--- new held-out tracks $1 ${sim:-float}"; grep -E "^  (level|pitch [-+]8|pitch [-+]12|shake 2) " runs/$1/newtracks_eval${2:+_$2}.log
}
real() {  # sim, then runs
  local sim=$1; shift
  for spec in "indoor-pov c80" "BHGP full" "race-chrono "; do
    local footage=${spec%% *} crop=${spec#* }
    PRIMAL_SIMULATE=$sim REAL_FOOTAGE=$footage REAL_CROP=$crop PYTHONPATH=. ./.venv/bin/python experiments/real_truth_orb.py "$@" 2>&1 | grep -E "single:" | sed "s/^/  ${sim:-float} $footage $crop: /"
  done
}
train $Q --seed 0 --init runs/$P/best.pt --steps 2000 --lr 1e-4 $COMMON --qat int8
evaluate $Q look
echo "=== Core ML agreement  $(date +%H:%M)"
PYTHONPATH=. ./.venv/bin/python experiments/coreml_int8.py v3_mobilenet_lr1_s0 $P $Q 2>&1 | grep "cosine min"
echo "=== accuracy, float and as on the NPU  $(date +%H:%M)"
newtracks $Q; newtracks $P int8; newtracks $Q int8
real "" $P $Q; real int8 $P $Q
echo "=== float16 release  $(date +%H:%M)"
PYTHONPATH=. ./.venv/bin/python -m train.release $Q --name primal-qat8-$(date +%Y%m%d) 2>&1 | grep -E "cosine_min|argmax_same|written|disagrees"
./.venv/bin/python -m experiments.index_runs
echo "qat done $(date +%H:%M)"
