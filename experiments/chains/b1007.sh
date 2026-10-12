#!/bin/zsh
# Batch b1007 (AC, 2026-10-07: full-width wander on the twelve replays in two weathers).
# The release recipe (v3_mobilenet_lr1_s0: MobileNet at the full rate, look re-renders, yaw
# augmentation 10) plus both wander sets through the virtual camera, turned back to the
# car's heading (the offset kept, the wander's turn undone: yaw is the head-angle search's
# job) with the release's own head jitter (10 deg yaw, 1 pitch, 6 roll). Two seeds. Wander
# pieces that could be the held-out G1 drives, and pieces under 5 s, stay out (train.py).
# Scored as the release (gates = wrong-reference control, G1/G2 streams, head turns, the
# virtual camera), on the lateral-offset test (experiments/lateral_offset.py) and on the
# real footage (indoor GoPro c80, BHGP full width, RaceChrono), against the release.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
B1007=(--wide data/packed_ac_b1007_full_own,data/packed_ac_b1007_full_alt --wide-derotate --head 10 1 6)
lateral() {
  [[ -f runs/$1/lateral_offset.json ]] || PYTHONPATH=. ./.venv/bin/python experiments/lateral_offset.py $1 > runs/$1/lateral_offset.log 2>&1
  echo "--- lateral $1"; grep -E "as rendered|turned back" runs/$1/lateral_offset.log | head -7
}
real() {  # runs: every model on every real-footage set, against the ORB truth
  for spec in "indoor-pov c80" "BHGP full" "race-chrono "; do
    local footage=${spec%% *} crop=${spec#* }
    REAL_FOOTAGE=$footage REAL_CROP=$crop PYTHONPATH=. ./.venv/bin/python experiments/real_truth_orb.py "$@" 2>&1 | grep -E "single:" | sed "s/^/  $footage $crop: /"
  done
}
lateral v3_mobilenet_lr1_s0
for seed in 0 1; do
  recipe b1007_mobilenet_s$seed $seed --with-look --aug-yaw 10 --encoder mobilenet --backbone-lr-scale 1.0 $B1007
  camera_eval b1007_mobilenet_s$seed
  lateral b1007_mobilenet_s$seed
done
echo "=== real footage  $(date +%H:%M)"
real v3_mobilenet_lr1_s0 b1007_mobilenet_s0 b1007_mobilenet_s1
./.venv/bin/python -m experiments.index_runs
echo "b1007 done $(date +%H:%M)"
