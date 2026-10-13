#!/bin/zsh
# The ten new circuits (packed_ac_b1007_new_live: 8 training tracks, 2 held out) with v3 and
# v3_wide, both wide sets through the (bicubic) virtual camera, the wander's turn undone,
# live clips paired with reference places at most 3 m sideways. Two runs on the release
# recipe (v3_mobilenet_lr1_s0), pitch first (Pavle's real session: pitch matters most):
#   newtracks_pitch_s0: held head pitch up to +-8 deg on wide clips (release +-1) and kart
#                       shake (level 0-1) on every live clip;
#   newtracks_s0:       the release's pose amplitudes, the new data alone.
# Scored as the release (gates, G1/G2 streams, head turns), on the virtual camera with the
# bicubic resampler (the release re-scored too), on the new held-out tracks level and
# pitched (experiments/newtracks_eval.py), and on the real footage.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
COMMON=(--with-look --aug-yaw 10 --encoder mobilenet --backbone-lr-scale 1.0
        --wide data/packed_ac_v3_wide,data/packed_ac_b1007_new_live --wide-derotate --max-lateral-gap 3)
cam() {
  local out=runs/$1/camera_v3_bicubic.json
  [[ -f $out ]] || ./.venv/bin/python -m train.eval camera --data $D --wide data/packed_ac_v3_wide --checkpoint runs/$1/best.pt --out $out > runs/$1/camera_v3_bicubic.log 2>&1
  echo "--- camera (bicubic) $1"; grep -E "^(level|yaw 7|pitch 5|roll 10|shake 2|same session: level) " runs/$1/camera_v3_bicubic.log
}
newtracks() {
  [[ -f runs/$1/newtracks_eval.json ]] || PYTHONPATH=. ./.venv/bin/python experiments/newtracks_eval.py $1 > runs/$1/newtracks_eval.log 2>&1
  echo "--- new held-out tracks $1"; grep -E "pitch|level," runs/$1/newtracks_eval.log
}
real() {
  for spec in "indoor-pov c80" "BHGP full" "race-chrono "; do
    local footage=${spec%% *} crop=${spec#* }
    REAL_FOOTAGE=$footage REAL_CROP=$crop PYTHONPATH=. ./.venv/bin/python experiments/real_truth_orb.py "$@" 2>&1 | grep -E "single:" | sed "s/^/  $footage $crop: /"
  done
}
cam v3_mobilenet_lr1_s0
recipe newtracks_pitch_s0 0 $COMMON --head 10 8 8 --shake 1; cam newtracks_pitch_s0; newtracks newtracks_pitch_s0
recipe newtracks_s0 0 $COMMON --head 10 1 6; cam newtracks_s0; newtracks newtracks_s0
echo "=== real footage  $(date +%H:%M)"
real v3_mobilenet_lr1_s0 newtracks_pitch_s0 newtracks_s0
./.venv/bin/python -m experiments.index_runs
echo "newtracks done $(date +%H:%M)"
