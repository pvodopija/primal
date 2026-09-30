#!/bin/zsh
cd /Users/pvodopija/code/primal
run() {  # name, extra args...
  local name=$1; shift
  [[ -f runs/$name/gates.json ]] && { echo "skip $name"; return; }
  echo "=== $name  $* $(date +%H:%M)"
  ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --name $name "$@" > runs/lc_logs/$name.train.log 2>&1 || { echo "train failed $name"; return; }
  ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --out runs/$name/gates.json > runs/$name/gates.log 2>&1
  grep -E "^G1|^G2" -A1 runs/$name/gates.log | grep -v "^--"; grep -E "PASS|FAIL" runs/$name/gates.log | head -1
}
run ft_clean --init runs/ab_cam2_mirror_long/best.pt --steps 1500 --lr 1e-4
./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/ft_clean/best.pt --speed-sigma 2.0 --out runs/ft_clean/stream_speed_seed0.json > runs/ft_clean/stream_speed.log 2>&1; tail -6 runs/ft_clean/stream_speed.log
common=(--camera-aug --aug-zoom 1.08 1.08 --aug-pitch 1.0 --mirror-p 0.5)
run abl_all $common --aug-parts pose,blur,occlude,vignette,jpeg
run abl_pose $common --aug-parts pose
run abl_pose_blur $common --aug-parts pose,blur
run abl_pose_occlude $common --aug-parts pose,occlude
run abl_pose_jpeg $common --aug-parts pose,vignette,jpeg
echo "done $(date +%H:%M)"
