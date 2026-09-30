#!/bin/zsh
# A/B arms on the five training tracks, 3000 steps each, compared with ac2_time_allframes
# (seed 0) and lc5s1 (seed 1). Gates run the wrong-reference control on every checkpoint.
cd /Users/pvodopija/code/primal; mkdir -p runs/lc_logs
FAST=${FAST:-}   # e.g. "--ref-grad-random 0.1" once the partial-gradient A/B passes
arms=(
  "ab_mirror|--mirror-p 0.5"
  "ab_camera|--camera-aug"
  "ab_camera_mirror|--camera-aug --mirror-p 0.5"
  "ab_wide|--width 64"
  "ab_long|--steps 6000"
  "ab_res74|--data data/packed_ac_v2_74x40"
  "ab_res111|--data data/packed_ac_v2_111x60"
)
for spec in $arms; do
  name=${spec%%|*}; extra=${spec#*|}
  [[ -f runs/$name/gates.json ]] && { echo "skip $name"; continue; }
  data=data/packed_ac_v2; [[ $extra == *--data* ]] && data=$(echo $extra | sed -E 's/.*--data ([^ ]+).*/\1/') && extra=$(echo $extra | sed -E 's/--data [^ ]+//')
  echo "=== $name  $extra  $FAST  $(date +%H:%M)"
  ./.venv/bin/python -u -m train.train --data $data --gate g1 --seed 0 --workers 3 --name $name ${=extra} ${=FAST} > runs/lc_logs/$name.train.log 2>&1 || { echo "train failed: $name"; continue; }
  ./.venv/bin/python -m train.eval gates --data $data --checkpoint runs/$name/best.pt --out runs/$name/gates.json > runs/$name/gates.log 2>&1 || echo "gates failed: $name"
  grep -E "s/step" runs/lc_logs/$name.train.log | tail -1
  grep -E "^G2" -A1 runs/$name/gates.log | tail -1; grep -E "PASS|FAIL" runs/$name/gates.log | head -1
done
echo "done $(date +%H:%M)"
