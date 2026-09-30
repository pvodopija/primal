#!/bin/zsh
cd /Users/pvodopija/code/primal
for pitch in 1 3; do
  base=recipe_p${pitch}; ft=${base}_ft
  if [[ ! -f runs/$base/best.pt ]]; then
    echo "=== $base $(date +%H:%M)"
    ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --camera-aug --aug-parts pose --aug-zoom 1.08 1.08 --aug-pitch $pitch --mirror-p 0.5 --steps 6000 --name $base > runs/lc_logs/$base.train.log 2>&1
  fi
  if [[ ! -f runs/$ft/gates.json ]]; then
    echo "=== $ft $(date +%H:%M)"
    ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --init runs/$base/best.pt --steps 1500 --lr 1e-4 --name $ft > runs/lc_logs/$ft.train.log 2>&1
    ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$ft/best.pt --out runs/$ft/gates.json > runs/$ft/gates.log 2>&1
    ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$ft/best.pt --speed-sigma 2.0 --out runs/$ft/stream_speed_seed0.json > runs/$ft/stream_speed.log 2>&1
  fi
  grep -E "PASS|FAIL" runs/$ft/gates.log | head -1; tail -6 runs/$ft/stream_speed.log
done
echo "done $(date +%H:%M)"
