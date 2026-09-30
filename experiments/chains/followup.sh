#!/bin/zsh
cd /Users/pvodopija/code/primal
for run in ab_camera_mirror ab_long ab_camera; do
  [[ -f runs/$run/stream_seed0.json ]] && continue
  echo "=== stream $run $(date +%H:%M)"
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$run/best.pt --out runs/$run/stream_seed0.json > runs/$run/stream.log 2>&1 || echo "stream failed $run"
done
name=ab_camera_mirror_long
if [[ ! -f runs/$name/gates.json ]]; then
  echo "=== $name $(date +%H:%M)"
  ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --workers 3 --camera-aug --mirror-p 0.5 --steps 6000 --name $name > runs/lc_logs/$name.train.log 2>&1
  ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --out runs/$name/gates.json > runs/$name/gates.log 2>&1
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --out runs/$name/stream_seed0.json > runs/$name/stream.log 2>&1
fi
echo "done $(date +%H:%M)"
