#!/bin/zsh
# 1) wait for the recipe runs, 2) pick the pitch by a fixed rule, 3) voting experiment on three
# models, 4) the winning recipe twice more with other seeds. All with both controls in gates.
cd /Users/pvodopija/code/primal
SP=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad
while [[ ! -f runs/recipe_p3_ft/stream_speed_seed0.json ]]; do sleep 120; done
# Rule, fixed before seeing results: fewer Silverstone ticks over 100 ms through the filter
# (vision only) wins; within one percentage point, the realistic +-3 deg wins.
winner=$(./.venv/bin/python -c "
import json
r = {p: json.load(open(f'runs/recipe_p{p}_ft/stream_speed_seed0.json'))['report']['G2 filter']['over_100ms'] for p in (1, 3)}
print(3 if r[3] <= r[1] + 0.01 else 1)")
echo "=== pitch winner: +-$winner deg  $(date +%H:%M)"
for ck in runs/ac2_time_allframes/best.pt runs/ft_clean/best.pt runs/recipe_p${winner}_ft/best.pt; do
  echo "=== voting $ck  $(date +%H:%M)"
  PYTHONPATH=. ./.venv/bin/python $SP/multiref.py $ck 5 2>&1 | grep -v objc | grep -E "^  G[12] [0-9]|whole laps"
done
for seed in 1 2; do
  base=recipe_p${winner}_s${seed}; ft=${base}_ft
  echo "=== $ft  $(date +%H:%M)"
  [[ -f runs/$base/best.pt ]] || ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed $seed --camera-aug --aug-parts pose --aug-zoom 1.08 1.08 --aug-pitch $winner --mirror-p 0.5 --steps 6000 --name $base > runs/lc_logs/$base.train.log 2>&1
  [[ -f runs/$ft/best.pt ]] || ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed $seed --init runs/$base/best.pt --steps 1500 --lr 1e-4 --name $ft > runs/lc_logs/$ft.train.log 2>&1
  ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$ft/best.pt --out runs/$ft/gates.json > runs/$ft/gates.log 2>&1
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$ft/best.pt --speed-sigma 2.0 --out runs/$ft/stream_speed_seed0.json > runs/$ft/stream_speed.log 2>&1
  grep -E "PASS|FAIL" runs/$ft/gates.log | head -1; tail -6 runs/$ft/stream_speed.log
done
echo "done $(date +%H:%M)"
