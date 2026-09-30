#!/bin/zsh
# After the overnight chain: lag-corrected whole-lap evals for every model, on the standard
# six Silverstone pairs and on all 23 laps (46 pairs).
cd /Users/pvodopija/code/primal
OUT=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/tasks/bg9wz73w3.output
until grep -q "^done" $OUT; do sleep 120; done
for run in ac2_time_allframes ft_clean recipe_p1_ft recipe_p3_ft recipe_p1_s1_ft recipe_p1_s2_ft recipe_p3_s1_ft recipe_p3_s2_ft; do
  [[ -f runs/$run/best.pt ]] || continue
  for lives in 3 23; do
    out=runs/$run/stream_lag_$lives.json
    [[ -f $out ]] && continue
    echo "=== $run, $lives Silverstone lives  $(date +%H:%M)"
    ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$run/best.pt --holdout-laps $lives --speed-sigma 2.0 --lag-k 0.5 --out $out 2>&1 | grep -v objc | tail -10
  done
done
echo "done $(date +%H:%M)"
