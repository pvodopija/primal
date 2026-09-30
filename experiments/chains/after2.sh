#!/bin/zsh
# After after.sh: every model at runtime stride 2 (the lag's cause is stride 4 sitting at the
# fast edge of training's 1-4), on all 23 Silverstone laps, with speed and the lag correction.
cd /Users/pvodopija/code/primal
OUT=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/tasks/bcvy3f8kh.output
until grep -q "^done" $OUT; do sleep 120; done
for run in recipe_p3_ft recipe_p3_s1_ft recipe_p3_s2_ft recipe_p1_s1_ft recipe_p1_s2_ft recipe_p1_ft ac2_time_allframes; do
  [[ -f runs/$run/best.pt ]] || continue
  out=runs/$run/stream_lag_23_stride2.json
  [[ -f $out ]] && continue
  echo "=== $run, stride 2, 23 Silverstone lives  $(date +%H:%M)"
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$run/best.pt --stride 2 --holdout-laps 23 --speed-sigma 2.0 --lag-k 0.5 --out $out 2>&1 | grep -v objc | tail -10
done
echo "done $(date +%H:%M)"
