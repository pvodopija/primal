#!/bin/zsh
# After the yaw runs: the recipe with head-turn re-renders plus fold-back clips (25%),
# against v3_look_s0..s2. Then the matcher's lag at runtime strides 4 and 2 (Silverstone,
# 46 pairs, experiments/lag.py) for the recipe and the fold model: folds should weaken
# the pace prior, so the lag should shrink, most visibly at stride 4.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until grep -q "^yaw done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_yaw.out 2>/dev/null; do sleep 120; done
recipe v3_fold_s0 0 --with-look --p-fold 0.25
cd /Users/pvodopija/code/primal
for run in v3_look_s0 v3_fold_s0; do
  for st in 4 2; do
    echo "=== lag $run stride $st"
    PYTHONPATH=. ./.venv/bin/python -u experiments/lag.py runs/$run/best.pt 23 $st 2>&1 | grep -v objc | grep -E "forward lag by|G2 filter  |G2 filter \+ speed" 
  done
done
./.venv/bin/python -m experiments.index_runs
echo "fold done $(date +%H:%M)"
