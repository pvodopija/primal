#!/bin/zsh
source /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_lib.sh
# 5) seeds 1 and 2 of the recipe with and without head-turn re-renders
until grep -q "^stage 1 done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_stage1.out 2>/dev/null; do sleep 120; done
for seed in 1 2; do
  recipe v3_recipe_s$seed $seed
  recipe v3_look_s$seed $seed --with-look
done
echo "stage 2 done $(date +%H:%M)"
