#!/bin/zsh
cd /Users/pvodopija/code/primal
for run in ft_clean recipe_p1_ft ac2_time_allframes; do
  echo "=== $run $(date +%H:%M)"
  PYTHONPATH=. ./.venv/bin/python -u /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/lag.py runs/$run/best.pt 23 2>&1 | grep -v objc | grep -v "^  G[12] ks_\|^  G1 magione"
done
