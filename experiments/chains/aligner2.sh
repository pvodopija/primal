#!/bin/zsh
# Two cheap aligner variants: from the pose + mirror recipe (better on unseen tracks), and
# twice as long from the clean finish. Wrong-reference control, then placement on Silverstone.
cd /Users/pvodopija/code/primal
SP=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad
run() {  # name init steps
  echo "=== $1  $(date +%H:%M)"
  [[ -f runs/$1/best.pt ]] || ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --init $2 --target-at middle --steps $3 --lr 3e-4 --name $1 > runs/lc_logs/$1.train.log 2>&1
  ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$1/best.pt --out runs/$1/gates.json > runs/$1/gates.log 2>&1
  grep -E "PASS|FAIL" runs/$1/gates.log | head -2; grep "best eval" runs/lc_logs/$1.train.log
  PYTHONPATH=. ./.venv/bin/python -u $SP/aligner_test.py runs/ft_clean/best.pt runs/$1/best.pt 2>&1 | grep -v objc | tail -3
}
run aligner_mid_p1 runs/recipe_p1_ft/best.pt 3000
run aligner_mid_long runs/ft_clean/best.pt 6000
echo "done $(date +%H:%M)"
