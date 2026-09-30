#!/bin/zsh
# Can training remove the lag? Fine-tune the clean-finish model 1500 steps with the head
# seeing weak-match correlations half the time, against the same fine-tune without it.
cd /Users/pvodopija/code/primal
SP=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad
for name in lagfix_deg lagfix_ctl; do
  extra=(); [[ $name == lagfix_deg ]] && extra=(--corr-degrade 0.5)
  echo "=== $name  $(date +%H:%M)"
  [[ -f runs/$name/best.pt ]] || ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --init runs/ft_clean/best.pt --steps 1500 --lr 1e-4 $extra --name $name > runs/lc_logs/$name.train.log 2>&1
  ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --out runs/$name/gates.json > runs/$name/gates.log 2>&1
  grep -E "PASS|FAIL" runs/$name/gates.log | head -2
  PYTHONPATH=. ./.venv/bin/python -u $SP/lag.py runs/$name/best.pt 23 2>&1 | grep -v objc | grep -v "^  G[12] ks_\|^  G1 magione"
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --speed-sigma 2.0 --lag-k 0.5 --out runs/$name/stream_lag_3.json 2>&1 | grep -v objc | tail -11
done
echo "done $(date +%H:%M)"
