#!/bin/zsh
# 1) the tail at runtime strides 2 and 3 (stride 4 was chosen for robustness to look-alikes);
# 2) fine-tune with training strides centred on the runtime stride, against lagfix_ctl
#    (the same fine-tune with the default strides 1-4).
cd /Users/pvodopija/code/primal
SP=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad
for st in 2 3; do
  echo "=== ft_clean at runtime stride $st, 23 Silverstone lives  $(date +%H:%M)"
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/ft_clean/best.pt --stride $st --holdout-laps 23 --speed-sigma 2.0 --lag-k 0.5 --out runs/ft_clean/stream_lag_23_stride$st.json 2>&1 | grep -v objc | tail -11
done
name=strides26
echo "=== $name  $(date +%H:%M)"
[[ -f runs/$name/best.pt ]] || ./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --init runs/ft_clean/best.pt --steps 1500 --lr 1e-4 --strides 2,3,4,5,6 --name $name > runs/lc_logs/$name.train.log 2>&1
./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --out runs/$name/gates.json > runs/$name/gates.log 2>&1
grep -E "PASS|FAIL" runs/$name/gates.log | head -2
PYTHONPATH=. ./.venv/bin/python -u $SP/lag.py runs/$name/best.pt 23 4 2>&1 | grep -v objc | grep -v "^  G[12] ks_\|^  G1 magione"
for run in $name lagfix_ctl; do
  echo "=== $run, 23 Silverstone lives, stride 4  $(date +%H:%M)"
  ./.venv/bin/python -m train.eval stream --data data/packed_ac_v2 --checkpoint runs/$run/best.pt --holdout-laps 23 --speed-sigma 2.0 --lag-k 0.5 --out runs/$run/stream_lag_23.json 2>&1 | grep -v objc | tail -11
done
echo "done $(date +%H:%M)"
