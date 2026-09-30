#!/bin/zsh
# After the partial-gradient A/B: a timing check of parallel data loading, the
# lower-resolution copies, the 8-bit check, then the A/B arms.
cd /Users/pvodopija/code/primal
SP=/private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad
while [[ ! -f runs/ab_refgrad/gates.json ]]; do sleep 60; done
echo "=== timing: 200 steps, baseline recipe, 3 data workers  $(date +%H:%M)"
./.venv/bin/python -u -m train.train --data data/packed_ac_v2 --gate g1 --seed 0 --workers 3 --steps 200 --eval-every 1000 --name timing_workers3 > runs/lc_logs/timing_workers3.log 2>&1
grep "s/step" runs/lc_logs/timing_workers3.log | tail -1
echo "=== downsampled copies  $(date +%H:%M)"
[[ -f data/packed_ac_v2_74x40/index.json ]] || ./.venv/bin/python $SP/downsample_pack.py data/packed_ac_v2 data/packed_ac_v2_74x40 74 40 > runs/lc_logs/down74.log 2>&1
[[ -f data/packed_ac_v2_111x60/index.json ]] || ./.venv/bin/python $SP/downsample_pack.py data/packed_ac_v2 data/packed_ac_v2_111x60 111 60 > runs/lc_logs/down111.log 2>&1
tail -1 runs/lc_logs/down74.log; tail -1 runs/lc_logs/down111.log
echo "=== 8-bit check  $(date +%H:%M)"
PYTHONPATH=. ./.venv/bin/python -u $SP/int8_check.py runs/ac2_time_allframes/best.pt data/packed_ac_v2 2>&1 | grep -v objc
echo "=== A/B arms  $(date +%H:%M)"
$SP/ab_batch.sh
