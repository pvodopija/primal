#!/bin/zsh
# After v3_wide.sh. Exact rotations on a fifth of the clips halved the pitch and roll
# losses but barely moved a held yaw (42% -> 38% at 7 deg, straights as bad as corners).
# Probe: the wide renders on 80% of the live clips, one seed, against v3_wide_s0..s2.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until grep -q "^wide done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_wide.out 2>/dev/null; do sleep 60; done
recipe v3_wide80_s0 0 --with-look --aug-yaw 10 --wide data/packed_ac_v3_wide --shake 2 --wide-share 0.8; camera_eval v3_wide80_s0
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "wide80 done $(date +%H:%M)"
