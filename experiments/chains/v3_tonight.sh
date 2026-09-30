#!/bin/zsh
# Tonight, after the stage-1 re-evaluations: the chosen recipe (pose + mirror + head-turn
# re-renders, clean finish) 1) without GroupNorm, for Halo's NPU, 2) with yaw augmentation
# widened from +-4 to +-10 deg, for head turns. Compare with v3_look_s0..s2.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until ! pgrep -f "train.eval stream --data data/packed_ac_v3 --checkpoint runs/v3_(base|recipe|look|curve)" > /dev/null; do sleep 60; done
recipe v3_nonorm_s0 0 --with-look --norm none
recipe v3_yaw10_s0 0 --with-look --aug-yaw 10
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "tonight done $(date +%H:%M)"
