#!/bin/zsh
# Yaw augmentation: +-10 deg beat the recipe under head turns on one seed (v3_yaw10_s0).
# Two more seeds, and +-15 deg once, all with the head-turn re-renders.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
for seed in 1 2; do recipe v3_yaw10_s$seed $seed --with-look --aug-yaw 10; done
recipe v3_yaw15_s0 0 --with-look --aug-yaw 15
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "yaw done $(date +%H:%M)"
