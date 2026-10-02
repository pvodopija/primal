#!/bin/zsh
# MobileNetV3-Small kept about half of ResNet-18's gain on one seed (unseen 10.0% with the
# two-mode reference-time tracker). A second seed, and the backbone at the full learning
# rate instead of 0.3x (more room to adapt to race tracks).
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
recipe v3_mobilenet_s1 1 --with-look --aug-yaw 10 --encoder mobilenet
recipe v3_mobilenet_lr1_s0 0 --with-look --aug-yaw 10 --encoder mobilenet --backbone-lr-scale 1.0
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "mobilenet2 done $(date +%H:%M)"
