#!/bin/zsh
# The ImageNet ResNet-18 encoder halved every error on one seed; a second seed to confirm.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
recipe v3_resnet_s1 1 --with-look --aug-yaw 10 --encoder resnet18
./.venv/bin/python -m train.eval stream --data $D --checkpoint runs/v3_resnet_s1/best.pt --holdout-laps 99 --speed-sigma 2.0 --traffic-test 0.3 --out runs/v3_resnet_s1/stream_traffic.json > runs/v3_resnet_s1/stream_traffic.log 2>&1
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "resnet s1 done $(date +%H:%M)"
