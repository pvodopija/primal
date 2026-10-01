#!/bin/zsh
# Night 2: Pavle's kart-ahead augmentation, and a pretrained encoder. All on the current
# recipe (pose + mirror + head-turn re-renders + yaw +-10, clean finish), against
# v3_yaw10_s0..s2. Every model also on the traffic test (a kart ahead for 30% of each lap,
# drawn by a silhouette training never used).
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
traffic_eval() {
  [[ -f runs/$1/stream_traffic.json ]] && return
  ./.venv/bin/python -m train.eval stream --data $D --checkpoint runs/$1/best.pt --holdout-laps 99 --speed-sigma 2.0 --traffic-test 0.3 --out runs/$1/stream_traffic.json > runs/$1/stream_traffic.log 2>&1
  echo "  traffic test: $(grep -E '^G2 filter, reference time, two modes  |^G2 filter  ' runs/$1/stream_traffic.log | sed 's/  */ /g' | tr '\n' '|')"
}
for s in 0 1 2; do echo "=== traffic test v3_yaw10_s$s"; traffic_eval v3_yaw10_s$s; done
recipe v3_traffic_s0 0 --with-look --aug-yaw 10 --traffic 0.3; traffic_eval v3_traffic_s0
recipe v3_resnet_s0 0 --with-look --aug-yaw 10 --encoder resnet18; traffic_eval v3_resnet_s0
recipe v3_fold10_s0 0 --with-look --aug-yaw 10 --p-fold 0.1
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "night2 done $(date +%H:%M)"
