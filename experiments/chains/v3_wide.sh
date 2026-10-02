#!/bin/zsh
# The virtual camera (train/camera.py) on the wide renders. B: today's models on the new
# tests (held head turns to 14 deg, shake, Halo's view). C: the small model trained with
# the wide renders seen at random head poses (+-14 yaw, +-4 pitch, +-8 roll) and shake on
# every clip (level 0-2), 3 seeds against v3_yaw10_s0..s2; then MobileNet-Small at the
# full rate the same way, against v3_mobilenet_lr1_s0.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until grep -q "^mobilenet large done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_mobilenet_large.out 2>/dev/null; do sleep 60; done
WIDE=(--wide data/packed_ac_v3_wide --shake 2)
for run in v3_yaw10_s0 v3_mobilenet_lr1_s0 v3_resnet_s0 v3_mobilenet_large_s0; do camera_eval $run; done
recipe v3_wide_s0 0 --with-look --aug-yaw 10 $WIDE; camera_eval v3_wide_s0
for run in v3_yaw10_s1 v3_yaw10_s2; do camera_eval $run; done
for seed in 1 2; do recipe v3_wide_s$seed $seed --with-look --aug-yaw 10 $WIDE; camera_eval v3_wide_s$seed; done
recipe v3_mobilenet_wide_s0 0 --with-look --aug-yaw 10 --encoder mobilenet --backbone-lr-scale 1.0 $WIDE; camera_eval v3_mobilenet_wide_s0
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "wide done $(date +%H:%M)"
