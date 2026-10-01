#!/bin/zsh
# After the distillation runs: the pretrained-encoder questions, on the current recipe.
# 1) MobileNetV3-Small: does a Halo-sized pretrained encoder keep ResNet-18's gain?
# 2) ResNet-18 frozen: off-the-shelf eyes, only projection and head learn
# 3) ResNet-18 from scratch: the architecture (residuals, depth) without ImageNet
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until grep -q "^distill done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_distill.out 2>/dev/null; do sleep 60; done
recipe v3_mobilenet_s0 0 --with-look --aug-yaw 10 --encoder mobilenet
recipe v3_resnet_frozen_s0 0 --with-look --aug-yaw 10 --encoder resnet18 --freeze-backbone
recipe v3_resnet_scratch_s0 0 --with-look --aug-yaw 10 --encoder resnet18 --encoder-weights none
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "encoders done $(date +%H:%M)"
