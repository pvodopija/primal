#!/bin/zsh
# After the ResNet replication: distil v3_resnet_s0 into Halo-sized students on the current
# recipe. 1) the 0.67M encoder as is, 2) without GroupNorm (what Halo's NPU can run).
# Compare with v3_yaw10_s0..s2 and v3_nonorm_s0.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until grep -q "^resnet s1 done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_resnet_s1.out 2>/dev/null; do sleep 120; done
T=runs/v3_resnet_s0/best.pt
recipe v3_distill_s0 0 --with-look --aug-yaw 10 --teacher $T
recipe v3_distill_nonorm_s0 0 --with-look --aug-yaw 10 --teacher $T --norm none
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "distill done $(date +%H:%M)"
