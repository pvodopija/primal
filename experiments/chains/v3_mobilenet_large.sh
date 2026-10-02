#!/bin/zsh
# After the MobileNet-Small follow-ups: MobileNetV3-Large cut to Halo's size.
source /Users/pvodopija/code/primal/experiments/chains/v3_lib.sh
until grep -q "^mobilenet2 done" /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_mobilenet2.out 2>/dev/null; do sleep 60; done
recipe v3_mobilenet_large_s0 0 --with-look --aug-yaw 10 --encoder mobilenet_large
cd /Users/pvodopija/code/primal && ./.venv/bin/python -m experiments.index_runs
echo "mobilenet large done $(date +%H:%M)"
