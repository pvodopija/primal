# Shared by the v3 chains. Every model: wrong-reference control (gates), then whole-lap
# streams at runtime stride 2 over every unseen lap, per unseen track, with true speed and
# the lag correction; recipe-type models also on the head-turn re-renders.
cd /Users/pvodopija/code/primal
D=data/packed_ac_v3
L=runs/lc_logs
RECIPE=(--camera-aug --aug-parts pose --aug-zoom 1.08 1.08 --aug-pitch 1 --mirror-p 0.5)
train() {  # name, then train.train arguments
  local name=$1; shift
  [[ -f runs/$name/best.pt ]] && return
  echo "=== train $name  $(date +%H:%M)"
  ./.venv/bin/python -u -m train.train --data $D --gate g1 "$@" --name $name > $L/$name.train.log 2>&1 || echo "!!! $name failed, see $L/$name.train.log"
}
evaluate() {  # name [look]
  local name=$1
  [[ -f runs/$name/best.pt ]] || return
  if [[ ! -f runs/$name/gates_v3.json ]]; then
    ./.venv/bin/python -m train.eval gates --data $D --checkpoint runs/$name/best.pt --out runs/$name/gates_v3.json > runs/$name/gates_v3.log 2>&1
  fi
  echo "--- $name  $(grep -E 'PASS|FAIL' runs/$name/gates_v3.log | head -1)"
  if [[ ! -f runs/$name/stream_v3.json ]]; then
    ./.venv/bin/python -m train.eval stream --data $D --checkpoint runs/$name/best.pt --holdout-laps 99 --speed-sigma 2.0 --lag-k 0.5 --out runs/$name/stream_v3.json > runs/$name/stream_v3.log 2>&1
  fi
  grep -E "^G1 filter  |^G1 filter \+ speed  |^G2 filter  |^G2 filter \+ speed  |^G2 .* filter  |^G2 .* filter \+ speed  " runs/$name/stream_v3.log
  if [[ $2 == look && ! -f runs/$name/stream_look.json ]]; then
    ./.venv/bin/python -m train.eval stream --data $D --checkpoint runs/$name/best.pt --look --holdout-laps 99 --speed-sigma 2.0 --out runs/$name/stream_look.json > runs/$name/stream_look.log 2>&1
    grep -E "^G2 filter  |^G2 filter \+ speed  " runs/$name/stream_look.log | sed 's/^/  head turns: /'
  fi
}
recipe() {  # name seed [extra train args]
  local name=$1 seed=$2; shift 2
  train ${name}_base --seed $seed $RECIPE --steps 6000 "$@"
  train $name --seed $seed --init runs/${name}_base/best.pt --steps 1500 --lr 1e-4 "$@"
  evaluate $name look
}
