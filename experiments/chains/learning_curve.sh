#!/bin/zsh
# Track-count learning curve: same recipe as runs/ac2_time_allframes (g1, 3000 steps,
# time bins, every frame supervised), trained on subsets of the five training tracks.
# Tracks left out, and Silverstone, are unseen at evaluation. Controls run in `gates`.
cd /Users/pvodopija/code/primal; mkdir -p runs/lc_logs
BC=ks_black_cat_county__layout_short; BH=ks_brands_hatch__indy; RB=ks_red_bull_ring__layout_national
VA=ks_vallelunga__club_circuit; MA=magione
runs=(
  "lc1a:$VA:0" "lc1b:$RB:0"
  "lc2a:$RB,$MA:0" "lc2b:$BH,$VA:0"
  "lc3a:$RB,$VA,$MA:0" "lc3b:$BC,$BH,$VA:0"
  "lc4a:$BH,$RB,$VA,$MA:0" "lc4b:$BC,$RB,$VA,$MA:0"
  "lc5s1:$BC,$BH,$RB,$VA,$MA:1"
)
for spec in $runs; do
  name=${spec%%:*}; rest=${spec#*:}; tracks=${rest%:*}; seed=${rest##*:}
  if [[ -f runs/$name/gates.json ]]; then echo "skip $name"; continue; fi
  echo "=== $name  tracks=$tracks  seed=$seed  $(date +%H:%M)"
  ./.venv/bin/python -m train.train --data data/packed_ac_v2 --gate g1 --tracks $tracks --seed $seed --name $name > runs/lc_logs/$name.train.log 2>&1 || { echo "train failed: $name"; continue; }
  ./.venv/bin/python -m train.eval gates --data data/packed_ac_v2 --checkpoint runs/$name/best.pt --out runs/$name/gates.json > runs/$name/gates.log 2>&1 || echo "gates failed: $name"

  grep -E "G2|wrong reference|PASS|FAIL" runs/$name/gates.log | head -6
done
echo "done $(date +%H:%M)"
