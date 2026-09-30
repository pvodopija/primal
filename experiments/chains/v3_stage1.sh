#!/bin/zsh
source /private/tmp/claude-501/-Users-pvodopija-code-primal/6c12a625-049e-47d4-8baa-f63390f6b6ec/scratchpad/v3_lib.sh
# 1) recipe, 2) plain baseline with the same steps, 3) recipe with head-turn re-renders
recipe v3_recipe_s0 0
train v3_base_s0_base --seed 0 --steps 6000
train v3_base_s0 --seed 0 --init runs/v3_base_s0_base/best.pt --steps 1500 --lr 1e-4
evaluate v3_base_s0 look
recipe v3_look_s0 0 --with-look
# 4) learning curve on the new metric: 5 (the v2 tracks), 7, 9 training tracks, plain, 3000 steps
OLD=ks_black_cat_county__layout_short,ks_brands_hatch__indy,ks_red_bull_ring__layout_national,ks_vallelunga__club_circuit,magione
for k in 5 7 9; do
  case $k in
    5) T=$OLD ;;
    7) T=$OLD,ks_highlands__layout_short,ks_laguna_seca ;;
    9) T=$OLD,ks_highlands__layout_short,ks_laguna_seca,ks_monza66__junior,ks_nurburgring__layout_sprint_a ;;
  esac
  train v3_curve_k$k --seed 0 --steps 3000 --tracks $T
  evaluate v3_curve_k$k
done
echo "stage 1 done $(date +%H:%M)"
