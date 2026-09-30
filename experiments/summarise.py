import json, sys
from pathlib import Path
runs = ["ac2_time_allframes", "ft_clean", "lagfix_ctl", "strides26", "recipe_p1_ft", "recipe_p3_ft", "recipe_p3_s1_ft", "recipe_p3_s2_ft"]
files = [("stride 4, 6 pairs", "stream_lag_3.json"), ("stride 4, 46 pairs", "stream_lag_23.json"), ("stride 2, 46 pairs", "stream_lag_23_stride2.json")]
keys = ["G1 filter", "G1 filter + speed", "G2 filter", "G2 filter, lag-corrected", "G2 filter + speed", "G2 filter + speed, lag-corrected"]
print(f"{'':44}" + "".join(f"{k.replace('filter', 'f').replace(', lag-corrected', ' lc'):>13}" for k in keys) + "   G2 f >10m")
for label, fn in files:
    print(f"-- {label}: % ticks over 100 ms")
    for r in runs:
        p = Path("runs") / r / fn
        if not p.exists():
            continue
        rep = json.loads(p.read_text())["report"]
        print(f"  {r:42}" + "".join(f"{100 * rep[k]['over_100ms']:12.1f}%" if k in rep else f"{'':13}" for k in keys) + f"{100 * rep['G2 filter']['over_10m']:10.1f}%")
