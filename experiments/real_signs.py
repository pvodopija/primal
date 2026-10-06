"""
Green or red: does PRIMAL tell the driver the right thing at the simplest level?

  - ahead or behind: the sign of the delta, the colour the display shows (green = ahead);
  - gaining or losing: the sign of the delta's change over the last 2 s and 5 s.

Each tick of every live lap (real footage, experiments/real_demo.lap_series) falls in one
of four cells: both green, both red, PRIMAL green but truly red (it flatters), PRIMAL red
but truly green (it discourages). Counted over every tick, then only where the truth is
clear (the true delta at least 0.1 s from zero, or its change at least 0.05 s), then also
only where PRIMAL shows a delta (confidence at least CONF_MIN).

Usage: REAL_FOOTAGE=BHGP REAL_CROP=full PYTHONPATH=. python experiments/real_signs.py RUN
       (writes ROOT/signs[_crop].json)
"""
import json
import sys

import cv2
import numpy as np

from experiments.real_demo import CONF_MIN, lap_series, setup
from experiments.real_footage import HZ, ROOT, SUFFIX, VIDEO


def cells(primal_green: np.ndarray, true_green: np.ndarray) -> dict:
    n = max(int(primal_green.size), 1)
    out = {"ticks": int(primal_green.size),
           "both green": float(np.sum(primal_green & true_green) / n), "both red": float(np.sum(~primal_green & ~true_green) / n),
           "PRIMAL green, truly red": float(np.sum(primal_green & ~true_green) / n),
           "PRIMAL red, truly green": float(np.sum(~primal_green & true_green) / n)}
    out["agree"] = out["both green"] + out["both red"]
    # what always showing the commoner colour would score, and the mean of the two hit rates
    truly_green = out["both green"] + out["PRIMAL red, truly green"]
    out["always one colour"] = max(truly_green, 1 - truly_green)
    out["balanced"] = 0.5 * (out["both green"] / max(truly_green, 1e-9) + out["both red"] / max(1 - truly_green, 1e-9))
    return out


def main() -> None:
    run = sys.argv[1]
    S = setup(run)
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    rows = {"delta": [], "change 2 s": [], "change 5 s": []}
    for lap_no in [k for k in range(1, len(S.laps) + 1) if k != S.r + 1]:
        L = lap_series(S, lap_no, fps)
        ok = np.isfinite(L.true)
        rows["delta"].append((L.delta[ok], L.true[ok], L.conf[ok]))
        for w in (2, 5):
            k = int(w * HZ)
            d, tr, c = L.delta[ok], L.true[ok], L.conf[ok]
            rows[f"change {w} s"].append((d[k:] - d[:-k], tr[k:] - tr[:-k], c[k:]))
    report = {}
    for name, parts in rows.items():
        p, q, c = (np.concatenate(x) for x in zip(*parts))
        clear = np.abs(q) >= (0.1 if name == "delta" else 0.05)
        green = lambda x: x < 0  # ahead, or gaining
        report[name] = {"every tick": cells(green(p), green(q)),
                        "truth clear": cells(green(p[clear]), green(q[clear])),
                        "truth clear, shown": cells(green(p[clear & (c >= CONF_MIN)]), green(q[clear & (c >= CONF_MIN)]))}
        label = "ahead or behind (the delta)" if name == "delta" else f"gaining or losing over {name[7:]}"
        print(f"\n{label}")
        for part, r in report[name].items():
            print(f"  {part:<18} {r['ticks']:>5} ticks: agree {100 * r['agree']:5.1f}% (always one colour {100 * r['always one colour']:4.1f}%, "
                  f"balanced {100 * r['balanced']:4.1f}%)  |  both green {100 * r['both green']:4.1f}%, "
                  f"both red {100 * r['both red']:4.1f}%, PRIMAL green/truly red {100 * r['PRIMAL green, truly red']:4.1f}%, "
                  f"PRIMAL red/truly green {100 * r['PRIMAL red, truly green']:4.1f}%")
    (ROOT / f"signs{SUFFIX}.json").write_text(json.dumps({"run": run, "report": report}, indent=2))


if __name__ == "__main__":
    main()
