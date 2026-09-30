"""How predictable is the kart in metres vs in reference-lap time? Labels only.

For stream-eval pairs (all unseen laps of v3, plus same-car vs other-car references):
predict 1 s ahead by carrying the last second's average rate, in (a) metres along the
track, (b) reference time. Error in reference milliseconds, the delta's own unit."""
from pathlib import Path
import numpy as np
from train.eval import _stream_pairs
from train.dataset import LapIndex

pairs = _stream_pairs(Path("data/packed_ac_v3"), "time", 99)
res = {"same car": {"m": [], "t": []}, "other car": {"m": [], "t": []}}
rates = {"same car": [], "other car": []}
for gate, ref, live in pairs:
    g = ref.reference_grid("time")
    L, T = g.track_length_m, g.lap_time_s
    s, t = live.s().astype(np.float64), live.t().astype(np.float64)
    # true position in metres (unwrapped) and in reference seconds (unwrapped)
    pos = np.unwrap(s * 2 * np.pi) / (2 * np.pi) * L
    bins = g.target(s)
    tau = np.unwrap(np.asarray(bins, float) * 2 * np.pi / g.n_bins) * T / (2 * np.pi)
    fps = live.fps
    k = int(round(fps))          # 1 s
    i = np.arange(2 * k, len(t) - k, 4)
    to_ms = lambda metres_at: np.interp(metres_at % L, np.r_[g.pos_m, g.pos_m[0] + L], np.r_[g.time_s, g.time_s[0] + T])
    # (a) metres: speed = last second's average, carried 1 s
    v = (pos[i] - pos[i - k]) / (t[i] - t[i - k])
    pred_pos = pos[i] + v * (t[i + k] - t[i])
    err_m = np.abs(((to_ms(pred_pos) - to_ms(pos[i + k])) + T / 2) % T - T / 2) * 1000
    # (b) reference time: rate = last second's average dtau/dt, carried 1 s
    r = (tau[i] - tau[i - k]) / (t[i] - t[i - k])
    pred_tau = tau[i] + r * (t[i + k] - t[i])
    err_t = np.abs(pred_tau - tau[i + k]) * 1000
    key = "same car" if ref.car_model == live.car_model else "other car"
    res[key]["m"].append(err_m); res[key]["t"].append(err_t); rates[key].append(r)
for key in res:
    if not res[key]["m"]:
        continue
    m, tt, r = np.concatenate(res[key]["m"]), np.concatenate(res[key]["t"]), np.concatenate(rates[key])
    print(f"{key:10s} ({len(res[key]['m'])} pairs): 1 s ahead, error in delta ms  "
          f"metres: median {np.median(m):5.1f}, p90 {np.percentile(m, 90):5.1f}, >100 ms {100*np.mean(m>100):4.1f}%   |   "
          f"reference time: median {np.median(tt):5.1f}, p90 {np.percentile(tt, 90):5.1f}, >100 ms {100*np.mean(tt>100):4.1f}%   "
          f"(rate {np.median(r):.3f}, 5-95% {np.percentile(r,5):.3f}-{np.percentile(r,95):.3f})")
