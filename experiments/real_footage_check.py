"""Is the raw-pixel truth trustworthy? Compare it per lap with a constant-pace warp, and see
where in the lap a model's disagreement with it sits."""
import numpy as np
from experiments.real_footage import frames, pixels, refine_crossings, viterbi, SPACING_S, HZ, BIN_S

f, t = frames()
px = pixels(f)
starts = refine_crossings(px, t)
laps = list(zip(starts[:-1], starts[1:]))
ref_start, ref_end = laps[0]; T = ref_end - ref_start
n = int(round(T / BIN_S)); bin_t = ref_start + np.arange(n) * T / n
bin_frame = np.array([int(np.argmin(np.abs(t - x))) for x in bin_t])
for k, (a, b) in enumerate(laps[1:], start=2):
    ticks = np.arange(a + 11 * SPACING_S, b, 1 / HZ)
    tf = np.array([int(np.argmin(np.abs(t - x))) for x in ticks])
    sim = np.mean([px[np.clip(tf + c, 0, len(t) - 1)] @ px[np.clip(bin_frame + c, 0, len(t) - 1)].T for c in (-4, -2, 0)], 0)
    path = viterbi(sim * 30.0, 4) * T / n
    linear = (ticks - a) * T / (b - a)
    dev = path - linear
    best = sim.max(1); at_path = sim[np.arange(len(path)), viterbi(sim * 30.0, 4)]
    # where the truth deviates most: seconds into the lap, and how sharp the pixel match is there
    worst = np.argsort(-np.abs(dev))[:1]
    print(f"lap {k}: truth - constant pace: median {np.median(np.abs(dev))*1000:5.0f} ms, p90 {np.percentile(np.abs(dev),90)*1000:5.0f} ms, max {np.abs(dev).max():5.2f} s at {ticks[worst][0]-a:5.1f} s into the lap; "
          f"pixel match on the path {np.median(at_path):.2f} (best anywhere {np.median(best):.2f})")
