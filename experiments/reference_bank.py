"""
A recognition bank (2026-10-17 prototype): the delta's clock stays one lap's (the timing lap), but
"where am I" matches the live clip against several laps, each mapped onto the timing lap's bins by
PRIMAL itself. A stretch the timing lap saw badly (a spin, traffic, a knocked camera) is then
recognised through the other laps.

Building it, as a phone could: each bank lap is run live (train/live.py LiveDelta) against the
timing lap; its confident ticks give, for every timing bin, the moment that lap passed it
(interpolated across unconfident stretches, steady pace assumed); its descriptor there is its entry.
The timing lap's own bins whose descriptor disagrees with the bank laps' at the same place (cosine
under JUNK_COS) are masked: it finds its own blind stretch. Matching: the clip against every lap,
"max" = each bin's best correlation then the head once; "mean" = the head per lap, the beliefs
averaged (a masked bin takes that lap's median correlation). The tracker and confidence as shipped.

Test (rotax-night, 30 fps): lap 1 the timing lap with GAP_S ruined at GAP_AT_S into it, "blind"
(black frames) or "spin" (the view from elsewhere on the lap); laps 2-3 the bank; laps 4.. live.
Every method is scored against the clean timing lap run alone (the comparison, there is no truth
for this video): the delta's difference over the lap and inside the ruined stretch, the share shown
there, the delta at the line. Controls: the bank on the clean laps (must not cost anything), and a
wrong-track bank (indoor-pov laps) against these laps (must stay unconfident).

Usage: [GAP_AT_S=20] [KINDS=blind,spin] [CONTROLS=1] PYTHONPATH=. python experiments/reference_bank.py RUN
       (writes data/real-footage/rotax-night/reference_bank_<run>.json)
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.speed_warp import lines
from train.dataset import _to_chw
from train.eval import load_model
from train.live import LiveDelta

FRAMES = "data/real-footage/rotax-night_148x80.npz"
START_S, BIN_S, PREROLL_S, FPS = 75.0, 0.052, 6.0, 30.0
GAP_AT_S, GAP_S = float(os.environ.get("GAP_AT_S", 20.0)), 4.0
KINDS = tuple(os.environ.get("KINDS", "blind,spin").split(","))
CONTROLS = os.environ.get("CONTROLS", "1") == "1"
BANK_LAPS, LIVE_LAPS = (1, 2), tuple(range(3, 11))  # lap k runs from line k to line k+1; lap 0 is the timing lap
JUNK_COS = 0.5
SHOWN = 0.5


class BankLiveDelta(LiveDelta):
    """LiveDelta matching against a bank (M, N, D) of laps on the timing lap's bins, mask (M, N) True = usable."""

    def __init__(self, *args, bank: torch.Tensor, mask: torch.Tensor, mode: str, **kw):
        super().__init__(*args, **kw)
        self.bank, self.mask, self.mode = bank, mask, mode

    def _logits(self, clip: torch.Tensor) -> torch.Tensor:
        corr = torch.einsum("kd,mnd->mkn", clip, self.bank)  # (M, K, N)
        fill = torch.stack([c[:, m].median() if m.any() else c.median() for c, m in zip(corr, self.mask)])
        corr = torch.where(self.mask[:, None, :], corr, fill[:, None, None])
        if self.mode == "max":
            masked = torch.where(self.mask[:, None, :], corr, torch.full_like(corr, -1e9)).max(0).values
            nobody = ~self.mask.any(0)
            masked[:, nobody] = corr.median()
            return self.model.head(masked[None] * self.scale)
        p = torch.stack([self.model.head(c[None] * self.scale).softmax(-1) for c in corr]).mean(0)
        return torch.log(p + 1e-12)


def run(engine: LiveDelta, d: torch.Tensor, t: np.ndarray, a: float, b: float):
    """The engine over PREROLL_S before line a, then the lap [a, b]; readings in the lap."""
    out = []
    for k in np.flatnonzero((t >= a - PREROLL_S) & (t < b)):
        r = engine.push_descriptor(d[k], float(t[k]))
        if r is not None and t[k] >= a:
            out.append((r.t, r.ref_time_s, r.confidence))
    return np.array(out)


def circ(x, T):
    return (x + T / 2) % T - T / 2


def main() -> None:
    run_name = sys.argv[1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    gpu, payload = load_model(Path("runs") / run_name / "best.pt", device)
    model, _ = load_model(Path("runs") / run_name / "best.pt", torch.device("cpu"))
    clip_len = int(payload["args"]["clip_len"])
    enc = lambda f: torch.cat([gpu.encoder(torch.from_numpy(_to_chw(f[i:i + 256])).to(device)).cpu() for i in range(0, len(f), 256)])
    z = np.load(FRAMES)
    f, t = z["frames"], z["t"]
    with torch.no_grad():
        d = enc(f)
        black = enc(np.zeros((1, 80, 148, 3), np.uint8))[0]
    c = lines(model, clip_len, d, t, START_S)
    a0, b0 = c[0], c[1]
    T = b0 - a0
    n = int(round(T / BIN_S))
    times = np.arange(n) * T / n
    near = lambda x: np.clip(np.round(np.interp(x, t, np.arange(len(t)))).astype(int), 0, len(t) - 1)
    gap_bins = (times >= GAP_AT_S) & (times < GAP_AT_S + GAP_S)
    gap_frames = (t >= a0 + GAP_AT_S) & (t < a0 + GAP_AT_S + GAP_S)
    engine = lambda ref, cls=LiveDelta, **kw: cls(model, None, times, T, clip_len, torch.device("cpu"), reference_descriptors=ref, **kw)

    def lap_descriptors(dd):
        return dd[near(a0 + times)]

    def build_bank(dd):
        """The timing lap (from dd) and the bank laps mapped onto its bins, with the junk mask."""
        timing = lap_descriptors(dd)
        rows, maps = [], []
        for k in BANK_LAPS:
            r = run(engine(timing), dd, t, c[k], c[k + 1])
            ok = r[:, 2] >= SHOWN
            rt, lt = r[ok, 1], r[ok, 0]
            order = np.argsort(rt)
            rt, lt = rt[order], lt[order]
            keep = np.r_[True, np.diff(np.maximum.accumulate(lt)) > 0]  # monotone: the lap moves one way
            maps.append(np.interp(times, rt[keep], lt[keep]))
            rows.append(dd[near(maps[-1])])
        others = torch.stack(rows)
        mean = torch.nn.functional.normalize(others.mean(0), dim=-1)
        agree = (timing * mean).sum(-1).numpy()
        mask = torch.ones(len(rows) + 1, n, dtype=torch.bool)
        mask[0] = torch.from_numpy(agree >= JUNK_COS)
        return timing, torch.cat([timing[None], others]), mask, agree, np.array(maps)

    report = {}
    clean_timing, clean_bank, clean_mask, clean_agree, clean_maps = build_bank(d)
    reference_runs = {k: run(engine(clean_timing), d, t, c[k], c[k + 1]) for k in LIVE_LAPS}

    def score(name: str, make, dd=d):
        diffs, gap_diffs, gap_shown, shown, at_line = [], [], [], [], []
        for k in LIVE_LAPS:
            ref_run, r = reference_runs[k], run(make(), dd, t, c[k], c[k + 1])
            m = min(len(ref_run), len(r))
            ref_run, r = ref_run[:m], r[:m]
            delta = circ((r[:, 0] - c[k]) - r[:, 1], T)
            truth = circ((ref_run[:, 0] - c[k]) - ref_run[:, 1], T)
            e = np.abs(delta - truth)
            into = r[:, 0] - c[k]
            inside = (into > 1) & (c[k + 1] - r[:, 0] > 1)
            in_gap = (ref_run[:, 1] >= GAP_AT_S) & (ref_run[:, 1] < GAP_AT_S + GAP_S)
            diffs.append(e[inside])
            gap_diffs.append(e[in_gap])
            gap_shown.append(r[in_gap, 2] >= SHOWN)
            shown.append(r[:, 2] >= SHOWN)
            at_line.append(1000 * abs(np.median(delta[-8:]) - np.median(truth[-8:])))
        e, g = np.concatenate(diffs), np.concatenate(gap_diffs)
        row = {"median_ms": 1000 * float(np.median(e)), "over_100ms": float(np.mean(e > 0.1)),
               "gap_median_ms": 1000 * float(np.median(g)), "gap_over_100ms": float(np.mean(g > 0.1)),
               "gap_shown": float(np.mean(np.concatenate(gap_shown))), "shown": float(np.mean(np.concatenate(shown))),
               "at_line_ms_median": float(np.median(at_line))}
        report[name] = row
        print(f"  {name:<44} vs the clean timing lap: median {row['median_ms']:5.0f} ms, {100 * row['over_100ms']:4.1f}% over 100 | "
              f"in the ruined stretch {row['gap_median_ms']:5.0f} ms, {100 * row['gap_over_100ms']:4.1f}% over 100, shown "
              f"{100 * row['gap_shown']:3.0f}% | shown {100 * row['shown']:3.0f}% | at the line {row['at_line_ms_median']:4.0f} ms", flush=True)

    print(f"{run_name}: lines {np.round(c[:12], 2).tolist()}; timing lap {T:.2f} s, {n} bins; ruined {GAP_AT_S}-{GAP_AT_S + GAP_S} s")
    if CONTROLS:
        print("clean laps (the bank must cost nothing):")
        score("one lap (as today)", lambda: engine(clean_timing))
        score("bank, max", lambda: engine(clean_timing, BankLiveDelta, bank=clean_bank, mask=clean_mask, mode="max"))
        score("bank, mean", lambda: engine(clean_timing, BankLiveDelta, bank=clean_bank, mask=clean_mask, mode="mean"))
    for kind in KINDS:
        dd = d.clone()
        idx = np.flatnonzero(gap_frames)
        if kind == "blind":
            dd[idx] = black
        else:  # the view from elsewhere: the same lap 20 s on, played backwards
            dd[idx] = d[near(t[idx] + 20.0)].flip(0)
        timing, bank, mask, agree, maps = build_bank(dd)
        found = ~mask[0].numpy()
        place = 1000 * np.abs(maps - clean_maps)  # where each bank lap's entry was taken, against the clean build
        print(f"{kind}: the bank laps' entries placed {np.median(place[:, gap_bins]):.0f} ms (median), {place[:, gap_bins].max():.0f} ms (max) "
              f"off inside the ruined stretch, {np.median(place[:, ~gap_bins]):.0f} ms elsewhere")
        report[f"{kind}: bank placement"] = {"gap_median_ms": float(np.median(place[:, gap_bins])), "gap_max_ms": float(place[:, gap_bins].max()),
                                            "elsewhere_median_ms": float(np.median(place[:, ~gap_bins]))}
        print(f"{kind}: the timing lap's own check masks {found[gap_bins].mean() * 100:.0f}% of the ruined bins and "
              f"{found[~gap_bins].mean() * 100:.1f}% of the rest (agreement median {np.median(agree[gap_bins]):.2f} ruined, "
              f"{np.median(agree[~gap_bins]):.2f} clean)")
        report[f"{kind}: junk mask"] = {"ruined_masked": float(found[gap_bins].mean()), "clean_masked": float(found[~gap_bins].mean())}
        score(f"{kind}: one lap (as today)", lambda: engine(timing), d)
        score(f"{kind}: bank, max", lambda: engine(timing, BankLiveDelta, bank=bank, mask=mask, mode="max"), d)
        score(f"{kind}: bank, mean", lambda: engine(timing, BankLiveDelta, bank=bank, mask=mask, mode="mean"), d)
        nomask = torch.ones_like(mask)
        score(f"{kind}: bank, max, no mask", lambda: engine(timing, BankLiveDelta, bank=bank, mask=nomask, mode="max"), d)

    out = Path("data/real-footage/rotax-night")
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if GAP_AT_S == 20.0 and CONTROLS else f"_gap{GAP_AT_S:g}"
    if not CONTROLS:
        (out / f"reference_bank_{run_name}{tag}.json").write_text(json.dumps(report, indent=2))
        return
    # control: a bank from another track (indoor-pov, three laps at their own pace) must stay unconfident
    os.environ["REAL_FOOTAGE"] = "indoor-pov"
    import experiments.real_footage as rf
    rf = importlib.reload(rf)
    fi, ti = rf.frames()
    with torch.no_grad():
        di = enc(fi)
    cr = list(rf.CROSSINGS_S)
    wrong = torch.stack([di[np.clip(np.round(np.interp(cr[j] + times * (cr[j + 1] - cr[j]) / T, ti, np.arange(len(ti)))).astype(int), 0, len(ti) - 1)]
                         for j in range(3)])
    wmask = torch.ones(3, n, dtype=torch.bool)
    for name, make in (("wrong track: one lap", lambda: engine(wrong[0])),
                       ("wrong track: bank, max", lambda: engine(wrong[0], BankLiveDelta, bank=wrong, mask=wmask, mode="max")),
                       ("wrong track: bank, mean", lambda: engine(wrong[0], BankLiveDelta, bank=wrong, mask=wmask, mode="mean"))):
        shown = np.concatenate([run(make(), d, t, c[k], c[k + 1])[:, 2] >= SHOWN for k in LIVE_LAPS])
        report[name] = {"shown": float(shown.mean())}
        print(f"  {name:<44} shown {100 * shown.mean():.1f}% of ticks (must stay near 0)", flush=True)
    (out / f"reference_bank_{run_name}{tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
