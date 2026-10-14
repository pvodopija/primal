"""
A real session replayed as the app runs it (train/session.py): START before the first line
crossing, the reference lap recorded and closed where the camera sees its start again, every
later lap timed against it live, a faster lap PRIMAL followed well becoming the reference. The
frames are a real-footage set's (experiments/real_footage.py: crop, masks) at 30 a second.
Per model: the lap times against a transponder's (the set's lap_times), when the reference
swapped, and each lap as two coloured bars along its progress through the reference:
  ahead or behind: the delta itself (green ahead of the reference, red behind, stronger further);
  gaining or losing: how fast the delta changes there (green taking time out of the reference,
                     red giving it away), over the last second;
grey where the delta was not shown (confidence under the threshold). START at the first
crossing; a hand close at the true line if PRIMAL has not closed the reference 15 s after it.
Also every lap against the session's fastest (the review after a session), the transponder's
delta at the line beside PRIMAL's. Writes ROOT/session_<run>.json, ROOT/session_bars_<run>.png
and ROOT/best_lap_bars_<run>.png.

Usage: REAL_FOOTAGE=night1 PYTHONPATH=. python experiments/session_bars.py RUN [RUN ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from experiments.real_footage import _S, CROSSINGS_S, ROOT, SUFFIX, frames
from train.dataset import _to_chw
from train.eval import load_model
from train.session import LapCompleted, Reading, ReferenceClosed, Session
from train.train import default_device

FPS = 30.0
# START at the first crossing (the frames before are the pit-exit lane, a view no later lap
# passes through); when PRIMAL has not closed the reference HAND_AFTER_S after its true end,
# the driver closes it by hand at the line, as the app lets them
HAND_AFTER_S = 15.0
DELTA_FULL_S, RATE_FULL = 0.5, 0.08  # colour saturates at this delta (s) / rate (s of delta per s of lap)
BARS = LinearSegmentedColormap.from_list("gain", ["#1a9850", "#f7f7f7", "#d73027"])  # ahead / gaining green, behind / losing red


def replay(run: str, f: np.ndarray, t: np.ndarray, device) -> dict:
    model, payload = load_model(Path("runs") / run / "best.pt", device)
    with torch.no_grad():
        d = torch.cat([model.encoder(torch.from_numpy(_to_chw(f[i:i + 256])).to(device)) for i in range(0, len(f), 256)]).cpu()
    cpu = torch.device("cpu")  # the session's engine on the CPU, the descriptors with it
    session = Session(model.to(cpu), int(payload["args"]["clip_len"]), cpu)
    session.start(float(t[0]))
    events = []
    for k in range(len(t)):
        if session.state == "reference" and t[k] > CROSSINGS_S[1] + HAND_AFTER_S:
            events += session.close_at(float(CROSSINGS_S[1]))
        events += session.push(d[k], float(t[k]))
    closed = next((e for e in events if isinstance(e, ReferenceClosed)), None)
    laps = [e for e in events if isinstance(e, LapCompleted)]
    readings = [e for e in events if isinstance(e, Reading)]
    return {"closed": closed, "laps": laps, "readings": readings}


# sectors per lap, ~4 s each on a 48 s lap: the delta jitters ±50 ms, so finer sectors show noise;
# full colour at this gain or loss across one
SECTORS, SECTOR_FULL_S = 12, 0.10


def draw(rows: list[dict], title: str, path: Path) -> None:
    """Each row a lap: frac (progress through the reference, 0-1), delta (s), shown, label, right."""
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * n + 1.6), sharey=True)
    grid = np.linspace(0, 1, 400)
    edges = np.linspace(0, 1, SECTORS + 1)
    for i, row in enumerate(rows):
        order = np.argsort(row["frac"])
        frac, delta, shown = row["frac"][order], row["delta"][order], row["shown"][order].astype(float)
        smooth = np.array([np.median(delta[max(0, k - 3):k + 4]) for k in range(delta.size)])  # half a second
        # ahead or behind, all along the lap
        v = np.interp(grid, frac, smooth)
        colours = BARS(np.clip(v / DELTA_FULL_S, -1, 1) * 0.5 + 0.5)
        colours[np.interp(grid, frac, shown) < 0.5] = (0.75, 0.75, 0.75, 1.0)
        axes[0].imshow(colours[None], extent=(0, 100, i - 0.38, i + 0.38), aspect="auto", interpolation="nearest")
        # gaining or losing, per mini-sector: the delta's change across it
        at = np.interp(edges, frac, smooth)
        gain = np.diff(at)
        seen = np.array([np.mean(shown[(frac >= a) & (frac < b)]) if np.any((frac >= a) & (frac < b)) else 0
                         for a, b in zip(edges[:-1], edges[1:])])
        colours = BARS(np.clip(gain / SECTOR_FULL_S, -1, 1) * 0.5 + 0.5)
        colours[seen < 0.5] = (0.75, 0.75, 0.75, 1.0)
        axes[1].imshow(colours[None], extent=(0, 100, i - 0.38, i + 0.38), aspect="auto", interpolation="nearest")
        axes[0].text(-1, i, row["label"], ha="right", va="center", fontsize=9)
        axes[1].text(101, i, row["right"], ha="left", va="center", fontsize=9)
    for ax, sub in ((axes[0], f"ahead (green) or behind (red)\nfull colour at ±{DELTA_FULL_S} s"),
                    (axes[1], f"time gained (green) or lost (red) per sector ({SECTORS} a lap)\nfull colour at ±{SECTOR_FULL_S} s")):
        ax.set_xlim(0, 100); ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([])
        ax.set_xlabel("progress through the reference lap, %"); ax.set_title(sub, fontsize=10)
    fig.suptitle(title + "  (grey: delta not shown)", fontsize=11)
    fig.tight_layout(rect=(0.09, 0, 0.88, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def session_rows(out: dict) -> list[dict]:
    """The live session's laps against the reference the app held at the time."""
    rows, reference_s = [], out["closed"].lap_s
    for info in out["laps"]:
        r = [x for x in out["readings"] if x.lap == info.lap]
        if len(r) < 10:
            continue
        delta = np.array([x.delta_s for x in r])
        rows.append({"frac": np.array([x.ref_time_s for x in r]) / reference_s, "delta": delta,
                     "shown": np.array([x.shown for x in r]),
                     "label": f"lap {info.lap + 2}  {info.lap_s:6.3f} s" + ("  * new ref" if info.became_reference else ""),
                     "right": f"{delta[-1]:+.2f} s at the line"})
        reference_s = info.lap_s if info.became_reference else reference_s
    return rows


def best_lap_rows(run: str) -> list[dict]:
    """Every lap against the session's fastest (the review after a session), with the transponder's
    delta at the line beside PRIMAL's."""
    from experiments.real_demo import CONF_MIN, lap_series, setup
    import cv2
    from experiments.real_footage import VIDEO
    S = setup(run)
    fps = cv2.VideoCapture(str(VIDEO)).get(cv2.CAP_PROP_FPS)
    truth = _S.get("lap_times")
    rows = []
    for lap_no in S.live:
        L = lap_series(S, lap_no, fps)
        line = f"  (transponder {truth[lap_no - 1] - truth[S.r]:+.2f})" if truth else ""
        at_line = float(np.median(L.delta[-8:]))
        if truth:
            print(f"  lap {lap_no:2d} against lap {S.r + 1}: PRIMAL {at_line:+.3f} s at the line, transponder "
                  f"{truth[lap_no - 1] - truth[S.r]:+.3f} s, error {1000 * (at_line - (truth[lap_no - 1] - truth[S.r])):+4.0f} ms")
        rows.append({"frac": L.ref_t / S.T, "delta": L.delta, "shown": L.conf >= CONF_MIN, "at_line": at_line,
                     "truth_at_line": truth[lap_no - 1] - truth[S.r] if truth else None,
                     "label": f"lap {lap_no}  {L.b - L.a:6.3f} s", "right": f"{np.median(L.delta[-8:]):+.2f} s at the line{line}"})
    return rows, S.r + 1


def main() -> None:
    device = torch.device(default_device())
    f, t = frames()
    step = max(int(round((1 / FPS) / np.median(np.diff(t)))), 1)
    keep = np.flatnonzero(t >= CROSSINGS_S[0])[::step]
    f, t = f[keep], t[keep]
    truth = _S.get("lap_times")
    for run in sys.argv[1:]:
        out = replay(run, f, t, device)
        c = out["closed"]
        print(f"\n{run}: reference closed by {c.placed_by if c else '-'} at {c.line_s if c else float('nan'):.2f} s, "
              f"reference lap {c.lap_s if c else float('nan'):.3f} s (transponder lap 1: {truth[0] if truth else '-'})")
        rows = []
        for lap in out["laps"]:
            k = lap.lap + 1  # the reference is transponder lap 1; live lap 0 is lap 2
            tr = truth[k] if truth and k < len(truth) else None
            err = (lap.lap_s - tr) * 1000 if tr else None
            rows.append({"lap": k + 1, "primal_s": lap.lap_s, "transponder_s": tr, "error_ms": err,
                         "shown_share": lap.shown_share, "became_reference": lap.became_reference, "voided": lap.voided})
            print(f"  lap {k + 1:2d}: PRIMAL {lap.lap_s:7.3f} s, transponder {tr if tr else '-':>7}, "
                  f"error {err:+6.0f} ms, delta shown {100 * lap.shown_share:3.0f}%" + ("  -> new reference" if lap.became_reference else ""))
        errs = np.array([r["error_ms"] for r in rows if r["error_ms"] is not None])
        if errs.size:
            print(f"  lap times: median |error| {np.median(np.abs(errs)):.0f} ms, max {np.abs(errs).max():.0f} ms over {errs.size} laps")
        (ROOT / f"session_{run}{SUFFIX}.json").write_text(json.dumps(
            {"reference": {"line_s": c.line_s, "lap_s": c.lap_s, "placed_by": c.placed_by} if c else None, "laps": rows,
             "readings": [{"t": r.t, "lap": r.lap, "ref_time_s": r.ref_time_s, "delta_s": r.delta_s, "confidence": r.confidence,
                           "shown": r.shown} for r in out["readings"]]}, indent=1))
        if c is not None:
            draw(session_rows(out), f"{run}: each lap against the reference the app held at the time",
                 ROOT / f"session_bars_{run}{SUFFIX}.png")
        rows, best = best_lap_rows(run)
        e = np.array([r["at_line"] - r["truth_at_line"] for r in rows if r["truth_at_line"] is not None]) * 1000
        if e.size:
            print(f"  delta at the line against the fastest lap: median |error| {np.median(np.abs(e)):.0f} ms, max {np.abs(e).max():.0f} ms")
        draw(rows, f"{run}: every lap against the fastest, lap {best}", ROOT / f"best_lap_bars_{run}{SUFFIX}.png")


if __name__ == "__main__":
    main()
