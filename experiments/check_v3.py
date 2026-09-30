"""Integrity and shape of packed_ac_v3: paths, array lengths, v2 laps unchanged, per-track summary."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

root = Path("data/packed_ac_v3")
ix = json.loads((root / "index.json").read_text())
laps = ix["laps"]
v2 = {l["lap_id"]: l for l in json.loads(Path("data/packed_ac_v2/index.json").read_text())["laps"]}
print("index keys:", [k for k in ix if k != "laps"], {k: ix[k] for k in ix if k != "laps" and not isinstance(ix[k], (list, dict))})
problems = []
for l in laps:
    d = root / l["path"] if not Path(l["path"]).is_absolute() else Path(l["path"])
    if not d.exists():
        d = root / "laps" / l["lap_id"]
    if not d.exists():
        problems.append(("missing", l["lap_id"])); continue
    f = np.load(d / "frames.npy", mmap_mode="r"); s = np.load(d / "s.npy"); t = np.load(d / "t.npy")
    if not (f.shape[0] == s.size == t.size == l["n_frames"]):
        problems.append(("length", l["lap_id"], f.shape, s.size, t.size, l["n_frames"]))
    if tuple(f.shape[1:]) != (80, 148, 3) or f.dtype != np.uint8:
        problems.append(("frame", l["lap_id"], f.shape, f.dtype))
    if not np.all(np.diff(t) > 0):
        problems.append(("time", l["lap_id"]))
    if l["lap_id"] in v2:
        old = v2[l["lap_id"]]
        diff = {k for k in old if k != "path" and old[k] != l.get(k)}
        if diff:
            problems.append(("v2 changed", l["lap_id"], sorted(diff)))
print("laps", len(laps), "new", sum(l["lap_id"] not in v2 for l in laps), "v2 kept", sum(l["lap_id"] in v2 for l in laps), "of", len(v2))
print("problems", len(problems)); [print(" ", p) for p in problems[:15]]
by = defaultdict(list)
for l in laps:
    by[(l["track"] + "__" + l["track_config"]) if l["track_config"] else l["track"]].append(l)
print(f"\n{'track':36} {'split':8} {'sess':>4} {'laps':>5} {'full':>5} {'frames':>8}  cars / fov / re-render sessions")
for k, g in sorted(by.items()):
    sessions = sorted({l["session_id"] for l in g})
    rerender = [s for s in sessions if "T20" in s.split("__")[-1][8:11] or s.split("__")[-1].startswith("20260929T2")]
    print(f"{k[:36]:36} {g[0]['split']:8} {len(sessions):4d} {len(g):5d} {sum(l['s_span'] > 0.98 for l in g):5d} {sum(l['n_frames'] for l in g):8d}  "
          f"{sorted({l['car_model'] for l in g})} {sorted({round(l['fov_h_deg'], 1) for l in g})} rerender={len(rerender)}")
