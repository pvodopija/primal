"""Wide renders of other sizes (batch b1007): each lap's own camera, turning a wandering
camera back to the car's heading, and the held-out-drive check on lap pieces."""
import json
from pathlib import Path

import numpy as np

from train import camera
from train.dataset import LapIndex, same_drive


def _write_lap(root: Path, name: str, t: np.ndarray, s: np.ndarray, size=(268, 144), fov=121.28, rig=None) -> dict:
    d = root / "laps" / name
    d.mkdir(parents=True)
    np.save(d / "t.npy", t.astype(np.float32))
    np.save(d / "s.npy", (s % 1.0).astype(np.float32))
    entry = {"lap_id": name, "track": "x", "n_frames": len(t), "track_length_m": 1000.0, "ref_bins": 500,
             "ref_spacing_m": 2.0, "fps": 60.0, "path": f"laps/{name}", "frame_size": list(size), "fov_h_deg": fov,
             "s_span": float(np.ptp(s))}
    if rig is not None:
        entry["rig"] = rig
    return entry


def _index(root: Path, entries: list[dict]) -> LapIndex:
    (root / "index.json").write_text(json.dumps({"schema": 1, "laps": entries}))
    return LapIndex.load(root, variants=None)


def test_each_wide_lap_gets_the_camera_that_rendered_it(tmp_path):
    t = np.arange(10) / 60
    old = _write_lap(tmp_path, "old", t, t / 100)  # packed_ac_v3_wide: no rig, rendered at 90 deg
    new = _write_lap(tmp_path, "new", t, t / 100, size=(320, 172), fov=129.47, rig={"fov_deg": 100.0})
    laps = {lap.lap_id: lap for lap in _index(tmp_path, [old, new]).laps}
    assert laps["old"].source_view == camera.WIDE
    view = laps["new"].source_view
    assert (view.width, view.height) == (320, 172)
    # same pixels per degree as the old renders, the horizon where AC's 100 deg puts it
    assert abs(view.focal - camera.WIDE.focal) < 0.5
    assert abs(view.cy - view.focal * np.tan(np.radians(50.0))) < 1e-6


def test_frame_poses_add_a_pose_per_frame():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (3, 144, 268, 3), dtype=np.uint8)
    times = np.arange(3) / 60
    plain = camera.render_clip(frames, times, rng, pose=(2.0, 1.0, 0.0))
    zeros = camera.render_clip(frames, times, rng, pose=(2.0, 1.0, 0.0), frame_poses=np.zeros((3, 3)))
    assert np.array_equal(plain, zeros)
    yaw = np.array([-5.0, 0.0, 7.0])
    turned = camera.render_clip(frames, times, rng, pose=(2.0, 1.0, 0.0), frame_poses=np.stack([yaw, 0 * yaw, 0 * yaw], 1))
    for i in range(3):
        assert np.array_equal(turned[i], camera.render(frames[i], camera.WIDE, camera.TRAINING, 2.0 + yaw[i], 1.0, 0.0))


def test_a_piece_of_a_held_out_drive_is_recognised(tmp_path):
    t = np.arange(0, 60, 1 / 60)
    rng = np.random.default_rng(1)
    pace = np.cumsum(30 + 5 * np.sin(t / 3) + rng.normal(0, 0.5, t.size)) / 60  # metres along the lap
    whole = _write_lap(tmp_path, "whole", t, pace / pace[-1] * 0.999)
    s_whole = pace / pace[-1] * 0.999
    # a re-render split by a capture drop: a later stretch of the same drive, on its own clock
    piece = _write_lap(tmp_path, "piece", t[1200:2400] - t[1200] + 7.0, s_whole[1200:2400])
    # another drive: the same place, a different pace through it
    other_pace = np.cumsum(30 + 5 * np.sin(t / 3 + 1.5)) / 60
    other = _write_lap(tmp_path, "other", t[1200:2400], (other_pace / other_pace[-1] * 0.999)[1200:2400])
    short = _write_lap(tmp_path, "short", t[1200:1300], s_whole[1200:1300])
    laps = {lap.lap_id: lap for lap in _index(tmp_path, [whole, piece, other, short]).laps}
    assert same_drive(laps["piece"], laps["whole"])
    assert same_drive(laps["whole"], laps["piece"])
    assert not same_drive(laps["other"], laps["whole"])
    assert not same_drive(laps["short"], laps["whole"])  # under 5 s: too short to tell
