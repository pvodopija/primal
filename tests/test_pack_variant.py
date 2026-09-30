import pandas as pd

from capture.pack import rig_variant


def _log(tmp_path, **columns):
    path = tmp_path / "rig_log.csv"
    base = {"car_spline": [0.1, 0.2, 0.3], "applied_lateral_m": [0.0, 0.5, 1.0]}
    pd.DataFrame({**base, **columns}).to_csv(path, index=False)
    return path


def test_an_older_rig_log_without_head_turns_is_a_live_drive(tmp_path):
    assert rig_variant(_log(tmp_path)) == "live"


def test_look_switched_off_is_a_live_drive(tmp_path):
    assert rig_variant(_log(tmp_path, look_gain=[0.0, 0.0, 0.0])) == "live"


def test_a_render_that_turns_the_head_is_a_look_rerender(tmp_path):
    assert rig_variant(_log(tmp_path, look_gain=[0.0, 0.55, 0.55])) == "look"
