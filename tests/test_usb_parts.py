from pathlib import Path

from capture.usb_parts import fill_in_order


def test_fill_in_order_keeps_lap_order_and_the_cap():
    units = [(Path("set"), [f"laps/l{i}"], size) for i, size in enumerate([9, 9, 5, 9, 2, 9])]
    parts = fill_in_order(units, cap=20)
    assert [total for total, _ in parts] == [18, 16, 9]
    flat = [unit[1][0] for _, members in parts for unit in members]
    assert flat == [f"laps/l{i}" for i in range(6)]
