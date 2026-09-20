"""
Assetto Corsa shared-memory reader, Windows only.

This is auxiliary metadata, not labels. Training labels come from the on-screen
timecode grid, which is immune to capture latency; this reader supplies the
things the overlay does not carry (track name, spline length, car model, speed,
world coordinates) for session bookkeeping and analysis.

The graphics page layout changed across AC versions: `carCoordinates` was a
single vec3 in early 1.x and became `activeCars` + `carCoordinates[60][3]` +
`carID[60]` later. Everything up to and including `normalizedCarPosition` has
been stable, so that prefix is read as one struct and the newer tail is read
separately and sanity-checked before being trusted.
"""

from __future__ import annotations

import ctypes
import mmap
import sys
from dataclasses import dataclass, field
from typing import Optional

PHYSICS_NAME = "Local\\acpmf_physics"
GRAPHICS_NAMES = ("Local\\acpmf_graphics", "Local\\acpmf_graphic")
STATIC_NAMES = ("Local\\acpmf_static", "Local\\acpmf_statics")

STATUS_OFF, STATUS_REPLAY, STATUS_LIVE, STATUS_PAUSE = 0, 1, 2, 3
MAX_CARS = 60


class PhysicsPage(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packet_id", ctypes.c_int32),
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("fuel", ctypes.c_float),
        ("gear", ctypes.c_int32),
        ("rpms", ctypes.c_int32),
        ("steer_angle", ctypes.c_float),
        ("speed_kmh", ctypes.c_float),
        ("velocity", ctypes.c_float * 3),
        ("acc_g", ctypes.c_float * 3),
        ("wheel_slip", ctypes.c_float * 4),
        ("wheel_load", ctypes.c_float * 4),
        ("wheels_pressure", ctypes.c_float * 4),
        ("wheel_angular_speed", ctypes.c_float * 4),
        ("tyre_wear", ctypes.c_float * 4),
        ("tyre_dirty_level", ctypes.c_float * 4),
        ("tyre_core_temperature", ctypes.c_float * 4),
        ("camber_rad", ctypes.c_float * 4),
        ("suspension_travel", ctypes.c_float * 4),
        ("drs", ctypes.c_float),
        ("tc", ctypes.c_float),
        ("heading", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("cg_height", ctypes.c_float),
        ("car_damage", ctypes.c_float * 5),
        ("number_of_tyres_out", ctypes.c_int32),
        ("pit_limiter_on", ctypes.c_int32),
        ("abs", ctypes.c_float),
        ("kers_charge", ctypes.c_float),
        ("kers_input", ctypes.c_float),
        ("auto_shifter_on", ctypes.c_int32),
        ("ride_height", ctypes.c_float * 2),
        ("turbo_boost", ctypes.c_float),
        ("ballast", ctypes.c_float),
        ("air_density", ctypes.c_float),
        ("air_temp", ctypes.c_float),
        ("road_temp", ctypes.c_float),
        ("local_angular_velocity", ctypes.c_float * 3),
        ("final_ff", ctypes.c_float),
        ("performance_meter", ctypes.c_float),
        ("engine_brake", ctypes.c_int32),
        ("ers_recovery_level", ctypes.c_int32),
        ("ers_power_level", ctypes.c_int32),
        ("ers_heat_charging", ctypes.c_int32),
        ("ers_is_charging", ctypes.c_int32),
        ("kers_current_kj", ctypes.c_float),
        ("drs_available", ctypes.c_int32),
        ("drs_enabled", ctypes.c_int32),
        ("brake_temp", ctypes.c_float * 4),
        ("clutch", ctypes.c_float),
        ("tyre_temp_i", ctypes.c_float * 4),
        ("tyre_temp_m", ctypes.c_float * 4),
        ("tyre_temp_o", ctypes.c_float * 4),
        ("is_ai_controlled", ctypes.c_int32),
        ("tyre_contact_point", (ctypes.c_float * 3) * 4),
        ("tyre_contact_normal", (ctypes.c_float * 3) * 4),
        ("tyre_contact_heading", (ctypes.c_float * 3) * 4),
        ("brake_bias", ctypes.c_float),
        ("local_velocity", ctypes.c_float * 3),
    ]


class GraphicsPage(ctypes.Structure):
    """Stable prefix, through `normalizedCarPosition`."""

    _pack_ = 4
    _fields_ = [
        ("packet_id", ctypes.c_int32),
        ("status", ctypes.c_int32),
        ("session", ctypes.c_int32),
        ("current_time", ctypes.c_wchar * 15),
        ("last_time", ctypes.c_wchar * 15),
        ("best_time", ctypes.c_wchar * 15),
        ("split", ctypes.c_wchar * 15),
        ("completed_laps", ctypes.c_int32),
        ("position", ctypes.c_int32),
        ("i_current_time", ctypes.c_int32),
        ("i_last_time", ctypes.c_int32),
        ("i_best_time", ctypes.c_int32),
        ("session_time_left", ctypes.c_float),
        ("distance_traveled", ctypes.c_float),
        ("is_in_pit", ctypes.c_int32),
        ("current_sector_index", ctypes.c_int32),
        ("last_sector_time", ctypes.c_int32),
        ("number_of_laps", ctypes.c_int32),
        ("tyre_compound", ctypes.c_wchar * 33),
        ("replay_time_multiplier", ctypes.c_float),
        ("normalized_car_position", ctypes.c_float),
    ]


class GraphicsTail(ctypes.Structure):
    """Post-1.16 continuation. Trusted only if `active_cars` looks sane."""

    _pack_ = 4
    _fields_ = [
        ("active_cars", ctypes.c_int32),
        ("car_coordinates", (ctypes.c_float * 3) * MAX_CARS),
        ("car_id", ctypes.c_int32 * MAX_CARS),
        ("penalty_time", ctypes.c_float),
        ("flag", ctypes.c_int32),
        ("ideal_line_on", ctypes.c_int32),
        ("is_in_pit_lane", ctypes.c_int32),
        ("surface_grip", ctypes.c_float),
    ]


class StaticPage(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("sm_version", ctypes.c_wchar * 15),
        ("ac_version", ctypes.c_wchar * 15),
        ("number_of_sessions", ctypes.c_int32),
        ("num_cars", ctypes.c_int32),
        ("car_model", ctypes.c_wchar * 33),
        ("track", ctypes.c_wchar * 33),
        ("player_name", ctypes.c_wchar * 33),
        ("player_surname", ctypes.c_wchar * 33),
        ("player_nick", ctypes.c_wchar * 33),
        ("sector_count", ctypes.c_int32),
        ("max_torque", ctypes.c_float),
        ("max_power", ctypes.c_float),
        ("max_rpm", ctypes.c_int32),
        ("max_fuel", ctypes.c_float),
        ("suspension_max_travel", ctypes.c_float * 4),
        ("tyre_radius", ctypes.c_float * 4),
        ("max_turbo_boost", ctypes.c_float),
        ("deprecated_1", ctypes.c_float),
        ("deprecated_2", ctypes.c_float),
        ("penalties_enabled", ctypes.c_int32),
        ("aid_fuel_rate", ctypes.c_float),
        ("aid_tyre_rate", ctypes.c_float),
        ("aid_mechanical_damage", ctypes.c_float),
        ("aid_allow_tyre_blankets", ctypes.c_int32),
        ("aid_stability", ctypes.c_float),
        ("aid_auto_clutch", ctypes.c_int32),
        ("aid_auto_blip", ctypes.c_int32),
        ("has_drs", ctypes.c_int32),
        ("has_ers", ctypes.c_int32),
        ("has_kers", ctypes.c_int32),
        ("kers_max_j", ctypes.c_float),
        ("engine_brake_settings_count", ctypes.c_int32),
        ("ers_power_controller_count", ctypes.c_int32),
        ("track_spline_length", ctypes.c_float),
        ("track_configuration", ctypes.c_wchar * 33),
    ]


@dataclass
class AcSnapshot:
    """One consistent-enough read of the pages we care about."""

    status: int
    track: str
    track_config: str
    track_length_m: float
    car_model: str
    ac_version: str
    completed_laps: int
    spline_pos: float
    speed_kmh: float
    heading: float
    pitch: float
    roll: float
    is_in_pit: bool
    is_ai_controlled: bool
    replay_time_multiplier: float
    world_pos: Optional[tuple[float, float, float]] = None
    aids: dict = field(default_factory=dict)

    @property
    def status_name(self) -> str:
        return {
            STATUS_OFF: "off",
            STATUS_REPLAY: "replay",
            STATUS_LIVE: "live",
            STATUS_PAUSE: "pause",
        }.get(self.status, f"unknown({self.status})")


def _open(names: tuple[str, ...] | str, size: int):
    if sys.platform != "win32":
        return None
    candidates = (names,) if isinstance(names, str) else names
    for name in candidates:
        try:
            return mmap.mmap(-1, size, tagname=name)
        except OSError:
            continue
    return None


class AcSharedMemory:
    """
    Best-effort reader. `snapshot()` returns None when AC or the shared-memory
    plugin is not running.
    """

    def __init__(self) -> None:
        self._buffers: list[mmap.mmap] = []
        self._physics: Optional[PhysicsPage] = None
        self._graphics: Optional[GraphicsPage] = None
        self._graphics_tail: Optional[GraphicsTail] = None
        self._static: Optional[StaticPage] = None

    @property
    def connected(self) -> bool:
        return self._graphics is not None

    def try_connect(self) -> bool:
        if self._graphics is not None:
            return True

        graphics_size = ctypes.sizeof(GraphicsPage) + ctypes.sizeof(GraphicsTail)
        buf = _open(GRAPHICS_NAMES, graphics_size)
        if buf is None:
            return False
        self._buffers.append(buf)
        self._graphics = GraphicsPage.from_buffer(buf)
        tail = GraphicsTail.from_buffer(buf, ctypes.sizeof(GraphicsPage))
        self._graphics_tail = tail if 0 <= tail.active_cars <= MAX_CARS else None

        buf = _open(PHYSICS_NAME, ctypes.sizeof(PhysicsPage))
        if buf is not None:
            self._buffers.append(buf)
            self._physics = PhysicsPage.from_buffer(buf)

        buf = _open(STATIC_NAMES, ctypes.sizeof(StaticPage))
        if buf is not None:
            self._buffers.append(buf)
            page = StaticPage.from_buffer(buf)
            self._static = page if 0.0 < page.track_spline_length < 1e5 else None
        return True

    def close(self) -> None:
        self._physics = None
        self._graphics = None
        self._graphics_tail = None
        self._static = None
        for buf in self._buffers:
            try:
                buf.close()
            except (BufferError, OSError):
                pass
        self._buffers.clear()

    def snapshot(self) -> Optional[AcSnapshot]:
        if not self.try_connect():
            return None
        graphics = self._graphics
        assert graphics is not None
        physics, static = self._physics, self._static

        spline = float(graphics.normalized_car_position)
        spline -= float(int(spline)) if spline >= 1.0 or spline < 0.0 else 0.0

        world = None
        if self._graphics_tail is not None:
            xyz = self._graphics_tail.car_coordinates[0]
            world = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

        return AcSnapshot(
            status=int(graphics.status),
            track=str(static.track) if static else "",
            track_config=str(static.track_configuration) if static else "",
            track_length_m=float(static.track_spline_length) if static else 0.0,
            car_model=str(static.car_model) if static else "",
            ac_version=str(static.ac_version) if static else "",
            completed_laps=int(graphics.completed_laps),
            spline_pos=spline,
            speed_kmh=float(physics.speed_kmh) if physics else 0.0,
            heading=float(physics.heading) if physics else 0.0,
            pitch=float(physics.pitch) if physics else 0.0,
            roll=float(physics.roll) if physics else 0.0,
            is_in_pit=bool(graphics.is_in_pit),
            is_ai_controlled=bool(physics.is_ai_controlled) if physics else False,
            replay_time_multiplier=float(graphics.replay_time_multiplier),
            world_pos=world,
            aids={
                "stability": float(static.aid_stability) if static else 0.0,
                "auto_clutch": bool(static.aid_auto_clutch) if static else False,
            },
        )


def main() -> None:
    """`python -m capture.ac_shm` prints a live one-line status, for checking setup."""
    import time

    shm = AcSharedMemory()
    try:
        while True:
            snap = shm.snapshot()
            if snap is None:
                print("waiting for AC shared memory...", end="\r")
            else:
                print(
                    f"{snap.status_name:>6}  {snap.track}/{snap.track_config or '-'}  "
                    f"{snap.track_length_m:7.1f} m  {snap.car_model:<24} "
                    f"lap {snap.completed_laps:>2}  s={snap.spline_pos:.5f}  "
                    f"{snap.speed_kmh:6.1f} km/h  ai={int(snap.is_ai_controlled)}   ",
                    end="\r",
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        shm.close()


if __name__ == "__main__":
    main()
