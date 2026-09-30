"""
Stop the OBS recording by itself after a number of laps.

    python -m capture.autostop --engage-bot          # hand the car to AC's AI and see it drive off
    python -m capture.autostop --start --laps 10     # record 10 completed laps

Watches the timecode app's per-frame log for the running AC launch and, once
AC's lap counter has gone up by `--laps`, presses OBS's Stop Recording hotkey
(F14, set in OBS under Settings > Hotkeys). `--start` presses Start Recording
(F13) first. F13 and F14 exist to Windows but not on keyboards, so nothing else
reacts to them: an earlier Ctrl+Alt+Shift+F9/F10 pair also toggled NVIDIA's
Instant Replay, which draws an icon into the game and loads the GPU. Keys go to
whichever window has focus, so they are only sent while AC or OBS is in front.
Other windows take focus mid-session (the Claude app when it shows a message,
NVIDIA's overlay); AC is then brought back to the front first. OBS polls its
hotkeys, so each is held briefly; a tap is missed while AC has focus.

The recording is also stopped if AC stops logging for `--stall-s` (paused,
crashed), if the car stays under 5 km/h for `--parked-s` (once, Ctrl+C had not
reached AC and a whole recording showed a parked car), or after `--max-min`.
`--engage-bot` sends Ctrl+C itself, with scan codes and AC in front, and waits
for the car to move.
"""

from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes
from pathlib import Path

import pandas as pd

from capture.install_overlay import find_ac_root

VK_F13, VK_F14, VK_CONTROL, VK_C = 0x7C, 0x7D, 0x11, 0x43
KEYEVENTF_KEYUP = 0x2
ALLOWED = {"acs.exe", "obs64.exe"}
# Other windows take focus mid-session: the Claude app whenever it shows a message (with Windows'
# text-input host behind its chat box), NVIDIA's overlay. AC is brought back to the front instead
# of sending a key to whatever holds focus.

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def foreground_exe() -> str:
    return window_exe(user32.GetForegroundWindow())


def window_exe(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return Path(buf.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)


def press(key: int, hold_s: float = 0.4, wait_s: float = 60.0) -> bool:
    """Hold `key` once AC or OBS has focus; False if neither does within wait_s."""
    deadline = time.time() + wait_s
    while (exe := foreground_exe()) not in ALLOWED:
        if time.time() > deadline:
            print(f"not sending: {exe or 'unknown'} has focus", flush=True)
            return False
        ac = user32.FindWindowW(None, "Assetto Corsa")
        if ac and window_exe(ac) == "acs.exe":
            user32.ShowWindow(ac, 9)  # SW_RESTORE
            user32.SetForegroundWindow(ac)
        time.sleep(0.5 if ac else 2)
    user32.keybd_event(key, 0, 0, 0)
    time.sleep(hold_s)
    user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
    return True


def focus_ac() -> bool:
    ac = user32.FindWindowW(None, "Assetto Corsa")
    if not ac or window_exe(ac) != "acs.exe":
        return False
    user32.ShowWindow(ac, 9)  # SW_RESTORE
    user32.SetForegroundWindow(ac)
    time.sleep(0.5)
    return foreground_exe() == "acs.exe"


def engage_bot(tries: int = 2) -> bool:
    """
    Hand the car to AC's AI with Ctrl+C and confirm from the telemetry that it
    drives off. AC takes the keys with their scan codes and only while it has
    focus, so both are ensured here rather than left to whoever sent the key.
    The timecode app logs only once the car is on track, so a stale log means
    AC is still in its session menu, and no key is sent.
    """
    launch = newest_launch()
    if time.time() - last_row(launch)[0] > 10:
        print("not sending Ctrl+C: no fresh telemetry, so AC is not driving yet (session menu?)", flush=True)
        return False
    for _ in range(tries):
        if not focus_ac():
            print(f"not sending Ctrl+C: {foreground_exe() or 'unknown'} has focus", flush=True)
            return False
        for vk in (VK_CONTROL, VK_C):
            user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), 0, 0)
        time.sleep(0.3)
        for vk in (VK_C, VK_CONTROL):
            user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0), KEYEVENTF_KEYUP, 0)
        for _ in range(8):
            time.sleep(2)
            if last_row(launch)[2] >= 5:
                return True
    return False


def newest_launch() -> Path:
    root = find_ac_root()
    if root is None:
        raise SystemExit("Assetto Corsa not found through Steam; is it installed?")
    base = root / "apps" / "lua" / "locamotif_timecode" / "frame_log"
    return max((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)


def last_row(launch: Path) -> tuple[float, int, float]:
    """Modification time of the newest chunk, and the lap count and top speed in it."""
    chunk = max(launch.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    rows = pd.read_csv(chunk, usecols=["lap_count", "speed_kmh"])
    return chunk.stat().st_mtime, int(rows.lap_count.iloc[-1]), float(rows.speed_kmh.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laps", type=int, default=0, help="completed laps to record from now")
    parser.add_argument("--engage-bot", action="store_true",
                        help="first hand the car to AC's AI (Ctrl+C) and wait until it drives off")
    parser.add_argument("--start", action="store_true", help="start the recording first")
    parser.add_argument("--stall-s", type=float, default=30.0)
    parser.add_argument("--max-min", type=float, default=60.0, help="stop after this long whatever the lap count")
    parser.add_argument("--parked-s", type=float, default=60.0, help="stop if the car stays under 5 km/h this long")
    args = parser.parse_args()
    began = time.time()

    launch = newest_launch()
    if args.engage_bot:
        if not engage_bot():
            raise SystemExit("the car did not drive off after Ctrl+C")
        print(f"{time.strftime('%H:%M:%S')} bot driving", flush=True)
        if not args.laps:
            return
    if args.laps < 1:
        # 0 is only for --engage-bot alone; recording 0 laps would stop at once.
        raise SystemExit("--laps must be at least 1 to record")
    _, start_lap, _ = last_row(launch)
    if args.start and not press(VK_F13):
        raise SystemExit(1)
    print(f"{time.strftime('%H:%M:%S')} recording; lap count {start_lap}, stopping at {start_lap + args.laps}", flush=True)
    reason = ""
    moved = time.time()
    while not reason:
        time.sleep(2)
        mtime, lap, speed = last_row(launch)
        if speed >= 5:
            moved = time.time()
        if lap >= start_lap + args.laps:
            reason = f"lap count {lap}"
        elif time.time() - mtime > args.stall_s:
            reason = f"STALLED: no telemetry for {time.time() - mtime:.0f} s at lap count {lap}"
        elif time.time() - moved > args.parked_s:
            # Once the bot failed to take over, and the recording was 30 minutes of a parked car.
            reason = f"PARKED: under 5 km/h for {time.time() - moved:.0f} s at lap count {lap}"
        elif time.time() - began > args.max_min * 60:
            reason = f"TIMEOUT after {args.max_min:.0f} min at lap count {lap}"
    ok = press(VK_F14)
    print(f"{time.strftime('%H:%M:%S')} {reason}; stop {'sent' if ok else 'NOT sent'}", flush=True)
    raise SystemExit(0 if ok and reason.startswith("lap count") else 1)


if __name__ == "__main__":
    main()
