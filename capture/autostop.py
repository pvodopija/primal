"""
Stop the OBS recording by itself after a number of laps.

    python -m capture.autostop --laps 10

Watches the timecode app's per-frame log for the running AC launch and, once
AC's lap counter has gone up by `--laps`, presses OBS's Stop Recording hotkey
(F14, set in OBS under Settings > Hotkeys). `--start` presses Start Recording
(F13) first. F13 and F14 exist to Windows but not on keyboards, so nothing else
reacts to them: an earlier Ctrl+Alt+Shift+F9/F10 pair also toggled NVIDIA's
Instant Replay, which draws an icon into the game and loads the GPU. Keys go to
whichever window has focus, so they are only sent while AC or OBS is in front.
The Claude app takes focus whenever it shows a message; when it has, AC is
brought back to the front first. If AC stops logging for `--stall-s` (paused,
crashed) or `--max-min` passes, the recording is stopped too. OBS polls its
hotkeys, so each is held briefly; a tap is missed while AC has focus.

Save the replay before AC's Hotlap session runs out (30 minutes): once it has,
ESC no longer opens the menu, and closing AC hung without writing the autosave.
"""

from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes
from pathlib import Path

import pandas as pd

from capture.install_overlay import find_ac_root

VK_F13, VK_F14 = 0x7C, 0x7D
KEYEVENTF_KEYUP = 0x2
ALLOWED = {"acs.exe", "obs64.exe"}
# The Claude app takes focus whenever it shows a message, and its chat box brings in Windows'
# text-input host. Either one in front means the machine is idle behind it, so AC is brought
# back to the front (allowed, since the app that has focus started this process) before a key goes out.
HANDBACK = {"claude.exe", "textinputhost.exe"}

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
        ac = user32.FindWindowW(None, "Assetto Corsa") if exe in HANDBACK else 0
        if ac and window_exe(ac) == "acs.exe":
            user32.ShowWindow(ac, 9)  # SW_RESTORE
            user32.SetForegroundWindow(ac)
        time.sleep(0.5 if ac else 2)
    user32.keybd_event(key, 0, 0, 0)
    time.sleep(hold_s)
    user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
    return True


def newest_launch() -> Path:
    root = find_ac_root()
    base = root / "apps" / "lua" / "locamotif_timecode" / "frame_log"
    return max((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)


def last_row(launch: Path) -> tuple[float, int]:
    """Modification time of the newest chunk and the lap count in its last row."""
    chunk = max(launch.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return chunk.stat().st_mtime, int(pd.read_csv(chunk, usecols=["lap_count"]).lap_count.iloc[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laps", type=int, required=True, help="completed laps to record from now")
    parser.add_argument("--start", action="store_true", help="start the recording first")
    parser.add_argument("--stall-s", type=float, default=30.0)
    parser.add_argument("--max-min", type=float, default=60.0, help="stop after this long whatever the lap count")
    args = parser.parse_args()
    began = time.time()

    launch = newest_launch()
    _, start_lap = last_row(launch)
    if args.start and not press(VK_F13):
        raise SystemExit(1)
    print(f"{time.strftime('%H:%M:%S')} recording; lap count {start_lap}, stopping at {start_lap + args.laps}", flush=True)
    reason = ""
    while not reason:
        time.sleep(2)
        mtime, lap = last_row(launch)
        if lap >= start_lap + args.laps:
            reason = f"lap count {lap}"
        elif time.time() - mtime > args.stall_s:
            reason = f"STALLED: no telemetry for {time.time() - mtime:.0f} s at lap count {lap}"
        elif time.time() - began > args.max_min * 60:
            reason = f"TIMEOUT after {args.max_min:.0f} min at lap count {lap}"
    ok = press(VK_F14)
    print(f"{time.strftime('%H:%M:%S')} {reason}; stop {'sent' if ok else 'NOT sent'}", flush=True)
    raise SystemExit(0 if ok and reason.startswith("lap count") else 1)


if __name__ == "__main__":
    main()
