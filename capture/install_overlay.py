"""
Copy the timecode overlay into Assetto Corsa's Lua apps folder.

    python -m capture.install_overlay
    python -m capture.install_overlay --ac-root "D:\\Steam\\steamapps\\common\\assettocorsa"

Finds AC through the Steam registry keys and any extra Steam library folders.
Requires Custom Shaders Patch, since the app uses the CSP Lua API.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

APP_NAME = "locamotif_timecode"
SOURCE = Path(__file__).parent / "ac_overlay"


def _steam_roots() -> list[Path]:
    if sys.platform != "win32":
        return []
    import winreg

    roots: list[Path] = []
    for hive, key, value in (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
    ):
        try:
            with winreg.OpenKey(hive, key) as handle:
                roots.append(Path(winreg.QueryValueEx(handle, value)[0]))
        except OSError:
            continue
    return roots


def _library_folders(steam_root: Path) -> list[Path]:
    vdf = steam_root / "steamapps" / "libraryfolders.vdf"
    if not vdf.exists():
        return []
    text = vdf.read_text(encoding="utf-8", errors="replace")
    return [Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', text)]


def find_ac_root() -> Path | None:
    candidates: list[Path] = []
    for steam in _steam_roots():
        candidates.append(steam)
        candidates.extend(_library_folders(steam))
    for base in candidates:
        target = base / "steamapps" / "common" / "assettocorsa"
        if (target / "acs.exe").exists() or (target / "apps").is_dir():
            return target
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ac-root", help="Assetto Corsa install folder")
    parser.add_argument("--force", action="store_true", help="overwrite an existing install")
    args = parser.parse_args()

    root = Path(args.ac_root) if args.ac_root else find_ac_root()
    if root is None:
        raise SystemExit("could not locate Assetto Corsa; pass --ac-root")
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    lua_apps = root / "apps" / "lua"
    if not lua_apps.is_dir():
        print(f"note: {lua_apps} does not exist yet; Custom Shaders Patch creates it")
    destination = lua_apps / APP_NAME

    if destination.exists() and not args.force:
        raise SystemExit(f"{destination} already exists; pass --force to overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.ini", f"{APP_NAME}.lua"):
        shutil.copy2(SOURCE / name, destination / name)

    print(f"installed to {destination}")
    print("Enable it in AC: app bar -> Timecode. See ml/README.md for the capture checklist.")


if __name__ == "__main__":
    main()
