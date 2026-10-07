"""
Split packed sets into parts that each fit one USB trip.

    python -m capture.usb_parts --dest D:/Documents/Transfer/batch1007 --prefix b1007 data/packed_ac_b1007_full_own ...

The USB stick holds 32 GB, so a set of 60 GB goes over in several trips. Parts
are filled first-fit with whole sessions (a session is split lap by lap only when
it alone exceeds the cap), and are made of hard links into the packed sets, so
they take no extra disk space; they must sit on the same volume as the packs.
Each part holds `<set>/laps/<lap>/...` for the laps it carries, each set's full
`index.json`, and a `MANIFEST.txt`. On the Mac, copying every part's `<set>/`
into `data/` (merging `laps/`) rebuilds each set.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def tree_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def plan_units(sets: list[Path], cap: float) -> list[tuple[Path, list[str], int]]:
    """(set, lap paths, bytes): one per session, or one per lap for a session over the cap."""
    units = []
    for packed in sets:
        by_session: dict[str, list[str]] = defaultdict(list)
        for lap in json.loads((packed / "index.json").read_text())["laps"]:
            by_session[lap["session_id"]].append(lap["path"])
        for paths in by_session.values():
            sizes = [tree_size(packed / p) for p in paths]
            if sum(sizes) <= cap:
                units.append((packed, paths, sum(sizes)))
            else:
                units.extend((packed, [p], size) for p, size in zip(paths, sizes))
    return units


def fill_parts(units: list[tuple[Path, list[str], int]], cap: float) -> list[tuple[int, list]]:
    """First-fit decreasing: the largest units first, each into the first part with room."""
    parts: list[list] = []
    for unit in sorted(units, key=lambda u: -u[2]):
        for part in parts:
            if part[0] + unit[2] <= cap:
                part[0] += unit[2]
                part[1].append(unit)
                break
        else:
            parts.append([unit[2], [unit]])
    return [(total, members) for total, members in sorted(parts, key=lambda p: -p[0])]


def link_tree(source: Path, target: Path) -> None:
    for f in source.rglob("*"):
        if f.is_file():
            dst = target / f.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(f, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", help="packed set directories")
    parser.add_argument("--dest", required=True, help="where the part folders go (same volume as the sets)")
    parser.add_argument("--prefix", required=True, help="parts are named usb_<prefix>_partN")
    parser.add_argument("--max-gb", type=float, default=24.0)
    args = parser.parse_args()

    cap = args.max_gb * 1e9
    parts = fill_parts(plan_units([Path(s) for s in args.sets], cap), cap)
    for i, (total, members) in enumerate(parts, 1):
        root = Path(args.dest) / f"usb_{args.prefix}_part{i}"
        carried = []
        for packed, paths, _ in members:
            (root / packed.name).mkdir(parents=True, exist_ok=True)
            index = root / packed.name / "index.json"
            if not index.exists():
                os.link(packed / "index.json", index)
            for p in paths:
                link_tree(packed / p, root / packed.name / p)
                carried.append(f"{packed.name}/{p}")
        (root / "MANIFEST.txt").write_text(
            f"{root.name}: {total / 1e9:.1f} GB, {len(carried)} lap folders.\n"
            "Copy each <set>/ folder into data/ (merging laps/ across parts); each index.json is the whole set's.\n\n"
            + "\n".join(sorted(carried)) + "\n"
        )
        print(f"{root.name}: {total / 1e9:.1f} GB, {len(carried)} laps, sets {sorted({p.name for p, _, _ in members})}")


if __name__ == "__main__":
    main()
