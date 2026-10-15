"""
A GoPro's own telemetry (GPMF, the "GoPro MET" track of an MP4 straight off the SD card;
a copy re-exported by Photos or iCloud has lost it): every stream's samples with a time
on the video's clock, as one .npz per file.

GPMF is key-length-value: a FourCC, a type byte, a struct size, a repeat count, the data
padded to 4 bytes; type 0 nests. Each MP4 sample of the track (about one a second) holds
one DEVC with the streams (STRM) since the last; a stream's samples are spread evenly over
its MP4 sample's time span, as Gyroflow and GoPro's own parser do, so they keep the
camera's clock without drift. SCAL divides, and ORIN names the axes of GYRO/ACCL in the
camera's frame.

Usage: PYTHONPATH=. python experiments/gpmf_read.py VIDEO.MP4 OUT.npz
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

TYPES = {"b": "b", "B": "B", "c": "c", "d": "d", "f": "f", "F": "4s", "j": "q", "J": "Q",
         "l": "i", "L": "I", "s": "h", "S": "H", "q": "i", "Q": "q"}


def klv(buf: bytes, start: int = 0, end: int | None = None):
    end = len(buf) if end is None else end
    i = start
    while i + 8 <= end:
        key = buf[i:i + 4].decode("latin1")
        kind, size, repeat = chr(buf[i + 4]), buf[i + 5], struct.unpack(">H", buf[i + 6:i + 8])[0]
        n = size * repeat
        yield key, kind, size, repeat, i + 8, i + 8 + n
        i += 8 + ((n + 3) & ~3)


def values(buf: bytes, kind: str, size: int, repeat: int, a: int, b: int):
    if kind == "c":
        return buf[a:b].decode("latin1").rstrip("\0")
    if kind not in TYPES:
        return buf[a:b]
    fmt = TYPES[kind]
    count = size // struct.calcsize(">" + fmt)
    data = np.array(struct.unpack(">" + fmt * count * repeat, buf[a:b]))
    if kind == "q":
        data = data / 65536.0
    return data.reshape(repeat, count) if count > 1 else data


def packets(video: Path, stream: int) -> list[tuple[float, float, bytes]]:
    """Each MP4 sample of the telemetry track: (start s, duration s, payload)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", str(stream), "-show_packets",
                          "-show_entries", "packet=pts_time,duration_time,pos,size", "-of", "json", str(video)],
                         capture_output=True, check=True, text=True).stdout
    found = []
    with open(video, "rb") as f:
        for p in json.loads(out)["packets"]:
            f.seek(int(p["pos"]))
            found.append((float(p["pts_time"]), float(p["duration_time"]), f.read(int(p["size"]))))
    return found


def gpmf_stream_index(video: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_tag_string", "-of", "json", str(video)],
                         capture_output=True, check=True, text=True).stdout
    for s in json.loads(out)["streams"]:
        if s.get("codec_tag_string") == "gpmd":
            return int(s["index"])
    raise SystemExit(f"{video}: no GoPro telemetry track (a re-exported copy loses it; use the file off the SD card)")


def read(video: Path) -> tuple[dict[str, np.ndarray], dict]:
    series: dict[str, list] = {}
    times: dict[str, list] = {}
    info: dict = {}
    for t0, dt, payload in packets(video, gpmf_stream_index(video)):
        for key, kind, size, repeat, a, b in klv(payload):
            if key != "DEVC":
                continue
            for k2, kind2, s2, r2, a2, b2 in klv(payload, a, b):
                if k2 == "DVNM":
                    info["camera"] = values(payload, kind2, s2, r2, a2, b2)
                if k2 != "STRM":
                    continue
                scal, orin, stnm, name, data = np.ones(1), None, None, None, None
                for k3, kind3, s3, r3, a3, b3 in klv(payload, a2, b2):
                    v = values(payload, kind3, s3, r3, a3, b3)
                    if k3 == "SCAL":
                        scal = np.asarray(v, float)
                    elif k3 == "ORIN":
                        orin = v
                    elif k3 == "STNM":
                        stnm = v
                    elif k3 in ("GPSF", "GPSP"):  # the GPS fix (0, 2D, 3D) and its precision (DOP x 100), once a payload
                        series.setdefault(k3, []).append(np.atleast_1d(np.asarray(v, float)))
                        times.setdefault(k3, []).append(np.array([t0 + dt / 2]))
                    elif k3 not in ("STMP", "TSMP", "SIUN", "UNIT", "TYPE", "TMPC", "EMPT", "MTRX", "ORIO") and kind3 != "\0":
                        name, data = k3, v
                if name is None or not isinstance(data, np.ndarray) or len(data) == 0:
                    continue
                data = np.asarray(data, float)
                data = data / scal.reshape(-1)[0] if scal.size == 1 or data.ndim == 1 else data / scal.reshape(1, -1)
                series.setdefault(name, []).append(data)
                times.setdefault(name, []).append(t0 + dt * (np.arange(len(data)) + 0.5) / len(data))
                info.setdefault("streams", {})[name] = {"name": stnm, "axes": orin}
    out = {k: np.concatenate(v) for k, v in series.items()}
    out.update({k + "_t": np.concatenate(v) for k, v in times.items()})
    return out, info


def main() -> None:
    video, target = Path(sys.argv[1]), Path(sys.argv[2])
    data, info = read(video)
    np.savez(target, **data)
    print(f"camera: {info.get('camera')}")
    for name, about in info["streams"].items():
        t = data[name + "_t"]
        print(f"  {name:<5} {about['name'] or '':<40} {len(t):7d} samples, {len(t) / (t[-1] - t[0]):7.1f} Hz, "
              f"axes {about['axes']}, shape {data[name].shape[1:]}")


if __name__ == "__main__":
    main()
