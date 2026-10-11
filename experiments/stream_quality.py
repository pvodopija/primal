"""
What does a livestream's quality cost PRIMAL? Re-encodes a real-footage video the way a
camera's livestream would deliver it: H.264 at a given frame height, bitrate and frame
rate, a keyframe a second, no B-frames, rate-capped (x264 veryfast, zerolatency). Writes
data/real-footage/<set>/stream_<name>.mp4 (overlays masked first) and prints the bitrate it
reached; experiments/real_truth_orb.py then scores it with REAL_STREAM=<name> against the
same ORB truth, which reads the original video.

Usage: REAL_FOOTAGE=indoor-pov PYTHONPATH=. python experiments/stream_quality.py 480p_1000k [...]
       names are <height>p_<kbps>k[_<fps>fps], 30 fps by default
"""
import re
import sys

import av
import cv2

from experiments.real_footage import ROOT, VIDEO, mask


def encode(name: str) -> None:
    h, kbps, fps = re.fullmatch(r"(\d+)p_(\d+)k(?:_(\d+)fps)?", name).groups()
    h, kbps, fps = int(h), int(kbps), int(fps or 30)
    w = int(round(h * 16 / 9 / 2)) * 2
    path = ROOT / f"stream_{name}.mp4"
    cap = cv2.VideoCapture(str(VIDEO))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt, s.bit_rate = w, h, "yuv420p", kbps * 1000
        s.options = {"preset": "veryfast", "tune": "zerolatency", "g": str(fps), "bf": "0",
                     "maxrate": f"{kbps}k", "bufsize": f"{kbps}k"}
        i = n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i / src_fps + 1e-6 >= n / fps:  # the source frame nearest the stream's clock
                frame = av.VideoFrame.from_ndarray(cv2.resize(mask(f), (w, h), interpolation=cv2.INTER_AREA), format="bgr24")
                out.mux(s.encode(frame))
                n += 1
            i += 1
        out.mux(s.encode())
    print(f"{name}: {w}x{h} at {fps} fps, {n} frames, reached {path.stat().st_size * 8 / (n / fps) / 1000:.0f} kbps", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        encode(name)
