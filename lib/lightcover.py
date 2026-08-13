#!/usr/bin/env python3
"""Re-encode an existing cover video to something a platform can actually serve.

The three shipped covers are 54 MB, 18.9 MB and 14 MB for 180 seconds of a nearly-static image —
all encoded at crf 17-18, which is a mastering setting, not a delivery one. A cover video is not
a film: it is one photograph with slow light on it, which is the most compressible material
there is. Encoded for delivery the same content lands in single-digit megabytes.

VP9 is the house codec (the project already ships VP9 teasers); h264 rides along as the fallback
for players that need it.

    python lightcover.py <in.mp4> [--seconds 12] [--height 1080]
"""
import argparse, subprocess, sys
from pathlib import Path


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-300:])


def encode(src, seconds=None, height=1080):
    src = Path(src)
    dur = f"-t {seconds} " if seconds else ""
    webm = src.with_name(src.stem + "_light.webm")
    mp4 = src.with_name(src.stem + "_light.mp4")
    vf = f"scale=-2:{height}:flags=lanczos"
    sh(f"ffmpeg -v error {dur}-i '{src}' -vf {vf} -c:v libvpx-vp9 -crf 33 -b:v 0 "
       f"-row-mt 1 -cpu-used 1 -g 240 -c:a libopus -b:a 128k '{webm}' -y")
    sh(f"ffmpeg -v error {dur}-i '{src}' -vf {vf} -c:v libx264 -preset veryslow -crf 28 "
       f"-pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k '{mp4}' -y")
    o = src.stat().st_size / 1048576
    return {"source_mb": round(o, 1),
            "webm_mb": round(webm.stat().st_size / 1048576, 2),
            "mp4_mb": round(mp4.stat().st_size / 1048576, 2),
            "saved_pct": round(100 * (1 - webm.stat().st_size / src.stat().st_size), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--seconds", type=float)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()
    r = encode(a.src, a.seconds, a.height)
    print(f"{Path(a.src).name}: {r['source_mb']} MB -> webm {r['webm_mb']} MB "
          f"/ mp4 {r['mp4_mb']} MB  ({r['saved_pct']}% smaller)")
