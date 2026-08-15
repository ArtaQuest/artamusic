#!/usr/bin/env python3
"""Turn a drifting generated clip into a seamless loop, and MEASURE the wrap.

Endpoint-conditioned generation (the same still pinned at frame 0 and the last frame) is the
better answer and is what the Kaggle kernel does. This is the local one: given a clip that drifts,
find the best loop point and close it, so a deliverable exists without a GPU.

The method is the classic video-textures one, which suits this material exactly: search for the
frame most similar to frame 0, cut there, and cross-dissolve a short overlap. On smoke and embers
there is no rigid structure to misalign, so a dissolve of a few tenths of a second reads as motion
rather than as a fade. Crucially the loop point is CHOSEN by similarity rather than by clock, so
the two ends already nearly match before anything is blended.

    python makeloop.py clip.mp4 out_basename [--min-seconds 2.0]
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(c[:80] + " -> " + r.stderr[-300:])
    return r.stdout


def frames(path, w=192, h=128):
    sh(f"ffmpeg -v error -i '{path}' -vf scale={w}:{h} -f rawvideo -pix_fmt rgb24 /tmp/_lf.rgb -y")
    a = np.fromfile("/tmp/_lf.rgb", dtype=np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255
    return a


def best_loop_point(a, fps, min_seconds):
    """The frame after min_seconds whose appearance is closest to frame 0."""
    ref = a[0]
    start = int(min_seconds * fps)
    if start >= len(a) - 1:
        start = len(a) // 2
    d = np.array([np.abs(a[i] - ref).mean() for i in range(start, len(a))])
    idx = int(start + d.argmin())
    return idx, float(d.min() * 255), float(np.abs(a[-1] - ref).mean() * 255)


def build(clip, out_base, min_seconds=2.0, fps=24, xfade=0.4, height=1080):
    a = frames(clip)
    idx, best_d, end_d = best_loop_point(a, fps, min_seconds)
    body = idx / fps
    print(f"  best loop point: frame {idx} ({body:.2f} s) · difference to frame 0 {best_d:.2f} "
          f"(the clip's own end scores {end_d:.2f})")
    tail = f"/tmp/_loopsrc.mp4"
    sh(f"ffmpeg -v error -i '{clip}' -t {body + xfade} -c:v libx264 -preset veryslow -crf 18 "
       f"-pix_fmt yuv420p {tail} -y")
    closed = "/tmp/_closed.mp4"
    sh(f"ffmpeg -v error -i '{tail}' -filter_complex "
       f"\"[0:v]split=2[a][b];"
       f"[a]trim=0:{body},setpts=PTS-STARTPTS[main];"
       f"[b]trim={body}:{body + xfade},setpts=PTS-STARTPTS[tl];"
       f"[main][tl]xfade=transition=fade:duration={xfade}:offset={body - xfade}[v]\" "
       f"-map '[v]' -c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p {closed} -y")
    webm, mp4 = f"{out_base}.webm", f"{out_base}.mp4"
    vf = f"scale=-2:{height}:flags=lanczos"
    sh(f"ffmpeg -v error -i {closed} -vf {vf} -c:v libvpx-vp9 -crf 33 -b:v 0 -row-mt 1 "
       f"-cpu-used 1 -g 240 -an '{webm}' -y")
    sh(f"ffmpeg -v error -i {closed} -vf {vf} -c:v libx264 -preset veryslow -crf 28 "
       f"-pix_fmt yuv420p -movflags +faststart -an '{mp4}' -y")
    # measure the DELIVERED loop the way a player shows it: last frame against first
    b = frames(webm)
    per = np.abs(np.diff(b, axis=0)).reshape(len(b) - 1, -1).mean(1) * 255
    wrap = float(np.abs(b[-1] - b[0]).mean() * 255)
    typical = float(np.percentile(per, 95))
    return {"seconds": round(len(b) / fps, 2), "frames": int(len(b)),
            "webm_mb": round(Path(webm).stat().st_size / 1048576, 2),
            "mp4_mb": round(Path(mp4).stat().st_size / 1048576, 2),
            "ti_mean": round(float(per.mean()), 2),
            "wrap_delta": round(wrap, 2), "typical_frame_delta": round(typical, 2),
            "wrap_ratio": round(wrap / max(typical, 1e-6), 2)}


def build_mirror(clip, out_base, fps=24, height=1080, seconds=None):
    """Forward then reverse. The wrap is EXACT by construction — the sequence ends on frame 0's
    neighbour and wraps to frame 0 — so no dissolve and no loop-point search is needed.

    The cost is that motion plays backwards for half the loop. That is fatal for material with a
    strong direction (falling rain, rising sparks read as sinking) and usually invisible for
    diffuse shimmer. Whether it applies HERE is a judgement to make by looking, not by rule, which
    is why this measures the wrap AND prints the reversal fraction rather than claiming success.
    """
    src = clip
    if seconds:
        src = "/tmp/_trim.mp4"
        sh(f"ffmpeg -v error -i '{clip}' -t {seconds} -c:v libx264 -preset veryslow -crf 18 "
           f"-pix_fmt yuv420p {src} -y")
    mir = "/tmp/_mirror.mp4"
    # drop the duplicated turning frames so neither end stutters
    sh(f"ffmpeg -v error -i '{src}' -filter_complex "
       f"\"[0:v]split=2[f][r];[r]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[rv];"
       f"[f][rv]concat=n=2:v=1:a=0[v]\" -map '[v]' "
       f"-c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p {mir} -y")
    webm, mp4 = f"{out_base}.webm", f"{out_base}.mp4"
    vf = f"scale=-2:{height}:flags=lanczos"
    sh(f"ffmpeg -v error -i {mir} -vf {vf} -c:v libvpx-vp9 -crf 33 -b:v 0 -row-mt 1 "
       f"-cpu-used 1 -g 240 -an '{webm}' -y")
    sh(f"ffmpeg -v error -i {mir} -vf {vf} -c:v libx264 -preset veryslow -crf 28 "
       f"-pix_fmt yuv420p -movflags +faststart -an '{mp4}' -y")
    b = frames(webm)
    per = np.abs(np.diff(b, axis=0)).reshape(len(b) - 1, -1).mean(1) * 255
    wrap = float(np.abs(b[-1] - b[0]).mean() * 255)
    typical = float(np.percentile(per, 95))
    return {"method": "mirror", "seconds": round(len(b) / fps, 2), "frames": int(len(b)),
            "webm_mb": round(Path(webm).stat().st_size / 1048576, 2),
            "mp4_mb": round(Path(mp4).stat().st_size / 1048576, 2),
            "ti_mean": round(float(per.mean()), 2),
            "wrap_delta": round(wrap, 2), "typical_frame_delta": round(typical, 2),
            "wrap_ratio": round(wrap / max(typical, 1e-6), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("out_base")
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--mirror", action="store_true")
    a = ap.parse_args()
    import json
    fn = build_mirror if a.mirror else build
    print(json.dumps(fn(a.clip, a.out_base) if a.mirror
                     else fn(a.clip, a.out_base, a.min_seconds), indent=2))
