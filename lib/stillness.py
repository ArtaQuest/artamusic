#!/usr/bin/env python3
"""Does the thing that must not move, move?

The cover loop shipped with a rubber sword. Every number the run published said it was fine —
zero cuts, a wrap 0.84x smoother than an ordinary frame step, motion energy in range — because
every one of them was a WHOLE-FRAME luma statistic, and the blade is 3.4% of the frame. It drifted
17x28 px, its silhouette swung 17%, and its own pixels moved 4.2x more than everything else. A
global average cannot see a local disaster, and "the subject holds still" was never measured at
all.

So: measure the subject, inside its own mask.

  drift_px      how far the subject travels across the loop, by template matching
  max_shift_px  the worst single frame's displacement from frame 0
  ssim_min      worst per-frame structural similarity to frame 0, inside the mask
  ratio         motion INSIDE the mask over motion OUTSIDE it — under 1.0 means the subject is
                the calmest thing in the picture, which is the whole point of a cinemagraph

    python stillness.py <video> [mask.png]      # mask defaults to the bright-subject heuristic
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def frames_of(path, size=None):
    """Decode a video to an (N, H, W, 3) uint8 array at its own resolution unless told otherwise."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip().split(",")
    w, h = (int(probe[0]), int(probe[1])) if len(probe) >= 2 else (size, size)
    if size:
        w = h = size
    raw = Path(tempfile.mkdtemp()) / "v.rgb"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw), "-y"], check=True)
    return np.fromfile(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def bright_subject_mask(frame0, pct=97.0, min_frac=0.005):
    """The subject, when it is the bright rigid thing: polished steel against coals and shadow.

    A percentile, not a fixed threshold — the same code then works on a darker or brighter still.
    """
    g = frame0.mean(2)
    thr = np.percentile(g, pct)
    m = g > thr
    while m.mean() < min_frac and pct > 80:
        pct -= 1.0
        m = g > np.percentile(g, pct)
    return m


def _deviation(a, b, mask):
    """Mean absolute change inside the mask, in grey levels — 0 for a frozen subject, whatever the
    texture. (A global SSIM was tried first and collapses on a flat region: with near-zero variance
    its covariance term is pure codec noise, so a perfectly frozen bar scored 0.748.)"""
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    return float(np.abs(x - y).mean()) if x.size else 0.0


def measure(video, mask=None, size=None):
    a = frames_of(video, size).astype(np.float32)
    g = a.mean(3)
    if mask is None:
        mask = bright_subject_mask(a[0])
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != g[0].shape:
        raise ValueError(f"mask {mask.shape} does not match frame {g[0].shape}")
    std = a.std(axis=0).mean(axis=2)
    inside, outside = float(std[mask].mean()), float(std[~mask].mean())
    # MEASURE DISPLACEMENT, DO NOT INFER IT FROM A SILHOUETTE. Two silhouette trackers were tried
    # and both followed the fire instead of the sword: a forge is full of pixels as bright as
    # polished steel, and once the blade is frozen its silhouette stops meaning anything anyway.
    # Template matching asks the question directly — how far must frame 0's subject be shifted to
    # sit on top of frame i's? — and fire cannot answer it, because fire is not in the template.
    ys0, xs0 = np.nonzero(mask)
    pad = 10
    y0, y1 = max(0, ys0.min() - pad), min(g.shape[1], ys0.max() + pad + 1)
    x0, x1 = max(0, xs0.min() - pad), min(g.shape[2], xs0.max() + pad + 1)
    tpl = g[0][y0:y1, x0:x1]
    tmask = mask[y0:y1, x0:x1]
    R = 8
    shifts = []
    for f in g:
        patch = f[y0:y1, x0:x1]
        best, bdy, bdx = None, 0, 0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                sh = np.roll(np.roll(patch, -dy, axis=0), -dx, axis=1)
                sad = float(np.abs(sh[tmask] - tpl[tmask]).mean())
                if best is None or sad < best:
                    best, bdy, bdx = sad, dy, dx
        shifts.append((bdy, bdx))
    shifts = np.array(shifts, dtype=float)
    drift = float(np.hypot(shifts[:, 1].max() - shifts[:, 1].min(),
                           shifts[:, 0].max() - shifts[:, 0].min()))
    swing = float(np.abs(shifts).max())        # worst single-frame displacement, in pixels

    devs = [_deviation(g[0], f, mask) for f in g]
    return {"frames": int(len(a)), "mask_px": int(mask.sum()),
            "mask_pct": round(100 * float(mask.mean()), 2),
            "drift_px": round(drift, 1), "max_shift_px": round(swing, 1),
            "max_dev": round(float(np.max(devs)), 2),
            "motion_inside": round(inside, 2), "motion_outside": round(outside, 2),
            "ratio": round(inside / max(outside, 1e-6), 2)}


def liveness(video, moving_mask, subject_mask, size=None):
    """The counter-gate. A test that only asks "did the subject hold still" is passed PERFECTLY by
    a still photograph, and passed well by a cut-out pasted over a frozen plate. So ask the two
    opposite questions too: does the fire actually live, and does its light still fall on the
    frozen subject? (The second is what separates a lit freeze from a cardboard cut-out — the
    blade's own pixels must change in BRIGHTNESS while never changing in PLACE.)"""
    a = frames_of(video, size).astype(np.float32)
    g = a.mean(3)
    mv = np.asarray(moving_mask, dtype=bool)
    sb = np.asarray(subject_mask, dtype=bool)
    fire = float(np.abs(np.diff(g, axis=0))[:, mv].mean()) if mv.any() else 0.0
    lum = np.array([f[sb].mean() for f in g]) if sb.any() else np.zeros(len(g))
    chroma = np.array([(f[..., 0] - f[..., 2])[sb].mean() for f in a]) if sb.any() else np.zeros(len(a))
    return {"fire_motion": round(fire, 2),
            "subject_light_std": round(float(lum.std()), 2),
            "subject_chroma_std": round(float(chroma.std()), 2)}


ALIVE = {"fire_motion": 1.0, "subject_light_std": 0.8}


def liveness_verdict(m):
    bad = []
    if m["fire_motion"] < ALIVE["fire_motion"]:
        bad.append(f"the fire barely moves ({m['fire_motion']}, want >= {ALIVE['fire_motion']})")
    if m["subject_light_std"] < ALIVE["subject_light_std"]:
        bad.append(f"no firelight plays on the subject ({m['subject_light_std']}, "
                   f"want >= {ALIVE['subject_light_std']}) — a frozen plate reads as a cut-out")
    return bad


# What "it does not move" means, in numbers. A rigid subject in a cinemagraph should be the
# STILLEST thing in the frame, not the busiest.
LIMIT = {"drift_px": 2.0, "max_shift_px": 1.0, "max_dev": 6.0, "ratio": 1.0}


def verdict(m):
    bad = []
    if m["drift_px"] > LIMIT["drift_px"]:
        bad.append(f"the subject wanders {m['drift_px']} px (limit {LIMIT['drift_px']})")
    if m["max_shift_px"] > LIMIT["max_shift_px"]:
        bad.append(f"one frame sits {m['max_shift_px']} px off (limit {LIMIT['max_shift_px']})")
    if m["max_dev"] > LIMIT["max_dev"]:
        bad.append(f"its worst frame differs by {m['max_dev']} grey levels (limit {LIMIT['max_dev']})")
    if m["ratio"] > LIMIT["ratio"]:
        bad.append(f"it moves {m['ratio']}x MORE than the rest of the frame (limit {LIMIT['ratio']}x)")
    return bad


def selftest(tmp=None):
    """A frozen subject over a boiling background must pass; the same subject drifting must not."""
    ok = True
    tmp = Path(tmp or tempfile.mkdtemp())
    rng = np.random.default_rng(4242)
    H = W = 256
    sub = np.zeros((H, W), bool)
    sub[110:140, 40:210] = True                       # a bar, standing in for the blade
    for name, shift in (("frozen subject", 0), ("subject drifting 4 px", 4)):
        d = tmp / name.replace(" ", "_"); d.mkdir(exist_ok=True)
        for i in range(24):
            # fire, not television static: low-frequency blobs that swell and drift
            small = rng.normal(0, 1, (16, 16))
            from PIL import Image as _I
            blob = np.asarray(_I.fromarray(small.astype(np.float32), mode="F").resize((W, H), _I.BILINEAR))
            bg = (90 + 45 * np.sin(i / 3 + blob * 2) + 12 * blob).clip(0, 255)
            fr = np.repeat(bg[:, :, None], 3, 2)
            off = int(round(shift * i / 23))
            s = np.roll(sub, off, axis=0)
            fr[s] = 240
            from PIL import Image
            Image.fromarray(fr.astype(np.uint8)).save(d / f"{i:03d}.png")
        vid = tmp / f"{d.name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "16", "-i", f"{d}/%03d.png",
                        "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(vid), "-y"],
                       check=True)
        m = measure(vid)
        bad = verdict(m)
        want_clean = shift == 0
        good = (not bad) if want_clean else bool(bad)
        ok &= good
        print(f"   {name:22s} drift {m['drift_px']:5.1f} px · shift {m['max_shift_px']:4.1f} px · "
              f"dev {m['max_dev']:5.2f} · ratio {m['ratio']:5.2f} -> "
              f"{'holds' if not bad else 'MOVES'}  {'ok' if good else 'FAIL'}")
    # and the counter-gate: a video of nothing moving must FAIL liveness, however still it is
    from PIL import Image as _I2
    d = tmp / "dead"; d.mkdir(exist_ok=True)
    plate = np.zeros((H, W, 3), np.uint8); plate[sub] = 240; plate[~sub] = 60
    for i in range(24):
        _I2.fromarray(plate).save(d / f"{i:03d}.png")
    vid = tmp / "dead.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "16", "-i", f"{d}/%03d.png",
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(vid), "-y"], check=True)
    lv = liveness(vid, ~sub, sub)
    bad = liveness_verdict(lv)
    good = bool(bad); ok &= good
    print(f"   {'a frozen photograph':22s} fire {lv['fire_motion']:5.2f} · light {lv['subject_light_std']:5.2f} -> "
          f"{'DEAD' if bad else 'alive'}  {'ok' if good else 'FAIL'}")
    print("   " + ("ALL PASSED" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "selftest":
        print("STILLNESS selftest — a frozen subject must pass, a drifting one must not")
        sys.exit(0 if selftest() else 1)
    mask = None
    if len(sys.argv) > 2:
        from PIL import Image
        mask = np.asarray(Image.open(sys.argv[2]).convert("L")) > 127
    m = measure(sys.argv[1], mask)
    print({k: v for k, v in m.items()})
    bad = verdict(m)
    print("VERDICT:", "the subject holds still" if not bad else "; ".join(bad))
    sys.exit(0 if not bad else 1)
