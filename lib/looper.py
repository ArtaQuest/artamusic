#!/usr/bin/env python3
"""Find the best cycle inside a generated clip, instead of assuming the whole clip loops.

A clip of something RHYTHMIC — a hammer rising and falling — only loops if the last frame leaves
the subject where the first frame found it. Nothing in the sampler arranges that. The old cover was
a slow fire where a dissolve could hide the mismatch; a hammer at the top of its swing dissolving
into a hammer at the bottom is a morph, and it reads as one.

So look for the cycle rather than imposing it: over every pair (i, j) far enough apart to be worth
keeping, score how alike frame i and frame j are, and take the pair that matches best. This is the
video-textures idea (Schödl et al., 2000) at its simplest — no transition graph, just the single
best cut point, which is all a cover loop needs.

Two details that matter more than the search:

  * Compare on a BLURRED, downscaled luma. Full-resolution differencing is dominated by film grain
    and codec noise, which are uncorrelated between any two frames and therefore drown the signal
    that a hammer is in the same place.
  * Score the frame pair AND their neighbours (i+1 vs j+1). Two frames can match by accident at the
    turning point of a motion while travelling in opposite directions — the hammer at the same
    height going up as it was coming down. Requiring the NEXT frames to match too demands the same
    position and the same velocity, which is what a seamless cut actually needs.

    python looper.py <video>        # prints the best cycle it can find
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _small(frames, size=64):
    """Downscale + blur to luma: grain and codec noise are not evidence about where a hammer is."""
    a = np.asarray(frames).astype(np.float32).mean(3)
    n, h, w = a.shape
    ky, kx = max(1, h // size), max(1, w // size)
    a = a[:, : (h // ky) * ky, : (w // kx) * kx]
    a = a.reshape(n, h // ky, ky, w // kx, kx).mean(axis=(2, 4))
    return a.reshape(n, -1)


def best_cycle(frames, min_frames=24, size=64):
    """(start, end, score) — the tightest pair of matching frames at least min_frames apart.

    `end` is EXCLUSIVE: frames[start:end] is the cycle, and frames[end] is the one that would have
    come next and looks like frames[start], which is exactly what makes the cut invisible.
    """
    v = _small(frames, size)
    v = v - v.mean(1, keepdims=True)
    v /= np.maximum(np.sqrt((v * v).mean(1, keepdims=True)), 1e-6)
    n = len(v)
    if n < min_frames + 2:
        return 0, n, float("inf")
    best = (0, n, float("inf"))
    for i in range(0, n - min_frames - 1):
        j = np.arange(i + min_frames, n - 1)
        if not len(j):
            continue
        # position AND velocity: frame i vs j, and the frame after each
        d = np.abs(v[j] - v[i]).mean(1) + np.abs(v[j + 1] - v[i + 1]).mean(1)
        k = int(d.argmin())
        if d[k] < best[2]:
            best = (i, int(j[k]), float(d[k]))
    return best


def all_cycles(frames, min_frames=24, size=64):
    """Every start's best partner, so a caller can trade length against seam quality."""
    v = _small(frames, size)
    v = v - v.mean(1, keepdims=True)
    v /= np.maximum(np.sqrt((v * v).mean(1, keepdims=True)), 1e-6)
    n = len(v)
    out = []
    for i in range(0, max(0, n - min_frames - 1)):
        j = np.arange(i + min_frames, n - 1)
        if not len(j):
            continue
        d = np.abs(v[j] - v[i]).mean(1) + np.abs(v[j + 1] - v[i + 1]).mean(1)
        k = int(d.argmin())
        out.append((i, int(j[k]), float(d[k])))
    return out


def cycle_report(frames, min_frames=24, prefer_longest=True):
    """Pick the LONGEST invisible cut, not the single best-scoring one.

    Taking the best score alone biases short: a hammer's rise-and-fall repeats, so the tightest
    match is usually one strike, and one strike at 24 fps is well under two seconds — which reads
    as a stutter rather than a loop. Any cut whose seam is less visible than an ordinary
    frame-to-frame step is already invisible, and among those the longest is simply better. So take
    the longest that clears that bar, and fall back to the best-scoring one when nothing does.
    """
    cands = all_cycles(frames, min_frames)
    v = _small(frames)
    v = v - v.mean(1, keepdims=True)
    v /= np.maximum(np.sqrt((v * v).mean(1, keepdims=True)), 1e-6)
    typical = float(np.abs(np.diff(v, axis=0)).mean() * 2)
    naive = float(np.abs(v[-1] - v[0]).mean() + np.abs(v[0] - v[1]).mean())
    # THE WHOLE CLIP IS A CANDIDATE TOO. It was not, and that cost a loop: on the epic hammer take
    # the search picked a 59-frame cycle seaming at 0.86x a normal step while the untouched 81
    # frames seamed at 0.78x — both longer AND cleaner, and never once considered, because
    # `all_cycles` can only propose cuts strictly inside the clip. Keeping the whole thing is the
    # option that needs no justification, so it has to be on the list.
    cands = cands + [(0, len(frames) - 1, naive)]
    i, j, score = min(cands, key=lambda c: c[2])
    # LONGEST, BUT NOT AT ANY PRICE. Preferring length alone lets a marginal seam beat an
    # excellent one: on the period-12 fixture the whole 59-frame clip scores 0.91 — under the bar,
    # so "clean" — while the true four-period cut scores 0.48, and taking the longer one puts a
    # visible hitch in a rhythm that had none. So a longer candidate has to be within a quarter of
    # the best seam available to displace it. On the epic hammer take the whole clip wins outright
    # at 0.77 against 0.86 for the best interior cut, which is the case this is meant to allow.
    if prefer_longest:
        best = min(c[2] for c in cands)
        clean = [c for c in cands if c[2] < typical and c[2] <= best * 1.25]
        if clean:
            i, j, score = max(clean, key=lambda c: (c[1] - c[0], -c[2]))
    return {"start": int(i), "end": int(j), "frames": int(j - i),
            "whole_clip_chosen": bool(i == 0 and j == len(frames) - 1),
            "candidates_clean": int(sum(1 for c in cands if c[2] < typical)),
            "seam_score": round(score, 4),
            "whole_clip_seam": round(naive, 4),
            "typical_step": round(typical, 4),
            # under 1.0 means the chosen cut is less visible than an ordinary frame-to-frame step
            "seam_vs_typical": round(score / max(typical, 1e-6), 2),
            "whole_vs_typical": round(naive / max(typical, 1e-6), 2)}


def selftest():
    """A clip built from a KNOWN cycle must be found; a clip with no cycle must not be claimed."""
    ok = True
    rng = np.random.default_rng(4242)
    H = W = 96
    bg = rng.normal(120, 8, (H, W))

    def hammer(t):                       # a bar whose height is periodic, plus grain
        y = int(20 + 22 * (1 - abs(((t % 12) / 12) * 2 - 1)))
        f = bg.copy()
        f[y:y + 6, 30:70] = 240
        f += rng.normal(0, 3, (H, W))     # grain: uncorrelated, must not dominate
        return np.repeat(f.clip(0, 255)[:, :, None], 3, 2).astype(np.uint8)

    clip = np.stack([hammer(t) for t in range(60)])
    r = cycle_report(clip, min_frames=8)
    found = r["frames"]
    good = found % 12 == 0 and r["seam_vs_typical"] < 1.0
    ok &= good
    print(f"   periodic clip (period 12)   found {found} frames · seam/typical "
          f"{r['seam_vs_typical']:.2f} -> {'ok' if good else 'FAIL'}")

    # The negative case has to change STRUCTURALLY and never come back. A global brightness ramp
    # is not that: every frame is normalised to zero mean and unit variance before comparison — on
    # purpose, so a flickering fire cannot masquerade as movement — which makes a pure brightness
    # drift structurally identical throughout, and any cut in it genuinely IS seamless. A bar that
    # translates steadily and never returns is the honest test.
    def sliding(t):
        f = bg.copy()
        x = 4 + int(t * 1.4)
        f[40:48, x:x + 18] = 240
        f += rng.normal(0, 3, (H, W))
        return np.repeat(f.clip(0, 255)[:, :, None], 3, 2).astype(np.uint8)

    drift = np.stack([sliding(t) for t in range(60)])
    r2 = cycle_report(drift, min_frames=8)
    good2 = r2["seam_vs_typical"] > 1.0
    ok &= good2
    print(f"   monotonic drift, no cycle   seam/typical {r2['seam_vs_typical']:.2f} "
          f"(want > 1.0, i.e. no cycle claimed) -> {'ok' if good2 else 'FAIL'}")
    print("   " + ("ALL PASSED" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "selftest":
        print("LOOPER selftest — a real cycle must be found, an absent one must not be invented")
        sys.exit(0 if selftest() else 1)
    import stillness as S
    fr = S.frames_of(sys.argv[1])
    print(cycle_report(fr))
