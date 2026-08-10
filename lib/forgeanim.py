#!/usr/bin/env python3
"""Animated cover: one still image, made to live for 180 seconds. No cuts, no montage.

The operator's brief was "simple animation — heat flow and steel burning". The temptation is an
ffmpeg filtergraph, and it is the wrong tool: `geq` is documented-slow and per-frame audio
reactivity can only be stepped at discrete `sendcmd` timestamps. So the motion is SIMULATED, not
filtered and not sampled from a video model:

    torch is the shader language. ffmpeg is the encoder, and nothing else.

A real (small) fluid solve runs at 24 fps over the still: heat is injected where the metal is,
buoyancy lifts it, divergence-free curl noise turbulates it, semi-Lagrangian advection moves it,
and the temperature field drives an incandescence ramp calibrated to how steel actually glows
(480 C faint red through 1380 C yellow-white). Everything composites in LINEAR light, which is
what stops a glow reading as an orange sticker laid over a photograph.

Determinism: one seed, no wall-clock, no RNG outside the seeded generator — the same inputs
produce the same 4320 frames, which the reproducibility gate requires.

CALM: every audio envelope is slew-limited to 6% change per frame, so a >10%-in-one-frame mean
luma jump is impossible BY CONSTRUCTION rather than by hope. Measured anyway in verify().

    python forgeanim.py render --image cover.png --audio master.wav --out cover.mp4 [--preview]
    python forgeanim.py verify --video cover.mp4 --audio master.wav
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SEED = 4242
FPS = 24

# Steel's own colour ladder, in LINEAR sRGB. Hand-tabulated from blacksmithing colour charts
# rather than a hue sweep: a linear hue ramp from red to yellow does not look like hot metal,
# because real incandescence gains luminance far faster than it gains hue.
LADDER_C = np.array([480, 600, 750, 900, 1100, 1250, 1380], dtype=np.float32)
LADDER_RGB = np.array([
    [0.09, 0.006, 0.002], [0.25, 0.02, 0.004], [0.55, 0.07, 0.01],
    [0.85, 0.18, 0.02], [1.00, 0.38, 0.05], [1.00, 0.58, 0.14],
    [1.00, 0.80, 0.42]], dtype=np.float32)


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[:90]} -> {r.stderr[-400:]}")
    return r.stdout


# ── Stage A: audio envelopes ────────────────────────────────────────────────────────────
def envelopes(audio_path, n_frames, sr=48000, seconds=None):
    """Four deterministic envelopes at frame rate. Normalised by whole-track percentiles —
    never a running AGC, which would make the same audio render differently depending on where
    it started."""
    import librosa
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True,
                        duration=seconds)   # a preview must be a true PREFIX
    hop = sr // FPS

    def band_rms(lo, hi):
        from scipy.signal import butter, sosfilt
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        b = sosfilt(sos, y)
        return librosa.feature.rms(y=b, frame_length=hop * 2, hop_length=hop)[0]

    e_forge = band_rms(20, 160)                      # the bellows: sub energy
    e_loud = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    perc = librosa.effects.percussive(y, margin=3.0)
    e_onset = librosa.onset.onset_strength(y=perc, sr=sr, hop_length=hop)

    def norm_slew(x):
        x = np.asarray(x, dtype=np.float32)
        x = np.interp(np.linspace(0, len(x) - 1, n_frames), np.arange(len(x)), x)
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
        a_att = math.exp(-1 / (FPS * 0.03))
        a_rel = math.exp(-1 / (FPS * 0.40))
        out = np.zeros_like(x)
        prev = float(x[0])
        for i, v in enumerate(x):
            a = a_att if v > prev else a_rel
            prev = a * prev + (1 - a) * v
            out[i] = prev
        # THE CALM GUARD, structural: 6% max change per frame makes a flash impossible.
        for i in range(1, len(out)):
            d = np.clip(out[i] - out[i - 1], -0.06, 0.06)
            out[i] = out[i - 1] + d
        return out.astype(np.float32)

    return {k: norm_slew(v) for k, v in
            (("forge", e_forge), ("loud", e_loud), ("onset", e_onset))}


# ── the metal mask, derived rather than hand-painted (the notebook must run for a stranger) ──
def metal_mask(img_lin, depth):
    """Where is the hot material? Everything incandescing — the blade's vein AND the coal bed.

    The first version normalised by luma.max(), which on this material is the yellow-white core
    of the blade vein. Dividing by the single hottest pixel crushed the coal bed to nothing and
    lit 4.7% of the frame; the animation was invisible and the measurements said so (TI 2.86
    against a target of 8-15). Normalising by a PERCENTILE instead keeps the dim-but-genuinely-
    hot regions, which on a forge scene are most of the heat.

    Warmth (r - b) is the load-bearing signal: a bright grey rock is not hot, a dim orange coal
    is. Luma only modulates. That ordering is what keeps the glow off the background.
    """
    r, g, b = img_lin[0], img_lin[1], img_lin[2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    warmth = (r - b).clamp(min=0)
    w_ref = torch.quantile(warmth[warmth > 0].flatten(), 0.90) if (warmth > 0).any() else warmth.max()
    l_ref = torch.quantile(luma.flatten(), 0.90)
    m = (warmth / (w_ref + 1e-9)).clamp(0, 1) ** 0.7
    m = m * (0.35 + 0.65 * (luma / (l_ref + 1e-9)).clamp(0, 1) ** 0.5)
    m = torch.where(m > 0.08, m, torch.zeros_like(m))   # a floor, not a median
    k = torch.ones(1, 1, 9, 9) / 81.0
    m = F.conv2d(m[None, None], k, padding=4)[0, 0]     # ~8 px feather
    return m.clamp(0, 1)


# ── Stage B: the heat field ─────────────────────────────────────────────────────────────
class HeatField:
    """Temperature + velocity on a coarse grid, stepped once per frame.

    No pressure projection. Curl noise is divergence-free by construction, and the
    unconditional stability of this scheme comes from semi-Lagrangian advection, not from the
    solve — so ~20 Jacobi iterations buy almost nothing on a plume rising off a static plate,
    and cost the hardest-to-debug component in the pipeline.
    """

    def __init__(self, h, w, mask, device, gen):
        self.h, self.w, self.dev = h, w, device
        self.T = torch.zeros(h, w, device=device)
        self.vx = torch.zeros(h, w, device=device)
        self.vy = torch.zeros(h, w, device=device)
        self.M = F.interpolate(mask[None, None], size=(h, w), mode="bilinear",
                               align_corners=False)[0, 0].to(device)
        # a seeded tileable potential for the curl noise: 3 octaves, the third axis is time
        n = torch.randn(64, 16, 16, generator=gen, device="cpu")
        self.psi = F.interpolate(n[None], size=(h, w), mode="bilinear",
                                 align_corners=False)[0].to(device)
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                torch.arange(w, device=device, dtype=torch.float32),
                                indexing="ij")
        self.gx, self.gy = xs, ys

    def _curl(self, t):
        """Divergence-free velocity from the gradient of a scalar potential.

        NORMALISED to unit std before use. Without this the amplitude silently depends on the
        upsample factor of the noise (measured: 0.003 px/frame — the turbulence did nothing and
        the plume froze into a steady state that reads as a still image).
        """
        i = int(t * FPS * 0.35) % self.psi.shape[0]
        p = self.psi[i]
        dpdy = torch.gradient(p, dim=0)[0]
        dpdx = torch.gradient(p, dim=1)[0]
        sd = dpdy.std() + dpdx.std() + 1e-9
        return dpdy / sd * 2.0, -dpdx / sd * 2.0

    def _advect(self, field, dt):
        x = (self.gx - dt * self.vx * FPS).clamp(0, self.w - 1)
        y = (self.gy - dt * self.vy * FPS).clamp(0, self.h - 1)
        gx = (x / (self.w - 1)) * 2 - 1
        gy = (y / (self.h - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)[None]
        return F.grid_sample(field[None, None], grid, mode="bilinear",
                             padding_mode="border", align_corners=False)[0, 0]

    def step(self, e_forge, t, dt=1.0 / FPS):
        self.T = self.T + dt * 6.0 * self.M * (0.30 + 0.70 * float(e_forge))
        self.vy = self.vy - dt * 9.0 * self.T                     # buoyancy (−y is up)
        cx, cy = self._curl(t)
        amp = 0.5 + 1.5 * self.T                                  # turbulence only where hot
        self.vx = self.vx + dt * cx * amp * 44.0   # px/s, measured: 22 gave TI 5.4, target 8-15
        self.vy = self.vy + dt * cy * amp * 44.0
        self.T = self._advect(self.T, dt)
        vx_new = self._advect(self.vx, dt)
        self.vy = self._advect(self.vy, dt)
        self.vx = vx_new
        self.T = self.T * math.exp(-1.8 * dt)
        self.vx = self.vx * math.exp(-0.9 * dt)
        self.vy = self.vy * math.exp(-0.9 * dt)
        k = torch.ones(1, 1, 3, 3, device=self.dev) / 9.0
        self.T = F.conv2d(self.T[None, None], k, padding=1)[0, 0].clamp(0, 1.4)


def incandescence(T, device):
    """Temperature -> emitted linear RGB, via steel's real colour ladder."""
    Tc = 480 + 900 * T.clamp(0, 1) ** 0.85
    c = torch.tensor(LADDER_C, device=device)
    rgb = torch.tensor(LADDER_RGB, device=device)
    idx = torch.searchsorted(c, Tc.flatten().contiguous()).clamp(1, len(c) - 1)
    lo, hi = idx - 1, idx
    w = ((Tc.flatten() - c[lo]) / (c[hi] - c[lo])).clamp(0, 1)[:, None]
    out = rgb[lo] * (1 - w) + rgb[hi] * w
    return out.reshape(*T.shape, 3).permute(2, 0, 1)


def bloom(x, device):
    """4-level pyramid bloom. In torch, so the gain is genuinely per-frame — a filtergraph
    could only step it at discrete sendcmd timestamps."""
    out = torch.zeros_like(x)
    cur = x[None]
    for w in (0.50, 0.30, 0.15, 0.05):
        cur = F.avg_pool2d(cur, 2) if cur.shape[-1] > 8 else cur
        up = F.interpolate(cur, size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        out = out + w * up
    return out


# ── Stage C: composite and render ───────────────────────────────────────────────────────
class Embers:
    """A fixed-budget particle system riding the same velocity field as the heat."""

    def __init__(self, n, H, W, device, gen):
        self.n, self.H, self.W, self.dev = n, H, W, device
        self.pos = torch.zeros(n, 2, device=device)
        self.vel = torch.zeros(n, 2, device=device)
        self.age = torch.full((n,), 1e9, device=device)     # all dead at t=0
        self.life = torch.ones(n, device=device)
        self.gen = gen

    def step(self, T, e_onset, dt=1.0 / FPS):
        dead = (self.age >= self.life).nonzero(as_tuple=True)[0]
        want = int(math.ceil(120 * float(e_onset)))
        if want and len(dead):
            k = min(want, len(dead))
            idx = dead[:k]
            flat = (T ** 3).flatten()                        # spawn from the hottest pixels
            tot = float(flat.sum())
            if tot > 1e-6:
                pick = torch.multinomial(flat / tot, k, replacement=True, generator=self.gen)
                self.pos[idx, 0] = (pick % T.shape[1]).float() * (self.W / T.shape[1])
                self.pos[idx, 1] = (pick // T.shape[1]).float() * (self.H / T.shape[0])
                self.vel[idx, 0] = torch.randn(k, generator=self.gen, device=self.dev) * 14.0
                self.vel[idx, 1] = -30.0 - torch.rand(k, generator=self.gen, device=self.dev) * 55.0
                self.age[idx] = 0.0
                self.life[idx] = 1.2 + torch.rand(k, generator=self.gen, device=self.dev) * 2.3

        live = self.age < self.life
        self.vel[live, 1] -= dt * 22.0                       # buoyant rise
        self.vel[live] += torch.randn(int(live.sum()), 2, generator=self.gen,
                                      device=self.dev) * 3.0 * dt * FPS
        self.pos[live] += self.vel[live] * dt
        self.age[live] += dt
        off = ((self.pos[:, 0] < 0) | (self.pos[:, 0] >= self.W) |
               (self.pos[:, 1] < 0) | (self.pos[:, 1] >= self.H))
        self.age[off] = 1e9

    def splat(self, device):
        """Additive 3x3 splat into a linear-light buffer, coloured by the same steel ladder."""
        buf = torch.zeros(3, self.H, self.W, device=device)
        live = (self.age < self.life).nonzero(as_tuple=True)[0]
        if not len(live):
            return buf
        frac = (self.age[live] / self.life[live]).clamp(0, 1)
        Tc = 1250 - 700 * frac                               # embers cool as they fly
        c = torch.tensor(LADDER_C, device=device)
        rgb = torch.tensor(LADDER_RGB, device=device)
        idx = torch.searchsorted(c, Tc.contiguous()).clamp(1, len(c) - 1)
        lo, hi = idx - 1, idx
        w = ((Tc - c[lo]) / (c[hi] - c[lo])).clamp(0, 1)[:, None]
        col = (rgb[lo] * (1 - w) + rgb[hi] * w).T             # (3, n_live)
        alpha = (1 - frac) ** 1.5
        xi = self.pos[live, 0].long().clamp(1, self.W - 2)
        yi = self.pos[live, 1].long().clamp(1, self.H - 2)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = 1.0 if (dx == 0 and dy == 0) else (0.5 if dx == 0 or dy == 0 else 0.25)
                lin = (yi + dy) * self.W + (xi + dx)
                for c in range(3):
                    buf[c].view(-1).index_put_((lin,), col[c] * alpha * w * 6.5, accumulate=True)
        return buf


def render(image, audio, out, seconds=180.0, preview=False, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    torch.manual_seed(SEED)
    W, H = (640, 360) if preview else (1920, 1080)
    n_frames = int(seconds * FPS)

    raw = Path("/tmp/_hero.rgb")
    _sh(f"ffmpeg -v error -i '{image}' -vf "
        f"\"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}\" "
        f"-f rawvideo -pix_fmt rgb24 '{raw}' -y")
    hero = torch.from_numpy(np.fromfile(raw, dtype=np.uint8).reshape(H, W, 3).copy())
    hero = (hero.permute(2, 0, 1).float() / 255.0).to(device)
    hero_lin = srgb_to_linear(hero)

    # Depth: a cheap luminance-and-position prior. Depth-Anything would be better and is the
    # documented upgrade, but a 25 MB model download is a dependency the notebook does not need
    # for a ±29 px parallax — at that disparity the depth only has to be roughly right.
    luma = (0.2126 * hero_lin[0] + 0.7152 * hero_lin[1] + 0.0722 * hero_lin[2])
    yy = torch.linspace(0, 1, H, device=device)[:, None].expand(H, W)
    depth = (0.6 * (luma / (luma.max() + 1e-9)) + 0.4 * yy).clamp(0, 1)
    k = torch.ones(1, 1, 15, 15, device=device) / 225.0
    depth = F.conv2d(depth[None, None], k, padding=7)[0, 0]

    M = metal_mask(hero_lin, depth)
    print(f"mask covers {100*float((M>0.05).float().mean()):.1f}% of frame", flush=True)

    env = envelopes(audio, n_frames, seconds=seconds)
    gh, gw = H // 2, W // 2
    field = HeatField(gh, gw, M, device, gen)
    for _ in range(240):                                   # warm-up: open on an established plume
        field.step(float(np.median(env["forge"])), 0.0)

    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    dmed = depth.median()
    grain_tile = torch.randn(512, 512, generator=gen).to(device)
    embers = Embers(9000, H, W, device, gen)

    proc = subprocess.Popen(
        f"ffmpeg -v error -f rawvideo -pix_fmt rgb24 -s {W}x{H} -r {FPS} -i - "
        f"-i '{audio}' -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p "
        f"-c:a aac -b:a 256k -shortest '{out}' -y",
        shell=True, stdin=subprocess.PIPE)

    for n in range(n_frames):
        t = n / FPS
        field.step(env["forge"][n], t)
        T = F.interpolate(field.T[None, None], size=(H, W), mode="bilinear",
                          align_corners=False)[0, 0]

        # C1 parallax — one full sine over the whole piece: starts and ends at zero, no seam,
        # and it is a perspective shift rather than a zoom, which is what a zoom cannot fake.
        cx = 0.015 * W * math.sin(2 * math.pi * t / seconds)
        cy = 0.006 * H * math.sin(4 * math.pi * t / seconds + math.pi / 3)
        dd = depth - dmed
        gx = ((xs + cx * dd) / (W - 1)) * 2 - 1
        gy = ((ys + cy * dd) / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)[None]
        base = F.grid_sample(hero_lin[None], grid, mode="bilinear",
                             padding_mode="border", align_corners=False)[0]
        Mw = F.grid_sample(M[None, None], grid, mode="bilinear",
                           padding_mode="border", align_corners=False)[0, 0]

        # C3 incandescence — superlinear in T so cool regions stay genuinely dark and the
        # flash budget is spent on the hot core instead of smeared over the whole frame
        E = incandescence(T, device) * (T ** 1.6)[None] * (0.9 + 1.1 * float(env["loud"][n]))
        breath = 0.85 + 0.30 * float(env["forge"][n])
        lit = base * breath + E * Mw[None]

        embers.step(field.T, env["onset"][n])
        lit = lit + embers.splat(device)

        # C4 bloom — the audio reactivity that reads from across the room
        B = (0.2126 * (E[0] * Mw) + 0.7152 * (E[1] * Mw) + 0.0722 * (E[2] * Mw) - 0.25).clamp(min=0)
        lit = lit + (0.6 + 1.2 * float(env["forge"][n])) * bloom(B[None].expand(3, H, W), device)

        # C6 grain, seeded and offset per frame — breaks banding in the dark background
        off = (n * 37) % 512
        g = torch.roll(grain_tile, shifts=(off, off * 3 % 512), dims=(0, 1))[:H % 512 or 512, :]
        g = F.interpolate(g[None, None], size=(H, W), mode="nearest")[0, 0]
        lit = lit + 0.006 * g[None]

        frame = (linear_to_srgb(lit).clamp(0, 1) * 255).to(torch.uint8)
        proc.stdin.write(frame.permute(1, 2, 0).cpu().numpy().tobytes())
        if n % (FPS * 20) == 0:
            print(f"  {t:6.1f}s  T_max {float(T.max()):.2f}  forge {env['forge'][n]:.2f}",
                  flush=True)

    proc.stdin.close()
    proc.wait()
    print(f"wrote {out}", flush=True)
    return out


def verify(video, audio, seconds=None):
    """Prove the claims: no cuts, no flashes, real motion, and glow that tracks the audio."""
    import librosa
    W = H = 256
    raw = _sh(f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,r_frame_rate "
              f"-of default=nw=1 '{video}'")
    print(raw.strip())
    tmp = "/tmp/_v.rgb"
    _sh(f"ffmpeg -v error -i '{video}' -vf scale={W}:{H} -f rawvideo -pix_fmt rgb24 '{tmp}' -y")
    a = np.fromfile(tmp, dtype=np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255.0
    luma = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    mean = luma.reshape(len(a), -1).mean(1)
    d = np.abs(np.diff(mean))
    ti = np.abs(np.diff(luma, axis=0)).reshape(len(a) - 1, -1).mean(1) * 255
    gy, gx = np.gradient(luma[0])
    si = float(np.sqrt(gx ** 2 + gy ** 2).std() * 255)
    print(f"  frames {len(a)} · cuts (Δmean>0.10) {int((d>0.10).sum())} "
          f"· flashes/s {(d>0.10).sum()/(len(a)/FPS):.2f}")
    print(f"  TI mean {ti.mean():.2f} p95 {np.percentile(ti,95):.2f}  (want 8-15: real motion)")
    print(f"  SI {si:.1f} · max Δmean {d.max():.4f}")
    n = len(mean)
    env = envelopes(audio, n, seconds=n / FPS)["forge"]
    r_drive = float(np.corrcoef(mean, env)[0, 1])

    y, sr = librosa.load(str(audio), sr=48000, mono=True, duration=n / FPS)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [20, 160], btype="band", fs=sr, output="sos")
    raw = librosa.feature.rms(y=sosfilt(sos, y), frame_length=4000, hop_length=sr // FPS)[0]
    raw = np.interp(np.linspace(0, len(raw) - 1, n), np.arange(len(raw)), raw)
    r_raw = float(np.corrcoef(mean, raw)[0, 1])
    lags = range(-int(FPS * 1.5), int(FPS * 1.5) + 1)
    best = max(lags, key=lambda L: abs(np.corrcoef(mean[max(0, L):n + min(0, L)],
                                                   raw[max(0, -L):n - max(0, L)])[0, 1]))
    r_best = float(np.corrcoef(mean[max(0, best):n + min(0, best)],
                               raw[max(0, -best):n - max(0, best)])[0, 1])
    print(f"  r(luma, DRIVE envelope)      = {r_drive:.3f}  (want >=0.6: the pipeline couples)")
    print(f"  r(luma, raw 20-160 Hz)       = {r_raw:.3f} · best {r_best:.3f} at {best/FPS:+.2f}s lag")
    print(f"     (heat physically lags the bellows; a positive best-lag is correct behaviour)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--image", required=True); r.add_argument("--audio", required=True)
    r.add_argument("--out", required=True); r.add_argument("--seconds", type=float, default=180.0)
    r.add_argument("--preview", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--video", required=True); v.add_argument("--audio", required=True)
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.image, a.audio, a.out, a.seconds, a.preview)
    else:
        verify(a.video, a.audio)
