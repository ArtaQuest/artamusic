#!/usr/bin/env python3
"""Hold the rigid subject perfectly still while the fire keeps moving.

An image-to-video model animates EVERYTHING it was given. Asked for heat haze and embers it also
re-draws the sword each frame: measured on the shipped cover, the blade drifted 18 px, its
silhouette swung 15%, and its own pixels moved 5x more than the rest of the picture. No prompt
fixes that — "the blade stays perfectly still" was already in the prompt.

So the subject is not asked to hold still, it is MADE to: its pixels come from the still
photograph, every frame, which is stillness by construction rather than by hope.

The trap is that a hard freeze looks dead. Polished steel beside a fire is mostly REFLECTED
FIRELIGHT, so a plate frozen at constant brightness reads as a cardboard cut-out pasted over a
living scene. The fix is to freeze the GEOMETRY and keep the LIGHT: each frame the frozen plate is
scaled by how bright the model made that region in that frame, so the blade still pulses with the
coals — it just never changes shape or place.

    freeze(frames, still, mask) -> frames with the masked region frozen and re-lit
"""
import numpy as np


def feather(mask, radius=3):
    """Soft alpha from a hard mask — a box blur, so no scipy. Hides the seam where fire meets steel."""
    a = mask.astype(np.float32)
    k = 2 * radius + 1
    # integral image with a zero row/col so every window has four corners inside the array
    acc = np.zeros((a.shape[0] + 2 * radius + 1, a.shape[1] + 2 * radius + 1), dtype=np.float64)
    acc[1:, 1:] = np.pad(a, radius, mode="edge").cumsum(0).cumsum(1)
    H, W = a.shape
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    out = (acc[ys + k, xs + k] - acc[ys, xs + k] - acc[ys + k, xs] + acc[ys, xs]) / (k * k)
    return np.clip(out.astype(np.float32), 0, 1)


def freeze(frames, still, mask, radius=3, relight=True, gain_clip=(0.75, 1.35)):
    """frames: (N,H,W,3) uint8 · still: (H,W,3) uint8 · mask: (H,W) bool — True where it must NOT move."""
    frames = np.asarray(frames)
    still = np.asarray(still).astype(np.float32)
    alpha = feather(np.asarray(mask, dtype=bool), radius)[..., None]
    m = np.asarray(mask, dtype=bool)
    ref = float(still[..., :3].mean(2)[m].mean()) if m.any() else 1.0
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        fl = f.astype(np.float32)
        plate = still
        if relight and m.any():
            # how bright did the model make this region in THIS frame? that is the firelight.
            gain = float(fl.mean(2)[m].mean()) / max(ref, 1e-6)
            plate = still * float(np.clip(gain, *gain_clip))
        out[i] = np.clip(alpha * plate + (1 - alpha) * fl, 0, 255).astype(np.uint8)
    return out


def fire_mask(still, warm_pct=88.0, grow=9, plume=90):
    """The only part of a forge photograph that should move: the coals, and the air above them.

    Built the other way round from the obvious one. Masking the SUBJECT means finding every rigid
    thing — blade, guard, grip, anvil, bench, wall — and any pixel missed keeps animating: a mask
    over just the bright steel left the sword's own silhouette swinging 13%. Masking the FIRE
    instead needs only one thing to be right, and everything not named stays frozen, which is what
    a locked-off camera means.

    warm = where the picture is orange (coals), grown outward, then extended UPWARD for the plume:
    smoke and sparks rise, so the moving region is a column above the fire, not a disc around it.
    """
    a = np.asarray(still).astype(np.float32)
    warm = a[..., 0] - a[..., 2]                       # red minus blue: fire, not steel
    m = warm > np.percentile(warm, warm_pct)
    m = feather(m, grow) > 0.08                        # grow outward
    up = m.copy()                                      # and upward, for smoke
    for dy in range(1, plume):
        up[:-dy] |= m[dy:]
    return feather(up, grow) > 0.05


def _label(binary):
    """Connected components, 8-connected, in numpy — a union-find over the True pixels.

    scipy.ndimage would do this in one call and is present on Kaggle, but it is absent from this
    laptop, and a mask this load-bearing has to be testable where it is written.
    """
    b = np.asarray(binary, dtype=bool)
    h, w = b.shape
    lab = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, c):
        ra, rc = find(a), find(c)
        if ra != rc:
            parent[max(ra, rc)] = min(ra, rc)

    nxt = 1
    for y in range(h):
        row, prev = b[y], b[y - 1] if y else None
        for x in np.nonzero(row)[0]:
            neigh = []
            if x and row[x - 1]:
                neigh.append(lab[y, x - 1])
            if prev is not None:
                for dx in (-1, 0, 1):
                    xx = x + dx
                    if 0 <= xx < w and prev[xx]:
                        neigh.append(lab[y - 1, xx])
            if neigh:
                lab[y, x] = min(neigh)
                for v in neigh:
                    union(lab[y, x], v)
            else:
                lab[y, x] = nxt
                parent.append(nxt)
                nxt += 1
    roots = {}
    out = np.zeros_like(lab)
    for y in range(h):
        for x in np.nonzero(lab[y])[0]:
            r = find(lab[y, x])
            if r not in roots:
                roots[r] = len(roots) + 1
            out[y, x] = roots[r]
    return out, len(roots)


def _fill_holes(m):
    """Fill enclosed holes: flood the background in from the border, whatever is left is inside."""
    b = np.asarray(m, dtype=bool)
    free = ~b
    lab, n = _label(free)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    outside = np.isin(lab, list(border))
    return b | (free & ~outside)


def _largest_elongated(binary, min_ratio=3.0):
    """The single connected blob that is long and thin — the blade, not a coal.

    Brightness and chroma alone cannot separate polished steel from a white-hot coal: both are
    bright, and the hottest coal cores are nearly achromatic too. What no coal is, is LONG. So take
    connected components, and keep the one whose principal axes are most unequal, preferring size
    among the elongated ones. (Measured on the shipped still, a colour-only mask claimed 20% of the
    frame including half the fire; this returns the sword.)
    """
    lab, n = _label(binary)
    if n == 0:
        return binary
    best, best_score = None, -1.0
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < 200:
            continue
        ys, xs = np.nonzero(m)
        y, x = ys - ys.mean(), xs - xs.mean()
        cov = np.cov(np.stack([y, x]))
        ev = np.linalg.eigvalsh(cov)
        ratio = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
        if ratio < min_ratio:
            continue
        score = area * ratio
        if score > best_score:
            best, best_score = m, score
    return _fill_holes(best) if best is not None else binary


def steel_mask(still, bright_pct=80.0, warm_max_pct=55.0, grow=7):
    """The rigid subject: polished steel is BRIGHT and NOT ORANGE.

    Brightness alone cannot find it — glowing coals are just as bright, which is how a mask meant
    for the blade ended up covering half the fire. Steel reflects the fire's light without taking
    its colour, so red-minus-blue separates them where luminance cannot.
    """
    a = np.asarray(still).astype(np.float32)
    lum = a.mean(2)
    warm = a[..., 0] - a[..., 2]
    m = (lum > np.percentile(lum, bright_pct)) & (warm < np.percentile(warm, warm_max_pct))
    m = _largest_elongated(m)                      # the sword is the long one; coals are many
    return feather(m, grow) > 0.10


def extend_along_axis(mask, still, frac=0.5, warm_max_pct=60.0):
    """Grow the blade mask along the sword's own axis, so the grip and pommel are held too.

    Colour finds the polished blade and nothing else: the leather grip and brass pommel are dark
    and warm, indistinguishable from the anvil by any threshold. But they lie ON THE BLADE'S LINE,
    so a dilation along the principal axis reaches them. The same dilation would also run off the
    tip into the coals, so the result is intersected with "not orange" — which the grip passes and
    fire does not.
    """
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return m
    y, x = ys - ys.mean(), xs - xs.mean()
    cov = np.cov(np.stack([y, x]))
    ev, evec = np.linalg.eigh(cov)
    axis = evec[:, int(np.argmax(ev))]                       # (dy, dx) unit vector
    length = 4.0 * float(np.sqrt(max(ev.max(), 1e-9)))       # ~ the blade's own length
    steps = int(frac * length)
    out = m.copy()
    for t in range(1, max(1, steps)):
        dy, dx = int(round(axis[0] * t)), int(round(axis[1] * t))
        for sy, sx in ((dy, dx), (-dy, -dx)):
            sh = np.roll(np.roll(m, sy, axis=0), sx, axis=1)
            if sy > 0:
                sh[:sy] = False
            elif sy < 0:
                sh[sy:] = False
            if sx > 0:
                sh[:, :sx] = False
            elif sx < 0:
                sh[:, sx:] = False
            out |= sh
    a = np.asarray(still).astype(np.float32)
    not_warm = (a[..., 0] - a[..., 2]) < np.percentile(a[..., 0] - a[..., 2], warm_max_pct)
    return (out & not_warm) | m


def moving_mask(still, **kw):
    """What may move: the fire and the air above it, minus the rigid subject standing in front."""
    fire = fire_mask(still, **{k: v for k, v in kw.items() if k in ("warm_pct", "grow", "plume")})
    steel = steel_mask(still)
    return fire & ~feather(steel, 4).astype(bool)


def blur(img, sigma):
    """Gaussian-ish blur by three box passes (central limit), numpy only.

    No cv2 anywhere in this file on purpose: it is one more pin for a run whose whole claim is that
    a stranger can re-run it, and every operation here is four lines of numpy.
    """
    a = np.asarray(img, dtype=np.float32)
    single = a.ndim == 2
    if single:
        a = a[..., None]
    r = max(1, int(round(sigma * 0.9)))
    k = 2 * r + 1
    for _ in range(3):
        pad = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
        acc = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1, pad.shape[2]), dtype=np.float64)
        acc[1:, 1:] = pad.cumsum(0).cumsum(1)
        ys = np.arange(a.shape[0])[:, None]
        xs = np.arange(a.shape[1])[None, :]
        a = ((acc[ys + k, xs + k] - acc[ys, xs + k] - acc[ys + k, xs] + acc[ys, xs]) / (k * k)).astype(np.float32)
    return a[..., 0] if single else a


def clean_plate(still, subject_mask, grow=10, levels=4, iters=60):
    """The still with the subject painted out — coals where the blade was.

    This is what makes the composite invisible rather than merely rigid. Animating the ORIGINAL
    still hands the model a sword to deform, and then the composite's feathered ring shows frozen
    coals beside moving ones. Animating a plate with no sword in it leaves nothing to deform, and
    the ring shows coals moving like all the others.

    The fill is a coarse-to-fine heat diffusion — deterministic, weightless, and only ever seen
    through a few pixels of feather, which is why it does not need to be a second neural model.
    """
    a = np.asarray(still).astype(np.float32)
    hole = feather(np.asarray(subject_mask, dtype=bool), grow) > 0.02
    small_a, small_m = a, hole
    stack = []
    for _ in range(levels):
        stack.append((small_a, small_m))
        small_a = small_a[::2, ::2]
        small_m = small_m[::2, ::2]
    fill = small_a.copy()
    for lvl in range(levels - 1, -1, -1):
        base, m = stack[lvl]
        if fill.shape[:2] != base.shape[:2]:
            fill = np.repeat(np.repeat(fill, 2, 0), 2, 1)[:base.shape[0], :base.shape[1]]
        cur = np.where(m[..., None], fill, base)
        for _ in range(iters):
            sm = blur(cur, 2.0)
            cur = np.where(m[..., None], sm, base)
        fill = cur
    return np.clip(fill, 0, 255).astype(np.uint8)


def relight_field(gen_t, gen_0, coal_mask, sigma=None, clip=(0.6, 1.8)):
    """A smooth, per-channel gain field: how much brighter and warmer this frame's fire is.

    Estimated ONLY from the coals and extrapolated inward by normalised convolution, so none of the
    generated frame's GEOMETRY can leak into the frozen subject — only its light. Per channel,
    because a fire flickers in colour temperature as much as in brightness, and one grey scalar
    makes frozen steel pulse grey.
    """
    h = np.asarray(gen_t).shape[0]
    sigma = sigma or max(4.0, 0.03 * h)
    w = np.asarray(coal_mask, dtype=np.float32)

    def field(img):
        num = blur(np.asarray(img, dtype=np.float32) * w[..., None], sigma)
        den = blur(w, sigma)[..., None] + 1e-6
        return num / den

    return np.clip(field(gen_t) / (field(gen_0) + 1e-6), *clip).astype(np.float32)


def freeze_lit(frames, still, subject_mask, coal_mask, radius=3, spec=0.8):
    """Composite the subject back over the generated frames: frozen in place, lit by their fire."""
    frames = np.asarray(frames)
    S = np.asarray(still).astype(np.float32)
    alpha = feather(np.asarray(subject_mask, dtype=bool), radius)[..., None]
    lum = S.mean(2)
    hi = np.clip(lum - blur(lum, 3.0), 0, None)[..., None]      # the still's own specular map
    out = np.empty_like(frames)
    g0 = frames[0].astype(np.float32)
    for i, f in enumerate(frames):
        fl = f.astype(np.float32)
        gain = relight_field(fl, g0, coal_mask)
        plate = S * gain
        if spec:
            pulse = spec * np.clip(gain.mean(2, keepdims=True) - 1.0, 0, None) * hi
            plate = 255.0 - (255.0 - plate) * (1.0 - np.clip(pulse / 255.0, 0, 0.9))
        out[i] = np.clip(alpha * plate + (1 - alpha) * fl, 0, 255).astype(np.uint8)
    return out
