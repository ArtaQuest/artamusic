# GATE 0 — the five-minute test that decides whether to spend anything on I2V.
#
# The operator rejected a procedural animation as unrealistic and asked for SOTA models. The
# research chose LTX-Video 0.9.8 2B distilled with diffusers' LTXI2VLongMultiPromptPipeline,
# which produces ONE continuous latent for the whole 180 s — no chaining, no cuts, no dissolves.
#
# But before any of that: the cover is a bed of live COALS and a spark field. LTX's VAE
# compresses 32x spatially. If the tokeniser destroys the sparks on the round trip, nothing
# downstream can bring them back — not the DiT, not the x2 spatial upscaler, not anchoring.
# So this loads ONLY the VAE, encodes the still and decodes it straight back at five base
# resolutions, and MEASURES what survived:
#
#   - spark count: connected components brighter than local median + 3 sigma in the lower third
#   - high-pass energy above 0.25 cyc/px in the coal-bed region
#   - max gradient across the fire-vein edge on the blade
#
# PASS = the smallest bucket retaining >=70% of sparks and >=60% of high-pass energy.
# If nothing clears it at or below 896 wide, this reports that and we do NOT spend a session.
import hashlib, json, os, subprocess, sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/kaggle/temp/hf"
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working")

SINGLE_FILE = ("https://huggingface.co/Lightricks/LTX-Video/resolve/"
               "8984fa25007f376c1a299016d0957a37a2f797bb/ltxv-2b-0.9.8-distilled.safetensors")

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1200:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1200:], flush=True)
    return r.returncode

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0

# deps BEFORE torch, so the cu126 line wins on Pascal
sh("pip install -q 'diffusers==0.39.0' 'transformers>=4.51.0,<4.58.0' accelerate "
   "safetensors imageio imageio-ffmpeg scipy hf_transfer 2>&1 | tail -2")
if PASCAL:
    sh("pip install -q torch==2.7.1 torchvision==0.22.1 --index-url "
       "https://download.pytorch.org/whl/cu126 2>&1 | tail -1")

import numpy as np
import torch
import diffusers
print(f"torch {torch.__version__} · diffusers {diffusers.__version__} · cap {CAP}", flush=True)
assert diffusers.__version__.startswith("0.39"), f"need diffusers 0.39.0, got {diffusers.__version__}"

# ── the still ────────────────────────────────────────────────────────────────────────────
import glob as _g
src = sorted(_g.glob("/kaggle/input/**/cover.png", recursive=True)) or \
      sorted(_g.glob("/kaggle/input/**/*.png", recursive=True))
assert src, "cover image not mounted"
STILL = src[0]
print("still:", STILL, flush=True)

from diffusers import AutoencoderKLLTXVideo
print("loading VAE from the single file (bundles both DiT and VAE tensors)...", flush=True)
vae = AutoencoderKLLTXVideo.from_single_file(SINGLE_FILE, torch_dtype=torch.bfloat16).to("cuda")
vae.eval()
print("VAE loaded", flush=True)

def load_still(w, h):
    raw = "/tmp/_still.rgb"
    sh(f"ffmpeg -v error -i '{STILL}' -vf "
       f"\"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}\" "
       f"-f rawvideo -pix_fmt rgb24 {raw} -y", quiet=True)
    a = np.fromfile(raw, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
    return a

def sparks(img):
    """Count small bright blobs in the lower third — the coal bed's live embers."""
    from scipy import ndimage
    g = img[..., 0] * 0.5 + img[..., 1] * 0.35 + img[..., 2] * 0.15
    band = g[int(g.shape[0] * 0.6):]
    med = ndimage.median_filter(band, size=9)
    hot = band > (med + 3.0 * band.std())
    lab, n = ndimage.label(hot)
    sizes = ndimage.sum(hot, lab, range(1, n + 1))
    return int((sizes >= 2).sum())

def highpass_energy(img):
    g = img.mean(-1)
    band = g[int(g.shape[0] * 0.6):]
    F = np.fft.rfft2(band)
    h, w = band.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    keep = np.sqrt(fy ** 2 + fx ** 2) > 0.25
    return float((np.abs(F) ** 2)[keep].sum())

rows = []
for w, h in [(512, 352), (640, 448), (704, 480), (768, 512), (896, 608)]:
    a = load_still(w, h)
    x = torch.from_numpy(a).permute(2, 0, 1)[None, :, None].to("cuda", torch.bfloat16) * 2 - 1
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.sample()
        back = vae.decode(lat).sample
    b = ((back[0, :, 0].float().permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    s0, s1 = sparks(a), sparks(b)
    e0, e1 = highpass_energy(a), highpass_energy(b)
    row = {"w": w, "h": h,
           "sparks_orig": s0, "sparks_rt": s1,
           "spark_keep": round(s1 / max(s0, 1), 3),
           "hp_keep": round(e1 / max(e0, 1e-9), 3),
           "latent_shape": list(lat.shape)}
    row["pass"] = bool(row["spark_keep"] >= 0.70 and row["hp_keep"] >= 0.60)
    rows.append(row)
    print(f"  {w}x{h}: sparks {s0}->{s1} ({row['spark_keep']:.0%}) · "
          f"high-pass {row['hp_keep']:.0%} · {'PASS' if row['pass'] else 'fail'}", flush=True)

(WORK / "gate0.json").write_text(json.dumps(rows, indent=2))
ok = [r for r in rows if r["pass"]]
print("\n=== GATE 0 VERDICT ===", flush=True)
if ok:
    best = min(ok, key=lambda r: r["w"])
    print(f"PASS at {best['w']}x{best['h']} — the tokeniser preserves the coal bed. "
          f"Build the 180 s run at this base resolution.", flush=True)
else:
    print("FAIL at every resolution up to 896 wide. LTX's VAE destroys the spark field before "
          "the model runs; no chaining, anchoring or upscaling recovers it. Do NOT spend a "
          "session — report and choose another approach.", flush=True)
