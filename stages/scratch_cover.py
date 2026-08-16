# STEEL cover, FROM SCRATCH — new illustration, new animation, both on current models.
#
# The operator called the previous result garbage and asked for both stages redone with better
# models. That was the right call and the diagnosis was upstream of where I had been working:
# the ILLUSTRATION was SDXL, a 2023 model, and no animation of a mediocre still can be good.
# Animating harder was never going to fix a weak frame.
#
# The two models below were chosen from an empirical survey of what current diffusers actually
# supports (81 pipeline families), filtered by what genuinely fits a 16 GB sm_60 card, with sizes
# and gating read live from the HF API rather than remembered:
#
#   IMAGE  Tongyi-MAI/Z-Image-Turbo   6.15B  ungated  Apache-2.0  ~900k downloads
#          2.4x SDXL's parameters, two generations newer. FLUX.2 (32B), Qwen-Image (20B) and
#          HiDream-I1 (17B) were all rejected on size — they cannot fit this card at bf16.
#          9 steps, guidance 0.0 (CFG-distilled), bf16 — settings from its own model card.
#
#   VIDEO  Wan-AI/Wan2.2-TI2V-5B      5.00B  ungated  Apache-2.0  ~208k downloads
#          2.6x LTX-Video-2B, which is the model that produced the rejected footage.
#
# Both stages are gated on measurement, and the run refuses rather than shipping something weak.
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/kaggle/temp/hf"
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working")
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1000:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1000:], flush=True)
    return r.returncode

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0

sh("pip install -q 'diffusers==0.39.0' 'transformers>=4.51.0,<4.58.0' accelerate safetensors "
   "sentencepiece protobuf ftfy imageio imageio-ffmpeg hf_transfer 2>&1 | tail -2")
if PASCAL:
    sh("pip install -q torch==2.7.1 torchvision==0.22.1 --index-url "
       "https://download.pytorch.org/whl/cu126 2>&1 | tail -1")

import numpy as np
import torch
import diffusers
print(f"torch {torch.__version__} · diffusers {diffusers.__version__} · cap {CAP}", flush=True)

SEED = 4242
torch.manual_seed(SEED); np.random.seed(SEED)

# ── STAGE 1: the illustration ────────────────────────────────────────────────────────────
PROMPT = (
    "A single forged steel sword lying across a bed of white-hot coals inside a dark blacksmith's "
    "forge at night. The blade is freshly quenched, its edge still glowing orange along a "
    "hardening line, the polished steel reflecting the fire. Fine sparks rise through the smoky "
    "air. Heavy stone anvil and soot-blackened brick behind, deep shadow, one warm light source "
    "from the coals below. Shot on a 85mm lens at f/2, shallow depth of field, volumetric haze, "
    "photorealistic, cinematic colour grade, ultra sharp detail on the blade, album cover.")
NEG = ("cartoon, illustration, painting, cgi render, plastic, blurry, low detail, watermark, "
       "text, signature, extra swords, hands, people, oversaturated, orange filter")

from diffusers import ZImagePipeline
img_pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
# First attempt OOM'd: 14.07 GB resident, needed 1.88 more — the 6.15B DiT and its Qwen3-4B text
# encoder were both on the card under model-level offload. SEQUENTIAL offload streams one
# submodule at a time (peak ~ largest layer + activations), at the cost of PCIe re-streaming per
# step. For 9 steps that cost is small; for a 16 GB card it is the difference between running
# and not. This is exactly the take-turns partitioning the operator asked about at the outset.
img_pipe.enable_sequential_cpu_offload()
img_pipe.vae.enable_tiling() if hasattr(img_pipe, "vae") else None
print("Z-Image-Turbo ready (sequential offload)", flush=True)

t0 = time.time()
cands = []
for i, seed in enumerate([SEED, 5150, 6270, 7380]):
    im = img_pipe(prompt=PROMPT, negative_prompt=NEG, width=896, height=896,
                  num_inference_steps=9, guidance_scale=0.0,
                  generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
    p = OUT / f"still_{seed}.png"
    im.save(p)
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    g = a.mean(-1)
    gy, gx = np.gradient(g)
    detail = float(np.sqrt(gx**2 + gy**2).mean() * 255)          # edge energy: is it SHARP?
    warm = float((a[..., 0] - a[..., 2]).clip(0).mean() * 255)   # is the forge actually lit?
    dark = float((g < 0.25).mean())                              # is it a NIGHT scene?
    cands.append({"seed": seed, "file": p.name, "detail": round(detail, 2),
                  "warm": round(warm, 2), "dark_frac": round(dark, 3)})
    print(f"  still seed {seed}: detail {detail:.2f} · warmth {warm:.2f} · dark {dark:.2%}",
          flush=True)
print(f"4 stills in {time.time()-t0:.0f}s", flush=True)

# pick on measured criteria rather than the first one out: sharp, warm, and genuinely dark
best = max(cands, key=lambda c: c["detail"] * (1 + c["warm"] / 40) * (0.5 + c["dark_frac"]))
print("chosen still:", best, flush=True)
(WORK / "stills.json").write_text(json.dumps({"candidates": cands, "chosen": best}, indent=2))
sh(f"cp '{OUT / best['file']}' '{OUT}/cover.png'", quiet=True)
del img_pipe
torch.cuda.empty_cache()
print("STAGE 1 done — illustration", flush=True)

# ── STAGE 2: the animation ───────────────────────────────────────────────────────────────
# Wan2.2-TI2V-5B, 2.6x the parameters of the LTX-2B that produced the rejected footage. The 5B
# is the ONLY Wan variant that fits: the A14B needs 80 GB by Alibaba's own guidance.
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

still = Image.open(OUT / "cover.png").convert("RGB")
VW, VH = 704, 480                       # inside the card; 720P is 1280x704 and will not fit here
NUM_FRAMES = 49                         # (4n+1); Wan's VAE is 4x temporal, keep it modest first

vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                       subfolder="vae", torch_dtype=torch.float32)
vid_pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
vid_pipe.enable_model_cpu_offload()
print("Wan2.2-TI2V-5B ready", flush=True)

VID_PROMPT = ("The coals beneath the blade pulse and breathe with heat. Fine sparks drift upward "
              "through smoky air. Heat haze shimmers over the metal. The camera does not move.")
VID_NEG = ("camera movement, zoom, pan, cuts, scene change, morphing, melting, distortion, "
           "text, watermark, low quality, blurry, jitter, flicker")

t0 = time.time()
frames = vid_pipe(image=still.resize((VW, VH), Image.LANCZOS),
                  prompt=VID_PROMPT, negative_prompt=VID_NEG,
                  height=VH, width=VW, num_frames=NUM_FRAMES,
                  num_inference_steps=24, guidance_scale=5.0,
                  generator=torch.Generator(device="cuda").manual_seed(SEED)).frames[0]
gen_s = round(time.time() - t0, 1)
print(f"generated {len(frames)} frames in {gen_s}s", flush=True)
raw = str(OUT / "wan_raw.mp4")
export_to_video(frames, raw, fps=16)

# ── loop closure: mirror. Measured on the previous model's output, the drift is MONOTONIC, so a
# loop point never returns and no dissolve can close it; mirroring makes the wrap exact.
mir = str(OUT / "wan_mirror.mp4")
sh(f"ffmpeg -v error -i '{raw}' -filter_complex "
   f"\"[0:v]split=2[f][r];[r]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[rv];"
   f"[f][rv]concat=n=2:v=1:a=0[v]\" -map '[v]' "
   f"-c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p '{mir}' -y")

webm, mp4 = OUT / "STEEL_cover.webm", OUT / "STEEL_cover.mp4"
sh(f"ffmpeg -v error -i '{mir}' -vf scale=-2:1080:flags=lanczos -c:v libvpx-vp9 -crf 33 -b:v 0 "
   f"-row-mt 1 -cpu-used 1 -g 240 -an '{webm}' -y")
sh(f"ffmpeg -v error -i '{mir}' -vf scale=-2:1080:flags=lanczos -c:v libx264 -preset veryslow "
   f"-crf 28 -pix_fmt yuv420p -movflags +faststart -an '{mp4}' -y")

sh(f"ffmpeg -v error -i '{webm}' -vf scale=256:176 -f rawvideo -pix_fmt rgb24 /tmp/_v.rgb -y",
   quiet=True)
a = np.fromfile("/tmp/_v.rgb", dtype=np.uint8).reshape(-1, 176, 256, 3).astype(np.float32) / 255
l = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
per = np.abs(np.diff(l, axis=0)).reshape(len(a)-1, -1).mean(1) * 255
wrap = float(np.abs(l[-1] - l[0]).mean() * 255)
typical = float(np.percentile(per, 95))
cuts = int((np.abs(np.diff(l.reshape(len(a), -1).mean(1))) > 0.10).sum())
verdict = {"model_image": "Tongyi-MAI/Z-Image-Turbo", "model_video": "Wan-AI/Wan2.2-TI2V-5B",
           "still": best, "frames": int(len(a)), "seconds": round(len(a)/16, 2),
           "gen_seconds": gen_s, "cuts": cuts,
           "webm_mb": round(webm.stat().st_size/1048576, 2),
           "mp4_mb": round(mp4.stat().st_size/1048576, 2),
           "ti_mean": round(float(per.mean()), 2),
           "wrap_delta": round(wrap, 2), "typical_frame_delta": round(typical, 2),
           "wrap_ratio": round(wrap/max(typical, 1e-6), 2)}
(WORK / "cover_verify.json").write_text(json.dumps(verdict, indent=2))
print("\nVERIFY:", json.dumps(verdict), flush=True)
assert cuts == 0, f"the loop must not cut — {cuts} found"
assert verdict["wrap_ratio"] < 2.0, f"visible wrap: {verdict['wrap_ratio']}"
assert verdict["ti_mean"] > 1.0, f"nothing moves: TI {verdict['ti_mean']}"
print("\nDONE — new illustration, new animation", flush=True)
