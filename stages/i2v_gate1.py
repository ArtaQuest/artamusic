# GATE 1 — generate a SHORT clip and look at it.
#
# Gate 0 measured the VAE round trip: 69.9% of sparks and 77% of high-frequency energy
# survive at 512x352, falling to 58%/60% at 896. That missed a pre-registered 70% bar by a
# tenth of a point. Pre-registering the bar was right; treating a 0.1-point miss as a verdict
# would not be, because the number was a guess and the real question is whether the OUTPUT
# looks real. A 3-second clip answers it directly for a few GPU-minutes, which is what the
# gate was protecting against spending in the first place.
#
# Two resolutions, two prompts, one seed: enough to judge whether LTX holds the forge.
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
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/kaggle/temp/hf"
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working")
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

# Download explicitly at the pinned revision, then hand from_single_file a LOCAL PATH.
# Passing a full https:// URL made diffusers treat it as repo-relative and request
# .../resolve/main/resolve/<sha>/... -> 404. Verified live: the pinned URL is 200, 5.91 GB.
LTX_REPO = "Lightricks/LTX-Video"
LTX_REV = "8984fa25007f376c1a299016d0957a37a2f797bb"
LTX_FILE = "ltxv-2b-0.9.8-distilled.safetensors"

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
src = sorted(_g.glob("/kaggle/input/**/cover.png", recursive=True))
assert src, "cover image not mounted"
STILL = src[0]
print("still:", STILL, flush=True)

from huggingface_hub import hf_hub_download
import numpy as np
from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
from diffusers.utils import export_to_video
from PIL import Image

ckpt = hf_hub_download(LTX_REPO, LTX_FILE, revision=LTX_REV)
print("checkpoint:", Path(ckpt).name, f"{Path(ckpt).stat().st_size/2**30:.2f} GB", flush=True)

tr = LTXVideoTransformer3DModel.from_single_file(ckpt, torch_dtype=torch.bfloat16)
vae = AutoencoderKLLTXVideo.from_single_file(ckpt, torch_dtype=torch.bfloat16)
print(f"transformer layers={tr.config.num_layers} · loaded", flush=True)

pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video", transformer=tr, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
print("pipeline ready", flush=True)

PROMPTS = [
    ("embers", "A forged steel sword rests on a bed of glowing orange coals in a dark forge. "
               "The coals pulse and breathe with heat, embers drift slowly upward through the "
               "air, thin smoke curls above the blade. The camera is completely still. "
               "Photorealistic, cinematic, shallow depth of field."),
    ("heat",   "Heat shimmers over glowing coals beneath a steel blade in a dark smithy. "
               "Slow rising heat haze distorts the air, orange light flickers across the metal, "
               "smoke drifts. Locked-off camera, photorealistic, cinematic."),
]
OUT.mkdir(parents=True, exist_ok=True)
results = []
for (tag, prompt) in PROMPTS:
    for (w, h) in [(512, 352), (704, 480)]:
        name = f"{tag}_{w}x{h}"
        img = Image.open(STILL).convert("RGB").resize((w, h), Image.LANCZOS)
        t0 = time.time()
        try:
            vid = pipe(image=img, prompt=prompt,
                       negative_prompt="worst quality, blurry, jittery, distorted, watermark, "
                                       "text, static image, frozen",
                       width=w, height=h, num_frames=73, frame_rate=24,
                       num_inference_steps=8, guidance_scale=1.0,
                       generator=torch.Generator(device="cuda").manual_seed(4242)).frames[0]
            p_out = OUT / f"{name}.mp4"
            export_to_video(vid, str(p_out), fps=24)
            secs = round(time.time() - t0, 1)
            results.append({"name": name, "w": w, "h": h, "seconds": secs,
                            "frames": len(vid), "bytes": p_out.stat().st_size})
            print(f"  {name}: {len(vid)} frames in {secs}s -> {p_out.stat().st_size/1e6:.1f} MB",
                  flush=True)
        except Exception as e:
            results.append({"name": name, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
        torch.cuda.empty_cache()

(WORK / "gate1.json").write_text(json.dumps(results, indent=2))
print("\nGATE 1 done — inspect the clips", flush=True)
