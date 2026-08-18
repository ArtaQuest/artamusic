# STEEL loop — Wan2.1-VACE-1.3B (Apache-2.0): the SMALL, SAFE seam-free recipe, and the hedge
# for the 14B GGUF kernel. It shares nothing with that kernel's machinery: plain safetensors, no
# quantisation, and fp32 end to end — the P100 has memory-efficient attention for fp32 (PyTorch
# 2.7.1 sdp_utils.cpp: below sm_80 only fp16/fp32 get it), so a 1.8B model runs at full range with
# no overflow and no O(N^2) attention. Slower than fp16, and it cannot go wrong numerically.
#
# The loop is closed BY CONSTRUCTION with the recipe from diffusers' own WanVACEPipeline docstring
# (first-and-last-frame): the still is frame 0 and frame N with a black mask (keep), grey frames
# and a white mask (generate) in between. Same photograph at both ends -> the wrap is exact.
import gc, glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/kaggle/temp/hf"
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
T_START = time.time()

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1200:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1200:], flush=True)
    return r.returncode

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
PASCAL = bool(smi) and float(smi.split(",")[1]) < 7.0
sh("free -g | head -2; df -h /kaggle/working /tmp | tail -2; nproc")

sh("pip install -q 'diffusers==0.39.0' 'transformers>=4.51.0,<4.58.0' accelerate safetensors "
   "sentencepiece protobuf ftfy imageio imageio-ffmpeg 2>&1 | tail -2")
if PASCAL:
    sh("pip install -q torch==2.7.1 torchvision==0.22.1 --index-url "
       "https://download.pytorch.org/whl/cu126 2>&1 | tail -1")

import numpy as np
import torch
from PIL import Image
print(f"torch {torch.__version__} · cuda cap {torch.cuda.get_device_capability()}", flush=True)
import diffusers, transformers
print(f"diffusers {diffusers.__version__} · transformers {transformers.__version__}", flush=True)
SEED = 4242
FPS = 16                                   # Wan's native rate; the loop is delivered at it

def vram(tag):
    a = torch.cuda.memory_allocated() / 2**30; m = torch.cuda.max_memory_allocated() / 2**30
    print(f"  [vram {tag}] now {a:.2f} GB · peak {m:.2f} GB · t+{(time.time()-T_START)/60:.1f} min",
          flush=True)

# ── the stills: the Z-Image-Turbo renders, mounted from their own public kernel ────────────
cands = sorted(glob.glob("/kaggle/input/**/still_*.png", recursive=True))
assert cands, "Z-Image stills not mounted (kernel source ashraasn/steel-scratch)"
STILLS = {Path(c).stem.split("_")[1]: c for c in cands}
print("stills:", {k: Image.open(v).size for k, v in STILLS.items()}, flush=True)

# ── weights: one repo, pinned by commit, hashed on arrival ───────────────────────────────
from huggingface_hub import snapshot_download

def sha(p, n=20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:n]

REPO, REV = "Wan-AI/Wan2.1-VACE-1.3B-diffusers", "ec4d2cb062b548996b179d493fdd05340de702a1"
t0 = time.time()
BASE = snapshot_download(REPO, revision=REV, allow_patterns=["*.json", "tokenizer/*", "text_encoder/*",
                                                              "transformer/*", "vae/*", "scheduler/*"])
print(f"downloaded in {time.time()-t0:.0f}s", flush=True)
HASHES = {Path(f).name: sha(f) for f in sorted(glob.glob(f"{BASE}/transformer/*.safetensors"))}
HASHES["vae"] = sha(f"{BASE}/vae/diffusion_pytorch_model.safetensors")
print("sha256:", HASHES, flush=True)

# ── prompt embeddings first, encoder freed before the DiT touches the card ────────────────
import ftfy, html, re
from transformers import AutoTokenizer, UMT5EncoderModel

def prompt_clean(t):
    t = html.unescape(html.unescape(ftfy.fix_text(t)))
    return re.sub(r"\s+", " ", t).strip()

PROMPT = ("Photograph, locked-off tripod shot, a dark blacksmith's forge at night. A polished steel "
          "sword lies across a bed of glowing orange coals in a stone hearth; an iron anvil stands "
          "behind it. Subtle, continuous, natural motion only: heat haze shimmers above the coals, "
          "the embers pulse and breathe with a slow orange glow, tiny sparks lift and fade, thin "
          "grey smoke drifts and curls upward, warm firelight flickers gently across the blade and "
          "the anvil. The blade, the hearth and the camera stay perfectly still. Cinematic, "
          "photorealistic, shallow depth of field, fine film grain, seamless loop.")
NEG = ("camera movement, pan, zoom, dolly, handheld shake, cut, scene change, morphing, deformed "
       "blade, blade moving, extra objects, people, hands, text, watermark, subtitles, blurry, low "
       "quality, JPEG artifacts, overexposed, oversaturated, cartoon, painting, flicker, jitter")

tok = AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16)
te = te.to("cuda").eval()
vram("umT5 on card")

def embed(text, max_len=512):
    ids = tok([prompt_clean(text)], padding="max_length", max_length=max_len, truncation=True,
              add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(ids.attention_mask.gt(0).sum(dim=1)[0])
    with torch.no_grad():
        h = te(ids.input_ids.cuda(), ids.attention_mask.cuda()).last_hidden_state[0].float().cpu()
    return torch.cat([h[:n], h.new_zeros(max_len - n, h.size(1))])[None], n

t0 = time.time()
PE, n_pos = embed(PROMPT); NE, n_neg = embed(NEG)
print(f"embeddings {tuple(PE.shape)} ({n_pos} / {n_neg} tokens) in {time.time()-t0:.1f}s · "
      f"finite={bool(torch.isfinite(PE).all())}", flush=True)
assert torch.isfinite(PE).all() and torch.isfinite(NE).all(), "text embeddings not finite"
del te; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
vram("encoder freed")

# ── the animator, fp32, resident on the card ─────────────────────────────────────────────
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline, WanVACETransformer3DModel

DT = torch.float32
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
tr = WanVACETransformer3DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=DT)
pipe = WanVACEPipeline.from_pretrained(BASE, transformer=tr, vae=vae, text_encoder=None,
                                       tokenizer=None, torch_dtype=DT)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)  # 480p
pipe.to("cuda")
pipe.vae.enable_tiling()
vram("pipeline on card")
print("pipeline ready (fp32, VACE-1.3B)", flush=True)

def flf(still, W, H, NF):
    """The docstring recipe: keep frame 0 and frame N-1 (black mask), generate the rest (white)."""
    img = still.resize((W, H), Image.LANCZOS)          # square -> square: no aspect distortion
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    frames = [img] + [grey] * (NF - 2) + [img]
    keep, gen = Image.new("L", (W, H), 0), Image.new("L", (W, H), 255)
    mask = [keep] + [gen] * (NF - 2) + [keep]
    return frames, mask, np.asarray(img)

def step_log(pipe_, i, t, kw):
    lat = kw["latents"]
    if i % 5 == 0 or i == pipe_.num_timesteps - 1:
        print(f"    step {i} · t={float(t):.0f} · finite={bool(torch.isfinite(lat).all())} · "
              f"t+{(time.time()-T_START)/60:.1f} min", flush=True)
    return {}

def generate(still, W, H, NF, steps, seed):
    video, mask, ref = flf(still, W, H, NF)
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(video=video, mask=mask, prompt_embeds=PE, negative_prompt_embeds=NE, height=H,
               width=W, num_frames=NF, num_inference_steps=steps, guidance_scale=5.0, generator=g,
               output_type="np", callback_on_step_end=step_log)
    fr = (np.clip(out.frames[0], 0, 1) * 255).round().astype(np.uint8)
    return fr, ref

# ── the loop, encoded and MEASURED on the delivered bytes ────────────────────────────────
def measure_loop(path, w=256):
    sh(f"ffmpeg -v error -i '{path}' -vf scale={w}:{w} -f rawvideo -pix_fmt rgb24 /tmp/v.rgb -y",
       quiet=True)
    a = np.fromfile("/tmp/v.rgb", dtype=np.uint8).reshape(-1, w, w, 3).astype(np.float32) / 255
    l = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    per = np.abs(np.diff(l, axis=0)).reshape(len(a) - 1, -1).mean(1) * 255
    wrap = float(np.abs(l[-1] - l[0]).mean() * 255)
    typical = float(np.percentile(per, 95))
    cuts = int((np.abs(np.diff(l.reshape(len(a), -1).mean(1))) > 0.10).sum())
    return {"frames": int(len(a)), "cuts": cuts, "ti_mean": round(float(per.mean()), 2),
            "wrap_delta": round(wrap, 2), "typical_frame_delta": round(typical, 2),
            "wrap_ratio": round(wrap / max(typical, 1e-6), 2),
            "luma_mean": round(float(l.mean()), 3), "luma_std_min": round(float(l.std(axis=(1, 2)).min()), 4)}

def deliver(frames, ref, tag, W, H, gen_s):
    loop = frames[:-1]                      # frame N-1 was pinned to frame 0: drop the duplicate
    fdir = Path(f"/tmp/frames_{tag}"); fdir.mkdir(exist_ok=True)
    for i, f in enumerate(loop):
        Image.fromarray(f).save(fdir / f"{i:04d}.png")
    raw = str(OUT / f"{tag}_raw.mp4")
    sh(f"ffmpeg -v error -framerate {FPS} -i '{fdir}/%04d.png' -c:v libx264 -crf 12 -preset slow "
       f"-pix_fmt yuv420p '{raw}' -y", quiet=True)
    webm, mp4 = OUT / f"{tag}.webm", OUT / f"{tag}.mp4"
    sh(f"ffmpeg -v error -i '{raw}' -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -cpu-used 1 -g 240 "
       f"-pix_fmt yuv420p -an '{webm}' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -c:v libx264 -preset veryslow -crf 20 -pix_fmt yuv420p "
       f"-movflags +faststart -an '{mp4}' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -vf scale=1080:1080:flags=lanczos -c:v libvpx-vp9 -crf 33 "
       f"-b:v 0 -row-mt 1 -cpu-used 1 -g 240 -pix_fmt yuv420p -an '{OUT}/{tag}_1080.webm' -y",
       quiet=True)
    idx = np.linspace(0, len(loop) - 1, 8).round().astype(int)
    Image.fromarray(np.concatenate([loop[i] for i in idx], axis=1)).save(OUT / f"{tag}_sheet.jpg",
                                                                        quality=88)
    ref_f = ref.astype(np.float32)
    m = measure_loop(str(webm))
    m.update({"tag": tag, "model": "Wan2.1-VACE-1.3B (diffusers, fp32)",
              "loop_closure": "first==last frame via VACE keep-mask; final duplicate frame dropped",
              "res": [W, H], "fps": FPS, "seconds": round(len(loop) / FPS, 2),
              "gen_seconds": round(gen_s, 1),
              "pin_mae_first": round(float(np.abs(frames[0].astype(np.float32) - ref_f).mean()), 2),
              "pin_mae_last": round(float(np.abs(frames[-1].astype(np.float32) - ref_f).mean()), 2),
              "webm_mb": round(webm.stat().st_size / 1048576, 2),
              "mp4_mb": round(mp4.stat().st_size / 1048576, 2)})
    print("MEASURED:", json.dumps(m), flush=True)
    return m

# ── the ladder ───────────────────────────────────────────────────────────────────────────
LADDER = [("6270", 512), ("5150", 512), ("6270", 640)]
NF, STEPS = 81, 30
BUDGET_MIN = 6 * 60
results = []
for still_id, size in LADDER:
    if (time.time() - T_START) / 60 > BUDGET_MIN:
        print(f"budget: skipping {still_id} {size}", flush=True); continue
    W = H = size
    tag = f"vace_{still_id}_{size}"
    still = Image.open(STILLS[still_id]).convert("RGB")
    try:
        torch.cuda.reset_peak_memory_stats()
        print(f"\n=== {tag}: {NF} frames · {STEPS} steps · cfg 5 · seed {SEED} · fp32 ===", flush=True)
        t0 = time.time()
        frames, ref = generate(still, W, H, NF, STEPS, SEED)
        gen_s = time.time() - t0
        vram(f"{tag} generated")
        print(f"{tag}: {len(frames)} frames in {gen_s:.0f}s", flush=True)
        results.append(deliver(frames, ref, tag, W, H, gen_s))
    except torch.cuda.OutOfMemoryError as e:
        print(f"{tag}: OOM — {str(e)[:220]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

(WORK / "loop_verify.json").write_text(json.dumps({"hashes": HASHES, "seed": SEED,
    "prompt": PROMPT, "negative": NEG, "results": results}, indent=2))
print("\nSUMMARY:", json.dumps(results, indent=1), flush=True)
assert results, "no rung produced a loop"
for m in results:
    assert m["cuts"] == 0, f"{m['tag']}: cut detected"
    assert m["luma_std_min"] > 0.01 and m["luma_mean"] > 0.02, f"{m['tag']}: black/degenerate frames"
    assert m["ti_mean"] > 0.5, f"{m['tag']}: nothing moves"
print("\nDONE", flush=True)
