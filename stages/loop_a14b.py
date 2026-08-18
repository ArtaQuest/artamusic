# STEEL loop — Wan2.2-I2V-A14B (Apache-2.0), the strongest open image-to-video model that fits a
# 16 GB card, closed into a SEAMLESS LOOP BY CONSTRUCTION rather than by mirroring: the same still
# is pinned as the FIRST and the LAST frame (`last_image`), so the wrap is exact and the model has
# to bring every ember and curl of smoke back to where it started. Nothing plays backwards.
#
# Why this recipe on a P100 (facts, each one a dead run if ignored):
#   * bf16 attention on sm_60 falls back to the O(N^2) MATH kernel (PyTorch 2.7.1 sdp_utils.cpp:
#     mem-efficient SDPA below sm_80 is fp16/fp32 ONLY). That is what OOM'd the 5B at 33 frames.
#     So compute is fp16, with an fp32 retry if fp16 ever produces a NaN latent.
#   * 14B does not fit at 16-bit (28 GB per expert). GGUF Q4_K_M is 9.66 GB per expert, one
#     expert resident at a time (model offload swaps high-noise -> low-noise at the boundary).
#   * lightx2v 4-step CFG-distilled merge: 4 forwards per clip instead of 40+, guidance 1.0.
#   * diffusers' single-file loader would guess the Wan2.1 I2V (CLIP-conditioned) config for a
#     36-channel 5120-dim checkpoint, so the 2.2 config is passed explicitly.
#   * Round 2 (after the T4 pair answered): on two GPUs each expert lives on its own card and the
#     low-noise forward is wrapped to move its inputs across — no expert ever swaps through host
#     RAM (the 640-square rung of round 1 died silently right after its last step, and every
#     9.66 GB swap is a chance for that). One GPU falls back to model offload.
#   * The wrap is delivered twice: the hard cut (duplicate pinned frame dropped) and a 6-frame
#     dissolve of tail into head. Round 1 measured the cut at 2.1x a typical frame step — the
#     ends match, but a spark burst and a smoke shape differ — so the prompt also gets a
#     steady-state variant without transient sparks.
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
   "sentencepiece protobuf 'gguf>=0.10.0' ftfy imageio imageio-ffmpeg 2>&1 | tail -2")
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
NGPU = torch.cuda.device_count()
print(f"cuda devices: {NGPU} · {[torch.cuda.get_device_name(i) for i in range(NGPU)]}", flush=True)

def mem(tag):
    g = subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
                       shell=True, text=True, capture_output=True).stdout.strip().replace("\n", " | ")
    r = subprocess.run("free -m | awk 'NR==2{print $3\"/\"$2\" MB\"}'", shell=True, text=True,
                       capture_output=True).stdout.strip()
    print(f"  [mem {tag}] gpu {g} · host {r} · t+{(time.time()-T_START)/60:.1f} min", flush=True)

def vram(tag):
    a = torch.cuda.memory_allocated() / 2**30; m = torch.cuda.max_memory_allocated() / 2**30
    print(f"  [vram {tag}] now {a:.2f} GB · peak {m:.2f} GB · t+{(time.time()-T_START)/60:.1f} min",
          flush=True)

# ── the stills: the Z-Image-Turbo renders, mounted from their own public kernel ────────────
cands = sorted(glob.glob("/kaggle/input/**/still_*.png", recursive=True))
assert cands, "Z-Image stills not mounted (kernel source ashraasn/steel-scratch)"
STILLS = {Path(c).stem.split("_")[1]: c for c in cands}
print("stills:", {k: Image.open(v).size for k, v in STILLS.items()}, flush=True)

# ── weights: pinned by commit, hashed on arrival ─────────────────────────────────────────
from huggingface_hub import hf_hub_download, snapshot_download

def sha(p, n=20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:n]

BASE_REPO, BASE_REV = "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
GGUF_REPO, GGUF_REV = ("jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
                       "338fb8eedd8f485c9188cf1b1de541721fc81d66")
HIGH_F = "high_noise_1030/wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030-Q4_K_M.gguf"
LOW_F = "low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf"

t0 = time.time()
BASE = snapshot_download(BASE_REPO, revision=BASE_REV,
                         allow_patterns=["model_index.json", "scheduler/*", "vae/*", "tokenizer/*",
                                         "text_encoder/*", "transformer/config.json",
                                         "transformer_2/config.json"])
HIGH = hf_hub_download(GGUF_REPO, HIGH_F, revision=GGUF_REV)
LOW = hf_hub_download(GGUF_REPO, LOW_F, revision=GGUF_REV)
print(f"downloaded in {time.time()-t0:.0f}s", flush=True)
HASHES = {"high_noise_gguf": sha(HIGH), "low_noise_gguf": sha(LOW),
          "vae": sha(f"{BASE}/vae/diffusion_pytorch_model.safetensors")}
print("sha256:", HASHES, flush=True)
sh("df -h /kaggle/temp | tail -1", quiet=True)

# ── prompt embeddings first, encoder freed before any DiT touches the card ────────────────
# umT5-XXL is 11.4 GB in bf16 — it and one expert cannot share 16 GB. Encode once, keep the
# 512x4096 tensor, free the encoder. Mirrors WanImageToVideoPipeline._get_t5_prompt_embeds.
import ftfy, html, re
from transformers import AutoTokenizer, UMT5EncoderModel

def prompt_clean(t):
    t = html.unescape(html.unescape(ftfy.fix_text(t)))
    return re.sub(r"\s+", " ", t).strip()

SCENE = ("Photograph, locked-off tripod shot, a dark blacksmith's forge at night. A polished steel "
         "sword lies across a bed of glowing orange coals in a stone hearth; an iron anvil stands "
         "behind it. ")
PROMPTS = {
    # A — round 1's prompt: sparks lift and fade (transient bursts, which is what pops at the seam)
    "sparks": SCENE + ("Subtle, continuous, natural motion only: heat haze shimmers above the coals, "
              "the embers pulse and breathe with a slow orange glow, tiny sparks lift and fade, thin "
              "grey smoke drifts and curls upward, warm firelight flickers gently across the blade and "
              "the anvil. The blade, the hearth and the camera stay perfectly still. Cinematic, "
              "photorealistic, shallow depth of field, fine film grain, seamless loop."),
    # B — steady state: nothing transient, and the DreamLoop-style closing sentence
    "steady": SCENE + ("Subtle, continuous, natural motion only: heat haze shimmers above the coals, "
              "the embers pulse and breathe with a slow orange glow, thin grey smoke drifts and curls "
              "gently in place, warm firelight flickers softly across the blade and the anvil. The "
              "blade, the hearth and the camera stay perfectly still. Cinematic, photorealistic, "
              "shallow depth of field, fine film grain. The motion is gentle and steady and the scene "
              "returns to exactly where it began, looping seamlessly."),
}
PROMPT = PROMPTS["steady"]
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
EMB = {k: embed(v) for k, v in PROMPTS.items()}
NE, n_neg = embed(NEG)
for k, (e, n) in EMB.items():
    print(f"embeddings[{k}] {tuple(e.shape)} ({n} tokens) finite={bool(torch.isfinite(e).all())}", flush=True)
    assert torch.isfinite(e).all(), f"text embeddings not finite: {k}"
print(f"negative {n_neg} tokens · all in {time.time()-t0:.1f}s", flush=True)
# The pipelines cast provided embeddings to the transformer's dtype but never MOVE them: a CPU
# tensor here dies at the first addmm ('cuda:0 and cpu'). Park them on the card now.
EMB = {k: e.to("cuda:0") for k, (e, n) in EMB.items()}
NE = NE.to("cuda:0")
del te; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
vram("encoder freed")

# ── the animator: two GGUF experts, one on the card at a time ────────────────────────────
from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                       WanImageToVideoPipeline, WanTransformer3DModel)

class NaNLatent(RuntimeError):
    pass

class PinnedSigmas(FlowMatchEulerDiscreteScheduler):
    PRE_SHIFT = [1.0, 0.75, 0.5, 0.25]         # -> [1.0, 0.9375, 0.8333, 0.625] after shift 5
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None,
                      timesteps=None):
        return super().set_timesteps(device=device, sigmas=list(self.PRE_SHIFT))

def build(dtype):
    vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
    q = GGUFQuantizationConfig(compute_dtype=dtype)
    high = WanTransformer3DModel.from_single_file(HIGH, quantization_config=q, config=BASE,
                                                  subfolder="transformer", torch_dtype=dtype)
    low = WanTransformer3DModel.from_single_file(LOW, quantization_config=q, config=BASE,
                                                 subfolder="transformer_2", torch_dtype=dtype)
    p = WanImageToVideoPipeline.from_pretrained(BASE, transformer=high, transformer_2=low, vae=vae,
                                                text_encoder=None, tokenizer=None, torch_dtype=dtype)
    p.vae.enable_tiling()
    if NGPU >= 2:
        # One expert per card, nothing swaps. The pipeline resolves its execution device from the
        # first module in signature order (transformer -> cuda:0); the low-noise expert on cuda:1
        # gets a forward wrapper that carries inputs across and brings the prediction back.
        high.to("cuda:0"); vae.to("cuda:0"); low.to("cuda:1")
        _fwd = low.forward
        def _across(*a, **k):
            a = [x.to("cuda:1") if torch.is_tensor(x) else x for x in a]
            k = {n: (v.to("cuda:1") if torch.is_tensor(v) else v) for n, v in k.items()}
            out = _fwd(*a, **k)
            if isinstance(out, tuple):
                return tuple(o.to("cuda:0") if torch.is_tensor(o) else o for o in out)
            return out.__class__(sample=out.sample.to("cuda:0"))
        low.forward = _across
        print("placement: high-noise expert cuda:0 · low-noise expert cuda:1 · vae cuda:0", flush=True)
    else:
        p.enable_model_cpu_offload()
        print("placement: single card, model offload (experts swap through host RAM)", flush=True)
    # 4-step distilled: Euler flow-matching at the EXACT points the distillation was trained on —
    # lightx2v's denoising_step_list is t=[1000, 750, 500, 250] under shift 5, i.e. sigmas
    # [1.0, 0.9375, 0.8333, 0.625] (ComfyUI "simple" spacing). diffusers' default 4-step spacing
    # would put the last step at sigma 0.005 — a wasted forward and a 0.715 -> 0.005 jump. The Wan
    # pipeline does not expose `sigmas`, so the scheduler is pinned; the scheduler applies the
    # shift itself, hence the pre-shift values below. boundary 0.9: t=1000, 937 high · 833, 625 low.
    p.scheduler = PinnedSigmas(shift=5.0)
    print(f"pipeline ready ({dtype}) · boundary {p.config.boundary_ratio} · "
          f"expand_timesteps {p.config.expand_timesteps}", flush=True)
    return p

def guard(pipe_, i, t, kw):
    lat = kw["latents"]
    if not torch.isfinite(lat).all():
        raise NaNLatent(f"non-finite latents after step {i} (t={float(t):.0f})")
    print(f"    step {i} done · t={float(t):.0f} · lat |mean| {lat.float().abs().mean():.3f} · "
          f"t+{(time.time()-T_START)/60:.1f} min", flush=True)
    if i == pipe_.num_timesteps - 1:
        mem("before decode")               # round 1 died silently right here at 640-square
    return {}

def generate(pipe_, still, W, H, NF, steps, seed, pkey):
    img = still.resize((W, H), Image.LANCZOS)          # square -> square: no aspect distortion
    g = torch.Generator(device="cuda:0").manual_seed(seed)
    out = pipe_(image=img, last_image=img, prompt_embeds=EMB[pkey], negative_prompt_embeds=NE,
                height=H, width=W, num_frames=NF, num_inference_steps=steps, guidance_scale=1.0,
                generator=g, output_type="np", callback_on_step_end=guard)
    fr = (np.clip(out.frames[0], 0, 1) * 255).round().astype(np.uint8)
    del out
    return fr, np.asarray(img)

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

def encode_loop(loop, base, W, H):
    """PNG frames -> lossless-ish h264 master -> vp9 webm + h264 mp4 (native) + 1080-square vp9 twin."""
    fdir = Path(f"/tmp/frames_{base.name}"); fdir.mkdir(exist_ok=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    for i, f in enumerate(loop):
        Image.fromarray(f).save(fdir / f"{i:04d}.png")
    raw = str(base) + "_raw.mp4"
    sh(f"ffmpeg -v error -framerate {FPS} -i '{fdir}/%04d.png' -c:v libx264 -crf 12 -preset slow "
       f"-pix_fmt yuv420p '{raw}' -y", quiet=True)
    webm, mp4 = Path(str(base) + ".webm"), Path(str(base) + ".mp4")
    sh(f"ffmpeg -v error -i '{raw}' -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -cpu-used 1 -g 240 "
       f"-pix_fmt yuv420p -an '{webm}' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -c:v libx264 -preset veryslow -crf 20 -pix_fmt yuv420p "
       f"-movflags +faststart -an '{mp4}' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -vf scale=1080:1080:flags=lanczos -c:v libvpx-vp9 -crf 33 "
       f"-b:v 0 -row-mt 1 -cpu-used 1 -g 240 -pix_fmt yuv420p -an '{base}_1080.webm' -y", quiet=True)
    return webm, mp4

XF = 6                                     # dissolve length in frames (0.375 s at 16 fps)

def deliver(frames, ref, tag, W, H, gen_s, dtype, pkey, seed):
    # frames[-1] was pinned to frames[0]'s still: drop it so the wrap does not hold a frame twice.
    L = frames[:-1]
    # Dissolve twin: the last XF frames fade into the first XF frames, and both ends are consumed by
    # the blend, so the seam becomes a short cross-dissolve between near-identical frames.
    w = (np.arange(1, XF + 1) / (XF + 1))[:, None, None, None]
    blend = ((1 - w) * L[-XF:].astype(np.float32) + w * L[:XF].astype(np.float32)).round().astype(np.uint8)
    XL = np.concatenate([L[XF:len(L) - XF], blend])
    out = {}
    for kind, loop in (("cut", L), ("xf", XL)):
        base = OUT / f"{tag}_{kind}"
        webm, mp4 = encode_loop(loop, base, W, H)
        idx = np.linspace(0, len(loop) - 1, 8).round().astype(int)
        Image.fromarray(np.concatenate([loop[i] for i in idx], axis=1)).save(
            OUT / f"{tag}_{kind}_sheet.jpg", quality=88)
        # the seam, at delivery size: last three frames beside the first three
        seam = np.concatenate([loop[i] for i in (-3, -2, -1, 0, 1, 2)], axis=1)
        Image.fromarray(seam).save(OUT / f"{tag}_{kind}_seam.jpg", quality=90)
        m = measure_loop(str(webm))
        m.update({"kind": kind, "webm_mb": round(webm.stat().st_size / 1048576, 2),
                  "mp4_mb": round(mp4.stat().st_size / 1048576, 2), "seconds": round(len(loop) / FPS, 2)})
        out[kind] = m
    ref_f = ref.astype(np.float32)
    rec = {"tag": tag, "still": tag.split("_")[1], "prompt": pkey, "seed": seed,
           "model": "Wan2.2-I2V-A14B lightx2v-4step Q4_K_M (jayn7 GGUF)",
           "loop_closure": "first==last frame pinned (last_image); cut = duplicate dropped, "
                           f"xf = {XF}-frame dissolve of tail into head",
           "dtype": str(dtype).replace("torch.", ""), "res": [W, H], "fps": FPS,
           "gen_seconds": round(gen_s, 1),
           "pin_mae_first": round(float(np.abs(frames[0].astype(np.float32) - ref_f).mean()), 2),
           "pin_mae_last": round(float(np.abs(frames[-1].astype(np.float32) - ref_f).mean()), 2),
           "cut": out["cut"], "xf": out["xf"]}
    print("MEASURED:", json.dumps(rec), flush=True)
    return rec

# ── the ladder ───────────────────────────────────────────────────────────────────────────
# Round 1 proved the machinery at 480-square (7.6 min on a T4, finite fp16 latents, zero cuts) and
# died without a word after the last step at 640-square. Round 2 spends its time on the
# deliverable size, both stills, both prompt variants, and a second seed if time allows; an OOM at
# a size steps down, a NaN steps to fp32.
RUNGS = [
    ("6270", [640, 576, 480], "steady", SEED),
    ("6270", [640, 576, 480], "sparks", SEED),
    ("5150", [640, 576, 480], "steady", SEED),
    ("6270", [640, 576, 480], "steady", 6270),
]
NF, STEPS = 81, 4
BUDGET_MIN = 100                          # do not start a rung after this many minutes
results, dtype, pipe = [], torch.float16, None
for still_id, sizes, pkey, seed in RUNGS:
    if (time.time() - T_START) / 60 > BUDGET_MIN:
        print(f"budget: skipping {still_id} {pkey} seed {seed}", flush=True); continue
    still = Image.open(STILLS[still_id]).convert("RGB")
    for size in sizes:
        W = H = size
        tag = f"a14b_{still_id}_{size}_{pkey}_s{seed}"
        done = False
        for attempt in range(2):                # a second attempt only after an fp16 -> fp32 rebuild
            try:
                if pipe is None:
                    pipe = build(dtype)
                for d in range(NGPU):
                    torch.cuda.reset_peak_memory_stats(d)
                print(f"\n=== {tag}: {NF} frames · {STEPS} steps · {dtype} ===", flush=True)
                mem("before generate")
                t0 = time.time()
                frames, ref = generate(pipe, still, W, H, NF, STEPS, seed, pkey)
                gen_s = time.time() - t0
                vram(f"{tag} generated"); mem("after generate")
                print(f"{tag}: {len(frames)} frames in {gen_s:.0f}s", flush=True)
                results.append(deliver(frames, ref, tag, W, H, gen_s, dtype, pkey, seed))
                del frames; gc.collect(); torch.cuda.empty_cache()
                done = True
                break
            except NaNLatent as e:
                print(f"{tag}: {e}", flush=True)
                if dtype is torch.float32:
                    print(f"{tag}: NaN even in fp32 — giving up on this size", flush=True); break
                print("  fp16 overflowed on this card — rebuilding the pipeline in fp32", flush=True)
                del pipe; pipe = None; gc.collect(); torch.cuda.empty_cache(); dtype = torch.float32
            except torch.cuda.OutOfMemoryError as e:
                print(f"{tag}: OOM — {str(e)[:220]}", flush=True)
                gc.collect(); torch.cuda.empty_cache(); break
        if done:
            break                                # first size that fits is the one delivered

(WORK / "loop_verify.json").write_text(json.dumps({"hashes": HASHES, "prompts": PROMPTS,
    "negative": NEG, "dissolve_frames": XF, "results": results}, indent=2))
print("\nSUMMARY:", json.dumps(results, indent=1), flush=True)
assert results, "no rung produced a loop"
for m in results:
    for kind in ("cut", "xf"):
        k = m[kind]
        assert k["cuts"] == 0, f"{m['tag']} {kind}: cut detected"
        assert k["luma_std_min"] > 0.01 and k["luma_mean"] > 0.02, f"{m['tag']} {kind}: black frames"
        assert k["ti_mean"] > 0.5, f"{m['tag']} {kind}: nothing moves"
print("\nDONE", flush=True)
