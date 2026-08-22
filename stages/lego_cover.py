# %% [markdown]
# # STEEL — the hammering, in LEGO
#
# The style changed, so the question changed with it. Searching Hugging Face for LEGO turns up 77
# repositories and exactly **one** that makes video: **`Remade-AI/Lego`**, a LoRA trained for LEGO
# animation, Apache-2.0 and ungated. Everything else is a Flux or SDXL image LoRA, or an unrelated
# code model that happens to have "lego" in its name.
#
# ## The thing this notebook is here to find out
#
# The LoRA is trained on **Wan2.1**-T2V-14B, not the Wan2.2 mixture-of-experts this project has been
# using — different architecture, so it belongs on its own base. Wan2.1-14B in fp16 is 28 GB, which
# does not fit a T4 pair with anything left for activations, so the base has to be the Q4_K_M GGUF.
#
# **And nobody has established that diffusers can put a LoRA onto a GGUF-quantised transformer.**
# The quantised weights are not ordinary linear layers. This is the same shape of obstacle that made
# Krea 2 unreachable, and it is not worth asserting either way from memory — so the run tries it,
# says plainly whether it took, and carries on either way rather than dying on the attempt.
#
# Three outcomes, all useful, all reported:
#
# * **the LoRA applies** — LEGO style from the adapter, which is what it was trained for;
# * **it does not** — the run continues on the base model with a prompt that describes bricks in
#   detail, and says so, so the frames are read as "Wan describing LEGO" rather than "the LoRA";
# * **neither fits** — the shape ladder says so and nothing is claimed.
#
# Everything is Apache-2.0: the base, the GGUF, and the adapter.

# %%
import gc, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
T0 = time.time()
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TMP / "hf")
SEED = 4242
PINS = {
    "wan_base": ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"),
    "wan_gguf": ("city96/Wan2.1-T2V-14B-gguf", "511cbce9f475a6ca0269be901d23b125f44f5c0d"),
    "wan_file": "wan2.1-t2v-14b-Q4_K_M.gguf",
    "lego_lora": ("Remade-AI/Lego", "3f7938015b2537238f9e4f17b8896ddceac9cbe7"),
    "lora_file": "lego_35_epochs.safetensors",
    "shot_sha": "712516cc1b1ae897ba0759602482aae994c2eba1",
    "tools_sha": "25d6c9f7496fb882f28d205b0870c48ec4d7e040",
}

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-800:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-800:], flush=True)
    return r.returncode

def clock(tag):
    print(f"  ⏱ {tag} · t+{(time.time()-T0)/60:.1f} min", flush=True)

# %%
# torchao>=0.16 IS THE WHOLE REASON THE LoRA DID NOT APPLY. The first run reported
#   ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only
#   versions above 0.16.0 are supported
# which is a dependency check, not an architectural limit — it failed BEFORE reaching the question
# this notebook exists to answer. Kaggle ships 0.10.0 preinstalled. Upgrading it is the difference
# between "a LoRA cannot attach to a quantised transformer" and "nobody installed the right
# torchao", and those are very different conclusions to draw.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.39.0",
                "transformers>=5.13", "accelerate", "safetensors", "sentencepiece", "protobuf",
                "gguf>=0.10.0", "peft", "torchao>=0.16.0", "ftfy", "imageio", "imageio-ffmpeg"],
               check=True)
import importlib.metadata as _md
try:
    print("  torchao:", _md.version("torchao"), flush=True)
except Exception:
    print("  torchao: not installed", flush=True)
import numpy as np, torch
from PIL import Image
np.random.seed(SEED); torch.manual_seed(SEED)
NG = torch.cuda.device_count()
print(f"torch {torch.__version__} · cap {torch.cuda.get_device_capability()} · {NG} gpu(s)", flush=True)
sh("free -g | head -2; df -h /tmp | tail -1; nproc")

TOOLS = TMP / "tools"; TOOLS.mkdir(exist_ok=True)
for _f in ("stillness.py", "looper.py"):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['tools_sha']}/lib/{_f}",
        str(TOOLS / _f))
sys.path.insert(0, str(TOOLS))
import stillness as S, looper as L
assert L.selftest(), "the cycle finder fails its own selftest — no cut it proposes is trustworthy"
clock("environment ready")

# %%
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['shot_sha']}/song/shot_lego.json",
    "/tmp/shot_lego.json")
SHOT = json.loads(Path("/tmp/shot_lego.json").read_text())
PROMPT, NEG, TRIGGER = SHOT["prompt"], SHOT["negative"], SHOT["trigger"]
assert TRIGGER in PROMPT, "the LoRA trigger phrase is missing from the prompt — the adapter would do nothing"
(WORK / "prompt.json").write_text(json.dumps({"shot": SHOT["name"], "prompt": PROMPT,
                                              "negative": NEG, "trigger": TRIGGER}, indent=2))
print(f"[shot] {SHOT['name']}\n[prompt] {PROMPT}\n[negative] {NEG}", flush=True)

# %% [markdown]
# ## The base, then the adapter
#
# UMT5 encodes and is freed before the transformer arrives. Then the LoRA is attempted, and the
# result is recorded as a fact rather than assumed: `LORA_APPLIED` travels into the manifest, so a
# reader can tell which of the two things they are looking at.

# %%
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan, GGUFQuantizationConfig
from transformers import UMT5EncoderModel, AutoTokenizer
from huggingface_hub import hf_hub_download
BASE, BREV = PINS["wan_base"]

tok = AutoTokenizer.from_pretrained(BASE, revision=BREV, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(BASE, revision=BREV, subfolder="text_encoder",
                                      torch_dtype=torch.float16).to("cuda:0")
def embed(text, n=512):
    ids = tok([text], padding="max_length", max_length=n, truncation=True, return_tensors="pt")
    k = int(ids.attention_mask.gt(0).sum(1)[0])
    with torch.inference_mode():
        h = te(ids.input_ids.to("cuda:0"), ids.attention_mask.to("cuda:0")).last_hidden_state[0].float().cpu()
    return torch.cat([h[:k], h.new_zeros(n - k, h.size(1))])[None]
PE, NE = embed(PROMPT), embed(NEG)
del te, tok; gc.collect(); torch.cuda.empty_cache()
clock("text encoded")

# %%
GREPO, GREV = PINS["wan_gguf"]
GGUF = hf_hub_download(GREPO, PINS["wan_file"], revision=GREV)
q = GGUFQuantizationConfig(compute_dtype=torch.float16)
tr = WanTransformer3DModel.from_single_file(GGUF, quantization_config=q, config=BASE,
                                            subfolder="transformer", torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(BASE, revision=BREV, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(BASE, revision=BREV, transformer=tr, vae=vae,
                                   text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
tr.to("cuda:0"); vae.to("cuda:0")
print(f"  base on card · {torch.cuda.memory_allocated(0)/2**30:.1f} GB", flush=True)

# THE EXPERIMENT. Nobody has established that a LoRA can be attached to a GGUF-quantised
# transformer in diffusers — the quantised weights are not ordinary linear layers. Try it, record
# the answer, and keep going either way: a run that dies here would tell us less than a run that
# finishes and says which model made the frames.
LREPO, LREV = PINS["lego_lora"]
LORA = hf_hub_download(LREPO, PINS["lora_file"], revision=LREV)
LORA_APPLIED, LORA_ERROR = False, None
try:
    pipe.load_lora_weights(LORA, adapter_name="lego")
    try:
        pipe.set_adapters(["lego"], adapter_weights=[1.0])
    except Exception:
        pass
    LORA_APPLIED = True
    print("  LEGO LoRA APPLIED to the GGUF transformer", flush=True)
except Exception as e:
    LORA_ERROR = f"{type(e).__name__}: {e}"
    print(f"  LEGO LoRA did NOT apply: {LORA_ERROR[:200]}", flush=True)
    print("  continuing on the base model — the prompt describes bricks in detail, so the frames "
          "are Wan DESCRIBING Lego rather than the adapter rendering it. Reported as such.",
          flush=True)
clock("model ready")

# %% [markdown]
# ## Two steps, then decide
#
# Time two steps on the real shot, let the budget pick the step count, and step the shape down on
# OOM rather than lose the session — twice as much as it sounds, because a two-step probe fitting
# has already proved not to guarantee that thirty will.

# %%
FPS = 16
SHAPES = [(640, 81), (576, 81), (512, 65), (448, 65)]
BUDGET_S = 150 * 60
H = W = 640
NF = 81

def run(steps, h, w, nf):
    with torch.inference_mode():
        return pipe(prompt_embeds=PE.to("cuda:0"), negative_prompt_embeds=NE.to("cuda:0"),
                    height=h, width=w, num_frames=nf, num_inference_steps=steps,
                    guidance_scale=5.0,
                    generator=torch.Generator("cuda:0").manual_seed(SEED), output_type="np")

per_step = None
for (px, nf) in SHAPES:
    try:
        t0 = time.time(); _p = run(2, px, px, nf); per_step = (time.time() - t0) / 2
        del _p; gc.collect(); torch.cuda.empty_cache()
        H = W = px; NF = nf
        print(f"  {px}x{px}x{nf} fits · {per_step:.0f} s/step", flush=True)
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  {px}x{px}x{nf} OOM", flush=True)
        gc.collect(); torch.cuda.empty_cache()
assert per_step is not None, "Wan2.1-14B Q4_K_M does not fit at any shape in " + str(SHAPES)
STEPS = max(12, min(30, int(BUDGET_S / max(per_step, 1e-6))))
print(f"  → {STEPS} steps ≈ {per_step*STEPS/60:.0f} min at {H}x{W}x{NF}", flush=True)

# %%
frames, gen_s = None, 0.0
for (px, nf) in [(H, NF)] + [s for s in SHAPES if s != (H, NF)]:
    try:
        t0 = time.time(); out = run(STEPS, px, px, nf); gen_s = time.time() - t0
        H = W = px; NF = nf
        frames = (np.clip(out.frames[0], 0, 1) * 255).round().astype(np.uint8)
        print(f"  rendered {len(frames)} frames at {px}x{px} in {gen_s/60:.1f} min", flush=True)
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  {px}x{px}x{nf} OOM on the full run — stepping down", flush=True)
        gc.collect(); torch.cuda.empty_cache()
assert frames is not None, "no shape completed the full run"
clock("generated")

# %%
def dissolve(fr, xf):
    if xf <= 0 or len(fr) < 2 * xf + 2:
        return fr
    w = (np.arange(1, xf + 1) / (xf + 1))[:, None, None, None]
    blend = ((1 - w) * fr[-xf:].astype(np.float32) + w * fr[:xf].astype(np.float32)).round().astype(np.uint8)
    return np.concatenate([fr[xf:len(fr) - xf], blend])

CYCLE = L.cycle_report(frames, min_frames=max(24, len(frames) // 3))
print(f"  cycle {CYCLE['start']}..{CYCLE['end']} ({CYCLE['frames']} frames) · seam "
      f"{CYCLE['seam_vs_typical']}x a normal step · whole clip {CYCLE['whole_vs_typical']}x · "
      f"whole chosen={CYCLE['whole_clip_chosen']}", flush=True)
if CYCLE["seam_vs_typical"] < 1.0:
    CYCLE["used"] = "cycle"
    loop = dissolve(frames[CYCLE["start"]:CYCLE["end"] + 1], 8 if CYCLE["whole_clip_chosen"] else 3)
else:
    CYCLE["used"] = "dissolve"
    loop = dissolve(frames[:-1], 8)
print(f"  closed by {CYCLE['used']} · {len(loop)} frames = {len(loop)/FPS:.2f}s", flush=True)

def encode(fr, base):
    d = Path(f"/tmp/f_{Path(base).name}"); d.mkdir(exist_ok=True)
    for f in d.glob("*.png"): f.unlink()
    for i, f in enumerate(fr): Image.fromarray(f).save(d / f"{i:04d}.png")
    sh(f"ffmpeg -v error -framerate {FPS} -i '{d}/%04d.png' -c:v libx264 -crf 12 -preset slow "
       f"-pix_fmt yuv420p '{base}_raw.mp4' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{base}_raw.mp4' -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -cpu-used 1 "
       f"-g 240 -pix_fmt yuv420p -an '{base}.webm' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{base}_raw.mp4' -c:v libx264 -preset veryslow -crf 20 -pix_fmt yuv420p "
       f"-movflags +faststart -an '{base}.mp4' -y", quiet=True)

encode(loop, str(OUT / "STEEL_cover_loop"))
encode(frames, str(OUT / "STEEL_cover_loop_asgenerated"))
Image.fromarray(loop[0]).save(OUT / "frame0.png")
Image.fromarray(loop[0]).save(OUT / "cover.png")
sh(f"ffmpeg -v error -i '{OUT}/cover.png' -vf scale=3000:3000:flags=lanczos '{OUT}/cover_3000.png' -y", quiet=True)
idx = np.linspace(0, len(loop) - 1, 8).round().astype(int)
Image.fromarray(np.concatenate([loop[i] for i in idx], 1)).save(OUT / "loop_sheet.jpg", quality=88)
Image.fromarray(np.concatenate([loop[i] for i in (-3, -2, -1, 0, 1, 2)], 1)).save(OUT / "loop_seam.jpg", quality=90)

rec = {"model": f"Wan2.1-T2V-14B Q4_K_M ({GREPO})", "lora": f"{LREPO}@{LREV[:7]}",
       "lora_applied": LORA_APPLIED, "lora_error": LORA_ERROR, "shot": SHOT["name"],
       "seed": SEED, "steps": STEPS, "seconds_per_step": round(per_step, 1), "guidance": 5.0,
       "res": [H, W], "fps": FPS, "frames": int(len(loop)),
       "seconds": round(len(loop) / FPS, 2), "gen_seconds": round(gen_s, 1), "cycle": CYCLE,
       "as_generated": {}, "frozen": {}, "froze_the_blade": False, "mask_ok": False,
       "alive": {}, "verdict_still": [], "verdict_alive": []}
(WORK / "loop_verify.json").write_text(json.dumps(rec, indent=2))
print("\nLOOP:", json.dumps(rec), flush=True)
print(f"\nA {len(loop)/FPS:.2f}s LEGO loop at {H}x{W}, closed by {CYCLE['used']}. "
      f"LoRA applied: {LORA_APPLIED}." + ("" if LORA_APPLIED else
      f" These frames are the BASE model describing bricks, not the adapter — {str(LORA_ERROR)[:120]}"),
      flush=True)
clock("DONE")
