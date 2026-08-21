# %% [markdown]
# # STEEL — the hammering, on a physics world model
#
# Three complaints on the last cover: **not epic enough**, and the **physics** and the **loop** are
# not realistic. Two of those were prompt work and are addressed in `shot_steel.json`. The third is
# a model question, and it has a real answer.
#
# ## What actually leads on physics
#
# The relevant benchmark is **VideoPhy-2** (ICLR 2026), which scores physical commonsense rather
# than prettiness, and it does not rank the way the "best open video model" articles do. On its
# full set **CogVideoX-5B leads at 67.2%**, HunyuanVideo takes 64.2%, and **Wan2.1-14B — the family
# this cover has been using — manages 60.5%**. On the hard split a 7B **Cosmos** diffusion model
# places second, beating the much larger HunyuanVideo-13B.
#
# It is worth stating the ceiling plainly: the best model on the hard split reaches **47.7%** joint
# physical-and-semantic adherence, and the paper singles out conservation of mass and momentum as
# where they fail. Realistic impact physics is not a solved thing that the right download unlocks.
#
# ## Why this model
#
# **NVIDIA Cosmos3-Edge.** Cosmos is a *world model* line built for physical AI rather than for
# pretty clips, which is the axis being complained about. Cosmos 3 shipped in May 2026 in three
# tiers and Edge is the 4B one — **8.5 GB for the whole repository**, against 49.7 GB for
# HunyuanVideo 1.5 and ~19 GB of Wan experts.
#
# That size is the point, not a compromise. It fits on **one** T4 with room to spare, so there is
# no sharding across cards, no offload, and no VAE waiting on the host — the three things that cost
# four attempts to get HunyuanVideo running at all. `Cosmos3OmniPipeline` is in diffusers 0.39.0,
# the version already pinned here. It takes a real negative prompt and real guidance, and it
# generates at **24 fps** rather than 16.
#
# Licence: **OpenMDW 1.1**, commercial and non-commercial use, ungated. That matters here — it is
# the same clean footing as Wan's Apache-2.0, and unlike HunyuanVideo's community licence it does
# not exclude the EU, UK or South Korea, so a stranger anywhere can re-run this.

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
    "cosmos": ("nvidia/Cosmos3-Edge", "e56fbcd06a7823969a25047cf50d5051ae436e88"),
    "shot_sha": "fa6f8dc3823879dcf42b2d712358520006a3b887",
    "tools_sha": "e43b03d4ddc8810e67f467f52feef9ce65ce9131",
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.39.0",
                "transformers>=5.13", "accelerate", "safetensors", "sentencepiece", "protobuf",
                "ftfy", "imageio", "imageio-ffmpeg"], check=True)
import numpy as np, torch
from PIL import Image
np.random.seed(SEED); torch.manual_seed(SEED)
NG = torch.cuda.device_count()
print(f"torch {torch.__version__} · cap {torch.cuda.get_device_capability()} · {NG} gpu(s)", flush=True)
sh("free -g | head -2; df -h /tmp | tail -1; nproc")

TOOLS = TMP / "tools"; TOOLS.mkdir(exist_ok=True)
for _f in ("stillness.py", "freeze.py", "looper.py"):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['tools_sha']}/lib/{_f}",
        str(TOOLS / _f))
sys.path.insert(0, str(TOOLS))
import stillness as S, looper as L
assert L.selftest(), "the cycle finder fails its own selftest — no cut it proposes is trustworthy"
clock("environment ready")

# %%
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['shot_sha']}/song/shot_steel.json",
    "/tmp/shot_steel.json")
SHOT = json.loads(Path("/tmp/shot_steel.json").read_text())
PROMPT, NEG = SHOT["prompt"], SHOT["negative"]
assert "anvil" in PROMPT and "hammer" in PROMPT, "the fetched shot is not the hammering shot"
(WORK / "prompt.json").write_text(json.dumps({"shot": SHOT["name"], "prompt": PROMPT,
                                              "negative": NEG}, indent=2))
print(f"[shot] {SHOT['name']}\n[prompt] {PROMPT}\n[negative] {NEG}", flush=True)

# %% [markdown]
# ## One model, one card
#
# 8.5 GB of weights on a card that holds 14.5. Everything that made the last model hard — splitting
# the transformer across both GPUs, balancing the halves, keeping the VAE on the host until the
# decode — is simply unnecessary here, which is worth more than it sounds: each of those was a
# failed run before it was a working one.

# %%
from diffusers import Cosmos3OmniPipeline
REPO, REV = PINS["cosmos"]
t0 = time.time()
pipe = Cosmos3OmniPipeline.from_pretrained(REPO, revision=REV, torch_dtype=torch.float16)
pipe.to("cuda:0")
if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing("auto")
if hasattr(pipe.vae, "enable_tiling"): pipe.vae.enable_tiling()
print(f"  loaded in {time.time()-t0:.0f}s · {torch.cuda.memory_allocated(0)/2**30:.1f} GB resident",
      flush=True)
clock("model on card")

# %% [markdown]
# ## Two steps, then decide — and a shape ladder under it
#
# Same discipline as every run before this one: time two steps on the real shot, let the budget
# pick the step count, and step the shape down rather than lose a session if the full render peaks
# over. A two-step probe fitting has already proved, twice, not to guarantee that thirty will.

# %%
FPS = 24
SHAPES = [(704, 97), (640, 97), (640, 73), (512, 73)]
BUDGET_S = 100 * 60
H = W = 704
NF = 97

def run(steps, h, w, nf):
    with torch.inference_mode():
        return pipe(prompt=PROMPT, negative_prompt=NEG, height=h, width=w, num_frames=nf,
                    fps=FPS, num_inference_steps=steps, guidance_scale=6.0,
                    generator=torch.Generator("cuda:0").manual_seed(SEED), output_type="np")

per_step, H, W, NF = None, None, None, None
for (px, nf) in SHAPES:
    try:
        t0 = time.time(); _p = run(2, px, px, nf); per_step = (time.time() - t0) / 2
        del _p; gc.collect(); torch.cuda.empty_cache()
        H = W = px
        NF = nf
        print(f"  {px}x{px}x{nf} fits · {per_step:.0f} s/step", flush=True)
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  {px}x{px}x{nf} OOM", flush=True)
        gc.collect(); torch.cuda.empty_cache()
assert per_step is not None, "Cosmos3-Edge does not fit at any shape in " + str(SHAPES)
STEPS = max(12, min(40, int(BUDGET_S / max(per_step, 1e-6))))
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
        print(f"  {px}x{px}x{nf} OOM on the full {STEPS}-step run — stepping down", flush=True)
        gc.collect(); torch.cuda.empty_cache()
assert frames is not None, "no shape completed the full run"
clock("generated")

# %% [markdown]
# ## Close it on a real cycle
#
# A hammer at the top of its swing cross-faded into a hammer at the bottom is a morph, and it reads
# as one. The clip is searched for the pair of frames that genuinely match — position *and*
# velocity, since two frames can match at a turning point while travelling in opposite directions —
# and the cut is taken only when it is less visible than an ordinary frame-to-frame step. If there
# is no such cut the whole clip is dissolved instead, and the report says which happened rather
# than claiming a seamless loop either way.

# %%
def dissolve(fr, xf):
    if xf <= 0 or len(fr) < 2 * xf + 2:
        return fr
    w = (np.arange(1, xf + 1) / (xf + 1))[:, None, None, None]
    blend = ((1 - w) * fr[-xf:].astype(np.float32) + w * fr[:xf].astype(np.float32)).round().astype(np.uint8)
    return np.concatenate([fr[xf:len(fr) - xf], blend])

CYCLE = L.cycle_report(frames, min_frames=max(24, len(frames) // 3))
print(f"  cycle {CYCLE['start']}..{CYCLE['end']} ({CYCLE['frames']} frames) · seam "
      f"{CYCLE['seam_vs_typical']}x a normal step · whole clip {CYCLE['whole_vs_typical']}x", flush=True)
if CYCLE["seam_vs_typical"] < 1.0:
    CYCLE["used"] = "cycle"; loop = dissolve(frames[CYCLE["start"]:CYCLE["end"] + 1], 3)
else:
    CYCLE["used"] = "dissolve"; loop = dissolve(frames[:-1], 8)
print(f"  closed by {CYCLE['used']} · {len(loop)} frames = {len(loop)/FPS:.2f}s", flush=True)

# %%
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

rec = {"model": f"NVIDIA Cosmos3-Edge ({REPO})", "revision": REV, "seed": SEED, "steps": STEPS,
       "seconds_per_step": round(per_step, 1), "guidance": 6.0, "res": [H, W], "fps": FPS,
       "frames": int(len(loop)), "generated_frames": int(len(frames)),
       "seconds": round(len(loop) / FPS, 2), "gen_seconds": round(gen_s, 1),
       "method": "text-to-video on a physics world model; no still image, no image conditioning",
       "cycle": CYCLE, "shot": SHOT["name"],
       "as_generated": {}, "frozen": {}, "froze_the_blade": False, "mask_ok": False,
       "alive": {}, "verdict_still": [], "verdict_alive": []}
(WORK / "loop_verify.json").write_text(json.dumps(rec, indent=2))
print("\nLOOP:", json.dumps(rec), flush=True)
print(f"\nA {len(loop)/FPS:.2f}s loop at {H}x{W}, closed by {CYCLE['used']}. The subject is MEANT "
      f"to move here, so no stillness claim is made about it.", flush=True)
clock("DONE")
