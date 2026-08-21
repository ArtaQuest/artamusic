# %% [markdown]
# # Can HunyuanVideo 1.5 be reached on a Kaggle T4 pair?
#
# The operator asked for a better model for an **epic loop of hammering the hot sword**. Among open
# weights there are three leaders and the choice between them is not close for this shot:
# **Wan2.2** leads on photorealism of human subjects, **LTXVideo** on speed, and **HunyuanVideo
# 1.5** on *natural motion and physics — fluid dynamics including fire, and object interactions*.
# A hammer striking hot steel and throwing sparks is that, exactly.
#
# It is also 8.3B against Wan's two 14B experts, which on this hardware buys steps.
#
# **The licence, honestly.** Tencent's community licence permits commercial use royalty-free under
# 100 million monthly active users, and ArtaQuest is a Canadian entity, so the grant covers us. But
# it explicitly does not apply in the **EU, UK and South Korea** — so a stranger there could not
# legally re-run this notebook, which is a real dent in the "anyone can Copy & Edit → Run All"
# claim even though the weights themselves are public and ungated. That is the operator's call to
# make, not a gate to hide behind, and it is recorded here so it is visible in the record.
#
# **The mechanical problem.** diffusers has `HunyuanVideoTransformer3DModel` in its single-file
# table but NOT `HunyuanVideo15Transformer3DModel` — the same trap that made Krea 2 unreachable —
# so the GGUF builds cannot be loaded here at all. The weights ship fp32: 31 GB of transformer,
# which is 7.76B parameters, so fp16 is 15.5 GB against roughly 15.6 GB usable on one T4. It does
# not fit on a card, with nothing left for activations. It has to be **sharded across the pair**.
#
# This probe answers one question — does that work, and how fast — and gets out of the way.

# %%
import gc, json, os, subprocess, sys, time
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
T0 = time.time()
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TMP / "hf")
SEED = 4242
REPO, REV = ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
             "286be7ce72277246578a3e3cc2487e95ddae5bcf")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.39.0",
                "transformers>=5.13", "accelerate", "safetensors", "sentencepiece", "protobuf",
                "ftfy", "imageio", "imageio-ffmpeg"], check=True)
import numpy as np, torch
from PIL import Image
np.random.seed(SEED); torch.manual_seed(SEED)
NG = torch.cuda.device_count()
def mem(tag):
    used = " · ".join(f"cuda:{i} {torch.cuda.memory_allocated(i)/2**30:.1f}G" for i in range(NG))
    print(f"  [{tag}] {used} · t+{(time.time()-T0)/60:.1f} min", flush=True)
print(f"torch {torch.__version__} · cap {torch.cuda.get_device_capability()} · {NG} gpu(s)", flush=True)
assert NG >= 2, "this needs the T4 PAIR — a single card cannot hold the transformer in fp16"

# %%
PROMPT = (
    "Locked-off static camera on a tripod, close on a heavy black iron anvil inside the dark stone "
    "interior of a blacksmith's forge at night — soot-blackened brick, the far walls lost in "
    "blackness, the hearth glowing orange behind. A sword blade, heated to glowing orange-white "
    "along its length, lies across the anvil gripped in a pair of long iron tongs. A big "
    "blacksmith's hammer swings down and strikes the hot steel: on each impact a burst of bright "
    "sparks explodes outward and showers down across the anvil and the floor, the struck metal "
    "flares yellow-white, and the hammer lifts and comes down again in a steady, powerful rhythm. "
    "Only the smith's gloved hands and bare forearms are in frame, no face and no full body. "
    "Shot on 65mm Kodak Vision3 500T film at T2.8, deep black shadow, the only light coming from "
    "the glowing steel and the hearth, high dynamic range, visible film grain, natural halation on "
    "the hot metal, sparks in sharp focus, slight motion blur on the hammer head. Photographic, "
    "epic, monumental.")
NEG = ("face, portrait, full body, whole person, campfire, bonfire, outdoors, open ground, "
       "3d render, cgi, video game, plastic, illustration, cartoon, anime, airbrushed, waxy, "
       "flat even lighting, washed out, camera motion, zoom, pan, handheld shake, extra fingers, "
       "deformed hands, text, watermark, blurry, low resolution")

# %% [markdown]
# ## Encode first, with no transformer in memory
#
# Two text encoders here, not one — a Qwen2.5-VL tower and a small ByT5 — and each returns an
# embedding AND a mask. They are loaded, used and thrown away before the transformer arrives,
# because 6.6 GB of encoder plus 15.5 GB of transformer is not a thing that fits. The encode runs
# under `inference_mode`: a pipeline's `__call__` carries the no-grad decorator, a component method
# called directly does not, and without it the activations are kept for a backward pass that never
# comes.

# %%
from diffusers import HunyuanVideo15Pipeline
pipe = HunyuanVideo15Pipeline.from_pretrained(REPO, revision=REV, transformer=None,
                                              torch_dtype=torch.float16)
pipe.text_encoder.to("cuda:0"); pipe.text_encoder_2.to("cuda:0")
mem("encoders on card")
with torch.inference_mode():
    pe, pem, pe2, pem2 = pipe.encode_prompt(prompt=PROMPT, device=torch.device("cuda:0"),
                                            dtype=torch.float16)
    ne, nem, ne2, nem2 = pipe.encode_prompt(prompt=NEG, device=torch.device("cuda:0"),
                                            dtype=torch.float16)
EMB = {k: v.detach().cpu() for k, v in
       dict(pe=pe, pem=pem, pe2=pe2, pem2=pem2, ne=ne, nem=nem, ne2=ne2, nem2=nem2).items()}
print("  embedded:", {k: tuple(v.shape) for k, v in EMB.items()}, flush=True)
pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
del pipe.text_encoder, pipe.text_encoder_2
pipe.text_encoder = None; pipe.text_encoder_2 = None
gc.collect(); torch.cuda.empty_cache()
mem("encoders freed")

# %% [markdown]
# ## The transformer, split across both cards
#
# `device_map="auto"` hands accelerate the job of cutting the 54 layers between the two T4s and
# inserting the hooks that move activations across as they go. That is the only configuration that
# fits: resident on one card is 0.1 GB short before a single activation is allocated, and offload
# is what made a comparable model cost 56 seconds a step here.

# %%
from diffusers import HunyuanVideo15Transformer3DModel
t0 = time.time()
# CAP EACH CARD, RATHER THAN LETTING IT FILL ONE. Left to itself, accelerate packed 10.0 GB onto
# cuda:0 and 7.9 onto cuda:1, then the denoise asked for 1.91 GiB with 1.49 free and died. The
# weights are 15.5 GB; the cards hold 14.56 each. Splitting them evenly and capping the halves
# leaves both with real headroom for activations, which flow through BOTH cards because the layers
# are split across them — so reserving room on only one would repeat the failure on the other.
tr = HunyuanVideo15Transformer3DModel.from_pretrained(
    REPO, revision=REV, subfolder="transformer", torch_dtype=torch.float16,
    device_map="auto", max_memory={0: "8GiB", 1: "8GiB"})
dm = getattr(tr, "hf_device_map", None) or {}
print(f"  transformer sharded in {time.time()-t0:.0f}s · "
      f"{sum(1 for v in dm.values() if v == 0)} blocks on cuda:0, "
      f"{sum(1 for v in dm.values() if v == 1)} on cuda:1", flush=True)
pipe.transformer = tr
# THE VAE STAYS ON THE HOST UNTIL IT IS NEEDED. It is 2.35 GB in fp16 and it does nothing at all
# until the denoise has finished, so parking it on a card is 2.35 GB taken from the activations
# that were 0.4 GB short.
pipe.vae.to("cpu")
if hasattr(pipe.vae, "enable_tiling"): pipe.vae.enable_tiling()
mem("transformer sharded, vae on host")

# %% [markdown]
# ## Two steps, then decide
#
# The way this fails is by taking an hour to tell you it is slow, so it is timed on the real shot
# at the real size before anything is committed.

# %%
# A LADDER, BECAUSE THE WEIGHTS CANNOT GET SMALLER. 15.5 GB across two 14.56 GiB cards is ~7.75 GB
# of weights per card by arithmetic, so the only thing left to trade is activations. Two steps ran
# fine at 480x480x61 and the full run then peaked ~1.9 GB over — so the run finds its own ceiling
# instead of me guessing it, and records which rung it landed on.
SHAPES = [(480, 61), (480, 45), (448, 45), (384, 45)]
DEVICE_MAP = {}
H = W = 480
NF = 61
def run(steps):
    with torch.inference_mode():
        return pipe(prompt_embeds=EMB["pe"].to("cuda:0"), prompt_embeds_mask=EMB["pem"].to("cuda:0"),
                    prompt_embeds_2=EMB["pe2"].to("cuda:0"), prompt_embeds_mask_2=EMB["pem2"].to("cuda:0"),
                    negative_prompt_embeds=EMB["ne"].to("cuda:0"), negative_prompt_embeds_mask=EMB["nem"].to("cuda:0"),
                    negative_prompt_embeds_2=EMB["ne2"].to("cuda:0"), negative_prompt_embeds_mask_2=EMB["nem2"].to("cuda:0"),
                    height=H, width=W, num_frames=NF, num_inference_steps=steps,
                    generator=torch.Generator("cuda:0").manual_seed(SEED), output_type="latent")


def decode(lat, chunk=16):
    """Decode after the transformer is GONE, and in temporal chunks.

    The generation itself succeeded and then the decode died asking for 1.91 GiB with 1.52 free —
    because 7.6 GB of transformer was still sitting on the card doing nothing. It is not needed
    once the latents exist, and a model kept alive out of habit is the most expensive kind of
    habit. Freeing it turns 6.8 GB free into most of a card.

    Chunking is the belt to that brace: 61 frames of 480x480 is one very large allocation however
    much is free, so the latents are decoded a slice at a time and joined. The slices overlap by
    nothing — this VAE's temporal compression means a clean cut on a chunk boundary, and the
    seam that would matter is between LATENT frames, not pixel ones.
    """
    # Keep the device map BEFORE dropping the model, because the manifest still wants it. Deleting
    # `tr` here and then reading it forty minutes later cost a completed 61-frame render its JSON:
    # the render, the decode and every image survived, and the run still ended in a NameError on a
    # reporting line. A static scope check cannot see this one — `tr` IS bound at module level, and
    # it is a runtime `del` that unbinds it — which is exactly why the value is copied out first
    # rather than the name being reached for again.
    global tr, DEVICE_MAP
    DEVICE_MAP = {k: str(v) for k, v in (getattr(tr, "hf_device_map", {}) or {}).items()}
    pipe.transformer = None
    try:
        del tr
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()
    free0 = torch.cuda.mem_get_info(0)[0] / 2**30
    dev = "cuda:0" if free0 > 4.0 else "cuda:1"
    print(f"  transformer freed · decoding on {dev} ({free0:.1f}G free)", flush=True)
    pipe.vae.to(dev)
    n = lat.shape[2]
    outs = []
    with torch.inference_mode():
        for i in range(0, n, chunk):
            piece = lat[:, :, i:i + chunk].to(dev, pipe.vae.dtype)
            outs.append(pipe.vae.decode(piece, return_dict=False)[0].float().cpu())
            del piece; gc.collect(); torch.cuda.empty_cache()
            print(f"    decoded latent frames {i}..{min(i+chunk, n)} of {n}", flush=True)
    out = torch.cat(outs, dim=2)
    return pipe.video_processor.postprocess_video(out, output_type="np")[0]

# attention slicing if this transformer offers it — it trades a little speed for a much smaller
# attention peak, which is exactly the thing that is 1.9 GB over
for _fn in ("enable_attention_slicing", "set_attention_slice"):
    if hasattr(pipe, _fn):
        try:
            getattr(pipe, _fn)("auto" if _fn == "enable_attention_slicing" else 1)
            print(f"  {_fn} on", flush=True); break
        except Exception as _e:
            print(f"  {_fn} unavailable: {str(_e)[:60]}", flush=True)

per_step = None
for (px, nf) in SHAPES:
    H = W = px; NF = nf
    try:
        t0 = time.time(); _probe = run(2); per_step = (time.time() - t0) / 2
        del _probe; gc.collect(); torch.cuda.empty_cache()
        print(f"  {px}x{px}x{nf} fits · {per_step:.0f} s/step", flush=True)
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  {px}x{px}x{nf} OOM", flush=True)
        gc.collect(); torch.cuda.empty_cache()
assert per_step is not None, (
    "HunyuanVideo 1.5 does not fit a T4 pair at any of " + str(SHAPES) + " even sharded. The "
    "weights alone are 7.75 GB a card; there is nothing left to give. This is the answer.")
mem("after 2 steps")
print(f"\n  {per_step:.0f} s/step at {H}×{W}×{NF} WITH guidance", flush=True)
BUDGET_S = 130 * 60
STEPS = max(8, min(30, int(BUDGET_S / max(per_step, 1e-6))))
print(f"  → {STEPS} steps ≈ {per_step*STEPS/60:.0f} min", flush=True)

# %%
# The two-step probe fitting does not guarantee thirty steps will: the first attempt cleared the
# probe and then peaked over on the real run. So step down the ladder again on OOM rather than lose
# the session, keeping whatever shape actually completes.
lat = None
for (px, nf) in [(H, NF)] + [s for s in SHAPES if (s[0], s[1]) != (H, NF)]:
    H = W = px; NF = nf
    try:
        t0 = time.time(); lat = run(STEPS); gen_s = time.time() - t0
        print(f"  rendered at {px}x{px}x{nf}", flush=True); break
    except torch.cuda.OutOfMemoryError:
        print(f"  {px}x{px}x{nf} OOM on the full {STEPS}-step run — stepping down", flush=True)
        gc.collect(); torch.cuda.empty_cache()
assert lat is not None, "no shape completed the full run; the ladder is above"
arr = (np.clip(decode(lat.frames), 0, 1) * 255).round().astype(np.uint8)
print(f"  {len(arr)} frames in {gen_s/60:.1f} min", flush=True)
Image.fromarray(arr[0]).save(OUT / "frame0.png")
d = Path("/tmp/f"); d.mkdir(exist_ok=True)
for i, f in enumerate(arr): Image.fromarray(f).save(d / f"{i:04d}.png")
subprocess.run(f"ffmpeg -v error -framerate 16 -i '{d}/%04d.png' -c:v libx264 -crf 12 "
               f"-pix_fmt yuv420p '{OUT}/hunyuan_raw.mp4' -y", shell=True, check=False)
subprocess.run(f"ffmpeg -v error -i '{OUT}/hunyuan_raw.mp4' -c:v libvpx-vp9 -crf 30 -b:v 0 "
               f"-row-mt 1 -cpu-used 2 -pix_fmt yuv420p -an '{OUT}/hunyuan.webm' -y",
               shell=True, check=False)
idx = np.linspace(0, len(arr)-1, 8).round().astype(int)
Image.fromarray(np.concatenate([arr[i] for i in idx], 1)).save(OUT / "sheet.jpg", quality=88)
(Path("/kaggle/working") / "hunyuan_probe.json").write_text(json.dumps(
    {"model": REPO, "revision": REV, "seconds_per_step": round(per_step, 1), "steps": STEPS,
     "res": [H, W], "frames": int(len(arr)), "gen_minutes": round(gen_s/60, 1),
     "device_map": DEVICE_MAP,
     "prompt": PROMPT, "negative": NEG}, indent=2))
print("\nHUNYUAN: done — look at sheet.jpg and frame0.png", flush=True)
print(f"  total t+{(time.time()-T0)/60:.1f} min", flush=True)
