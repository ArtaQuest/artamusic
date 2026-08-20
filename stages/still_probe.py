# %% [markdown]
# # Is the BASE model reachable on this hardware?
#
# The cover ships stills from **Z-Image Turbo** — eight steps, no classifier-free guidance, no
# negative prompt. **Z-Image base** is the better model for a directed image: it takes real CFG and
# a real negative prompt, which is most of the difference between a catalogue frame and one that
# obeys a brief. It was tried once here and rendered at **56 seconds a step**, so two stills filled
# an hour and the stage timed out.
#
# That measurement was taken under `enable_sequential_cpu_offload`, which streams one submodule at
# a time across the PCIe bus, on a card with no hardware bf16. Those two costs multiply. But this
# notebook already runs a 14-billion-parameter video model on the same machine by putting **one
# component on each card** and never offloading anything — and nobody has tried that here.
#
# So: encode the prompts with the text encoder, free it, give the transformer a whole card to
# itself, decode on the other. Then time **two steps** and print seconds-per-step BEFORE committing
# to four images, because the way this fails is by taking an hour to tell you it is slow.
#
# The gated alternatives were considered and rejected. FLUX.1-Krea-dev is the obvious "more
# aesthetic" model and its GGUF build would sidestep the 4-bit dispatch problem that killed Krea 2
# here — but its base repo is gated, and so is FLUX.1-schnell, whose VAE and text encoders it would
# borrow. The ungated copies are unofficial re-uploads. A gated input fails our own "every input is
# public" check, and routing around a licence gate with a mirror is not a thing to do. Z-Image base
# is Apache-2.0 and ungated.

# %%
import json, os, sys, time
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
T0 = time.time()
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
SEED = 4242
REPO, REV = "Tongyi-MAI/Z-Image", "04cc4abb7c5069926f75c9bfde9ef43d49423021"

# %%
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.39.0",
                "transformers>=5.13", "accelerate", "safetensors", "sentencepiece", "protobuf",
                "ftfy"], check=True)
import numpy as np, torch
from PIL import Image
np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__} · {torch.cuda.device_count()} gpu(s) · cap {torch.cuda.get_device_capability()}", flush=True)

# %% [markdown]
# ## The same four briefs, unchanged
# A model comparison is only a comparison if nothing else moves: same briefs, same seed, same size.

# %%
SCENE = ("A cinematic still from a 70mm feature film. A single forged steel sword lies across a bed "
         "of white-hot coals in the black interior of a mountain forge at night. ")
LIGHT = ("The only light is the fire itself: a hard low key raking in from camera-left, the right "
         "third falling away into unlit black, thin volumetric shafts of smoke crossing the beam, "
         "a sharp rim of orange along the blade's spine. ")
OPTIC = ("Shot on 65mm Kodak Vision3 500T, anamorphic 40mm at T2.8, gentle horizontal flare, "
         "halation blooming off the hot metal, fine grain, deep shadow detail held, high contrast, "
         "solemn and monumental. An uncluttered frame with one subject and clean falloff to black.")
BRIEFS = {
    "low_hero": SCENE + ("The camera is low and close, almost at the level of the coals, looking up "
                         "the length of the blade so it runs diagonally out of the bottom-left "
                         "corner and the tip crosses the centre of the frame. The upper half is "
                         "quiet dark air and drifting smoke, left empty for a title. ") + LIGHT + OPTIC,
    "wide_forge": SCENE + ("A wide shot from across the workshop: the anvil and the hearth sit in "
                           "the lower third, the sword laid over the coals in silhouette against "
                           "their glow, and the whole upper two thirds is the forge's black "
                           "cavernous air with one shaft of light. ") + LIGHT + OPTIC,
    "edge_macro": SCENE + ("Extreme close on the blade's edge crossing the frame on a shallow "
                           "diagonal, the hardening line and hammer marks legible in the steel, "
                           "coals blurred to molten bokeh behind, sparks suspended. The top-right "
                           "quarter is empty darkness. ") + LIGHT + OPTIC,
    "overhead": SCENE + ("Directly overhead, looking straight down: the sword lies across the "
                         "glowing bed like a line drawn on fire, the coals a field of cracked "
                         "orange and grey around it, the frame square and symmetrical about the "
                         "blade with dark stone at the corners. ") + LIGHT + OPTIC,
}
# The thing Turbo cannot be told. Every item here is a defect seen in an actual take of this cover.
NEG = ("charcoal briquettes, barbecue, uniform round lumps, video game render, plastic sheen, "
       "flat even lighting, washed out, low contrast, grey haze, cluttered workshop, tools "
       "everywhere, multiple swords, hands, people, text, watermark, blurry, soft focus, "
       "oversaturated orange, cartoon, illustration, HDR glow")

# %% [markdown]
# ## One component per card
# The text encoder does its work first and is then thrown away, so the transformer gets a whole T4
# to itself and never streams. The VAE decodes on the second card in fp32.

# %%
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(REPO, revision=REV, torch_dtype=torch.bfloat16)
pipe.text_encoder.to("cuda:0")
embeds = {}
for name, brief in BRIEFS.items():
    pe, ne = pipe.encode_prompt(prompt=brief, device=torch.device("cuda:0"),
                                do_classifier_free_guidance=True, negative_prompt=NEG)
    embeds[name] = ([t.detach().cpu() for t in pe], [t.detach().cpu() for t in ne])
    print(f"  embedded {name}", flush=True)
pipe.text_encoder.to("cpu"); del pipe.text_encoder; pipe.text_encoder = None
import gc; gc.collect(); torch.cuda.empty_cache()

NG = torch.cuda.device_count()
DEC = "cuda:1" if NG >= 2 else "cuda:0"
pipe.transformer.to("cuda:0")
pipe.vae.to(DEC, torch.float32)
if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing("auto")
print(f"  transformer on cuda:0 · vae on {DEC} · "
      f"{torch.cuda.memory_allocated(0)/2**30:.1f} GB resident", flush=True)

# %% [markdown]
# ## Two steps, then decide
# This is the whole point of the probe. A model that needs 56 s/step cannot render four stills in a
# Kaggle session, and the honest way to learn that is to measure two steps and multiply.

# %%
H = W = 896
def render(name, steps):
    pe, ne = embeds[name]
    pe = [t.to("cuda:0") for t in pe]; ne = [t.to("cuda:0") for t in ne]
    lat = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=H, width=W,
               num_inference_steps=steps, guidance_scale=5.0, output_type="latent",
               generator=torch.Generator("cuda:0").manual_seed(SEED)).images
    lat = lat.to(DEC, torch.float32)
    with torch.no_grad():
        x = pipe.vae.decode(lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor,
                            return_dict=False)[0]
    return pipe.image_processor.postprocess(x, output_type="pil")[0]

t0 = time.time(); render("wide_forge", 2); per_step = (time.time() - t0) / 2
print(f"\n  {per_step:.1f} s/step  ->  40 steps = {per_step*40/60:.1f} min/still, "
      f"four stills = {per_step*40*4/60:.1f} min", flush=True)
BUDGET_MIN = 150
STEPS = 40
assert per_step * STEPS * 4 / 60 < BUDGET_MIN, (
    f"Z-Image base needs {per_step:.1f} s/step on this hardware — four stills would take "
    f"{per_step*STEPS*4/60:.0f} minutes, past the session. The pair-of-cards idea does not rescue "
    f"it; Turbo remains the only base-quality-per-minute that fits. This is the answer, not a bug.")

# %%
rec = {"model": f"Z-Image base ({REPO})", "revision": REV, "steps": STEPS, "guidance": 5.0,
       "seconds_per_step": round(per_step, 2), "candidates": {}}
for name in BRIEFS:
    t0 = time.time(); im = render(name, STEPS)
    f = OUT / f"base_{name}.png"; im.save(f)
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    g = a.mean(-1); gy, gx = np.gradient(g)
    rec["candidates"][name] = {"file": f.name, "seconds": round(time.time() - t0),
        "detail": round(float(np.sqrt(gx**2 + gy**2).mean() * 255), 2),
        "warm": round(float((a[..., 0] - a[..., 2]).clip(0).mean() * 255), 2),
        "dark_frac": round(float((g < 0.25).mean()), 3),
        "contrast": round(float(g.std() * 255), 2)}
    print(f"  {name}: {rec['candidates'][name]['seconds']}s  "
          f"contrast {rec['candidates'][name]['contrast']}", flush=True)

# A BLANK IMAGE IS NOT A RESULT — four pure-black stills once shipped from this pipeline because
# the numbers were recorded and never compared against.
dead = [n for n, c in rec["candidates"].items() if c["contrast"] < 2.0 or c["dark_frac"] > 0.995]
assert not dead, f"{len(dead)} stills are blank ({', '.join(dead)}) — refusing to report them"

sheet = np.concatenate([np.asarray(Image.open(OUT / f"base_{n}.png").convert("RGB").resize((448, 448)))
                        for n in BRIEFS], 1)
Image.fromarray(sheet).save(OUT / "base_sheet.jpg", quality=90)
(OUT / "base_stills.json").write_text(json.dumps(rec, indent=2))
print("\nBASE:", json.dumps(rec), flush=True)
print(f"total {(time.time()-T0)/60:.1f} min", flush=True)
