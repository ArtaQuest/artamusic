# %% [markdown]
# # STEEL — the cover, rebuilt
#
# The first cover was rejected on three counts, and each one is answered here by a change to the
# method rather than to the prompt.
#
# | Rejected | Why it happened | What this notebook does |
# |---|---|---|
# | "the dagger should not move" | an image‑to‑video model animates *everything*; the blade drifted 18 px, its silhouette swung 15%, and its own pixels moved 5× more than the rest of the frame | the sword is painted OUT of the still, a plate with nothing to deform is animated, and the real sword is composited back — frozen by construction, re‑lit each frame so it still flickers |
# | "make the still more epic" | a fast distilled 6B model, prompted in catalogue tag‑soup, lit flat and composed dead‑centre | **Krea 2 Turbo** (12.9B, trained with an explicit aesthetic reward) and a written brief that names medium, light direction, optics and composition |
# | "not good overall" | nothing measured what was actually wrong: whole‑frame statistics cannot see a 3% subject, and no gate looked at the picture at all | two new instruments — `stillness.py` (does the rigid thing move? plus a liveness counter‑gate, since a frozen photograph passes any stillness test) — and this run stops at the cover so it can be looked at before the song is spent |
#
# Everything is pinned by revision and the run refuses rather than shipping something it cannot
# defend. It ends with a contact sheet: four stills, the chosen loop, and the numbers.

# ── environment ──────────────────────────────────────────────────────────────────────────
# Kaggle's T4 pair is the right box for this: sm_75 has fp16 tensor cores (the P100 does not),
# bitsandbytes' cu128 wheels support it, and two 15 GB cards mean the image model and the video
# model never have to share one. The default torch is kept — reinstalling the Pascal line here
# would cost the tensor cores that make this affordable.
import gc, glob, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(TMP / "hf")
T_START = time.time()

def want_fast_downloads():
    """HF_HUB_ENABLE_HF_TRANSFER=1 is a REQUEST THAT CANNOT FAIL SOFTLY: huggingface_hub raises
    ValueError on every download if the package is missing, rather than falling back to plain
    HTTP. Asked for and absent, it killed a run at second thirty. Set it only if it is importable."""
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except Exception:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        return False

PINS = {
    # The image model. Krea 2 Turbo, 12.9B, in an NF4 diffusers layout that is ungated and 11.2 GB
    # — the GGUF builds of this model are ComfyUI-only (Krea2Transformer2DModel is absent from
    # diffusers' single-file table), so NF4 is the route that works from python.
    "image": ("OzzyGT/Krea_2_Turbo_bnb_nf4", "5458debf8356a6646a5aa814de28dcea881f8a6d"),
    "image_fallback": ("Tongyi-MAI/Z-Image", "04cc4abb7c5069926f75c9bfde9ef43d49423021"),
    "wan_base": ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "596658fd9ca6b7b71d5057529bbf319ecbc61d74"),
    "wan_gguf": ("jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
                 "338fb8eedd8f485c9188cf1b1de541721fc81d66"),
    "wan_high": "high_noise_1030/wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030-Q4_K_M.gguf",
    "wan_low": "low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf",
    "tools_sha": "3bceffe194ac0029a8c212480be7c40bf4550519",  # ArtaQuest/artamusic lib/{stillness,freeze}.py
}
SEED = 4242

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1200:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1200:], flush=True)
    return r.returncode

def clock(tag):
    print(f"  ⏱ {tag} · t+{(time.time()-T_START)/60:.1f} min", flush=True)

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0
sh("free -g | head -2; df -h /tmp | tail -1; nproc")

sh("pip install -q 'diffusers==0.39.0' 'transformers>=4.51.0,<4.58.0' accelerate safetensors "
   "sentencepiece protobuf 'gguf>=0.10.0' ftfy imageio imageio-ffmpeg bitsandbytes hf_transfer "
   "2>&1 | tail -2")
if PASCAL:
    # sm_60 needs the cu126 torch line, and bitsandbytes' cu128 wheels drop sm60 with it
    sh("pip install -q torch==2.7.1 torchvision==0.22.1 --index-url "
       "https://download.pytorch.org/whl/cu126 2>&1 | tail -1")

print("fast downloads:", want_fast_downloads(), flush=True)

import numpy as np
import torch
np.random.seed(SEED); torch.manual_seed(SEED)
import diffusers, transformers
NGPU = torch.cuda.device_count()
print(f"torch {torch.__version__} · diffusers {diffusers.__version__} · transformers "
      f"{transformers.__version__} · {NGPU} gpu(s) {[torch.cuda.get_device_name(i) for i in range(NGPU)]}",
      flush=True)
clock("environment ready")

# %% [markdown]
# ## The brief — written, not tagged
#
# The rejected still was prompted the way people prompted CLIP models in 2023: a comma-separated
# pile of quality words ("photorealistic, cinematic, ultra detailed, album cover"). Krea 2 is
# conditioned by a vision-language model, and its own guide asks for **one dense paragraph of
# natural language**, ordered subject → composition → light → style → crop.
#
# Four things separate an epic frame from a catalogue photograph, and each is stated as a fact
# rather than wished for:
#
# * **The medium is declared first** — "a cinematic still from a 70 mm feature", never "a photo of".
# * **The light names a source, a direction and a quality.** Catalogue light is soft, frontal and
#   shadowless; epic light is a single hard key from the side with the rest falling into black.
# * **The composition reserves space.** A cover needs somewhere for type to live, and a subject
#   that does not sit dead centre by accident.
# * **The optics are specific.** A 65 mm anamorphic at T2.8 with halation reads as cinema; "8k
#   masterpiece" reads as nothing at all.
#
# Krea 2 Turbo is guidance‑free and takes **no negative prompt**, so every exclusion is phrased as
# a positive fact — "an uncluttered frame falling to black", not "no clutter". Four different
# framings of one scene are rendered, so the sheet offers real alternatives rather than four
# seeds of the same idea.

# ── the four briefs ──────────────────────────────────────────────────────────────────────
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
HUMAN_PICK = "low_hero"       # the operator's, and overridable — the scorer only ever advises
(WORK / "briefs.json").write_text(json.dumps(BRIEFS, indent=2))
for k, v in BRIEFS.items():
    print(f"\n[{k}] {v[:150]}…", flush=True)

# %% [markdown]
# ## The two stages, each in its own process
#
# Nine runs of the previous notebook died before this lesson stuck: a notebook that loads a model
# keeps its host memory after the model is freed, and the next stage is then killed by the OOM
# killer — twice so hard that Kaggle saved neither log nor outputs. The weights of three model
# stages also cannot coexist on the container's disk, so each stage drops its own when it is done.
#
# So the notebook below never loads a model. It writes a stage script, runs it, and reads JSON
# back. One notebook, one Run All, and no model in this process.

# ── the stage script ─────────────────────────────────────────────────────────────────────
STAGE_SRC = r'''#!/usr/bin/env python3
import gc, glob, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
CFG = json.load(open("/tmp/aq_cfg.json"))
PINS, SEED = CFG["pins"], CFG["seed"]
TMP, WORK, OUT = Path(CFG["tmp"]), Path(CFG["work"]), Path(CFG["out"])
os.environ["HF_HOME"] = CFG["hf_home"]
try:
    import hf_transfer  # noqa: F401       # asking for it without having it is a hard error
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except Exception:
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
sys.path.insert(0, CFG["tools"])
T0 = time.time()

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
np.random.seed(SEED); torch.manual_seed(SEED)
NGPU = torch.cuda.device_count()
CAP = torch.cuda.get_device_capability()
print(f"[{sys.argv[1]}] torch {torch.__version__} · cap {CAP} · {NGPU} gpu(s)", flush=True)

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1000:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1000:], flush=True)
    return r.returncode

def drop(*needles):
    hub = Path(os.environ["HF_HOME"]) / "hub"
    freed = 0.0
    for d in list(hub.glob("models--*")):
        if any(n.lower() in d.name.lower() for n in needles):
            sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30
            shutil.rmtree(d, ignore_errors=True); freed += sz
            print(f"  dropped {d.name} ({sz:.1f} GB)", flush=True)
    print(f"  [drop] freed {freed:.1f} GB", flush=True)

def sha20(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:20]


def stage_still():
    # Krea 2 Turbo, NF4, resident. Two things decide whether this is fast or unusable on a T4:
    # the repo bakes bnb_4bit_compute_dtype="bfloat16", and neither the T4 nor the P100 has bf16
    # hardware — left alone, every quantised matmul falls back to fp32. And the VAE stays fp32,
    # which costs almost nothing and is the documented cure for black decodes.
    repo, rev = PINS["image"]
    ok = True
    try:
        p = snapshot_download(repo, revision=rev)
    except Exception as e:
        print(f"  image repo unavailable ({str(e)[:100]})", flush=True); ok = False
    imgs, model_used = {}, None
    if ok:
        try:
            for sub in ("transformer/config.json", "text_encoder/config.json"):
                f = Path(p) / sub
                if f.exists():
                    c = json.loads(f.read_text())
                    q = c.get("quantization_config") or {}
                    if q.get("bnb_4bit_compute_dtype") == "bfloat16":
                        q["bnb_4bit_compute_dtype"] = "float16"
                        c["quantization_config"] = q
                        f.write_text(json.dumps(c))
                        print(f"  patched {sub}: compute dtype -> float16", flush=True)
            from diffusers import Krea2Pipeline
            pipe = Krea2Pipeline.from_pretrained(p, torch_dtype=torch.float16)
            pipe.to("cuda")
            pipe.vae.to(torch.float32)
            if hasattr(pipe, "enable_vae_tiling"): pipe.enable_vae_tiling()
            model_used = f"Krea-2-Turbo NF4 ({repo})"
            print("  Krea 2 Turbo ready (NF4, resident, fp16 compute)", flush=True)
            for name, brief in CFG["briefs"].items():
                t0 = time.time()
                im = pipe(prompt=brief, height=1024, width=1024, num_inference_steps=8,
                          guidance_scale=0.0,
                          generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
                imgs[name] = im
                print(f"  {name}: {time.time()-t0:.0f}s", flush=True)
            del pipe
        except Exception as e:
            print(f"  Krea 2 failed ({type(e).__name__}: {str(e)[:200]}) — falling back", flush=True)
            imgs, model_used = {}, None
            gc.collect(); torch.cuda.empty_cache()
    if not imgs:
        # Z-Image BASE, not Turbo: it takes a real negative prompt and CFG, which Turbo cannot.
        from diffusers import ZImagePipeline
        repo, frev = PINS["image_fallback"]
        pipe = ZImagePipeline.from_pretrained(repo, revision=frev, torch_dtype=torch.float16)
        pipe.to("cuda"); pipe.vae.to(torch.float32)
        if hasattr(pipe, "enable_vae_tiling"): pipe.enable_vae_tiling()
        NEG = ("flat frontal studio lighting, evenly lit backdrop, stock photo staging, centred "
               "symmetrical subject, cluttered background, oversaturated, plastic, watermark, text")
        model_used = f"Z-Image base ({repo})"
        print("  Z-Image base ready (fp16 resident)", flush=True)
        for name, brief in CFG["briefs"].items():
            t0 = time.time()
            im = pipe(prompt=brief, negative_prompt=NEG, height=1024, width=1024,
                      num_inference_steps=32, guidance_scale=4.0,
                      generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
            imgs[name] = im
            print(f"  {name}: {time.time()-t0:.0f}s", flush=True)
        del pipe
    rec = {"model": model_used, "candidates": {}}
    for name, im in imgs.items():
        f = OUT / f"still_{name}.png"; im.save(f)
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        g = a.mean(-1); gy, gx = np.gradient(g)
        rec["candidates"][name] = {
            "file": f.name,
            "detail": round(float(np.sqrt(gx**2 + gy**2).mean() * 255), 2),
            "warm": round(float((a[..., 0] - a[..., 2]).clip(0).mean() * 255), 2),
            "dark_frac": round(float((g < 0.25).mean()), 3),
            "contrast": round(float(g.std() * 255), 2)}
    pick = CFG["human_pick"] if CFG["human_pick"] in imgs else list(imgs)[0]
    rec["shipped"] = pick
    imgs[pick].save(OUT / "cover.png")
    sh(f"ffmpeg -v error -i '{OUT}/cover.png' -vf scale=3000:3000:flags=lanczos '{OUT}/cover_3000.png' -y", quiet=True)
    (WORK / "stills.json").write_text(json.dumps(rec, indent=2))
    print("STILL:", json.dumps(rec)[:400], flush=True)
    drop("Krea", "Z-Image")
    gc.collect(); torch.cuda.empty_cache()


def stage_loop():
    import freeze as F
    import stillness as S
    from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler,
                           GGUFQuantizationConfig, WanImageToVideoPipeline, WanTransformer3DModel)
    from transformers import AutoTokenizer, UMT5EncoderModel
    import ftfy, html

    still = np.asarray(Image.open(OUT / "cover.png").convert("RGB"), dtype=np.uint8)
    W = H = 640
    small = np.asarray(Image.fromarray(still).resize((W, H), Image.LANCZOS), dtype=np.uint8)

    # The masks. Everything the operator's complaint depends on lives in these three lines: the
    # sword is found by connectivity and elongation (colour alone claimed 20% of the frame), the
    # grip and pommel are caught by extending along the sword's own axis, and the plate is the
    # still with the sword painted out so the model has nothing to deform.
    blade = F.steel_mask(small)
    sword = F.feather(F.extend_along_axis(blade, small), 3) > 0.3
    coals = F.fire_mask(small, plume=25)
    plate = F.clean_plate(small, sword, grow=8)
    Image.fromarray(plate).save(OUT / "clean_plate.png")
    prev = small.copy(); prev[sword] = (0.35 * prev[sword] + np.array([0, 0, 170])).clip(0, 255)
    Image.fromarray(np.concatenate([small, prev.astype(np.uint8), plate], 1)).save(OUT / "mask_check.jpg")
    print(f"  sword mask {100*sword.mean():.1f}% · coals {100*coals.mean():.1f}%", flush=True)

    base, brev = PINS["wan_base"]; gg, grev = PINS["wan_gguf"]
    BASE = snapshot_download(base, revision=brev,
                             allow_patterns=["model_index.json", "scheduler/*", "vae/*", "tokenizer/*",
                                             "text_encoder/*", "transformer/config.json",
                                             "transformer_2/config.json"])
    HIGH = hf_hub_download(gg, PINS["wan_high"], revision=grev)
    LOW = hf_hub_download(gg, PINS["wan_low"], revision=grev)
    hashes = {"high": sha20(HIGH), "low": sha20(LOW)}
    print("  wan sha256:", hashes, flush=True)

    PROMPT = ("Photograph, locked-off tripod shot of a blacksmith's forge at night. A bed of glowing "
              "orange coals burns in a stone hearth. The coals pulse and breathe, heat haze shimmers "
              "above them, thin grey smoke drifts and curls upward, embers glow and fade. The camera "
              "does not move. Cinematic, photorealistic, fine film grain.")
    NEG = ("camera movement, pan, zoom, dolly, handheld shake, cut, scene change, morphing, melting, "
           "people, hands, text, watermark, blurry, low quality, oversaturated, cartoon, flicker")

    def clean(t):
        return re.sub(r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(t)))).strip()

    tok = AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(BASE, subfolder="text_encoder",
                                          torch_dtype=torch.bfloat16).to("cuda:0").eval()
    def embed(t, n=512):
        ids = tok([clean(t)], padding="max_length", max_length=n, truncation=True,
                  add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        k = int(ids.attention_mask.gt(0).sum(1)[0])
        with torch.no_grad():
            h = te(ids.input_ids.to("cuda:0"), ids.attention_mask.to("cuda:0")).last_hidden_state[0].float().cpu()
        return torch.cat([h[:k], h.new_zeros(n - k, h.size(1))])[None]
    PE, NE = embed(PROMPT).to("cuda:0"), embed(NEG).to("cuda:0")
    del te; gc.collect(); torch.cuda.empty_cache()

    class Pinned(FlowMatchEulerDiscreteScheduler):
        PRE = [1.0, 0.75, 0.5, 0.25]        # lightx2v's trained points under shift 5
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None,
                          timesteps=None):
            return super().set_timesteps(device=device, sigmas=list(self.PRE))

    q = GGUFQuantizationConfig(compute_dtype=torch.float16)
    hi = WanTransformer3DModel.from_single_file(HIGH, quantization_config=q, config=BASE,
                                                subfolder="transformer", torch_dtype=torch.float16)
    lo = WanTransformer3DModel.from_single_file(LOW, quantization_config=q, config=BASE,
                                                subfolder="transformer_2", torch_dtype=torch.float16)
    vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(BASE, transformer=hi, transformer_2=lo, vae=vae,
                                                   text_encoder=None, tokenizer=None,
                                                   torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    if NGPU >= 2:
        hi.to("cuda:0"); vae.to("cuda:0"); lo.to("cuda:1")
        _f = lo.forward
        def across(*a, **k):
            a = [x.to("cuda:1") if torch.is_tensor(x) else x for x in a]
            k = {n: (v.to("cuda:1") if torch.is_tensor(v) else v) for n, v in k.items()}
            o = _f(*a, **k)
            return o.__class__(sample=o.sample.to("cuda:0")) if hasattr(o, "sample") else o
        lo.forward = across
        print("  one expert per card", flush=True)
    else:
        pipe.enable_model_cpu_offload()
    pipe.scheduler = Pinned(shift=5.0)

    NF, FPS, XF = 65, 16, 6                 # shorter than 81: drift compounds with frame count
    img = Image.fromarray(plate)
    t0 = time.time()
    out = pipe(image=img, last_image=img, prompt_embeds=PE, negative_prompt_embeds=NE,
               height=H, width=W, num_frames=NF, num_inference_steps=4, guidance_scale=1.0,
               generator=torch.Generator("cuda:0").manual_seed(SEED), output_type="latent")
    if NGPU < 2:
        pipe.transformer.to("cpu"); pipe.transformer_2.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
    v = pipe.vae.to("cuda:0")
    lat = out.frames.to(v.dtype)
    mean = torch.tensor(v.config.latents_mean).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    std = 1.0 / torch.tensor(v.config.latents_std).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    with torch.no_grad():
        dec = v.decode(lat / std + mean, return_dict=False)[0]
    frames = (np.clip(pipe.video_processor.postprocess_video(dec, output_type="np")[0], 0, 1) * 255).round().astype(np.uint8)
    gen_s = time.time() - t0
    print(f"  {len(frames)} frames in {gen_s:.0f}s", flush=True)
    del pipe, hi, lo; gc.collect(); torch.cuda.empty_cache()
    drop("Wan-AI", "jayn7")

    # freeze the sword back in, lit by the fire the model made
    lit = F.freeze_lit(frames, small, sword, coals, radius=3)
    L = lit[:-1]
    w = (np.arange(1, XF + 1) / (XF + 1))[:, None, None, None]
    blend = ((1 - w) * L[-XF:].astype(np.float32) + w * L[:XF].astype(np.float32)).round().astype(np.uint8)
    loop = np.concatenate([L[XF:len(L) - XF], blend])

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
        sh(f"ffmpeg -v error -i '{base}_raw.mp4' -vf scale=1080:1080:flags=lanczos -c:v libvpx-vp9 "
           f"-crf 33 -b:v 0 -row-mt 1 -cpu-used 1 -pix_fmt yuv420p -an '{base}_1080.webm' -y", quiet=True)

    encode(loop, str(OUT / "STEEL_cover_loop"))
    encode(np.concatenate([frames[:-1]]), str(OUT / "STEEL_cover_loop_unfrozen"))   # the comparison
    idx = np.linspace(0, len(loop) - 1, 8).round().astype(int)
    Image.fromarray(np.concatenate([loop[i] for i in idx], 1)).save(OUT / "loop_sheet.jpg", quality=88)
    Image.fromarray(np.concatenate([loop[i] for i in (-3, -2, -1, 0, 1, 2)], 1)).save(OUT / "loop_seam.jpg", quality=90)

    rec = {"model": "Wan2.2-I2V-A14B lightx2v-4step Q4_K_M", "hashes": hashes, "seed": SEED,
           "frames": int(len(loop)), "fps": FPS, "gen_seconds": round(gen_s, 1),
           "method": "sword painted out of the still; the plate animated; the sword composited back "
                     "frozen and re-lit per frame from the coals",
           "sword_mask_pct": round(100 * float(sword.mean()), 2),
           "frozen": S.measure(str(OUT / "STEEL_cover_loop.webm"), mask=blade),
           "unfrozen": S.measure(str(OUT / "STEEL_cover_loop_unfrozen.webm"), mask=blade),
           "alive": S.liveness(str(OUT / "STEEL_cover_loop.webm"), coals, blade)}
    rec["verdict_still"] = S.verdict(rec["frozen"])
    rec["verdict_alive"] = S.liveness_verdict(rec["alive"])
    (WORK / "loop_verify.json").write_text(json.dumps(rec, indent=2))
    print("LOOP:", json.dumps(rec), flush=True)


if __name__ == "__main__":
    {"still": stage_still, "loop": stage_loop}[sys.argv[1]]()
    print(f"[{sys.argv[1]}] done in {(time.time()-T0)/60:.1f} min", flush=True)
'''

# ── run the stages ───────────────────────────────────────────────────────────────────────
TOOLS = TMP / "tools"; TOOLS.mkdir(exist_ok=True)
import urllib.request
for f in ("stillness.py", "freeze.py"):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['tools_sha']}/lib/{f}",
        str(TOOLS / f))
sys.path.insert(0, str(TOOLS))
import stillness as S
assert S.selftest(), "the stillness instrument fails its own selftest — no number here is trustworthy"

Path("/tmp/aq_stage.py").write_text(STAGE_SRC)
Path("/tmp/aq_cfg.json").write_text(json.dumps({
    "pins": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PINS.items()},
    "seed": SEED, "tmp": str(TMP), "work": str(WORK), "out": str(OUT),
    "hf_home": str(TMP / "hf"), "tools": str(TOOLS),
    "briefs": BRIEFS, "human_pick": HUMAN_PICK}))

def run_stage(name, minutes):
    t0 = time.time()
    print(f"\n=== stage {name} (own process) ===", flush=True)
    r = subprocess.run([sys.executable, "/tmp/aq_stage.py", name], text=True, timeout=minutes * 60)
    print(f"stage {name}: rc={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode

assert run_stage("still", 75) == 0, "the still stage failed — see its output above"
stills = json.loads((WORK / "stills.json").read_text())
print("stills by:", stills["model"], "· shipped:", stills["shipped"], flush=True)
clock("stills done")

assert run_stage("loop", 120) == 0, "the loop stage failed — see its output above"
loop = json.loads((WORK / "loop_verify.json").read_text())
clock("loop done")

# %% [markdown]
# ## The verdict — and the comparison that proves it
#
# The loop is measured twice: once as the model made it, and once after the sword is composited
# back. Both numbers are published, because "the blade holds still now" is only a claim worth
# anything beside the number it used to be. The liveness counter-gate runs on the same file — a
# frozen photograph would pass every stillness test ever written, so the fire must be shown to
# move and the firelight must be shown to still play on the steel.

# ── verdict ──────────────────────────────────────────────────────────────────────────────
fr, un, al = loop["frozen"], loop["unfrozen"], loop["alive"]
print(f"\n{'':22s} {'as generated':>14s} {'after the freeze':>18s}")
for k, label in (("drift_px", "sword drift (px)"), ("max_shift_px", "worst frame (px)"),
                 ("max_dev", "change in its pixels"), ("ratio", "its motion / the rest")):
    print(f"{label:22s} {un[k]:>14} {fr[k]:>18}")
print(f"\nfire motion {al['fire_motion']} · firelight on the steel {al['subject_light_std']} "
      f"· colour swing {al['subject_chroma_std']}", flush=True)

manifest = {f.name: {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:32]}
            for f in sorted(OUT.iterdir()) if f.is_file()}
(WORK / "manifest.json").write_text(json.dumps(manifest, indent=2))

problems = list(loop["verdict_still"]) + list(loop["verdict_alive"])
(WORK / "cover_verdict.json").write_text(json.dumps(
    {"stills": stills, "loop": loop, "problems": problems}, indent=2))
if problems:
    print("\nREFUSED:", "; ".join(problems), flush=True)
else:
    print("\nThe sword holds still, the fire lives, and the loop closes.", flush=True)
assert not problems, "the cover does not meet its own gates: " + "; ".join(problems)
clock("DONE")
