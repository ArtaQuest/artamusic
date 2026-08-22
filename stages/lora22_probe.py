# %% [markdown]
# # Does the LEGO LoRA attach to Wan2.2-A14B — and does it change the frames?
#
# The adapter is trained for Wan2.1-T2V-14B. It should reach Wan2.2-A14B because the two
# transformers are the same shape — 40 layers, inner dim 5120, ffn 13824 — and the adapter's
# `lora_A` is `[32, 5120]` against both. Wan2.2 is a mixture of TWO experts, each denoising half
# the schedule, so an adapter on `transformer` alone would leave the second half unstyled.
#
# Attaching is not the question, though. A LoRA can load, report success, and change nothing —
# `set_adapters` can silently no-op, and a quantised base can swallow the delta. So this probe
# generates the SAME seed twice, once with the adapter disabled and once enabled, and measures
# the difference between the two clips. No difference means not applied, whatever the log said.

# %%
import gc, json, os, subprocess, sys, time
from pathlib import Path
T0 = time.time()
def sh(c): subprocess.run(c, shell=True, check=True)
sh(f"{sys.executable} -m pip install -q -U 'torchao>=0.16.0' 'safetensors>=0.7.0' "
   f"'transformers>=4.51.0,<4.58.0' diffusers==0.39.0 gguf accelerate peft hf_transfer")
os.environ.update(HF_HOME="/tmp/hf", HF_HUB_ENABLE_HF_TRANSFER="1")

import numpy as np, torch, diffusers, torchao
from huggingface_hub import hf_hub_download
print(f"torch {torch.__version__} · diffusers {diffusers.__version__} · torchao {torchao.__version__} "
      f"· gpus {torch.cuda.device_count()}", flush=True)

PINS = {"wan_base": ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "main"),
        "wan_gguf": ("QuantStack/Wan2.2-T2V-A14B-GGUF", "main"),
        "wan_high": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
        "wan_low":  "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
        "lego_lora": ("Remade-AI/Lego", "3f7938015b2537238f9e4f17b8896ddceac9cbe7"),
        "lora_file": "lego_35_epochs.safetensors"}
PROMPT = ("l3g0_5ty13 Lego animation style, a Lego minifigure blacksmith with a brown beard and a "
          "leather apron hammers a glowing translucent orange Lego sword on a Lego anvil in a Lego "
          "castle forge, glossy plastic bricks with visible studs, stop-motion Lego animation")
NEG = "photorealistic, live action, human skin, realistic hands, blurry, text, watermark"

# %%
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan, GGUFQuantizationConfig
from transformers import UMT5EncoderModel, AutoTokenizer
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
print(f"  prompt encoded · t+{(time.time()-T0)/60:.1f} min", flush=True)

# %%
GREPO, GREV = PINS["wan_gguf"]
HIGH = hf_hub_download(GREPO, PINS["wan_high"], revision=GREV)
LOW  = hf_hub_download(GREPO, PINS["wan_low"],  revision=GREV)
q = GGUFQuantizationConfig(compute_dtype=torch.float16)
hi = WanTransformer3DModel.from_single_file(HIGH, quantization_config=q, config=BASE,
                                            subfolder="transformer", torch_dtype=torch.float16)
lo = WanTransformer3DModel.from_single_file(LOW, quantization_config=q, config=BASE,
                                            subfolder="transformer_2", torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(BASE, revision=BREV, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(BASE, revision=BREV, transformer=hi, transformer_2=lo, vae=vae,
                                   text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
NGPU = torch.cuda.device_count()
if NGPU >= 2:
    hi.to("cuda:0"); vae.to("cuda:0"); lo.to("cuda:1")
    _f = lo.forward
    def across(*a, **k):
        a = [x.to("cuda:1") if torch.is_tensor(x) else x for x in a]
        k = {n: (v.to("cuda:1") if torch.is_tensor(v) else v) for n, v in k.items()}
        o = _f(*a, **k)
        if isinstance(o, tuple):
            return tuple(x.to("cuda:0") if torch.is_tensor(x) else x for x in o)
        if torch.is_tensor(o):
            return o.to("cuda:0")
        return o.__class__(sample=o.sample.to("cuda:0"))
    lo.forward = across
else:
    pipe.enable_model_cpu_offload()
print(f"  experts loaded · t+{(time.time()-T0)/60:.1f} min", flush=True)

# %%
LREPO, LREV = PINS["lego_lora"]
LORA = hf_hub_download(LREPO, PINS["lora_file"], revision=LREV)
APPLIED, ERR, WHERE = False, None, []
try:
    pipe.load_lora_weights(LORA, adapter_name="lego")
    WHERE.append("transformer")
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.load_lora_weights(LORA, adapter_name="lego", load_into_transformer_2=True)
        WHERE.append("transformer_2")
    APPLIED = True
    print(f"  LoRA loaded into: {', '.join(WHERE)}", flush=True)
except Exception as e:
    ERR = f"{type(e).__name__}: {e}"
    print(f"  LoRA did NOT load: {ERR[:300]}", flush=True)

# COUNT THE INJECTED LAYERS. "Loaded" is the library's word for it; the modules are the fact.
def count(m):
    return sum(1 for n, _ in m.named_modules() if n.endswith("lora_A.lego"))
NLAYERS = {"transformer": count(hi), "transformer_2": count(lo)}
print("  lora_A modules per expert:", NLAYERS, flush=True)

# %%
def run(steps=4, nf=17, hw=384):
    g = torch.Generator("cuda").manual_seed(4242)
    with torch.inference_mode():
        out = pipe(prompt_embeds=PE.to("cuda:0", torch.float16),
                   negative_prompt_embeds=NE.to("cuda:0", torch.float16),
                   height=hw, width=hw, num_frames=nf, num_inference_steps=steps,
                   guidance_scale=4.0, guidance_scale_2=3.0, generator=g, output_type="np")
    return np.asarray(out.frames[0])

# THE SAME SEED, ADAPTER OFF THEN ON. If the frames are identical the adapter is decorative.
try:
    pipe.set_adapters(["lego"], adapter_weights=[0.0])
    off = run()
    pipe.set_adapters(["lego"], adapter_weights=[1.0])
    on = run()
    d = float(np.abs(on.astype(np.float32) - off.astype(np.float32)).mean())
    rel = d / max(float(off.astype(np.float32).std()), 1e-6)
    CHANGED = rel > 0.02
    print(f"  adapter off vs on: mean abs diff {d:.4f} ({rel:.3f} of a std) -> "
          f"{'THE ADAPTER CHANGES THE FRAMES' if CHANGED else 'NO EFFECT — decorative'}", flush=True)
except Exception as e:
    CHANGED, d, rel = None, None, None
    print(f"  A/B failed: {type(e).__name__}: {e}", flush=True)

VERDICT = {"lora_loaded": APPLIED, "lora_error": ERR, "loaded_into": WHERE,
           "lora_A_modules": NLAYERS, "changes_frames": CHANGED,
           "mean_abs_diff": d, "diff_over_std": rel,
           "minutes": round((time.time() - T0) / 60, 1)}
Path("/kaggle/working/verdict.json").write_text(json.dumps(VERDICT, indent=1))
print("VERDICT:", json.dumps(VERDICT), flush=True)
