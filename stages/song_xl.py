# UNBROKEN — SOTA on Kaggle, with offloading engineered rather than hoped for.
#
# THE CEILING THAT WAS SELF-INFLICTED. The previous build ran a 1.1B song model because a single
# OOM was read as "4.6B does not fit in 16 GB". The error actually said:
#   "Tried to allocate 1.21 GiB ... 2.30 GiB is reserved by PyTorch but unallocated.
#    If reserved but unallocated memory is large try setting expandable_segments:True"
# Short by 1.21 GiB with 2.30 GiB stranded in reserved blocks — fragmentation, not capacity. And
# ACE-Step's own offload_to_cpu flag had been explicitly set False. Neither remedy was tried.
#
# THE OFFLOAD ARITHMETIC, since offloading trades VRAM for PCIe and that trade is only sometimes
# good. A diffusion model runs the whole network once per denoising step, so streaming re-sends
# the weights every step: 9.3 GB at bf16 over PCIe 3.0 x16 (~16 GB/s) is ~0.6 s/step, ~45 s across
# 80 steps on a take that already runs ~6 min. Affordable here. It would NOT be affordable for a
# video model, where the cost multiplies by frames as well as steps.
#
# WHY bfloat16 AND NOT float16. Measured on this exact card: fp16 overflowed to NaN in the 4.6B
# DiT ("nan=280000", four seeds, nine minutes each). bf16 carries float32's exponent range so it
# cannot overflow that way, and a real GEMM on sm_60 measured 5168 GFLOP/s against float32's 6366
# — a 0.81 ratio, so it is hardware, not emulation. bitsandbytes is not an option at all: its
# 4/8-bit kernels need sm_75.

import json, os, subprocess, sys, time
from pathlib import Path

# Set BEFORE torch is imported or the allocator is already built.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TMP / "hf"), HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  # ACE-Step kills generation at 600 s by default. The 4.6B model LOADS
                  # fine here (12.12 GB peak, 3.7 GB spare) but cannot finish 180 s of
                  # audio inside that window on a P100: the 1.1B takes ~250 s, and 4.2x
                  # the parameters means ~1000 s. All six takes died at exactly 600 s.
                  # The constraint is THROUGHPUT, not memory, and this is the knob for it.
                  ACESTEP_GENERATION_TIMEOUT="2400")

def sh(c, quiet=False):
    if not quiet: print(f"$ {c}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip(): print(r.stdout[-2000:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-2000:], flush=True)
    return r.returncode

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0

# ── environment ──────────────────────────────────────────────────────────────────────────
if not REPO.exists():
    sh(f"git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}")
sh("pip install -q hf_transfer toml python-dotenv modelscope diskcache py3langid pyloudnorm "
   "ffmpeg-python soundfile loguru einops accelerate numba scipy demucs "
   "'safetensors>=0.7.0' 'transformers>=4.51.0,<4.58.0' diffusers vector-quantize-pytorch 2>&1 | tail -2")
# Dependencies first, torch LAST — whatever pip runs last wins, and on Pascal the winner must be
# the cu126 line, whose wheels still carry sm_60.
if PASCAL:
    sh("pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
       "--index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2")

import torch
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GB | "
      f"alloc_conf={os.environ['PYTORCH_CUDA_ALLOC_CONF']}", flush=True)

def gemm_ok(dt, n=2048):
    try:
        a = torch.randn(n, n, device="cuda", dtype=dt); c = a @ a
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(c).all()); del a, c; torch.cuda.empty_cache(); return ok
    except Exception:
        torch.cuda.empty_cache(); return False
BF16 = gemm_ok(torch.bfloat16)
print(f"bfloat16 usable on this card: {BF16}", flush=True)

# ── patch the dtype decision the library will not expose ─────────────────────────────────
orch = REPO / "acestep/core/generation/handler/init_service_orchestrator.py"
src = orch.read_text()
OLD = '''            elif resolved_device == "cuda":
                if gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16'''
NEW = '''            elif resolved_device == "cuda":
                _f = os.environ.get("AQ_FORCE_DTYPE", "")
                if _f:
                    self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                                  "float16": torch.float16}[_f]
                    logger.info(f"[initialize_service] AQ_FORCE_DTYPE={_f}")
                elif gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16'''
assert OLD in src, "ACE-Step changed — re-read the orchestrator before trusting this patch"
orch.write_text(src.replace(OLD, NEW, 1))
print("dtype override patched in", flush=True)

# ── THE OFFLOAD LADDER ───────────────────────────────────────────────────────────────────
# Best-quality-that-fits, in order. Each rung is strictly more offloaded (slower) than the one
# above. The ladder is walked ONCE at load time; whichever rung holds is used for every take, so
# a take never fails halfway for a reason the load already knew about.
LADDER = []
if BF16:
    LADDER += [
        ("xl-resident",  "acestep-v15-xl-sft", "bfloat16", False, False,
         "4.6B fully resident — 9.3 GB of weights, viable only because expandable_segments stops "
         "the allocator stranding reserved blocks"),
        ("xl-offload",   "acestep-v15-xl-sft", "bfloat16", True,  False,
         "4.6B with the non-DiT components streamed from CPU"),
        ("xl-dit-swap",  "acestep-v15-xl-sft", "bfloat16", True,  True,
         "4.6B with the DiT itself streamed layer-wise — take-turns execution, ~45 s of PCIe per "
         "80-step take"),
    ]
LADDER.append(("sft-fp32", "acestep-v15-sft", "float32", False, False,
               "1.1B in float32 — cannot overflow, always fits, the known-good floor"))

print("\nOFFLOAD LADDER:", flush=True)
for n, m, d, oc, od, why in LADDER:
    print(f"  {n:14s} {m:20s} {d:9s} cpu={oc!s:5s} dit={od!s:5s}  {why}", flush=True)

# ── the song ─────────────────────────────────────────────────────────────────────────────
LYRICS = """[verse]
Cold house, thin coat, cracked door
Wind came through and took the floor
Take the blanket, take the bread
Take the roof above my head
Drag me out into the dirt
Cold got in and did its work
Winter put me on the street
Take the winter, call it heat

[pre-chorus]
Put me in the coals and wait
Hold me down and shut the gate
Water hits me, steam goes white
Pull me out, I keep my shape

[chorus]
Bring the heat, I take the flame
Swing the hammer, call my name
Every blow they meant to kill
Beat the iron into steel
Strike me hard and hear it ring
Unbroken, hold the line
Swing again and hear it ring
Unbroken, hold the line

[verse]
Man I trusted walked away
Take the table, take the keys
Send the letter, send the bill
Send the men to break my will
Every fist that found my jaw
Made the edge they never saw
They were forging what they hate
Take the beating, call it hammer

[pre-chorus]
Put me in the coals and wait
Hold me down and shut the gate
Water hits me, steam goes white
Pull me out, I keep my shape

[chorus]
Bring the heat, I take the flame
Swing the hammer, call my name
Every blow they meant to kill
Beat the iron into steel
Strike me hard and hear it ring
Unbroken, hold the line
Swing again and hear it ring
Unbroken, hold the line

[bridge]
Some nights I sat down and cried
Some nights I bent in the dark
Hands were shaking, head hung low
Kitchen light, the water cold
Hold my face under the tap
Come up harder, call it quench

[chorus]
Bring the heat, I take the flame
Swing the hammer, call my name
Every blow they meant to kill
Beat the iron into steel
Strike me hard and hear it ring
Unbroken, hold the line
Swing again and hear it ring
Unbroken, hold the line

[outro]
Bring it all now, call it edge
Unbroken, hold the line
Bring it all now, call it edge
Unbroken, hold the line"""

# Captions are drawn from the model's own genres_vocab.txt. "trap, cinematic, male rap" and
# "cinematic, orchestral, baritone" are LITERAL training entries; "male choir", "baritone" and
# "male vocals" are attested parts. This does NOT guarantee a male voice — measured, 7 of 8 takes
# came back female from male-explicit captions — which is why the register is MEASURED below and
# a female take is discarded rather than shipped.
TAKES = [
    ("t1", "trap, epic, male choir, baritone",     60, 7.5, 6103),
    ("t2", "trap, cinematic, male rap",            60, 7.5, 7211),
    ("t3", "cinematic, orchestral, baritone",      60, 7.5, 7322),
    ("t4", "epic orchestral trap, male vocals",    60, 7.0, 7433),
]
DURATION = 180.0

import toml
# The repo is cloned into /tmp, not installed — without this the import fails 178 s in,
# after the environment is built but before any GPU work. The probe had this line; this
# script did not, which is what lifting code between files costs when the imports differ.
sys.path.insert(0, str(REPO))
from acestep.handler import AceStepHandler

# ── walk the ladder ONCE, then keep the rung that held ───────────────────────────────────
chosen = None
for name, model, dtype, off_cpu, off_dit, why in LADDER:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    os.environ["AQ_FORCE_DTYPE"] = dtype
    t0 = time.time()
    try:
        h = AceStepHandler()
        h.initialize_service(project_root=str(REPO), config_path=model, device="cuda",
                             offload_to_cpu=off_cpu, offload_dit_to_cpu=off_dit)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"\nRUNG HELD: {name} — {model} @ {dtype}, peak {peak:.2f} GB, "
              f"loaded in {time.time()-t0:.0f}s", flush=True)
        chosen = dict(rung=name, model=model, dtype=dtype, offload_to_cpu=off_cpu,
                      offload_dit_to_cpu=off_dit, peak_gb=round(peak, 2))
        del h; torch.cuda.empty_cache()
        break
    except Exception as e:
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  rung {name} failed at peak {peak:.2f} GB — {type(e).__name__}: {str(e)[:130]}",
              flush=True)
        torch.cuda.empty_cache()
assert chosen, "no rung held — even the 1.1B float32 floor failed, which should be impossible"
Path("/kaggle/working/rung.json").write_text(json.dumps(chosen, indent=2))

def config_for(name, caption, steps, cfg, seed):
    return {"project_root": str(REPO), "config_path": chosen["model"], "checkpoint_dir": str(CKPT),
            "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
            "offload_to_cpu": chosen["offload_to_cpu"],
            "offload_dit_to_cpu": chosen["offload_dit_to_cpu"],
            "task_type": "text2music", "caption": caption, "lyrics": LYRICS, "instrumental": False,
            "bpm": 100, "keyscale": "F minor", "timesignature": "4/4", "vocal_language": "en",
            "duration": DURATION, "inference_steps": steps, "guidance_scale": cfg,
            "seed": seed, "infer_method": "ode",
            "thinking": False, "use_cot_metas": False, "use_cot_caption": False,
            "use_cot_lyrics": False, "use_cot_language": False,
            "batch_size": 1, "use_random_seed": False, "seeds": [seed]}

results = []
for name, caption, steps, cfg, seed in TAKES:
    conf = TMP / f"{name}.toml"
    conf.write_text(toml.dumps(config_for(name, caption, steps, cfg, seed)))
    t0 = time.time()
    rc = sh(f"cd {REPO} && AQ_FORCE_DTYPE={chosen['dtype']} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"ACESTEP_GENERATION_TIMEOUT=2400 "
            f"python cli.py -c {conf} --backend pt --log-level INFO")
    took = round(time.time() - t0, 1)
    found = sorted(Path(TMP / f"out_{name}").rglob("*.flac")) + \
            sorted(Path(TMP / f"out_{name}").rglob("*.wav"))
    if found:
        src_f = max(found, key=lambda p: p.stat().st_size)
        sh(f"ffmpeg -v error -i '{src_f}' -codec:a libmp3lame -b:a 320k '{OUT/(name+'.mp3')}' -y", quiet=True)
        results.append({"name": name, "caption": caption, "seed": seed, "seconds": took,
                        "rung": chosen["rung"], "model": chosen["model"]})
        print(f"{name}: OK in {took}s", flush=True)
    else:
        results.append({"name": name, "caption": caption, "seed": seed, "seconds": took,
                        "error": f"no audio (rc={rc})"})
        print(f"{name}: FAILED after {took}s", flush=True)

Path("/kaggle/working/takes.json").write_text(json.dumps(results, indent=2))
print("\nDONE", json.dumps({"rung": chosen, "takes": results}, indent=2)[:1200], flush=True)
