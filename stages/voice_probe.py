# UNBROKEN — the VOICE experiment: mechanism vs lottery.
#
# Across 12 takes and two model sizes, male-explicit captions produced ONE male vocal. Captions
# are not a control. ACE-Step exposes an actual mechanism: task_type="cover" conditions on
# reference_audio with audio_cover_strength — and a verified-male take exists (the published
# 1.1B UNBROKEN, 158 Hz on an isolated stem). It is mounted from the PUBLIC kernel
# arash0ash/unbroken, so the reference itself stays public — which the ArtaQuest checklist
# demands of every input anyway.
#
# Block 1 probes cover strengths (the docs are vague about semantics, so measure, never guess):
# same seed, three strengths, v1 lyric IDENTICAL to what the reference sings — one variable.
# Block 2 is the fallback lottery, solo-voice captions only (choir words in the caption produced
# multi-voice stems that made register UNMEASURABLE: IQR 215 on t1), sung from the v2 lyric so
# the takes double as machine-transcription evidence for the provisional 'Never broken' hook.
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
import glob as _glob
_ref = sorted(_glob.glob("/kaggle/input/**/UNBROKEN.mp3", recursive=True))
MALE_REF = _ref[0] if _ref else None
print("male reference:", MALE_REF or "NOT FOUND", flush=True)
assert MALE_REF, ("the male reference is not mounted — this kernel is POINTLESS without it; "
                  "fail loudly at second 5 rather than skip-complete at minute 40")
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
    # Cover-only re-run: the solo block already rendered in v1 (s1-s3, fetched). This version
    # exists because v1 went up without the kernel source mounted and the cover rows self-skipped
    # exactly as designed — better a visible skip than a silent text2music fallback.
    ("c090", "cover", "trap, epic, male choir, baritone", "v1", 0.90, 60, 7.5, 6103),
    ("c060", "cover", "trap, epic, male choir, baritone", "v1", 0.60, 60, 7.5, 6103),
    ("c030", "cover", "trap, epic, male choir, baritone", "v1", 0.30, 60, 7.5, 6103),
]
LYRICS_V2 = """[verse]
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
Never broken, hold the line
Swing again and hear it ring
Never broken, hold the line

[verse]
Man I trusted walked away
They were forging what they hate
Send the notice, send the bill
Send the men to break my will
Every fist that found my jaw
Made the edge they never saw
Take the table, take the ladder
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
Never broken, hold the line
Swing again and hear it ring
Never broken, hold the line

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
Never broken, hold the line
Swing again and hear it ring
Never broken, hold the line

[outro]
Bring it all now, call it edge
Never broken, hold the line
Bring it all now, call it edge
Never broken, hold the line"""

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

def config_for(name, task, caption, lyr_key, strength, steps, cfg, seed):
    c = {"project_root": str(REPO), "config_path": chosen["model"], "checkpoint_dir": str(CKPT),
         "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
         "offload_to_cpu": chosen["offload_to_cpu"],
         "offload_dit_to_cpu": chosen["offload_dit_to_cpu"],
         "task_type": task, "caption": caption,
         "lyrics": LYRICS if lyr_key == "v1" else LYRICS_V2, "instrumental": False,
         "bpm": 100, "keyscale": "F minor", "timesignature": "4/4", "vocal_language": "en",
         "duration": DURATION, "inference_steps": steps, "guidance_scale": cfg,
         "seed": seed, "infer_method": "ode",
         "thinking": False, "use_cot_metas": False, "use_cot_caption": False,
         "use_cot_lyrics": False, "use_cot_language": False,
         "batch_size": 1, "use_random_seed": False, "seeds": [seed]}
    if task == "cover":
        c["reference_audio"] = MALE_REF
        c["audio_cover_strength"] = strength
    return c

results = []
for name, task, caption, lyr_key, strength, steps, cfg, seed in TAKES:
    if task == "cover" and not MALE_REF:
        results.append({"name": name, "skipped": "no male reference mounted"}); continue
    conf = TMP / f"{name}.toml"
    conf.write_text(toml.dumps(config_for(name, task, caption, lyr_key, strength, steps, cfg, seed)))
    t0 = time.time()
    rc = sh(f"cd {REPO} && AQ_FORCE_DTYPE={chosen['dtype']} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"ACESTEP_GENERATION_TIMEOUT=2400 "
            f"python cli.py -c {conf} --backend pt --log-level INFO")
    took = round(time.time() - t0, 1)
    found = sorted(Path(TMP / f"out_{name}").rglob("*.flac")) + \
            sorted(Path(TMP / f"out_{name}").rglob("*.wav"))
    row = {"name": name, "task": task, "caption": caption, "lyrics": lyr_key,
           "cover_strength": strength, "steps": steps, "cfg": cfg, "seed": seed,
           "seconds": took, "rung": chosen["rung"], "model": chosen["model"],
           "dtype": chosen["dtype"], "timeout_s": 2400}
    if found:
        src_f = max(found, key=lambda p: p.stat().st_size)
        rc2 = sh(f"ffmpeg -v error -i '{src_f}' -codec:a libmp3lame -b:a 320k '{OUT/(name+'.mp3')}' -y", quiet=True)
        mp3 = OUT / (name + ".mp3")
        # A take is only OK if the mp3 EXISTS and is big enough to be 3 minutes of audio — a row
        # that says OK without the artefact verified is how a broken take ships.
        if rc2 == 0 and mp3.exists() and mp3.stat().st_size > 3_000_000:
            row["ok"] = True; row["mp3_bytes"] = mp3.stat().st_size
            print(f"{name}: OK in {took}s", flush=True)
        else:
            row["ok"] = False; row["error"] = "mp3 conversion failed or undersized"
            print(f"{name}: CONVERSION FAILED after {took}s", flush=True)
    else:
        row["ok"] = False; row["error"] = f"no audio (rc={rc})"
        print(f"{name}: FAILED after {took}s", flush=True)
    results.append(row)
    Path("/kaggle/working/takes.json").write_text(json.dumps(results, indent=2))

Path("/kaggle/working/takes.json").write_text(json.dumps(results, indent=2))
print("\nDONE", json.dumps({"rung": chosen, "takes": results}, indent=2)[:1200], flush=True)
