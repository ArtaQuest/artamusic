# XL-ON-P100 PROBE — can the 4.6B model be made to fit, without paying for bigger hardware?
#
# The previous run's OOM was NOT a "model too big" failure, which is why this is worth testing:
#   "Tried to allocate 1.21 GiB. GPU has 15.89 GiB of which 139.12 MiB is free.
#    2.30 GiB is reserved by PyTorch but unallocated."
# Short by 1.21 GiB while 2.30 GiB sat reserved-and-unused. That is fragmentation, not capacity.
# Two things were never tried:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the error message's own suggestion, which
#      lets the allocator grow a segment instead of stranding reserved blocks.
#   2. ACE-Step's own offload_to_cpu flag, which the previous kernel explicitly set to False.
# This probe measures peak VRAM at each setting instead of guessing, and stops before generating
# audio — the question is whether the weights and one forward pass FIT, not how they sound.

import json, os, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
os.environ["HF_HOME"] = str(TMP / "hf")
os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(CKPT)
os.environ["ACESTEP_PROJECT_ROOT"] = str(REPO)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def sh(c):
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip(): print(r.stdout[-1500:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1500:], flush=True)
    return r.returncode

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0

if not REPO.exists():
    sh(f"git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}")
sh("pip install -q hf_transfer toml python-dotenv modelscope diskcache py3langid pyloudnorm "
   "ffmpeg-python soundfile loguru einops accelerate numba scipy "
   "'safetensors>=0.7.0' 'transformers>=4.51.0,<4.58.0' diffusers vector-quantize-pytorch 2>&1 | tail -2")
if PASCAL:
    sh("pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
       "--index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2")

import torch
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GB", flush=True)
print("expandable_segments:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), flush=True)

# Patch the dtype decision so bfloat16 can be forced on a card the library would give float16.
orch = REPO / "acestep/core/generation/handler/init_service_orchestrator.py"
src = orch.read_text()
OLD = """            elif resolved_device == "cuda":
                if gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16"""
NEW = """            elif resolved_device == "cuda":
                _f = os.environ.get("AQ_FORCE_DTYPE", "")
                if _f:
                    self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                                  "float16": torch.float16}[_f]
                elif gpu_config.cuda_supports_bfloat16():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16"""
if OLD in src:
    orch.write_text(src.replace(OLD, NEW, 1)); print("dtype patch applied", flush=True)

sys.path.insert(0, str(REPO))
from acestep.handler import AceStepHandler

results = []
# Ordered cheapest-to-most-degraded. The first that fits wins; nothing after it needs running.
TRIALS = [
    ("xl / bfloat16 / no offload",   "acestep-v15-xl-sft", "bfloat16", False, False),
    ("xl / bfloat16 / cpu offload",  "acestep-v15-xl-sft", "bfloat16", True,  False),
    ("xl / bfloat16 / dit offload",  "acestep-v15-xl-sft", "bfloat16", True,  True),
]
for label, model, dtype, off_cpu, off_dit in TRIALS:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    os.environ["AQ_FORCE_DTYPE"] = dtype
    t0 = time.time()
    try:
        h = AceStepHandler()
        h.initialize_service(project_root=str(REPO), config_path=model, device="cuda",
                             offload_to_cpu=off_cpu, offload_dit_to_cpu=off_dit)
        peak = torch.cuda.max_memory_allocated() / 2**30
        res = {"trial": label, "loaded": True, "peak_gb": round(peak, 2),
               "seconds": round(time.time() - t0, 1)}
        print(f"  OK   {label:32s} peak {peak:5.2f} GB in {res['seconds']}s", flush=True)
        results.append(res)
        del h
        break                                   # first success is the answer
    except Exception as e:
        peak = torch.cuda.max_memory_allocated() / 2**30
        msg = f"{type(e).__name__}: {str(e)[:150]}"
        print(f"  FAIL {label:32s} peak {peak:5.2f} GB — {msg}", flush=True)
        results.append({"trial": label, "loaded": False, "peak_gb": round(peak, 2), "error": msg})
    finally:
        torch.cuda.empty_cache()

Path("/kaggle/working/xl_probe.json").write_text(json.dumps(results, indent=2))
print("\nRESULT", json.dumps(results, indent=2), flush=True)
