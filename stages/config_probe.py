# What did the three shipped songs ACTUALLY run with? Zero GPU-seconds, and it settles the
# largest open question in the project.
#
# The research flagged that ACE-Step's `thinking` parameter DEFAULTS TO TRUE and gates the 5 Hz
# LM planner — the stage that emits semantic audio codes and plans structure before the diffusion
# transformer runs. Every kernel here set thinking=False, copied forward from an early speed
# experiment and never questioned. If that suppressed the planner, it plausibly explains the
# intelligibility ceiling that seeds and conditioning tricks have been fighting all week.
#
# It is also possible the flag does nothing on this path, or that the CLI overrides it. So this
# prints the resolved config the library actually uses — not what the TOML asked for — before a
# single GPU-hour is spent on the theory.
import json, subprocess, sys
from pathlib import Path

TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "ACE-Step-1.5"

def sh(c):
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    print(r.stdout[-1200:] or r.stderr[-600:], flush=True)
    return r.returncode

if not REPO.exists():
    sh(f"git clone -q https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}")
    sh(f"cd {REPO} && git checkout -q 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0")
sh("pip install -q toml python-dotenv loguru 2>&1 | tail -1")
sys.path.insert(0, str(REPO))

# 1. What are the library's own defaults?
from acestep.inference import InferenceParams  # dataclass of every knob
d = InferenceParams()
keys = ["task_type", "thinking", "use_cot_metas", "use_cot_caption", "use_cot_lyrics",
        "use_cot_language", "infer_method", "inference_steps", "guidance_scale", "shift",
        "audio_cover_strength", "cover_noise_strength", "lm_backend"]
print("\n=== LIBRARY DEFAULTS ===", flush=True)
for k in keys:
    print(f"  {k:22s} {getattr(d, k, '<absent>')!r}", flush=True)

# 2. What did our songs send? (the values every shipped kernel used)
OURS = {"task_type": "text2music", "thinking": False, "use_cot_metas": False,
        "use_cot_caption": False, "use_cot_lyrics": False, "use_cot_language": False,
        "infer_method": "ode", "inference_steps": 80, "guidance_scale": 7.5}
print("\n=== WHAT THE SHIPPED SONGS SENT ===", flush=True)
for k, v in OURS.items():
    lib = getattr(d, k, "<absent>")
    flag = "  <-- DIFFERS FROM DEFAULT" if lib != v else ""
    print(f"  {k:22s} ours={v!r:12} default={lib!r}{flag}", flush=True)

# 3. Does the planner actually load and run when thinking=True?
print("\n=== DOES thinking=True CHANGE THE RESOLVED PIPELINE? ===", flush=True)
import inspect
from acestep.core.generation.handler import init_service_orchestrator as orch
src = inspect.getsource(orch)
for term in ("thinking", "lm_", "planner", "semantic"):
    hits = [l.strip()[:120] for l in src.splitlines() if term in l.lower()]
    print(f"  '{term}': {len(hits)} references", flush=True)
    for h in hits[:4]:
        print(f"      {h}", flush=True)
Path("/kaggle/working/config_probe.json").write_text(json.dumps(
    {"defaults": {k: repr(getattr(d, k, None)) for k in keys}, "ours": OURS}, indent=2))
print("\nprobe complete", flush=True)
