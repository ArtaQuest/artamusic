# %% [markdown]
# # STEEL song audition — the lyric on trial, fast, on a compute account
#
# The operator approves the LYRIC BY EAR before the production slot is spent: two takes of the
# song only (the cover is already approved), male-register read on each, mp3s out. Production on
# artafather runs after approval, not before.

# %%
import json, os, subprocess, sys, time
from pathlib import Path
T0 = time.time()
def sh(c, quiet=False): subprocess.run(c, shell=True, check=True,
    stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
def clock(w): print(f"  ⏱ {w} · t+{(time.time()-T0)/60:.1f} min", flush=True)

PINS = {
    "ace_step_code": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
    "song_model": "acestep-v15-xl-sft",
    "measure_sha": "199535aa517324d8021667b5a34a799aedd19353",
    "lyric_sha": "d6d1e4c2393f69d3009d252ff838b1ff484af2dd",   # song/lyrics_steel_run.txt — Not Today (short sha OK: raw URL accepts it)
}
TMP = Path("/tmp/aq"); TMP.mkdir(exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TMP/"hf"), HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  ACESTEP_GENERATION_TIMEOUT="2400")
if not REPO.exists():
    sh(f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}", quiet=True)
    sh(f"cd {REPO} && git checkout {PINS['ace_step_code']}")
sh("pip install -q hf_transfer toml python-dotenv modelscope diskcache py3langid pyloudnorm "
   "ffmpeg-python soundfile loguru einops accelerate numba scipy demucs "
   "'safetensors>=0.7.0' 'transformers>=4.51.0,<4.58.0' vector-quantize-pytorch ftfy "
   "sentencepiece protobuf 2>&1 | tail -1")
clock("installed")

import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['measure_sha']}/lib/measure.py",
    "/tmp/measure.py")
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/song/lyrics_steel_run.txt",
    "/tmp/lyrics.txt")
sys.path.insert(0, "/tmp")
import numpy as np, torch
import measure as M
LYRICS = Path("/tmp/lyrics.txt").read_text().strip()
assert LYRICS.startswith("[Intro]") and "I am steel" in LYRICS, "wrong lyric at pin"
_ref = sorted(Path("/kaggle/input").rglob("*KEEPTHEKEY*.mp3"))
assert _ref, "male reference not mounted (kernel source artafather/keep-the-key)"
MALE_REF = str(_ref[0])
CAPTION = ("Powerful marching anthem. A heavy stomping kick on every beat like boots on stone, "
           "pounding floor toms and a hard bassline, handclaps on the backbeat. A strong deep "
           "male voice out in front of the mix, urgent and clear, with a male gang-vocal "
           "shouting the answers in the chorus. Big open power chords and brass swells lift the "
           "choruses; the bridge drops to drums and a lone voice before the last chorus lands "
           "twice as heavy. Motivational, relentless, triumphant, ending at full force with no "
           "fade.")
BPM, KEYSCALE, DURATION = 128, "A minor", 180.0

# %%
sys.path.insert(0, str(REPO))
import toml
BF16 = torch.cuda.is_bf16_supported()
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
assert OLD in src, "ACE-Step changed under its pin"
orch.write_text(src.replace(OLD, NEW, 1))

def render_conf(name, seed, rung, steps):
    return {"project_root": str(REPO), "config_path": rung["model"], "checkpoint_dir": str(CKPT),
            "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
            "offload_to_cpu": rung["offload_to_cpu"], "offload_dit_to_cpu": rung["offload_dit_to_cpu"],
            "task_type": "text2music", "reference_audio": MALE_REF,
            "caption": CAPTION, "lyrics": LYRICS, "instrumental": False,
            "bpm": BPM, "keyscale": KEYSCALE, "timesignature": "4", "vocal_language": "en",
            "duration": DURATION, "inference_steps": steps, "guidance_scale": 7.5,
            "shift": 1.0, "thinking": True,
            "use_cot_metas": False, "use_cot_caption": False, "use_cot_lyrics": False,
            "use_cot_language": False, "seed": seed, "infer_method": "ode",
            "batch_size": 1, "use_random_seed": False, "seeds": [seed]}

def cli_render(name, conf_dict, dtype):
    conf = TMP / f"{name}.toml"; conf.write_text(toml.dumps(conf_dict))
    rc = subprocess.run(
        f"cd {REPO} && AQ_FORCE_DTYPE={dtype} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"python cli.py -c {conf} --backend pt --log-level INFO > /tmp/cli_{name}.txt 2>&1",
        shell=True).returncode
    found = sorted((TMP / f"out_{name}").rglob("*.flac")) + sorted((TMP / f"out_{name}").rglob("*.wav"))
    return rc, found

LADDER = ([("xl-resident", PINS["song_model"], "bfloat16", False, False),
           ("xl-offload",  PINS["song_model"], "bfloat16", True,  False),
           ("xl-dit-swap", PINS["song_model"], "bfloat16", True,  True)] if BF16 else []) + \
         [("sft-fp32", "acestep-v15-sft", "float32", False, False)]
chosen = None
for name, model, dtype, oc, od in LADDER:
    rung = dict(rung=name, model=model, dtype=dtype, offload_to_cpu=oc, offload_dit_to_cpu=od)
    rc, found = cli_render(f"probe_{name}", render_conf(f"probe_{name}", 7001, rung, 2), dtype)
    if found:
        print(f"RUNG HELD: {name}", flush=True); chosen = rung; break
assert chosen, "no rung held"
clock("rung held")

# %%
report = []
for seed in (6001, 6002):
    name = f"aud{seed}"
    rc, found = cli_render(name, render_conf(name, seed, chosen, 80), chosen["dtype"])
    assert found, f"{name}: no audio (rc {rc}) — {Path(f'/tmp/cli_{name}.txt').read_text()[-260:]}"
    mp3 = OUT / f"{name}.mp3"
    sh(f"ffmpeg -v error -i '{found[0]}' -codec:a libmp3lame -b:a 320k '{mp3}' -y")
    # register read for disclosure — the audition is by ear; production re-gates everything
    import demucs.separate, shlex, tempfile as _tf
    td = _tf.mkdtemp()
    demucs.separate.main(shlex.split(
        f'--two-stems vocals -n htdemucs --shifts 0 --device cpu -o "{td}" "{mp3}"'))
    stem = next(Path(td).rglob("vocals.wav"), None)
    reg = M.classify_f0(M.finite_f0(M.f0_yin(*M.load(str(stem), mono=True)))) if stem else {}
    row = {"seed": seed, "register": reg.get("register"), "lead_hz": reg.get("lead_hz")}
    report.append(row)
    (WORK / "audition.json").write_text(json.dumps(report, indent=1))
    print(f"  {name}: register {reg.get('register')} · lead {reg.get('lead_hz')} Hz", flush=True)
    clock(f"{name} done")
print("AUDITION:", json.dumps(report), flush=True)
clock("DONE")
