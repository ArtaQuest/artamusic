# KEEP THE KEY — the publication run, hardened by the adversarial review of its predecessor.
#
# What changed and why, each item a confirmed defect in the UNBROKEN kernel:
#
#   GATE THE DELIVERED FILE.  UNBROKEN's gate measured cand<seed>.mp3; EQ, gain and a limiter then
#   rewrote it into the published master, which nobody measured. Here a VERIFY stage runs LAST,
#   on the exact bytes that get published, and it is the verify stage that authorises the run.
#
#   DISTRIBUTION, NOT VERDICT.  The register instrument returns median + quartiles + semitone
#   spread + lead mode + band occupancy. The median classifies (corpus-validated: covers 149-157 Hz
#   male, caption takes 197-233 female); the modes are DISCLOSED so a male lead under a male choir
#   is claimed as exactly that, never as a solo. `trusted` no longer exists to be ignored.
#
#   SOTA WHERE IT DECIDES.  Intelligibility gates on whisper large-v3 (the pinned model the old
#   kernel ignored in favour of small.en) — GPU float32 if the card cooperates (CT2 fp16 needs
#   cc>=7.0; the P100 is 6.0), CPU int8 otherwise, and the run RECORDS which one judged.
#   Artwork tries FLUX.1-schnell (12B, Apache, ungated) under sequential CPU offload with SDXL as
#   the recorded fallback.
#
#   PINNED EVERYTHING.  The ACE-Step CODE repo is checked out at a recorded sha (a floating clone
#   of main was an unpinned input to a reproducibility claim); measure.py is fetched from the
#   public repo at a commit sha; pip installs carry versions; python/numpy/torch are seeded.
#   pins.json stops being decorative: this file IS the pin set, inlined and asserted.
#
# The claims that travel with the published files:
#   male lead    — measured on a demucs-isolated stem with YIN, on the DELIVERED master, with the
#                  full pitch distribution published beside the verdict.
#   intelligible — whisper large-v3 word accuracy against the exact lyric below, gate 75%.
#   dynamic      — static gain + 4x-oversampled true-peak limiter, then measure-then-correct on
#                  the encoded mp3 (LAME overshoots; asserting a target you never re-measure is
#                  how UNBROKEN shipped at -0.8 dBTP against a -1.0 promise).

import json, os, random, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── pins: every input that can move, nailed down ────────────────────────────────────────
PINS = {
    "ace_step_code": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",   # github.com/ACE-Step/ACE-Step-1.5
    "song_model": "acestep-v15-xl-sft",
    "measure_sha": "5e9880ac0486bef21033b999a9c7bf3a4b7bf0f6",      # ArtaQuest/artamusic lib/measure.py
    "asr": "large-v3",
    "image": "black-forest-labs/FLUX.1-schnell",
    "image_fallback": "stabilityai/stable-diffusion-xl-base-1.0",
    "torch": "2.7.1", "cuda_line": "cu126",
}
SEED = 4242
random.seed(SEED)

TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/working")
os.environ.update(HF_HOME=str(TMP / "hf"), HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  ACESTEP_GENERATION_TIMEOUT="2400")

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:160]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1600:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1600:], flush=True)
    return r.returncode

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0

# ── environment, pinned; deps BEFORE torch so the cu126 line wins on Pascal ─────────────
if not REPO.exists():
    sh(f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}")
    sh(f"cd {REPO} && git checkout {PINS['ace_step_code']} && git log -n1 --format='code pin OK %h'")
sh("pip install -q demucs==4.0.1 faster-whisper==1.2.1 hf_transfer toml python-dotenv modelscope "
   "diskcache py3langid pyloudnorm ffmpeg-python soundfile loguru einops accelerate numba scipy "
   "'safetensors>=0.7.0' 'transformers>=4.51.0,<4.58.0' diffusers==0.35.2 "
   "vector-quantize-pytorch 2>&1 | tail -2")
if PASCAL:
    sh(f"pip install -q torch=={PINS['torch']} torchvision==0.22.1 torchaudio=={PINS['torch']} "
       f"--index-url https://download.pytorch.org/whl/{PINS['cuda_line']} 2>&1 | tail -2")

import numpy as np
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
print(f"torch {torch.__version__} | seeded {SEED}", flush=True)

def gemm_ok(dt, n=2048):
    try:
        a = torch.randn(n, n, device="cuda", dtype=dt); c = a @ a
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(c).all()); del a, c; torch.cuda.empty_cache(); return ok
    except Exception:
        torch.cuda.empty_cache(); return False
BF16 = gemm_ok(torch.bfloat16)
print("bfloat16 usable:", BF16, flush=True)

# ── the instruments, pinned, and PROVEN before anything expensive runs ──────────────────
import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['measure_sha']}/lib/measure.py",
    "/tmp/measure.py")
sys.path.insert(0, "/tmp")
import measure as M
# The self-test is the entry point, not an afterthought: every estimator in this project that
# was not validated against known-truth signals turned out to be wrong. 3 seconds, or no run.
assert M.selftest(), "measurement selftest FAILED — no number from this build can be trusted"

_WH = [None, None]   # model, judged_by — loaded PER CALL and unloaded after, never resident:
# a 6 GB float32 large-v3 parked on the GPU starved the next candidate's 12.2 GB render into
# silent garbage (rc 0, no file — ACE-Step's #924 guard suppresses rather than crashes).
def asr():
    from faster_whisper import WhisperModel
    if _WH[0] is None:
        try:
            _WH[0] = WhisperModel(PINS["asr"], device="cuda", compute_type="float32")
            _WH[1] = f"{PINS['asr']}/cuda-float32"
        except Exception as e:
            print(f"GPU ASR unavailable ({str(e)[:80]}) — CPU int8 fallback", flush=True)
            _WH[0] = WhisperModel(PINS["asr"], device="cpu", compute_type="int8")
            _WH[1] = f"{PINS['asr']}/cpu-int8"
        print("ASR judge:", _WH[1], flush=True)
    return _WH[0]

def asr_release():
    import gc
    _WH[0] = None
    gc.collect()
    torch.cuda.empty_cache()

def _vocal_stem(mp3):
    """One separation, shared by register and words — the stem is where the voice is."""
    import demucs.separate, shlex, tempfile as _tf, shutil
    td = _tf.mkdtemp()
    demucs.separate.main(shlex.split(f'--two-stems vocals -n htdemucs --device cpu -o "{td}" "{mp3}"'))
    src = next(Path(td).rglob("vocals.wav"), None)
    if src is None:
        return None
    keep = Path(_tf.mkdtemp()) / "vocals.wav"
    shutil.copy(src, keep)
    return str(keep)


def word_accuracy(mp3, lyric_text, stem=None):
    import re as _re
    # ICME-2025 recipe, worth ~3 pp on sung lyrics: transcribe the SEPARATED STEM, chunked at
    # quiet points into ~20 s segments, instead of the full mix under the 808s.
    target = stem or mp3
    segs, _ = asr().transcribe(str(target), beam_size=5, vad_filter=True,
                               vad_parameters=dict(min_silence_duration_ms=400),
                               chunk_length=20, condition_on_previous_text=False)
    hyp = _re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
    asr_release()   # never leave 6 GB parked in front of the next render
    ref = _re.findall(r"[a-z']+", _re.sub(r"\[[^\]]*\]", " ", lyric_text.lower()))
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    d[:,0] = np.arange(len(ref)+1); d[0,:] = np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    return 1.0 - min(1.0, float(d[-1,-1])/max(1,len(ref)))

def register_gate(mp3):
    """male on the median-classifier = PASS; 'mixed'/'ambiguous'/'female' = FAIL; and the
    distribution travels with the verdict so the claim can never outrun the evidence."""
    reg = M.register(str(mp3))
    ok = reg.get("register") == "male"
    return ok, reg

# ── the male voice reference, mounted from the PUBLIC kernel ────────────────────────────
import glob as _glob
# The style anchor is KEEP THE KEY's lead — the cleanest male vocal this pipeline owns
# (156 Hz median, 6.65 st spread vs UNBROKEN's 10.06): a better anchor makes a better bias.
_ref = sorted(_glob.glob("/kaggle/input/**/KEEPTHEKEY.mp3", recursive=True)) or \
       sorted(_glob.glob("/kaggle/input/**/UNBROKEN.mp3", recursive=True))
MALE_REF = _ref[0] if _ref else None
assert MALE_REF, "male reference not mounted — fail at second five, not minute forty"
print("male reference:", MALE_REF, flush=True)

LYRICS = """[intro]
Call me Steel — cold flame.
Call me the anvil's claim.
Call me the drawn oath
a hammer swore before.
Call me the bright weight
your bleeding fathers bore.

[verse]
Cut! by the coal that roared,
Cut! by the heat it poured,
Deep in the grain, old fire sleeps,
What the flame gave, the blade keeps.
Cut! born of the crushing blow,
Cut! born of the blinding glow,
Folded in dark, drawn to light,
Beaten till the core ran bright.

[verse]
Cut! every mark stays true,
Cut! every blow shows through,
A bright wave runs down my spine,
The fire signed me, line by line.
Cut! from the water's scream,
Cut! from the rising steam,
Hot at heart, I met the cold,
What the cold caught, cold will hold.

[chorus]
I am the weight you bear.
I am the oath you swear.
I am the tempered edge.
I am the standing pledge.
The fire is out and cold.
The forge is lost to dark.
Of all the flame once told,
I hold the living spark.

[verse]
Cut! when the horn calls clear,
Cut! through the choke of fear,
The smith gave form. The flame gave speed.
The hand gives cause. I am the deed.
Cut! down the darkened field,
Cut! through the splintered shield,
The hand that holds me holds my trust,
Steel serves the cause, or steel is rust.

[bridge]
Heat is gone.
Cold has come.
Deep in the steel
the hammers drum.
Hands can shake.
Hearts can hide.
Steel stays calm
at your side.
Grip me tight.
Swing me through.
What I keep
I keep for you.
Bend me far.
Feel me spring.
Tempered deep,
hear me ring.

[verse]
Cut! with a willow's give,
Cut! with a will to live,
Hard steel shatters, soft steel bends,
Temper is where the trembling ends.
Cut! and the weight comes due,
Cut! and the debt bleeds through,
All that I take, I also keep,
Hearts may forgive — the scars lie deep.

[chorus]
I am the weight you bear.
I am the oath you swear.
I am the tempered edge.
I am the standing pledge.
The fire is out and cold.
The forge is lost to dark.
Of all the flame once told,
I hold the living spark.

[verse]
Cut! till the sun burns low,
Cut! till the young hands grow,
Bearers change and banners fade,
The hand goes home. I stay the blade.
Cut! for the oath I keep,
Cut! while the makers sleep,
From hand to hand the burden runs,
Passed from your fathers to your sons.

[chorus]
I am the weight you bear.
I am the oath you swear.
I am the tempered edge.
I am the standing pledge.
The fire is dead and cold.
The last hand has gone still.
The spark the flame once told,
I kept — and always will."""
_OLD = """[verse]
Back the truck up on the drive
Gate down under a grey sky
Take the cups down off the shelf
Cups he shined and stacked himself
Glasses wrapped and taped up tight
Boxes marked and stacked just right
Kitchen's boxed and in the hall
Man held on to it, that's all

[pre-chorus]
Table tips and corners scrape
Door too tight, the paint gives way
Take the weight, keep it moving
All of it comes out this house

[chorus]
Take the table out the door
Roll the rug up off the floor
Wrap the glass and sweep the corner
Truck out front, sitting lower
Rooms get bigger when they're bare
Dad's coat hangs there on the stair
Coat rides up front with me
Shut the door, keep the key

[verse]
Frames come down off every wall
Pale squares up and down the hall
Sweep the boards and bag the grit
Haul the boxes, get on with it
Coats and boots and empty hooks
His grey coat stays on its hook
Rest of it goes on the truck
Room I slept in, swept and shut

[pre-chorus]
Fold the bed up, lean the frame
Bare top floor still creaks the same
Take the weight, keep it moving
All of it comes out this house

[chorus]
Take the table out the door
Roll the rug up off the floor
Wrap the glass and sweep the corner
Truck out front, sitting lower
Rooms get bigger when they're bare
Dad's coat hangs there on the stair
Coat rides up front with me
Shut the door, keep the key

[bridge]
Stand there in the room and wait
Boots and chairs go by, it's late
Head goes back, the paint's gone yellow
Pale square, nail hole, dust below
Someone asks me if I'm good
I say yeah. I lift the wood

[chorus]
Take the table out the door
Roll the rug up off the floor
Wrap the glass and sweep the corner
Truck out front, sitting lower
Rooms get bigger when they're bare
Dad's coat hangs there on the stair
Coat rides up front with me
Shut the door, keep the key

[outro]
Truck pulls out and down the drive
Coat up front, same as always
New name where his used to be
Shut the door, keep the key"""

CAPTION = "Dark chant anthem. Pounding war drums and anvil strikes on the beat, massive unison male chant choir answering a deep gravelly lead vocal, low strings and war horns, sparse and martial, minor key, solemn and heavy, 100 BPM."
DURATION = 180.0
BPM, KEYSCALE = 100, "F minor"   # match the conditioning source; a key fight with src_audio is
                                 # an experiment, and a publication run is not where you run it

# ── load the 4.6B model (resident rung; ladder kept as insurance) ───────────────────────
sys.path.insert(0, str(REPO))
import toml
from acestep.handler import AceStepHandler

LADDER = ([("xl-resident", PINS["song_model"], "bfloat16", False, False),
           ("xl-offload",  PINS["song_model"], "bfloat16", True,  False),
           ("xl-dit-swap", PINS["song_model"], "bfloat16", True,  True)] if BF16 else []) + \
         [("sft-fp32", "acestep-v15-sft", "float32", False, False)]

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
assert OLD in src, "ACE-Step changed under its pin — impossible unless the checkout failed"
orch.write_text(src.replace(OLD, NEW, 1))

chosen = None
for name, model, dtype, oc, od in LADDER:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    os.environ["AQ_FORCE_DTYPE"] = dtype
    t0 = time.time()
    try:
        h = AceStepHandler()
        h.initialize_service(project_root=str(REPO), config_path=model, device="cuda",
                             offload_to_cpu=oc, offload_dit_to_cpu=od)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"RUNG HELD: {name} — {model} @ {dtype}, peak {peak:.2f} GB in {time.time()-t0:.0f}s",
              flush=True)
        chosen = dict(rung=name, model=model, dtype=dtype, offload_to_cpu=oc,
                      offload_dit_to_cpu=od, peak_gb=round(peak, 2))
        del h; torch.cuda.empty_cache(); break
    except Exception as e:
        print(f"rung {name} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        torch.cuda.empty_cache()
assert chosen, "no rung held"
(WORK / "rung.json").write_text(json.dumps({**chosen, "pins": PINS, "seed": SEED}, indent=2))

# ── candidates: render, gate, keep the first that passes ────────────────────────────────
# STYLE TRANSFER is the production mechanism — measured, it breaks cover mode's two-knob trap:
# text2music + reference_audio keeps the words text-driven (89% at strength 0.2 where cover mode
# managed 71%) while the reference biases timbre (male median 145.9 Hz at 0.35, spread 7.14 st —
# the healthy single-lead texture). Ordered so better words ship if the register holds:
# 0.30 is the interpolated sweet spot, 0.35 the proven-male fallback, then a fresh seed.
# THE A/B. One seed, one caption, one lyric, one strength — the ONLY variable is `thinking`.
# Source evidence (ACE-Step inference.py): `thinking: bool = True` is the library default;
# skip_lm_tasks = {cover, cover-nofsq, repaint, extract} does NOT include text2music, so on our
# style-transfer path the flag genuinely gates the 5 Hz LM planner:
#   use_lm  = (params.thinking or need_lm_for_cot) and llm_initialized and not skip_lm
#   infer_type = "llm_dit" if need_audio_codes and params.thinking else "dit"
# Every shipped song ran with thinking=False — i.e. plain `dit`, no semantic codes, no planning.
# This measures what that cost, at a fixed seed, before any of it is believed.
CANDIDATES = [(6003, 0.35), (6003, 0.35)]
THINKING = [True, False]   # proven strength; seeds are the lottery now   # (seed, cover_strength) — escalate toward the voice if the lyric pulls register off
WINNER, WINNER_ACC, gate_log = None, None, []
for _i, (seed, strength) in enumerate(CANDIDATES):
    think = THINKING[_i]
    name_suffix = "think" if think else "nothink"
    name = f"cand{seed}_{name_suffix}"
    conf = TMP / f"{name}.toml"
    conf.write_text(toml.dumps({
        "project_root": str(REPO), "config_path": chosen["model"], "checkpoint_dir": str(CKPT),
        "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
        "offload_to_cpu": chosen["offload_to_cpu"],
        "offload_dit_to_cpu": chosen["offload_dit_to_cpu"],
        "task_type": "text2music", "reference_audio": MALE_REF,
        "audio_cover_strength": strength,
        "caption": CAPTION, "lyrics": LYRICS, "instrumental": False,
        "bpm": BPM, "keyscale": KEYSCALE, "timesignature": "4/4", "vocal_language": "en",
        "duration": DURATION, "inference_steps": 80, "guidance_scale": 7.5,
        "seed": seed, "infer_method": "ode",
        # `thinking` and the `use_cot_*` flags are NOT the same switch, and conflating them cost a run:
        #   thinking       -> the 5 Hz LM emits SEMANTIC AUDIO CODES for the DiT (the quality lever)
        #   use_cot_lyrics -> the LM WRITES THE LYRICS for you, discarding the ones supplied
        #   use_cot_caption/metas/language -> likewise delegate authorship of those fields
        # Setting them together made the CLI try to generate a lyric from the caption and then
        # fail validation ("--use_cot_lyrics requires the LM handler"). Had it succeeded it would
        # have sung a lyric nobody wrote. The A/B tests the LEVER, with authorship kept.
        "thinking": think, "use_cot_metas": False, "use_cot_caption": False,
        "use_cot_lyrics": False, "use_cot_language": False,
        "batch_size": 1, "use_random_seed": False, "seeds": [seed],
        # pt backend: the library already forces this on pre-Volta cards
        # (gpu_config._apply_lm_backend_compatibility_overrides -> pt_only).
        "lm_backend": "pt", "offload_lm_to_cpu": True}))
    t0 = time.time()
    # No pipes here: PIPESTATUS is bash-only and shell=True is dash on this image — a pipe
    # would silently report tail's exit code as the render's. Redirect whole-output to a file;
    # the line's rc is then genuinely the CLI's.
    rc = sh(f"cd {REPO} && AQ_FORCE_DTYPE={chosen['dtype']} "
       f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACESTEP_GENERATION_TIMEOUT=2400 "
       f"python cli.py -c {conf} --backend pt --log-level INFO > /tmp/cli_{name}.txt 2>&1", quiet=True)
    found = sorted(Path(TMP / f"out_{name}").rglob("*.flac")) + \
            sorted(Path(TMP / f"out_{name}").rglob("*.wav"))
    row = {"seed": seed, "thinking": think, "seconds": round(time.time()-t0, 1), "steps": 80,
           "cover_strength": strength, "model": chosen["model"], "dtype": chosen["dtype"]}
    if not found:
        tail = Path(f"/tmp/cli_{name}.txt").read_text()[-800:] if Path(f"/tmp/cli_{name}.txt").exists() else ""
        row["verdict"] = f"no audio (rc={rc})"; row["cli_tail"] = tail
        gate_log.append(row)
        (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
        print(f"{name}: no audio rc={rc}\n{tail[-400:]}", flush=True); continue
    mp3 = OUT / f"{name}.mp3"
    sh(f"ffmpeg -v error -i '{max(found, key=lambda p: p.stat().st_size)}' "
       f"-codec:a libmp3lame -b:a 320k '{mp3}' -y", quiet=True)
    stem = _vocal_stem(mp3)
    ok_reg, reg = register_gate(mp3)
    acc = word_accuracy(mp3, LYRICS, stem=stem)
    row.update(register=reg, word_accuracy=round(acc, 3), asr_judge=_WH[1])
    ok_words = acc >= 0.75
    row["verdict"] = "ACCEPTED" if (ok_reg and ok_words) else \
        f"REJECTED ({reg.get('register')}{'' if ok_words else f' words {acc*100:.0f}%'})"
    gate_log.append(row)
    (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
    print(f"{name}: median={reg.get('f0_hz')}Hz [{reg.get('register')}] "
          f"spread={reg.get('spread_st')}st lead={reg.get('lead_hz')}Hz@{reg.get('lead_frac')} "
          f"words={acc*100:.1f}% -> {row['verdict']}", flush=True)
    # no early exit: an A/B needs BOTH arms measured

print("\n=== THINKING A/B ===", flush=True)
for r in gate_log:
    reg = r.get("register") or {}
    print(f"  thinking={str(r.get('thinking')):5s} words={r.get('word_accuracy')} "
          f"register={reg.get('register')} median={reg.get('f0_hz')}Hz "
          f"spread={reg.get('spread_st')}st lead_frac={reg.get('lead_frac')} "
          f"secs={r.get('seconds')}", flush=True)
(WORK / "ab.json").write_text(json.dumps(gate_log, indent=2))
sys.exit(0)
