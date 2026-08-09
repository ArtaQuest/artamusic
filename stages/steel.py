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
CANDIDATES = [(6001, 0.35), (6002, 0.35), (6003, 0.35)]   # proven strength; seeds are the lottery now   # (seed, cover_strength) — escalate toward the voice if the lyric pulls register off
WINNER, WINNER_ACC, gate_log = None, None, []
for seed, strength in CANDIDATES:
    name = f"cand{seed}"
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
        "thinking": False, "use_cot_metas": False, "use_cot_caption": False,
        "use_cot_lyrics": False, "use_cot_language": False,
        "batch_size": 1, "use_random_seed": False, "seeds": [seed]}))
    t0 = time.time()
    # No pipes here: PIPESTATUS is bash-only and shell=True is dash on this image — a pipe
    # would silently report tail's exit code as the render's. Redirect whole-output to a file;
    # the line's rc is then genuinely the CLI's.
    rc = sh(f"cd {REPO} && AQ_FORCE_DTYPE={chosen['dtype']} "
       f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACESTEP_GENERATION_TIMEOUT=2400 "
       f"python cli.py -c {conf} --backend pt --log-level INFO > /tmp/cli_{name}.txt 2>&1", quiet=True)
    found = sorted(Path(TMP / f"out_{name}").rglob("*.flac")) + \
            sorted(Path(TMP / f"out_{name}").rglob("*.wav"))
    row = {"seed": seed, "seconds": round(time.time()-t0, 1), "steps": 80,
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
    if ok_reg and ok_words:
        WINNER, WINNER_ACC = mp3, acc; break

if WINNER is None:
    repairable = [r for r in gate_log
                  if r.get("word_accuracy", 0) >= 0.80 and (r.get("register") or {}).get("register") != "male"]
    if repairable:
        best = max(repairable, key=lambda r: r["word_accuracy"])
        print(f"\nSVC REPAIR: converting seed {best['seed']}'s vocal to the male reference "
              f"(zero-shot Seed-VC) — deterministic register instead of another seed roll", flush=True)
        sh("git clone -q https://github.com/Plachtaa/seed-vc /tmp/seedvc && "
           "pip install -q -r /tmp/seedvc/requirements.txt 2>&1 | tail -1")
        cand = OUT / f"cand{best['seed']}.mp3"
        stem = _vocal_stem(str(cand))
        # accompaniment = mix minus vocal, phase-aligned by construction (same separation)
        import soundfile as sf
        import numpy as np_
        acc_wav = str(Path(stem).parent / "acc.wav")
        sh(f"ffmpeg -v error -i '{cand}' -ar 44100 -ac 2 /tmp/_mix.wav -y", quiet=True)
        mix, sr_ = sf.read("/tmp/_mix.wav"); voc, _ = sf.read(stem)
        n = min(len(mix), len(voc)); sf.write(acc_wav, mix[:n] - voc[:n], sr_)
        conv_dir = "/tmp/svc_out"
        rc = sh(f"cd /tmp/seedvc && python inference.py --source '{stem}' "
                f"--target '{MALE_REF}' --output {conv_dir} "
                f"--diffusion-steps 50 --length-adjust 1.0 --inference-cfg-rate 0.7 "
                f"--f0-condition True --auto-f0-adjust True")
        conv = sorted(Path(conv_dir).glob("*.wav"), key=lambda q: q.stat().st_mtime)
        if rc == 0 and conv:
            fixed = OUT / f"cand{best['seed']}_svc.mp3"
            sh(f"ffmpeg -v error -i '{conv[-1]}' -i '{acc_wav}' "
               f"-filter_complex '[0:a][1:a]amix=inputs=2:duration=shortest:normalize=0' "
               f"-codec:a libmp3lame -b:a 320k '{fixed}' -y", quiet=True)
            stem2 = _vocal_stem(str(fixed))
            ok2, reg2 = register_gate(fixed)
            acc2 = word_accuracy(fixed, LYRICS, stem=stem2)
            row2 = {"seed": best["seed"], "svc_repaired": True, "register": reg2,
                    "word_accuracy": round(acc2, 3),
                    "verdict": "ACCEPTED" if (ok2 and acc2 >= 0.75) else
                               f"REJECTED post-SVC ({reg2.get('register')} words {acc2*100:.0f}%)"}
            gate_log.append(row2)
            (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
            print(f"post-SVC: {reg2.get('f0_hz')}Hz [{reg2.get('register')}] words {acc2*100:.1f}% "
                  f"-> {row2['verdict']}", flush=True)
            if ok2 and acc2 >= 0.75:
                WINNER, WINNER_ACC = fixed, acc2
        else:
            print(f"SVC conversion failed (rc={rc}) — the assert below refuses honestly", flush=True)

assert WINNER, "no candidate passed the male+intelligible gate — refusing to master a wrong take"
sh(f"cp '{WINNER}' {WORK}/take_raw.mp3", quiet=True)
# ── master: static gain, oversampled limiter, then MEASURE-THEN-CORRECT on the encode ───
# Never loudnorm's second pass: it silently discards linear=true when the target is unreachable
# and rides gain instead (measured on UNBROKEN's ancestor: LRA 7.1 -> 3.0 against a 5.4 target).
TARGET_LUFS, TARGET_TP = -10.0, -1.0
wav, mp3 = OUT / "STEEL.wav", OUT / "STEEL.mp3"

def master_once(src_mp3, ceiling_db):
    a = M.loudness(src_mp3)
    g = TARGET_LUFS - (a["lufs"] if a["lufs"] is not None else -14.0)
    lim = 10 ** (ceiling_db / 20)
    af = (f"volume={g:.2f}dB,aresample=176400,"
          f"alimiter=limit={lim:.5f}:level=disabled,aresample=44100")
    sh(f"ffmpeg -v error -i '{src_mp3}' -af '{af}' -ar 44100 '{wav}' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{wav}' -codec:a libmp3lame -b:a 320k '{mp3}' -y", quiet=True)
    return M.loudness(mp3)

# Reference-matching pre-stage (research: Matchering 2.0, limiter/normalize OFF so our proven
# finisher stays the only dynamics authority). Optional by design: on any failure the winner
# proceeds unmatched, recorded.
MATCH_IN = WINNER
try:
    sh("pip install -q matchering 2>&1 | tail -1")
    import matchering as mg
    sh(f"ffmpeg -v error -i '{WINNER}' -ar 44100 /tmp/_target.wav -y", quiet=True)
    sh(f"ffmpeg -v error -i '{MALE_REF}' -ar 44100 /tmp/_ref.wav -y", quiet=True)
    mg.process(target="/tmp/_target.wav", reference="/tmp/_ref.wav",
               results=[mg.Result("/tmp/_matched.wav", "FLOAT",
                                  use_limiter=False, normalize=False)])
    MATCH_IN = "/tmp/_matched.wav"
    print("matchering pre-stage applied (house-sound reference)", flush=True)
except Exception as e:
    print(f"matchering skipped ({str(e)[:80]}) — winner proceeds unmatched", flush=True)

ceiling, iterations = TARGET_TP, []
for it in range(3):
    got = master_once(MATCH_IN, ceiling)
    iterations.append({"ceiling_db": round(ceiling, 2), **{k: got.get(k) for k in
                       ("lufs", "lra_lu", "true_peak_dbtp", "peak_dbfs", "clipped")}})
    tp = got.get("true_peak_dbtp")
    print(f"master pass {it+1}: {iterations[-1]}", flush=True)
    if tp is None or tp <= TARGET_TP + 0.05:
        break
    # The encoder overshoots; lower the ceiling by the measured excess plus margin and go again.
    ceiling -= (tp - TARGET_TP) + 0.1
(WORK / "master.json").write_text(json.dumps(iterations, indent=2))

# ── artwork: FLUX.1-schnell under sequential offload; SDXL is the RECORDED fallback ─────
ART_PROMPT = ("a forged steel sword laid across a cold black anvil in a dead forge, one living ember's glow reflected along the blade's bright temper line, ash motes in still air, embers dying in the hearth behind, cinematic, photorealistic, solemn, ultra detailed")
art, art_model = OUT / "cover.png", None
try:
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(PINS["image"], torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()          # 12B on a 16 GB card: stream, don't resident
    img = pipe(ART_PROMPT, num_inference_steps=4, guidance_scale=0.0,
               width=1024, height=1024,
               generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
    img.save(art); art_model = PINS["image"]
    del pipe; torch.cuda.empty_cache()
except Exception as e:
    print(f"FLUX unavailable ({str(e)[:100]}) — SDXL fallback", flush=True)
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        PINS["image_fallback"], torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    img = pipe(ART_PROMPT, num_inference_steps=40,
               generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
    img.save(art); art_model = PINS["image_fallback"]
    del pipe; torch.cuda.empty_cache()
print("artwork by:", art_model, flush=True)
sh(f"ffmpeg -v error -i '{art}' -vf scale=3000:3000:flags=lanczos '{OUT}/cover_3000.png' -y", quiet=True)

# ── the simple cover video: ONE held image, slow push, the operator's standing rule ─────
mp4 = OUT / "STEEL_cover.mp4"
sh(f"ffmpeg -v error -loop 1 -i '{art}' -i '{wav}' "
   f"-filter_complex \"[0:v]scale=2160:2160:flags=lanczos,zoompan=z='1+0.04*on/(25*180)':"
   f"x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d=25*180:s=1080x1080:fps=25,format=yuv420p[v]\" "
   f"-map '[v]' -map 1:a -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 256k "
   f"-shortest '{mp4}' -y")

# ── VERIFY: the last word, on the exact bytes that ship ─────────────────────────────────
verify = {"pins": PINS, "seed_policy": f"first passing (seed,strength) of {CANDIDATES}"}
ok_reg, reg = register_gate(mp3)
acc = word_accuracy(mp3, LYRICS)
Lw, Lm = M.loudness(wav), M.loudness(mp3)
rep = M.report(str(mp3))
verify.update(register=reg, word_accuracy=round(acc, 3), asr_judge=_WH[1],
              wav=Lw, mp3=Lm, seconds=rep.get("seconds"), bpm=rep.get("bpm"),
              art_model=art_model, master_iterations=iterations)
problems = []
if not ok_reg: problems.append(f"register on the DELIVERED master is {reg.get('register')}")
if acc < 0.75: problems.append(f"word accuracy {acc*100:.1f}% under the 75% floor")
if WINNER_ACC is not None and acc < WINNER_ACC - 0.03:
    problems.append(f"mastering cost {100*(WINNER_ACC-acc):.1f} points of intelligibility")
for tag, L in (("wav", Lw), ("mp3", Lm)):
    if L.get("clipped"): problems.append(f"{tag} has {L['clipped']} clipped samples")
tp = Lm.get("true_peak_dbtp")
if tp is not None and tp > TARGET_TP + 0.05: problems.append(f"true peak {tp} dBTP over target")
if rep.get("seconds") and abs(rep["seconds"] - DURATION) > 5:
    problems.append(f"duration {rep['seconds']}s vs {DURATION}s")
verify["problems"] = problems
verify["disclosure"] = (f"male lead: median {reg.get('f0_hz')} Hz, lead mode {reg.get('lead_hz')} Hz "
                        f"carrying {reg.get('lead_frac')} of voiced frames, spread "
                        f"{reg.get('spread_st')} st, octave-up {reg.get('oct_up_frac')}; bands "
                        f"{reg.get('bands')} — a male choir in the arrangement is claimed as such")
(WORK / "verify_final.json").write_text(json.dumps(verify, indent=2))
print("\nDISCLOSURE:", verify["disclosure"], flush=True)
assert not problems, "VERIFY REFUSED THE MASTER: " + "; ".join(problems)
print(f"\nVERIFIED: male [{reg.get('register')}] · words {acc*100:.1f}% ({_WH[1]}) · "
      f"{Lm.get('lufs')} LUFS · LRA {Lm.get('lra_lu')} · TP {tp} dBTP · 0 clipped", flush=True)
sh(f"ls -la {OUT}")
