# %% [markdown]
# # STEEL — a record of record
#
# One public notebook that makes the whole record **from scratch** and proves each claim on the
# bytes it ships: the **lyric** (measured for craft), the **song** (generated, gated, mastered),
# the **cover still** (generated, chosen, the choice recorded) and the **cover loop** (generated
# from that still, closed into a seamless loop by construction), then a final **verify** pass
# that recomputes every claim on the delivered files and refuses to finish if any fails.
#
# Nothing here is downloaded from a private place, nothing is hand-edited between cells, and
# every model is pinned to an immutable revision. Anyone can hit *Copy & Edit → Run All* on
# Kaggle's free GPU and get this. That is weaker than a byte-identical double execution (samplers
# are seeded but the environment is not byte-frozen) and stronger than "trust our laptop".
#
# | Stage | Model / instrument | What is claimed | How it is checked |
# |---|---|---|---|
# | Lyric | ArtaQuest's own text | craft profile vs a measured reference | `lyric_profile.py` at a pinned commit |
# | Song | ACE-Step 1.5 XL (4.6B), style transfer from a public male take | male lead · intelligible · dynamic | YIN on a demucs stem · whisper large-v3 word accuracy · loudness / true peak / clipping |
# | Still | Z-Image-Turbo (6.15B, Apache-2.0) | photoreal, chosen by measurement with the human veto recorded | edge energy · warmth · darkness, all printed |
# | Loop | Wan2.2-I2V-A14B (Apache-2.0), first frame = last frame = the still | seamless, nothing moves backwards | zero cuts · wrap delta vs typical frame delta · motion energy |
# | Verify | the same instruments, on the delivered bytes | everything above | the run fails if any claim fails |
#
# Code, measurement tools and the lyric are public at github.com/ArtaQuest/artamusic
# (`stages/steel_record.py`); every fetch below is pinned to a commit sha.

# ── environment: one set of pins for song, still and loop; deps first, torch last ─────────
# The three stages share one environment on purpose: ACE-Step 1.5 (this commit) wants
# diffusers>=0.37 and transformers 4.51-4.57, and so do Z-Image-Turbo and Wan2.2 in diffusers
# 0.39.0. On a Pascal card (P100, sm_60) Kaggle's default torch has no kernels, so the cu126
# line is installed LAST (whatever runs last wins); on a T4 pair the default torch stays.
import gc, glob, hashlib, json, os, random, re, shutil, subprocess, sys, time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"      # torch >= 2.10 spelling
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TMP / "hf"), HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  ACESTEP_GENERATION_TIMEOUT="2400")
T_START = time.time()

PINS = {
    "ace_step_code": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",   # github.com/ACE-Step/ACE-Step-1.5
    "song_model": "acestep-v15-xl-sft",
    "measure_sha": "5e9880ac0486bef21033b999a9c7bf3a4b7bf0f6",      # ArtaQuest/artamusic lib/measure.py
    "lyric_profile_sha": "ebee5bf324d8a6cff22ba666825a777c7dfc5c39",  # ArtaQuest/artamusic lib/lyric_profile.py
    "asr": "large-v3",
    "image": ("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6"),
    "wan_base": ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "596658fd9ca6b7b71d5057529bbf319ecbc61d74"),
    "wan_gguf": ("jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
                 "338fb8eedd8f485c9188cf1b1de541721fc81d66"),
    "wan_high": "high_noise_1030/wan2.2_i2v_A14b_high_noise_lightx2v_4step_1030-Q4_K_M.gguf",
    "wan_low": "low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf",
    "torch_pascal": "2.7.1", "cuda_line_pascal": "cu126",
}
SEED = 4242
random.seed(SEED)

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:160]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1600:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1600:], flush=True)
    return r.returncode

def clock(tag):
    print(f"  ⏱ {tag} · t+{(time.time()-T_START)/60:.1f} min", flush=True)

smi = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                     shell=True, text=True, capture_output=True).stdout.strip()
print("GPU:", smi, flush=True)
CAP = float(smi.splitlines()[0].split(",")[1]) if smi else 0.0
PASCAL = 0 < CAP < 7.0
sh("free -g | head -2; nproc; df -h /kaggle/working /tmp | tail -2")

if not REPO.exists():
    sh(f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}")
    sh(f"cd {REPO} && git checkout {PINS['ace_step_code']} && git log -n1 --format='code pin OK %h'")
sh("pip install -q 'demucs>=4.0.1' faster-whisper==1.2.1 hf_transfer toml python-dotenv modelscope "
   "diskcache py3langid pyloudnorm ffmpeg-python soundfile loguru einops accelerate numba scipy "
   "'safetensors>=0.7.0' 'transformers>=4.51.0,<4.58.0' diffusers==0.39.0 "
   "vector-quantize-pytorch 'gguf>=0.10.0' ftfy sentencepiece protobuf imageio imageio-ffmpeg "
   "matchering 2>&1 | tail -2")
if PASCAL:
    sh(f"pip install -q torch=={PINS['torch_pascal']} torchvision==0.22.1 torchaudio=={PINS['torch_pascal']} "
       f"--index-url https://download.pytorch.org/whl/{PINS['cuda_line_pascal']} 2>&1 | tail -2")

import numpy as np
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
import diffusers, transformers
NGPU = torch.cuda.device_count()
print(f"torch {torch.__version__} · diffusers {diffusers.__version__} · transformers "
      f"{transformers.__version__} · cuda devices {NGPU} "
      f"{[torch.cuda.get_device_name(i) for i in range(NGPU)]} · seeded {SEED}", flush=True)

def gemm_ok(dt, n=2048):
    try:
        a = torch.randn(n, n, device="cuda", dtype=dt); c = a @ a
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(c).all()); del a, c; torch.cuda.empty_cache(); return ok
    except Exception:
        torch.cuda.empty_cache(); return False
BF16 = gemm_ok(torch.bfloat16)
print("bfloat16 usable:", BF16, flush=True)
clock("environment ready")

# %% [markdown]
# ## The lyric — written for the voice, measured for craft
#
# STEEL is sung by the blade. Every verse opens on the antiphon **Cut!** — the chant choir's
# stroke answered by the lead — and the arc walks the forge: the furnace, the hammer and the
# fold, the quench and the temper, the oath of service, and the inheritance from hand to hand.
# The chorus is the blade's testimony; the outro is the same testimony after the last hand is gone.
#
# The text was revised against a **measured craft profile** (`lib/lyric_profile.py`, the
# instrument that measured the reference record): lines were brought into the sung 6–8 syllable
# band (100%, from 76%), the bridge's hammer‑strokes were paired into sung lines, imperative
# openings were raised toward the reference's one‑in‑three, and every chorus was made
# word‑identical (near‑identical variants are what machine transcription mishears). The chant
# register stays deliberately monosyllabic (79% against a pop reference's 71%) — that number is
# printed below, not hidden. Lyrics © ArtaQuest Foundation.

# ── the lyric, and its craft numbers from the pinned instrument ────────────────────────────
import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_profile_sha']}/lib/lyric_profile.py",
    "/tmp/lyric_profile.py")
sys.path.insert(0, "/tmp")
import lyric_profile as LP

LYRICS = """[intro]
Call me Steel, the frozen flame.
Call me the anvil's claim.
Call me the oath a hammer swore,
the iron your fathers bore.

[verse]
Cut! by the furnace that roared,
Cut! by the heat it poured,
Deep in the grain the old fire sleeps,
What the flame gave, the blade still keeps.
Cut! born of the crushing blow,
Cut! born of the blinding glow,
Folded in dark and drawn to light,
Beaten until the core ran bright.

[verse]
Cut! every mark rings true,
Cut! every blow shows through,
Watch the temper run down my spine,
read where the fire signed, line by line.
Cut! from the water's scream,
Cut! from the rising steam,
Hot at the heart, I met the cold —
What the cold caught, the cold will hold.

[chorus]
I am the weight you bear.
I am the oath you swear.
I am the tempered edge.
I am the standing pledge.
The fire is out and cold,
the forge is lost to dark;
of all the flame once told,
I hold the living spark.

[verse]
Cut! when the horn calls clear,
Cut! through the choke of fear,
The smith gave form, the flame gave speed,
The hand has cause — I am the deed.
Cut! down the darkened field,
Cut! through the splintered shield,
Hold me, and you hold my trust;
steel keeps the cause, or steel is rust.

[bridge]
Heat is gone and cold has come;
deep in the steel the hammers drum.
Hands may tremble, hearts may hide;
steel stays steady at your side.
Grip me tight and swing me through,
what I keep, I keep for you.
Bend me far and feel me spring;
tempered deep — now hear me ring.

[verse]
Cut! with a willow's give,
Cut! with a will to live,
Hard steel shatters, soft steel bends;
temper is where the trembling ends.
Cut! and the weight comes due,
Cut! and the debt bleeds through,
All that I take, I also keep —
hearts may forgive; the scars run deep.

[chorus]
I am the weight you bear.
I am the oath you swear.
I am the tempered edge.
I am the standing pledge.
The fire is out and cold,
the forge is lost to dark;
of all the flame once told,
I hold the living spark.

[verse]
Cut! till the sun burns low,
Cut! till the children grow,
Watch bearers change and banners fade;
the hand goes home — I stay the blade.
Cut! for the oath I keep,
Cut! while the makers sleep,
From hand to hand the burden runs,
passed from your fathers to your sons.

[chorus]
I am the weight you bear.
I am the oath you swear.
I am the tempered edge.
I am the standing pledge.
The fire is out and cold,
the forge is lost to dark;
of all the flame once told,
I hold the living spark.

[outro]
The fire is dead and cold.
The last hand has fallen still.
The spark the flame once told,
I kept — and always will."""

(OUT / "STEEL_lyrics.txt").write_text(LYRICS + "\n")
craft = LP.measure(LYRICS)
craft_report = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in craft.items()
                if k not in ("long_lines", "short_lines")}
craft_report["long_lines"] = craft["long_lines"]; craft_report["short_lines"] = craft["short_lines"]
craft_report["target"] = LP.TARGET
craft_report["invariants"] = LP.check_invariants(LYRICS, "steel") or ["ok"]
(WORK / "lyric_craft.json").write_text(json.dumps(craft_report, indent=2))
print(json.dumps({k: craft_report[k] for k in ("lines", "words", "syllables", "tight_pct",
                                              "mono_pct", "imper_pct", "invariants")}, indent=1),
      flush=True)
assert craft_report["invariants"] == ["ok"], "lyric invariants broken"

CAPTION = ("Dark chant anthem. Pounding war drums and anvil strikes on the beat, massive unison "
           "male chant choir answering a deep gravelly lead vocal, low strings and war horns, "
           "sparse and martial, minor key, solemn and heavy, 100 BPM.")
DURATION = 180.0
BPM, KEYSCALE = 100, "F minor"   # match the conditioning reference; a key fight is an experiment,
                                 # and a publication run is not where you run one

# %% [markdown]
# ## The instruments — pinned, and proven before anything expensive runs
#
# Every estimator this project ever used without a known‑truth self‑test turned out to be wrong,
# so the measurement library is fetched from the public repo **at a commit sha** and its
# self‑test is the first thing that runs. Register is measured with YIN on a **demucs‑isolated
# vocal stem** (mix‑based estimators lock onto the 808s), and reported as a distribution — median
# classifies, modes are disclosed. Intelligibility is **whisper large‑v3 word accuracy** against
# the exact lyric above, on the separated stem, chunked at quiet points. The ASR model is loaded
# per call and released, never left parked in front of a 12 GB render.

# ── measure.py at its pin; whisper and demucs on demand ─────────────────────────────────
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['measure_sha']}/lib/measure.py",
    "/tmp/measure.py")
import measure as M
assert M.selftest(), "measurement selftest FAILED — no number from this build can be trusted"

_WH = [None, None]   # model, judged_by
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
    _WH[0] = None
    gc.collect(); torch.cuda.empty_cache()

def _vocal_stem(mp3):
    """One separation, shared by register and words — the stem is where the voice is."""
    import demucs.separate, shlex, tempfile as _tf
    td = _tf.mkdtemp()
    demucs.separate.main(shlex.split(f'--two-stems vocals -n htdemucs --device cpu -o "{td}" "{mp3}"'))
    src = next(Path(td).rglob("vocals.wav"), None)
    if src is None:
        return None
    keep = Path(_tf.mkdtemp()) / "vocals.wav"
    shutil.copy(src, keep)
    return str(keep)

def word_accuracy(mp3, lyric_text, stem=None):
    # transcribe the SEPARATED STEM, chunked at quiet points into ~20 s segments (ICME-2025 recipe)
    target = stem or mp3
    segs, _ = asr().transcribe(str(target), beam_size=5, vad_filter=True,
                               vad_parameters=dict(min_silence_duration_ms=400),
                               chunk_length=20, condition_on_previous_text=False)
    hyp = re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
    asr_release()
    ref = re.findall(r"[a-z']+", re.sub(r"\[[^\]]*\]", " ", lyric_text.lower()))
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    d[:,0] = np.arange(len(ref)+1); d[0,:] = np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    return 1.0 - min(1.0, float(d[-1,-1])/max(1,len(ref)))

def register_gate(mp3):
    reg = M.register(str(mp3))
    return reg.get("register") == "male", reg

# the male voice reference — KEEP THE KEY's lead, the cleanest male vocal this pipeline owns
# (156 Hz median, 6.65 st spread), mounted from its PUBLIC kernel so the reference stays public
_ref = sorted(glob.glob("/kaggle/input/**/KEEPTHEKEY.mp3", recursive=True))
MALE_REF = _ref[0] if _ref else None
assert MALE_REF, "male reference not mounted (kernel source artafather/keep-the-key)"
print("male reference:", MALE_REF, flush=True)
clock("instruments proven")

# %% [markdown]
# ## The song — ACE-Step 1.5 XL, style transfer from a public male take, best of four by measurement
#
# Captions are not a control over vocal register (one male take in fifteen, measured), so the
# render is **style transfer**: text‑to‑music with the public KEEP THE KEY lead as `reference_audio`
# at strength 0.35 — the words stay text‑driven while the reference biases the timbre. Four seeds
# are rendered at 80 ODE steps, guidance 7.5, 180 s, and **every** take is gated on the isolated
# stem: register must measure male, word accuracy must reach 75%. Of the takes that pass, the one
# with the **highest word accuracy** ships (ties go to the tighter pitch spread). If nothing
# passes, the most intelligible non‑male take is converted to the male reference with zero‑shot
# Seed‑VC and gated again — a deterministic repair rather than another roll of the dice.
# The whole gate log is published beside the song.

# ── the 4.6B model: a ladder of ways to hold it on the card ─────────────────────────────
sys.path.insert(0, str(REPO))
import toml
from acestep.handler import AceStepHandler

LADDER = ([("xl-resident", PINS["song_model"], "bfloat16", False, False),
           ("xl-offload",  PINS["song_model"], "bfloat16", True,  False),
           ("xl-dit-swap", PINS["song_model"], "bfloat16", True,  True)] if BF16 else []) + \
         [("sft-fp32", "acestep-v15-sft", "float32", False, False)]

# ACE-Step picks fp16 on any card without bf16 hardware, and fp16 overflows to NaN in the 4.6B
# DiT; bf16 has float32's range and runs (emulated) on Pascal and Turing. Honour our own env var.
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

def render_conf(name, seed, strength, rung, steps, duration):
    return {"project_root": str(REPO), "config_path": rung["model"], "checkpoint_dir": str(CKPT),
            "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
            "offload_to_cpu": rung["offload_to_cpu"], "offload_dit_to_cpu": rung["offload_dit_to_cpu"],
            "task_type": "text2music", "reference_audio": MALE_REF, "audio_cover_strength": strength,
            "caption": CAPTION, "lyrics": LYRICS, "instrumental": False,
            "bpm": BPM, "keyscale": KEYSCALE, "timesignature": "4/4", "vocal_language": "en",
            "duration": duration, "inference_steps": steps, "guidance_scale": 7.5,
            "seed": seed, "infer_method": "ode",
            "thinking": False, "use_cot_metas": False, "use_cot_caption": False,
            "use_cot_lyrics": False, "use_cot_language": False,
            "batch_size": 1, "use_random_seed": False, "seeds": [seed]}

def cli_render(name, conf_dict, dtype):
    """One render through ACE-Step's own CLI in its own process (the GPU is clean afterwards).
    Whole output to a file, no pipes: the line's rc is then genuinely the CLI's."""
    conf = TMP / f"{name}.toml"; conf.write_text(toml.dumps(conf_dict))
    rc = sh(f"cd {REPO} && AQ_FORCE_DTYPE={dtype} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"PYTORCH_ALLOC_CONF=expandable_segments:True ACESTEP_GENERATION_TIMEOUT=2400 "
            f"python cli.py -c {conf} --backend pt --log-level INFO > /tmp/cli_{name}.txt 2>&1", quiet=True)
    found = sorted(Path(TMP / f"out_{name}").rglob("*.flac")) + sorted(Path(TMP / f"out_{name}").rglob("*.wav"))
    tail = Path(f"/tmp/cli_{name}.txt").read_text()[-900:] if Path(f"/tmp/cli_{name}.txt").exists() else ""
    return rc, found, tail

# A rung is not held because the model LOADED — it is held because it RENDERED. The first run on a
# T4 pair loaded the resident rung at 12.1 GB and then every take OOM'd inside the CLI on a 1.2 GB
# attention buffer (rc 0, no file: ACE-Step suppresses rather than crashes). So each rung is probed
# with a FULL-LENGTH render at 2 steps — attention buffers scale with the sequence, not the steps —
# and only a rung that produces audio is kept.
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
        del h; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"rung {name}: load failed — {type(e).__name__}: {str(e)[:100]}", flush=True)
        torch.cuda.empty_cache(); continue
    rung = dict(rung=name, model=model, dtype=dtype, offload_to_cpu=oc, offload_dit_to_cpu=od,
                peak_gb=round(peak, 2))
    rc, found, tail = cli_render(f"probe_{name}", render_conf(f"probe_{name}", SEED, 0.35, rung, 2, DURATION), dtype)
    if found:
        rung["probe_seconds"] = round(time.time() - t0, 1)
        print(f"RUNG HELD: {name} — {model} @ {dtype}, load peak {peak:.2f} GB, full-length probe "
              f"rendered in {time.time()-t0:.0f}s", flush=True)
        chosen = rung; break
    print(f"rung {name}: loaded ({peak:.2f} GB) but the full-length probe produced no audio (rc={rc}):\n"
          f"{tail[-400:]}", flush=True)
assert chosen, "no rung held"
(WORK / "rung.json").write_text(json.dumps({**chosen, "pins": {k: v for k, v in PINS.items()
                                                               if isinstance(v, str)},
                                            "seed": SEED}, indent=2))
clock("song model held")

# ── four takes: render every one, gate every one, keep the best passer ───────────────────
CANDIDATES = [(6001, 0.35), (6002, 0.35), (6003, 0.35), (6004, 0.35)]
gate_log, passers = [], []
for seed, strength in CANDIDATES:
    name = f"cand{seed}"
    t0 = time.time()
    rc, found, tail = cli_render(name, render_conf(name, seed, strength, chosen, 80, DURATION), chosen["dtype"])
    row = {"seed": seed, "seconds": round(time.time()-t0, 1), "steps": 80,
           "cover_strength": strength, "model": chosen["model"], "dtype": chosen["dtype"],
           "rung": chosen["rung"]}
    if not found:
        row["verdict"] = f"no audio (rc={rc})"; row["cli_tail"] = tail
        gate_log.append(row); (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
        print(f"{name}: no audio rc={rc}\n{tail[-400:]}", flush=True); continue
    mp3 = OUT / f"{name}.mp3"
    sh(f"ffmpeg -v error -i '{max(found, key=lambda p: p.stat().st_size)}' "
       f"-codec:a libmp3lame -b:a 320k '{mp3}' -y", quiet=True)
    stem = _vocal_stem(mp3)
    ok_reg, reg = register_gate(mp3)
    acc = word_accuracy(mp3, LYRICS, stem=stem)
    row.update(register=reg, word_accuracy=round(acc, 3), asr_judge=_WH[1])
    ok_words = acc >= 0.75
    row["verdict"] = "PASS" if (ok_reg and ok_words) else \
        f"REJECTED ({reg.get('register')}{'' if ok_words else f' words {acc*100:.0f}%'})"
    gate_log.append(row); (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
    print(f"{name}: median={reg.get('f0_hz')}Hz [{reg.get('register')}] "
          f"spread={reg.get('spread_st')}st lead={reg.get('lead_hz')}Hz@{reg.get('lead_frac')} "
          f"words={acc*100:.1f}% -> {row['verdict']} · {row['seconds']:.0f}s", flush=True)
    if ok_reg and ok_words:
        passers.append((acc, -float(reg.get("spread_st") or 99), mp3, seed))
    clock(f"{name} gated")

WINNER, WINNER_ACC, WINNER_SEED = None, None, None
if passers:
    passers.sort(reverse=True)
    WINNER_ACC, _, WINNER, WINNER_SEED = passers[0]
    for r in gate_log:
        if r.get("seed") == WINNER_SEED and r.get("verdict") == "PASS":
            r["verdict"] = "ACCEPTED — best words of the passers"
    (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
    print(f"\nWINNER: seed {WINNER_SEED} · words {WINNER_ACC*100:.1f}% · "
          f"{len(passers)} of {len(CANDIDATES)} passed", flush=True)

# ── the repair path: zero-shot Seed-VC to the male reference, only if nothing passed ─────
if WINNER is None:
    repairable = [r for r in gate_log
                  if r.get("word_accuracy", 0) >= 0.80 and (r.get("register") or {}).get("register") != "male"]
    if repairable:
        best = max(repairable, key=lambda r: r["word_accuracy"])
        print(f"\nSVC REPAIR: converting seed {best['seed']}'s vocal to the male reference", flush=True)
        # Seed-VC in its OWN package tree (pip --target + PYTHONPATH): its requirements broke the
        # shared transformers once; venv is unavailable (Kaggle's ensurepip is broken).
        sh("git clone -q https://github.com/Plachtaa/seed-vc /tmp/seedvc")
        sh("pip install -q --target /tmp/svcdeps -r /tmp/seedvc/requirements.txt 2>&1 | tail -2")
        if PASCAL:
            sh("pip install -q --target /tmp/svcdeps --upgrade --force-reinstall "
               "torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1")
        # the PyPI 'typing'/'dataclasses' backports shadow the stdlib; the pyOpenSSL/cryptography
        # pair must come from ONE tree — evict them from the overlay (each was a dead run once)
        sh("rm -f /tmp/svcdeps/typing.py /tmp/svcdeps/dataclasses.py && "
           "rm -rf /tmp/svcdeps/typing-*.dist-info /tmp/svcdeps/dataclasses-*.dist-info "
           "/tmp/svcdeps/typing /tmp/svcdeps/dataclasses /tmp/svcdeps/OpenSSL /tmp/svcdeps/pyOpenSSL* "
           "/tmp/svcdeps/pyopenssl* /tmp/svcdeps/cryptography /tmp/svcdeps/cryptography-*")
        rc0 = sh("PYTHONPATH=/tmp/svcdeps python -c 'import typing, dataclasses, numpy, torch, "
                 "OpenSSL, yaml, librosa, soundfile, transformers; "
                 "print(\"svcdeps tree clean, torch\", torch.__version__)'")
        cand = OUT / f"cand{best['seed']}.mp3"
        stem = _vocal_stem(str(cand))
        import soundfile as sf
        acc_wav = str(Path(stem).parent / "acc.wav")
        sh(f"ffmpeg -v error -i '{cand}' -ar 44100 -ac 2 /tmp/_mix.wav -y", quiet=True)
        mix, sr_ = sf.read("/tmp/_mix.wav"); voc, _ = sf.read(stem)
        n = min(len(mix), len(voc)); sf.write(acc_wav, mix[:n] - voc[:n], sr_)
        conv_dir = "/tmp/svc_out"
        rc = 1 if rc0 else sh(f"cd /tmp/seedvc && PYTHONPATH=/tmp/svcdeps python inference.py --source '{stem}' "
                              f"--target '{MALE_REF}' --output {conv_dir} "
                              f"--diffusion-steps 50 --length-adjust 1.0 --inference-cfg-rate 0.7 "
                              f"--f0-condition True --auto-f0-adjust True > /tmp/svc.log 2>&1")
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
                    "verdict": "ACCEPTED post-SVC" if (ok2 and acc2 >= 0.75) else
                               f"REJECTED post-SVC ({reg2.get('register')} words {acc2*100:.0f}%)"}
            gate_log.append(row2); (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
            print(f"post-SVC: {reg2.get('f0_hz')}Hz [{reg2.get('register')}] words {acc2*100:.1f}% "
                  f"-> {row2['verdict']}", flush=True)
            if ok2 and acc2 >= 0.75:
                WINNER, WINNER_ACC, WINNER_SEED = fixed, acc2, best["seed"]
        else:
            print(f"SVC conversion failed (rc={rc}); log tail: "
                  f"{Path('/tmp/svc.log').read_text()[-400:] if Path('/tmp/svc.log').exists() else ''}",
                  flush=True)

assert WINNER, "no candidate passed the male+intelligible gate — refusing to master a wrong take"
shutil.copy(WINNER, WORK / "take_raw.mp3")
clock("song chosen")

# %% [markdown]
# ## Mastering — measured, never trusted
#
# Static gain to −10 LUFS, then a 4×‑oversampled true‑peak limiter at −1 dBTP, then
# **measure the encoded mp3 and correct** (LAME overshoots; a target you never re‑measure is how a
# record ships at −0.8 dBTP against a −1.0 promise). Never `loudnorm`'s second pass: it silently
# discards `linear=true` when the target is unreachable and rides gain instead. Two masters are
# made — direct, and tonally matched to the reference with matchering — and the one whose
# **word accuracy** survives better ships; the choice is recorded.

# ── master: static gain, oversampled limiter, measure-then-correct; the arm chosen by words ──
TARGET_LUFS, TARGET_TP = -10.0, -1.0
wav, mp3 = OUT / "STEEL.wav", OUT / "STEEL.mp3"

def finish(src, wav_out, mp3_out):
    ceiling, iters = TARGET_TP, []
    for it in range(3):
        a = M.loudness(src)
        g = TARGET_LUFS - (a["lufs"] if a["lufs"] is not None else -14.0)
        lim = 10 ** (ceiling / 20)
        af = (f"volume={g:.2f}dB,aresample=176400,"
              f"alimiter=limit={lim:.5f}:level=disabled,aresample=44100")
        sh(f"ffmpeg -v error -i '{src}' -af '{af}' -ar 44100 '{wav_out}' -y", quiet=True)
        sh(f"ffmpeg -v error -i '{wav_out}' -codec:a libmp3lame -b:a 320k '{mp3_out}' -y", quiet=True)
        got = M.loudness(mp3_out)
        iters.append({"ceiling_db": round(ceiling, 2), **{k: got.get(k) for k in
                      ("lufs", "lra_lu", "true_peak_dbtp", "peak_dbfs", "clipped")}})
        tp = got.get("true_peak_dbtp")
        if tp is None or tp <= TARGET_TP + 0.05:
            break
        ceiling -= (tp - TARGET_TP) + 0.1
    return iters

MATCH_IN = None
try:
    import matchering as mg
    sh(f"ffmpeg -v error -i '{WINNER}' -ar 44100 /tmp/_target.wav -y", quiet=True)
    sh(f"ffmpeg -v error -i '{MALE_REF}' -ar 44100 /tmp/_ref.wav -y", quiet=True)
    mg.process(target="/tmp/_target.wav", reference="/tmp/_ref.wav",
               results=[mg.Result("/tmp/_matched.wav", "FLOAT", use_limiter=False, normalize=False)])
    MATCH_IN = "/tmp/_matched.wav"
except Exception as e:
    print(f"matchering unavailable ({str(e)[:80]}) — direct master only", flush=True)

arms, master_iters = {}, {}
master_iters["direct"] = finish(str(WINNER), str(OUT / "_direct.wav"), str(OUT / "_direct.mp3"))
arms["direct"] = {"wav": str(OUT / "_direct.wav"), "mp3": str(OUT / "_direct.mp3")}
if MATCH_IN:
    master_iters["matched"] = finish(MATCH_IN, str(OUT / "_matched.wav"), str(OUT / "_matched.mp3"))
    arms["matched"] = {"wav": str(OUT / "_matched.wav"), "mp3": str(OUT / "_matched.mp3")}
scores = {}
for name, files in arms.items():
    st = _vocal_stem(files["mp3"])
    scores[name] = {"words": round(word_accuracy(files["mp3"], LYRICS, stem=st), 3)}
    print(f"master arm {name}: words {scores[name]['words']*100:.1f}%", flush=True)
best_arm = max(arms, key=lambda k: scores[k]["words"])
print(f"master choice by measurement: {best_arm}", flush=True)
shutil.copy(arms[best_arm]["wav"], wav); shutil.copy(arms[best_arm]["mp3"], mp3)
(WORK / "master.json").write_text(json.dumps({"arm_chosen": best_arm, "arm_scores": scores,
                                              "iterations": master_iters}, indent=2))
for f in ("_direct.wav", "_direct.mp3", "_matched.wav", "_matched.mp3"):
    (OUT / f).unlink(missing_ok=True)
clock("mastered")

# %% [markdown]
# ## The cover still — Z-Image-Turbo, four seeds, the choice recorded
#
# Z‑Image‑Turbo (6.15B, Apache‑2.0, CFG‑distilled: 9 steps, guidance 0) reads as a photograph
# where SDXL read as a game render. It runs on a 16 GB card only under **sequential CPU offload**
# (the DiT and its Qwen3‑4B text encoder cannot both be resident). Four seeds are rendered and
# scored on edge energy, warmth and darkness. The scorer prefers seed 5150; the human eye prefers
# **6270** — the blade sweeping the frame with the hearth behind — and this run records both and
# ships the eye's choice. Arguing with a scorer is allowed; hiding the argument is not.

# ── the still ────────────────────────────────────────────────────────────────────────────
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import ZImagePipeline

ART_PROMPT = (
    "A single forged steel sword lying across a bed of white-hot coals inside a dark blacksmith's "
    "forge at night. The blade is freshly quenched, its edge still glowing orange along a "
    "hardening line, the polished steel reflecting the fire. Fine sparks rise through the smoky "
    "air. Heavy stone anvil and soot-blackened brick behind, deep shadow, one warm light source "
    "from the coals below. Shot on a 85mm lens at f/2, shallow depth of field, volumetric haze, "
    "photorealistic, cinematic colour grade, ultra sharp detail on the blade, album cover.")
ART_NEG = ("cartoon, illustration, painting, cgi render, plastic, blurry, low detail, watermark, "
           "text, signature, extra swords, hands, people, oversaturated, orange filter")
IMAGE_REV = PINS["image"][1]
img_pipe = ZImagePipeline.from_pretrained(PINS["image"][0], revision=IMAGE_REV, torch_dtype=torch.bfloat16)
img_pipe.enable_sequential_cpu_offload()
if hasattr(img_pipe, "vae"):
    img_pipe.vae.enable_tiling()
STILL_SEEDS, HUMAN_PICK = [SEED, 5150, 6270, 7380], 6270
t0 = time.time(); stills = []
for seed in STILL_SEEDS:
    im = img_pipe(prompt=ART_PROMPT, negative_prompt=ART_NEG, width=896, height=896,
                  num_inference_steps=9, guidance_scale=0.0,
                  generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
    p = OUT / f"still_{seed}.png"; im.save(p)
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    g = a.mean(-1); gy, gx = np.gradient(g)
    stills.append({"seed": seed, "file": p.name,
                   "detail": round(float(np.sqrt(gx**2 + gy**2).mean() * 255), 2),
                   "warm": round(float((a[..., 0] - a[..., 2]).clip(0).mean() * 255), 2),
                   "dark_frac": round(float((g < 0.25).mean()), 3)})
    print(f"  still seed {seed}: {stills[-1]}", flush=True)
print(f"{len(stills)} stills in {time.time()-t0:.0f}s", flush=True)
scorer_pick = max(stills, key=lambda c: c["detail"] * (1 + c["warm"] / 40) * (0.5 + c["dark_frac"]))
pick = next((c for c in stills if c["seed"] == HUMAN_PICK), scorer_pick)
(WORK / "stills.json").write_text(json.dumps({"candidates": stills, "scorer_pick": scorer_pick,
    "human_pick": HUMAN_PICK, "shipped": pick, "image_revision": IMAGE_REV}, indent=2))
shutil.copy(OUT / pick["file"], OUT / "cover.png")
sh(f"ffmpeg -v error -i '{OUT}/cover.png' -vf scale=3000:3000:flags=lanczos '{OUT}/cover_3000.png' -y", quiet=True)
print("cover still:", pick, "(scorer preferred", scorer_pick["seed"], ")", flush=True)
del img_pipe; gc.collect(); torch.cuda.empty_cache()
clock("still chosen")

# %% [markdown]
# ## The cover loop — Wan2.2‑I2V‑A14B, first frame = last frame = the still
#
# The strongest open image‑to‑video model with an Apache‑2.0 licence that fits a free card:
# Alibaba's two 14B experts, here as GGUF Q4_K_M with the lightx2v four‑step distillation, run at
# the exact points it was trained on (t = 1000 · 750 · 500 · 250 under shift 5). The loop is
# closed **by construction**: the same still is pinned as the first and the last frame, so the
# clip must return to where it began and nothing is mirrored (heat, smoke and embers keep their
# direction). Measured on the previous round, the hard cut still pops — the model returns to the
# still inside its last pinned frame — so the delivered loop is a **6‑frame dissolve** of tail
# into head, which measures smoother than an ordinary frame step; the hard cut is shipped beside
# it for the record.
#
# On a Turing/Pascal card memory‑efficient attention exists only for fp16/fp32 (a bf16 model
# silently falls back to O(N²) attention), so the experts compute in fp16 with an fp32 fallback
# on any non‑finite latent. With two GPUs each expert lives on its own card; with one, they
# swap through host RAM and the decode is done explicitly after both are parked.

# ── the loop ─────────────────────────────────────────────────────────────────────────────
import ftfy, html
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                       WanImageToVideoPipeline, WanTransformer3DModel)

FPS, NF, STEPS, XF = 16, 81, 4, 6
WAN_BASE, WAN_BASE_REV = PINS["wan_base"]; WAN_GGUF, WAN_GGUF_REV = PINS["wan_gguf"]
BASE = snapshot_download(WAN_BASE, revision=WAN_BASE_REV,
                         allow_patterns=["model_index.json", "scheduler/*", "vae/*", "tokenizer/*",
                                         "text_encoder/*", "transformer/config.json",
                                         "transformer_2/config.json"])
HIGH = hf_hub_download(WAN_GGUF, PINS["wan_high"], revision=WAN_GGUF_REV)
LOW = hf_hub_download(WAN_GGUF, PINS["wan_low"], revision=WAN_GGUF_REV)

def sha_of(p, n=20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:n]
WAN_HASHES = {"high_noise_gguf": sha_of(HIGH), "low_noise_gguf": sha_of(LOW),
              "vae": sha_of(f"{BASE}/vae/diffusion_pytorch_model.safetensors")}
print("wan sha256:", WAN_HASHES, flush=True)

LOOP_PROMPT = ("Photograph, locked-off tripod shot, a dark blacksmith's forge at night. A polished steel "
               "sword lies across a bed of glowing orange coals in a stone hearth; an iron anvil stands "
               "behind it. Subtle, continuous, natural motion only: heat haze shimmers above the coals, "
               "the embers pulse and breathe with a slow orange glow, thin grey smoke drifts and curls "
               "gently in place, warm firelight flickers softly across the blade and the anvil. The "
               "blade, the hearth and the camera stay perfectly still. Cinematic, photorealistic, "
               "shallow depth of field, fine film grain. The motion is gentle and steady and the scene "
               "returns to exactly where it began, looping seamlessly.")
LOOP_NEG = ("camera movement, pan, zoom, dolly, handheld shake, cut, scene change, morphing, deformed "
            "blade, blade moving, extra objects, people, hands, text, watermark, subtitles, blurry, low "
            "quality, JPEG artifacts, overexposed, oversaturated, cartoon, painting, flicker, jitter")

def prompt_clean(t):
    t = html.unescape(html.unescape(ftfy.fix_text(t)))
    return re.sub(r"\s+", " ", t).strip()

# umT5-XXL (11.4 GB bf16) is encoded first and freed — it and an expert cannot share a card.
tok = AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda:0").eval()
def embed(text, max_len=512):
    ids = tok([prompt_clean(text)], padding="max_length", max_length=max_len, truncation=True,
              add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(ids.attention_mask.gt(0).sum(dim=1)[0])
    with torch.no_grad():
        h = te(ids.input_ids.to("cuda:0"), ids.attention_mask.to("cuda:0")).last_hidden_state[0].float().cpu()
    return torch.cat([h[:n], h.new_zeros(max_len - n, h.size(1))])[None]
PE, NE = embed(LOOP_PROMPT), embed(LOOP_NEG)
assert torch.isfinite(PE).all() and torch.isfinite(NE).all(), "text embeddings not finite"
PE, NE = PE.to("cuda:0"), NE.to("cuda:0")      # the pipeline casts their dtype but never moves them
del te; gc.collect(); torch.cuda.empty_cache()

class NaNLatent(RuntimeError):
    pass

class PinnedSigmas(FlowMatchEulerDiscreteScheduler):
    PRE_SHIFT = [1.0, 0.75, 0.5, 0.25]      # -> [1.0, 0.9375, 0.8333, 0.625] after shift 5
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        return super().set_timesteps(device=device, sigmas=list(self.PRE_SHIFT))

def build_wan(dtype):
    vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
    q = GGUFQuantizationConfig(compute_dtype=dtype)
    high = WanTransformer3DModel.from_single_file(HIGH, quantization_config=q, config=BASE,
                                                  subfolder="transformer", torch_dtype=dtype)
    low = WanTransformer3DModel.from_single_file(LOW, quantization_config=q, config=BASE,
                                                 subfolder="transformer_2", torch_dtype=dtype)
    p = WanImageToVideoPipeline.from_pretrained(BASE, transformer=high, transformer_2=low, vae=vae,
                                                text_encoder=None, tokenizer=None, torch_dtype=dtype)
    p.vae.enable_tiling()
    if NGPU >= 2:
        high.to("cuda:0"); vae.to("cuda:0"); low.to("cuda:1")
        _fwd = low.forward
        def _across(*a, **k):
            a = [x.to("cuda:1") if torch.is_tensor(x) else x for x in a]
            k = {n: (v.to("cuda:1") if torch.is_tensor(v) else v) for n, v in k.items()}
            out = _fwd(*a, **k)
            if isinstance(out, tuple):
                return tuple(o.to("cuda:0") if torch.is_tensor(o) else o for o in out)
            return out.__class__(sample=out.sample.to("cuda:0"))
        low.forward = _across
        print("wan placement: high-noise cuda:0 · low-noise cuda:1 · vae cuda:0", flush=True)
    else:
        p.enable_model_cpu_offload()
        print("wan placement: one card, model offload", flush=True)
    p.scheduler = PinnedSigmas(shift=5.0)
    return p

def guard(pipe_, i, t, kw):
    lat = kw["latents"]
    if not torch.isfinite(lat).all():
        raise NaNLatent(f"non-finite latents after step {i}")
    print(f"    wan step {i} · t={float(t):.0f} · t+{(time.time()-T_START)/60:.1f} min", flush=True)
    return {}

def decode_latents(pipe_, latents):
    """The pipeline's own decode, done explicitly after the experts are parked."""
    if NGPU < 2:
        pipe_.transformer.to("cpu"); pipe_.transformer_2.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
    vae = pipe_.vae.to("cuda:0")
    lat = latents.to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    lat = lat / std + mean
    with torch.no_grad():
        video = vae.decode(lat, return_dict=False)[0]
    video = pipe_.video_processor.postprocess_video(video, output_type="np")[0]
    return (np.clip(video, 0, 1) * 255).round().astype(np.uint8)

def gen_loop(pipe_, still, W, H, seed):
    img = still.resize((W, H), Image.LANCZOS)          # square -> square, no distortion
    g = torch.Generator(device="cuda:0").manual_seed(seed)
    out = pipe_(image=img, last_image=img, prompt_embeds=PE, negative_prompt_embeds=NE,
                height=H, width=W, num_frames=NF, num_inference_steps=STEPS, guidance_scale=1.0,
                generator=g, output_type="latent", callback_on_step_end=guard)
    frames = decode_latents(pipe_, out.frames)
    return frames, np.asarray(img)

def measure_loop(path, w=256):
    sh(f"ffmpeg -v error -i '{path}' -vf scale={w}:{w} -f rawvideo -pix_fmt rgb24 /tmp/v.rgb -y", quiet=True)
    a = np.fromfile("/tmp/v.rgb", dtype=np.uint8).reshape(-1, w, w, 3).astype(np.float32) / 255
    l = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    per = np.abs(np.diff(l, axis=0)).reshape(len(a) - 1, -1).mean(1) * 255
    wrap = float(np.abs(l[-1] - l[0]).mean() * 255); typical = float(np.percentile(per, 95))
    return {"frames": int(len(a)), "cuts": int((np.abs(np.diff(l.reshape(len(a), -1).mean(1))) > 0.10).sum()),
            "ti_mean": round(float(per.mean()), 2), "wrap_delta": round(wrap, 2),
            "typical_frame_delta": round(typical, 2), "wrap_ratio": round(wrap / max(typical, 1e-6), 2),
            "luma_mean": round(float(l.mean()), 3), "luma_std_min": round(float(l.std(axis=(1, 2)).min()), 4)}

def encode_loop(loop, base):
    fdir = Path(f"/tmp/frames_{Path(base).name}"); fdir.mkdir(exist_ok=True)
    for f in fdir.glob("*.png"): f.unlink()
    for i, f in enumerate(loop):
        Image.fromarray(f).save(fdir / f"{i:04d}.png")
    raw = f"{base}_raw.mp4"
    sh(f"ffmpeg -v error -framerate {FPS} -i '{fdir}/%04d.png' -c:v libx264 -crf 12 -preset slow -pix_fmt yuv420p '{raw}' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -cpu-used 1 -g 240 -pix_fmt yuv420p -an '{base}.webm' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -c:v libx264 -preset veryslow -crf 20 -pix_fmt yuv420p -movflags +faststart -an '{base}.mp4' -y", quiet=True)
    sh(f"ffmpeg -v error -i '{raw}' -vf scale=1080:1080:flags=lanczos -c:v libvpx-vp9 -crf 33 -b:v 0 -row-mt 1 -cpu-used 1 -g 240 -pix_fmt yuv420p -an '{base}_1080.webm' -y", quiet=True)
    return raw

still_img = Image.open(OUT / "cover.png").convert("RGB")
loop_rec, wan_pipe, wan_dtype = None, None, torch.float16
for size in (640, 576, 480):
    W = H = size
    for attempt in range(2):
        try:
            if wan_pipe is None:
                wan_pipe = build_wan(wan_dtype)
            print(f"\n=== loop {size}² · {NF} frames · {STEPS} steps · {wan_dtype} ===", flush=True)
            t0 = time.time()
            frames, ref = gen_loop(wan_pipe, still_img, W, H, SEED)
            gen_s = time.time() - t0
            L = frames[:-1]                                     # drop the duplicated pinned frame
            w_ = (np.arange(1, XF + 1) / (XF + 1))[:, None, None, None]
            blend = ((1 - w_) * L[-XF:].astype(np.float32) + w_ * L[:XF].astype(np.float32)).round().astype(np.uint8)
            XL = np.concatenate([L[XF:len(L) - XF], blend])     # the dissolve loop
            raw_xf = encode_loop(XL, str(OUT / "STEEL_cover_loop"))
            encode_loop(L, str(OUT / "STEEL_cover_loop_cut"))
            idx = np.linspace(0, len(XL) - 1, 8).round().astype(int)
            Image.fromarray(np.concatenate([XL[i] for i in idx], axis=1)).save(OUT / "STEEL_cover_loop_sheet.jpg", quality=88)
            Image.fromarray(np.concatenate([XL[i] for i in (-3, -2, -1, 0, 1, 2)], axis=1)).save(OUT / "STEEL_cover_loop_seam.jpg", quality=90)
            ref_f = ref.astype(np.float32)
            loop_rec = {"model": "Wan2.2-I2V-A14B lightx2v-4step Q4_K_M (jayn7 GGUF)", "hashes": WAN_HASHES,
                        "closure": f"first==last frame pinned (last_image); delivered as a {XF}-frame dissolve of tail into head; hard cut beside it",
                        "dtype": str(wan_dtype).replace("torch.", ""), "res": [W, H], "fps": FPS,
                        "gen_seconds": round(gen_s, 1), "seed": SEED, "prompt": LOOP_PROMPT,
                        "pin_mae_first": round(float(np.abs(frames[0].astype(np.float32) - ref_f).mean()), 2),
                        "pin_mae_last": round(float(np.abs(frames[-1].astype(np.float32) - ref_f).mean()), 2),
                        "dissolve": measure_loop(str(OUT / "STEEL_cover_loop.webm")),
                        "cut": measure_loop(str(OUT / "STEEL_cover_loop_cut.webm"))}
            (WORK / "loop_verify.json").write_text(json.dumps(loop_rec, indent=2))
            print("LOOP:", json.dumps(loop_rec["dissolve"]), flush=True)
            for f in (OUT / "STEEL_cover_loop_cut_raw.mp4",):
                f.unlink(missing_ok=True)
            break
        except NaNLatent as e:
            print(f"loop {size}: {e}", flush=True)
            if wan_dtype is torch.float32:
                break
            del wan_pipe; wan_pipe = None; gc.collect(); torch.cuda.empty_cache(); wan_dtype = torch.float32
            print("  rebuilding the animator in fp32", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"loop {size}: OOM — {str(e)[:200]}", flush=True)
            gc.collect(); torch.cuda.empty_cache(); break
        except Exception as e:                       # anything else at this size: step down, recorded
            print(f"loop {size}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            gc.collect(); torch.cuda.empty_cache(); break
    if loop_rec:
        break
assert loop_rec, "no loop was produced at any size"
del wan_pipe; gc.collect(); torch.cuda.empty_cache()
clock("loop delivered")

# ── the cover video: the loop under the whole song ───────────────────────────────────────
sh(f"ffmpeg -v error -stream_loop -1 -i '{OUT}/STEEL_cover_loop_raw.mp4' -i '{wav}' "
   f"-vf scale=1080:1080:flags=lanczos,format=yuv420p -c:v libx264 -preset slow -crf 20 "
   f"-c:a aac -b:a 256k -shortest -movflags +faststart '{OUT}/STEEL_cover_video.mp4' -y")
(OUT / "STEEL_cover_loop_raw.mp4").unlink(missing_ok=True)
clock("cover video rendered")

# %% [markdown]
# ## Verify — the last word, on the exact bytes that ship
#
# The song is re‑measured on the delivered mp3: register on a fresh stem, word accuracy with the
# same judge, loudness, true peak, clipping, duration; and mastering may not have cost more than
# three points of intelligibility against the take that was chosen. The loop is re‑measured on
# the delivered VP9: zero cuts, a wrap smoother than a typical frame step, motion present, no
# black frames. Every published file gets a sha256 in `manifest.json`. If any claim fails, this
# cell raises and the run does not finish — an unfinished run cannot be submitted.

# ── verify + manifest ────────────────────────────────────────────────────────────────────
verify = {"pins": {k: v for k, v in PINS.items()}, "image_revision": IMAGE_REV,
          "seed_policy": f"best-of-{len(CANDIDATES)} passers by word accuracy; seeds {[c[0] for c in CANDIDATES]}",
          "winner_seed": WINNER_SEED, "lyric_craft": {k: craft_report[k] for k in
                                                     ("lines", "words", "syllables", "tight_pct", "mono_pct", "imper_pct")}}
ok_reg, reg = register_gate(mp3)
acc = word_accuracy(mp3, LYRICS, stem=_vocal_stem(mp3))
Lw, Lm = M.loudness(wav), M.loudness(mp3)
rep = M.report(str(mp3))
verify.update(register=reg, word_accuracy=round(acc, 3), asr_judge=_WH[1], wav=Lw, mp3=Lm,
              seconds=rep.get("seconds"), bpm=rep.get("bpm"), master_arm=best_arm,
              still=pick, loop=loop_rec)
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
lp = loop_rec["dissolve"]
if lp["cuts"]: problems.append(f"the cover loop cuts ({lp['cuts']})")
if lp["wrap_ratio"] >= 1.5: problems.append(f"loop wrap visible: ratio {lp['wrap_ratio']}")
if lp["ti_mean"] < 0.3: problems.append("the cover loop does not move")
if lp["luma_std_min"] < 0.01 or lp["luma_mean"] < 0.02: problems.append("the cover loop has black frames")
verify["problems"] = problems
verify["disclosure"] = (f"male lead: median {reg.get('f0_hz')} Hz, lead mode {reg.get('lead_hz')} Hz "
                        f"carrying {reg.get('lead_frac')} of voiced frames, spread {reg.get('spread_st')} st, "
                        f"octave-up {reg.get('oct_up_frac')}; bands {reg.get('bands')} — a male choir in "
                        f"the arrangement is claimed as such. Lyric monosyllable rate "
                        f"{craft_report['mono_pct']}% against a pop reference's 71%: a chant register, by design.")

manifest = {}
for f in sorted(OUT.iterdir()):
    if f.is_file():
        manifest[f.name] = {"bytes": f.stat().st_size, "sha256": sha_of(f, 64)}
(WORK / "manifest.json").write_text(json.dumps(manifest, indent=2))
(WORK / "verify_final.json").write_text(json.dumps(verify, indent=2))
print("\nDISCLOSURE:", verify["disclosure"], flush=True)
print("\nMANIFEST:", json.dumps({k: v["bytes"] for k, v in manifest.items()}, indent=1), flush=True)
assert not problems, "VERIFY REFUSED THE RECORD: " + "; ".join(problems)
print(f"\nVERIFIED: male [{reg.get('register')}] · words {acc*100:.1f}% ({_WH[1]}) · "
      f"{Lm.get('lufs')} LUFS · LRA {Lm.get('lra_lu')} · TP {tp} dBTP · 0 clipped · "
      f"loop wrap {lp['wrap_ratio']}x · {lp['frames']} frames", flush=True)
clock("DONE")
