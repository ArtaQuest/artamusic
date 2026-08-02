# UNBROKEN — the publication run. One public notebook that generates, VERIFIES and masters the
# record, and renders its cover — so the published files are provably this run's outputs, and the
# claims travel with them:
#
#   male voice   — not requested but MEASURED, in this notebook, on a demucs-isolated stem with
#                  YIN. Captions proved to be a 7% lottery (1 male in 15); conditioning on the
#                  published male take (task_type='cover', src_audio) went 3-for-3 male in the
#                  strength probe. The gate runs per candidate seed and a female take is REJECTED
#                  in the log, never shipped.
#   intelligible — Whisper word accuracy against the exact lyric below, in this notebook; gate 75%.
#   dynamic      — mastered with a STATIC gain + 4x-oversampled true-peak limiter. Never loudnorm's
#                  second pass: it silently discards linear=true when the target is unreachable
#                  and rides gain (measured: it collapsed loudness range 7.1 -> 3.0 LU).
#
# The measurement code is fetched from the public repo github.com/ArtaQuest/artamusic PINNED TO A
# COMMIT SHA, so the instruments used to verify this run are themselves versioned and inspectable.
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
sh("pip install -q demucs faster-whisper hf_transfer toml python-dotenv modelscope diskcache py3langid pyloudnorm "
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

# ── the instruments, pinned by commit ────────────────────────────────────────────────────
MEASURE_SHA = "60f9ce5629fcca0e83d1d84c46e870cce9b79ef5"
import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{MEASURE_SHA}/lib/measure.py",
    "/tmp/measure.py")
sys.path.insert(0, "/tmp")
import measure as M

def register_of(mp3):
    """demucs-isolated stem + YIN — the only register measurement that survived validation."""
    return M.register(mp3)

def wer_of(mp3, lyric_text):
    import re as _re
    from faster_whisper import WhisperModel
    global _WH
    try: _WH
    except NameError: _WH = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = _WH.transcribe(str(mp3), beam_size=5, vad_filter=False,
                             condition_on_previous_text=False)
    hyp = _re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
    t = _re.sub(r"\[[^\]]*\]", " ", lyric_text.lower())
    ref = _re.findall(r"[a-z']+", t)
    import numpy as _np
    d = _np.zeros((len(ref)+1, len(hyp)+1), dtype=_np.int32)
    d[:,0] = _np.arange(len(ref)+1); d[0,:] = _np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    return 1.0 - min(1.0, float(d[-1,-1])/max(1,len(ref)))

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

DURATION = 180.0
import toml

# Two candidate seeds; the gate decides, not hope. Strength 0.6 measured best in the probe
# (87.3% words, male 159.1 Hz); 80 steps per the measured policy: sweep at 60, FINALIZE at 80.
CANDIDATES = [4242, 4343]
WINNER = None
gate_log = []
for seed in CANDIDATES:
    name = f"cand{seed}"
    conf = TMP / f"{name}.toml"
    conf.write_text(toml.dumps({
        "project_root": str(REPO), "config_path": chosen["model"], "checkpoint_dir": str(CKPT),
        "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
        "offload_to_cpu": chosen["offload_to_cpu"],
        "offload_dit_to_cpu": chosen["offload_dit_to_cpu"],
        "task_type": "cover", "src_audio": MALE_REF, "audio_cover_strength": 0.6,
        "caption": "trap, epic, male choir, baritone", "lyrics": LYRICS, "instrumental": False,
        "bpm": 100, "keyscale": "F minor", "timesignature": "4/4", "vocal_language": "en",
        "duration": DURATION, "inference_steps": 80, "guidance_scale": 7.5,
        "seed": seed, "infer_method": "ode",
        "thinking": False, "use_cot_metas": False, "use_cot_caption": False,
        "use_cot_lyrics": False, "use_cot_language": False,
        "batch_size": 1, "use_random_seed": False, "seeds": [seed]}))
    t0 = time.time()
    sh(f"cd {REPO} && AQ_FORCE_DTYPE={chosen['dtype']} "
       f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACESTEP_GENERATION_TIMEOUT=2400 "
       f"python cli.py -c {conf} --backend pt --log-level INFO")
    found = sorted(Path(TMP / f"out_{name}").rglob("*.flac")) + \
            sorted(Path(TMP / f"out_{name}").rglob("*.wav"))
    row = {"seed": seed, "seconds": round(time.time()-t0, 1)}
    if not found:
        row["verdict"] = "no audio"; gate_log.append(row)
        print(f"{name}: no audio", flush=True); continue
    src_f = max(found, key=lambda p: p.stat().st_size)
    mp3 = OUT / f"{name}.mp3"
    sh(f"ffmpeg -v error -i '{src_f}' -codec:a libmp3lame -b:a 320k '{mp3}' -y", quiet=True)

    # THE GATE. A female take is rejected in this log, never shipped. This is the platform's own
    # thesis applied to ourselves: the claim is checked in the run that makes the artefact.
    reg = register_of(str(mp3))
    acc = wer_of(str(mp3), LYRICS)
    row.update(register=reg, word_accuracy=round(acc, 3))
    male = reg.get("register") == "male"
    ok_words = acc >= 0.75
    row["verdict"] = ("ACCEPTED" if (male and ok_words) else
                      f"REJECTED ({'register '+reg.get('register','?') if not male else ''}"
                      f"{' words %.0f%%' % (acc*100) if not ok_words else ''})")
    gate_log.append(row)
    print(f"{name}: f0={reg.get('f0_hz')} {reg.get('register')} · words {acc*100:.1f}% "
          f"-> {row['verdict']}", flush=True)
    if male and ok_words:
        WINNER = mp3; break

Path("/kaggle/working/gate.json").write_text(json.dumps(gate_log, indent=2))
assert WINNER, "no candidate passed the male+intelligible gate — REFUSING to master a wrong take"
sh(f"cp '{WINNER}' /kaggle/working/take_raw.mp3", quiet=True)

def write_wav(path, st, sr):
    x = np.clip(st, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1] if x.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())



def match_fir(err_db, sr, taps=4097, floor_hz=40.0, cap_boost=3.0, cap_cut=1.0):
    """Linear-phase FIR that REPAIRS deficits against the reference without flattening surpluses.

    err_db is measured MINUS reference, so the correction is its negative.

    Boosts and cuts are capped ASYMMETRICALLY, and that is a deliberate judgement rather than
    symmetry for its own sake. The winning take measured +3.3 to +5.8 dB above 1 kHz against the
    reference and -2.4 dB at 60-120 Hz. A symmetric match would have pulled the top down by the
    full cap — cutting precisely the brightness that carried 86% word intelligibility, which is
    the one thing this record was rejected for last time. The reference is a STYLE target, not a
    specification: where the take is thin, fill it; where it is livelier than the reference, let
    it be.

    The old floor at 120 Hz was also wrong for this material. It existed to stop the EQ gutting an
    808, but the take turned out to be LIGHT down there, so the floor was blocking the one
    correction that helps. It now only protects the true sub (below ~40 Hz), where a boost buys
    nothing audible and eats headroom.
    """
    n = taps
    f = np.fft.rfftfreq(n, 1 / sr)
    gain_db = np.zeros_like(f)
    centres = np.array([np.sqrt(lo * hi) for lo, hi in J.BANDS])
    corr = np.clip(-err_db, -cap_cut, cap_boost)
    # Interpolate the per-band correction smoothly across frequency (log spacing, as heard).
    gain_db = np.interp(np.log(np.maximum(f, 1.0)), np.log(centres), corr,
                        left=corr[0], right=corr[-1])
    # Fade the correction out below floor_hz so the true sub is never touched.
    gain_db *= np.clip((f - floor_hz / 2) / floor_hz, 0.0, 1.0)
    mag = 10 ** (gain_db / 20)
    h = np.fft.irfft(mag, n)
    h = np.roll(h, n // 2) * np.hanning(n)
    return h / max(1e-9, np.abs(np.fft.rfft(h)).max()) * 1.0



def shelf_fir(sr, hz=1800.0, gain_db=2.0, taps=2049):
    f = np.fft.rfftfreq(taps, 1 / sr)
    mag = 10 ** ((gain_db * np.clip((f - hz) / hz, 0.0, 1.0) *
                  np.clip((6000 - f) / 6000 + 0.4, 0.0, 1.0)) / 20)
    h = np.fft.irfft(mag, taps)
    return np.roll(h, taps // 2) * np.hanning(taps)



def convolve_stereo(st, h):
    n = len(st) + len(h) - 1
    n_fft = 1 << int(np.ceil(np.log2(n)))
    H = np.fft.rfft(h, n_fft)
    out = np.zeros((len(st), st.shape[1]))
    lag = len(h) // 2
    for c in range(st.shape[1]):
        y = np.fft.irfft(np.fft.rfft(st[:, c], n_fft) * H, n_fft)
        out[:, c] = y[lag:lag + len(st)]
    return out
# ── master the winning take, in-run ──────────────────────────────────────────────────────
# Everything below used to happen on a laptop, which meant the published file was NOT the file
# this notebook produced. For a platform whose whole claim is "these exact files came from this
# run", that gap is the difference between a true statement and a false one. So the mastering
# happens here.
# Module-level imports the LIFTED functions need. write_wav uses `wave` at module scope while
# _load imports it inside itself — which made a static check think it was bound and cost a
# whole GPU run to discover. Import everything the lifted bodies touch, here, once.
import numpy as np, json as _json, wave, tempfile
BANDS = [(20,60),(60,120),(120,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,16000)]

def _load(path, sr=44100, mono=False):
    import wave, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t: tmp=t.name
    subprocess.run(["ffmpeg","-v","error","-i",str(path),"-ac","1" if mono else "2",
                    "-ar",str(sr),"-c:a","pcm_s16le",tmp,"-y"],check=True)
    with wave.open(tmp) as w:
        raw=np.frombuffer(w.readframes(w.getnframes()),"<i2").astype(np.float64)/32768.0
        ch=w.getnchannels()
    os.unlink(tmp)
    return (raw if ch==1 else raw.reshape(-1,2)), sr

def _bands(x, sr):
    n=1<<15; hop=n//2
    if len(x)<n: return np.zeros(len(BANDS))
    idx=np.arange(1+(len(x)-n)//hop)[:,None]*hop+np.arange(n)[None,:]
    S=(np.abs(np.fft.rfft(x[idx]*np.hanning(n),axis=1))**2).mean(0)
    f=np.fft.rfftfreq(n,1/sr)
    return np.array([S[(f>=lo)&(f<hi)].sum() for lo,hi in BANDS])

_db = lambda v: 10*np.log10(np.maximum(v,1e-20))

# The mastering functions above were lifted VERBATIM from the tested chain rather than retyped,
# which is the right call — a retyped copy drifts. But they were written as part of a module that
# imports the judging harness as `J`, and lifting code without lifting what it depends on is how
# this run died with NameError after the song had already generated. A shim satisfies the
# references without editing the lifted bodies, so they stay byte-identical to the tested versions.
class J:
    BANDS = BANDS
    db = staticmethod(_db)
    band_energies = staticmethod(lambda x, sr=44100: _bands(x, sr))

take = WINNER
st, sr = _load(take)
tb=_bands(st.mean(1),sr); tdb=_db(tb/tb.sum())
# No reference record ships with this notebook, so the repair EQ targets a published trap curve
# rather than a file a reader cannot obtain. Stated plainly instead of silently skipped.
TARGET_DB = np.array([-3.1,-5.5,-11.7,-12.1,-12.9,-16.1,-18.8,-20.1,-25.1])
y = convolve_stereo(st, match_fir(tdb - TARGET_DB, sr))
y = y/max(1e-9,np.abs(y).max())*0.89
stage="/tmp/stage.wav"; write_wav(stage,y,sr)

probe=subprocess.run(["ffmpeg","-v","info","-i",stage,"-af",
    "loudnorm=I=-10:TP=-1:LRA=9:print_format=json","-f","null","-"],capture_output=True,text=True).stderr
stats=_json.loads(probe[probe.rindex("{"):probe.rindex("}")+1])
# STATIC gain, never loudnorm's second pass: `linear=true` is silently discarded when the target
# is unreachable (it requires target_I <= target_TP-(input_TP-input_I)), and ffmpeg then runs a
# gain RIDER. Measured on an earlier master: +9.9 dB on the intro against +0.6 dB on the drop,
# collapsing loudness range from 7.1 to 3.0 LU. The limiter is 4x oversampled because alimiter at
# base rate is a sample-peak, not a true-peak, limiter.
g = -10.0 - float(stats["input_i"])
subprocess.run(["ffmpeg","-v","error","-i",stage,"-af",
    f"volume={g:.2f}dB,aresample=176400,alimiter=limit=0.8913:level=disabled,aresample=44100",
    "-ar","44100","-c:a","pcm_s16le","/kaggle/working/UNBROKEN.wav","-y"],check=True)
subprocess.run(["ffmpeg","-v","error","-i","/kaggle/working/UNBROKEN.wav","-codec:a","libmp3lame",
    "-b:a","320k","/kaggle/working/UNBROKEN.mp3","-y"],check=True)
print(f"mastered: static gain {g:+.2f} dB from {stats['input_i']} LUFS")# ── the cover: one still, held, with a slow push ─────────────────────────────────────────
from diffusers import StableDiffusionXLPipeline
sdxl = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
img = sdxl(prompt="a bed of glowing coals in a dark forge, embers pulsing orange, fine ash "
                  "drifting, macro, shallow depth of field, deep cold blue shadows, single warm "
                  "gold key light, heavy film grain, cinematic, photorealistic",
           negative_prompt="cartoon, illustration, 3d render, text, watermark, blurry, low quality",
           width=1024, height=576, num_inference_steps=40, guidance_scale=7.0,
           generator=torch.Generator("cuda").manual_seed(4105)).images[0]
img.save("/kaggle/working/cover.png")

dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of",
    "csv=p=0","/kaggle/working/UNBROKEN.mp3"],capture_output=True,text=True).stdout.strip())
GRADE=("eq=contrast=1.10:saturation=0.88,curves=r='0/0 0.25/0.22 0.75/0.80 1/1':"
       "g='0/0.01 0.25/0.22 0.75/0.77 1/1':b='0/0.04 0.25/0.28 0.75/0.74 1/0.96',"
       "vignette=PI/5,noise=alls=4:allf=t+u")
over=1.06
subprocess.run(["ffmpeg","-v","error","-loop","1","-i","/kaggle/working/cover.png",
  "-i","/kaggle/working/UNBROKEN.mp3","-filter:v",
  f"scale=2150:1210:force_original_aspect_ratio=increase,crop=2150:1210,"
  f"crop=w='iw/(1+{over-1:.3f}*min(t/{dur:.2f},1))':h='ih/(1+{over-1:.3f}*min(t/{dur:.2f},1))':"
  f"x='(iw-ow)/2':y='(ih-oh)/2',scale=1920:1080,{GRADE},fps=24",
  "-map","0:v","-map","1:a","-t",f"{dur:.3f}","-c:v","libx264","-crf","19","-preset","medium",
  "-pix_fmt","yuv420p","-movflags","+faststart","-c:a","aac","-b:a","320k",
  "/kaggle/working/UNBROKEN_cover.mp4","-y"],check=True)

for p in Path("/kaggle/working/out").glob("*"): p.unlink()
Path("/kaggle/working/out").rmdir()
sh("ls -la /kaggle/working")
print("DONE — UNBROKEN.mp3, UNBROKEN.wav, UNBROKEN_cover.mp4, cover.png")