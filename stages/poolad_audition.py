# %% [markdown]
# # پولاد (POOLAD) — the Persian STEEL, on audition
#
# The lyric of STEEL, rewritten as Persian epic verse — the Shahnameh's own metre, بحر متقارب
# (fa'ūlun fa'ūlun fa'ūlun fa'ul), rhymed couplets — sung by the same model at the same
# production settings that made the English record, with the vocal language switched to Persian.
# Two arms, two seeds each, so the operator can approve BY EAR before the production slot is spent:
#
# | Arm | Conditioning | Why |
# |---|---|---|
# | A `anchor` | the published STEEL lead as `reference_audio` | the reference is what held the male register 22/22 times when tags alone did not; it also carries the march |
# | B `caption` | caption only | the register lottery, but the arrangement is free to be Persian |
#
# Model: ACE-Step 1.5 XL (4.6B, sft) at the pinned commit with the 4B structure planner — the
# newest weights on the ACE-Step org (the later "-diffusers" entries are format ports). The
# settings are the production record's, not the CLI's defaults: shift 1.0 (the CLI hardcodes the
# turbo 3.0), thinking on, 80 ODE steps, guidance 7.5, timesignature "4".

# %%
import json, os, re, subprocess, sys, time, shutil
from pathlib import Path
T0 = time.time()
def sh(c, quiet=False): subprocess.run(c, shell=True, check=True,
    stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
def clock(w): print(f"  ⏱ {w} · t+{(time.time()-T0)/60:.1f} min", flush=True)

PINS = {
    "ace_step_code": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",   # github.com/ACE-Step/ACE-Step-1.5
    "song_model": "acestep-v15-xl-sft",
    "planner": "acestep-5Hz-lm-4B",
    "measure_sha": "199535aa517324d8021667b5a34a799aedd19353",     # ArtaQuest/artamusic lib/measure.py
    "lyric_sha": "5f335210565f25188058b82f4d493c3db965163d",                            # ArtaQuest/artamusic song/lyrics_poolad_fa.txt
    "torch_pascal": "2.7.1", "cuda_line_pascal": "cu126",
    "asr": "large-v3",
}
TMP = Path("/tmp/aq"); TMP.mkdir(exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TMP/"hf"), HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  ACESTEP_GENERATION_TIMEOUT="2400",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  PYTORCH_ALLOC_CONF="expandable_segments:True")
PY = sys.executable   # `pip` on PATH is a different interpreter from the one running this notebook
# The card is read in a SUBPROCESS: importing torch here, before the installs, would pin this
# notebook to Kaggle's stock build and the cu126 line below would never be what runs (a v1 death).
_probe = subprocess.run([PY, "-c", "import torch; c=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0); "
                         "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('%d.%d' % c)"],
                        capture_output=True, text=True, check=True).stdout.split()
CAP = float(_probe[-1]); print("card:", " ".join(_probe[:-1]), CAP, flush=True)
PASCAL = 0 < CAP < 7.0

if not REPO.exists():
    sh(f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git {REPO}", quiet=True)
    sh(f"cd {REPO} && git checkout {PINS['ace_step_code']} && git log -n1 --format='code pin OK %h'")
sh(f"{PY} -m pip install -q 'demucs>=4.0.1' faster-whisper==1.2.1 hf_transfer toml python-dotenv "
   "modelscope diskcache py3langid pyloudnorm ffmpeg-python soundfile loguru einops accelerate numba "
   "scipy 'safetensors>=0.7.0' 'transformers>=4.51.0,<4.58.0' vector-quantize-pytorch ftfy "
   "sentencepiece protobuf 2>&1 | tail -1")
if PASCAL:
    # Kaggle's stock torch has no kernels for sm_60; the cu126 line goes LAST so it is what runs.
    sh(f"{PY} -m pip install -q torch=={PINS['torch_pascal']} torchvision==0.22.1 "
       f"torchaudio=={PINS['torch_pascal']} --index-url https://download.pytorch.org/whl/{PINS['cuda_line_pascal']} "
       "2>&1 | tail -1")
clock("installed")

import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['measure_sha']}/lib/measure.py",
    "/tmp/measure.py")
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/song/lyrics_poolad_fa.txt",
    "/tmp/lyrics_fa.txt")
sys.path.insert(0, "/tmp")
import numpy as np, torch
a = torch.randn(256, 256, device="cuda"); assert torch.isfinite(a @ a).all(), "CUDA matmul failed — wrong torch for this card"
import measure as M
LYRICS = Path("/tmp/lyrics_fa.txt").read_text(encoding="utf-8").strip()
assert LYRICS.startswith("[Intro]") and "پولاد" in LYRICS, "wrong lyric at pin"
_ref = sorted(Path("/kaggle/input").rglob("STEEL.mp3"))
assert _ref, "the STEEL lead is not mounted (kernel source artafather/steel-record-final)"
MALE_REF = str(_ref[0])
print(f"card capability {CAP} · pascal {PASCAL} · reference {MALE_REF}", flush=True)

# The caption is the only channel the model parses for arrangement; the instruction+caption+metas
# template truncates SILENTLY at 256 tokens, so this stays well under ~700 characters.
CAPTION = ("Persian epic anthem in the spirit of the Shahnameh: a deep, powerful male voice "
           "declaiming Persian verse out in front of the mix, proud and relentless. War drums, "
           "kus and dohol, pound a heavy march on every beat with daf and tombak driving under "
           "them; santur and kamancheh carry the melody and a ney answers each line; low brass "
           "and a male choir swell the choruses. The bridge drops to drums and the lone voice "
           "before the last chorus lands twice as heavy. Chahargah heroic mode, cinematic, "
           "triumphant, ending at full force with no fade.")
assert len(CAPTION) < 700, len(CAPTION)
BPM, KEYSCALE, DURATION = 112, "D minor", 180.0
clock("inputs ready")

# %%
sys.path.insert(0, str(REPO))
import toml
from huggingface_hub import snapshot_download
# The structure planner must be ON DISK and NAMED: absent, the CLI returns 0, prints
# "5Hz LM not initialized" and writes no audio. The 4B is the production record's planner.
snapshot_download(f"ACE-Step/{PINS['planner']}", local_dir=str(CKPT / PINS["planner"]))
_small = CKPT / "acestep-5Hz-lm-0.6B"
if _small.exists():
    shutil.move(str(_small), str(TMP / "lm-0.6B-parked"))
assert (CKPT / PINS["planner"]).exists(), "planner not on disk"
clock("planner on disk")

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
assert OLD in src, "ACE-Step changed under its pin"
orch.write_text(src.replace(OLD, NEW, 1))
BF16 = torch.cuda.is_bf16_supported()

def render_conf(name, seed, rung, steps, reference):
    conf = {"project_root": str(REPO), "config_path": rung["model"], "checkpoint_dir": str(CKPT),
            "lm_model_path": PINS["planner"],
            "save_dir": str(TMP / f"out_{name}"), "audio_format": "flac", "device": "cuda",
            "offload_to_cpu": rung["offload_to_cpu"], "offload_dit_to_cpu": rung["offload_dit_to_cpu"],
            "task_type": "text2music",
            "caption": CAPTION, "lyrics": LYRICS, "instrumental": False,
            "bpm": BPM, "keyscale": KEYSCALE, "timesignature": "4",
            "vocal_language": "fa",                      # constants.py VALID_LANGUAGES includes 'fa'
            "duration": DURATION, "inference_steps": steps, "guidance_scale": 7.5, "use_adg": False,
            "shift": 1.0,                                # sft/base schedule; the CLI's default is turbo 3.0
            "thinking": True,                            # the planner plans structure; every CoT field stays ours
            "use_cot_metas": False, "use_cot_caption": False, "use_cot_lyrics": False,
            "use_cot_language": False, "seed": seed, "infer_method": "ode",
            "batch_size": 1, "use_random_seed": False, "seeds": [seed]}
    if reference:
        conf.update({"reference_audio": reference, "audio_cover_strength": 0.35})
    return conf

def cli_render(name, conf_dict, dtype):
    conf = TMP / f"{name}.toml"; conf.write_text(toml.dumps(conf_dict), encoding="utf-8")
    rc = subprocess.run(
        f"cd {REPO} && AQ_FORCE_DTYPE={dtype} {PY} cli.py -c {conf} --backend pt --log-level INFO "
        f"> /tmp/cli_{name}.txt 2>&1", shell=True).returncode
    found = sorted((TMP / f"out_{name}").rglob("*.flac")) + sorted((TMP / f"out_{name}").rglob("*.wav"))
    if not found:
        log = Path(f"/tmp/cli_{name}.txt").read_text(errors="replace")
        hits = [l for l in log.splitlines() if re.search(r"LM|Initializ|CUDA|Error|error|nan", l)][-12:]
        print(f"  {name}: NO AUDIO (rc {rc}) — " + " | ".join(h[:120] for h in hits), flush=True)
    return rc, found

LADDER = ([("xl-resident", PINS["song_model"], "bfloat16", False, False),
           ("xl-offload",  PINS["song_model"], "bfloat16", True,  False),
           ("xl-dit-swap", PINS["song_model"], "bfloat16", True,  True)] if BF16 else []) + \
         [("sft-fp32", "acestep-v15-sft", "float32", False, False)]
chosen = None
for name, model, dtype, oc, od in LADDER:
    rung = dict(rung=name, model=model, dtype=dtype, offload_to_cpu=oc, offload_dit_to_cpu=od)
    rc, found = cli_render(f"probe_{name}", render_conf(f"probe_{name}", 7001, rung, 2, MALE_REF), dtype)
    if found:
        print(f"RUNG HELD: {name} — {model} @ {dtype}", flush=True); chosen = rung; break
assert chosen, "no rung held"
assert chosen["model"] == PINS["song_model"], f"the XL did not hold on this card — {chosen['rung']} is not the production model"
clock("rung held")

# %%
# The deterministic judge, Persian: demucs --shifts 0 on CPU after a pure-gain normalise to -14 LUFS,
# whisper large-v3 at temperature zero with the language FIXED to fa. Disclosure only — the
# audition is by ear; production re-gates everything.
_WH = [None]
def asr():
    from faster_whisper import WhisperModel
    if _WH[0] is None:
        _WH[0] = WhisperModel(PINS["asr"], device="cpu", compute_type="int8")
    return _WH[0]
def _fa_tokens(text):
    t = re.sub(r"\[[^\]]*\]", " ", text)
    t = t.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    t = re.sub(r"[ً-ْٰ]", "", t)             # harakat, tashdid, sukun
    t = re.sub(r"[،؛؟!\.:,\-—–\"'«»()]", " ", t)
    return [w for w in t.split() if w]
REF_TOKENS = _fa_tokens(LYRICS)
def word_accuracy_fa(stem):
    segs, _ = asr().transcribe(str(stem), language="fa", beam_size=5, vad_filter=True,
                               vad_parameters=dict(min_silence_duration_ms=400),
                               chunk_length=20, condition_on_previous_text=False, temperature=0.0)
    hyp = _fa_tokens(" ".join(s.text for s in segs))
    ref = REF_TOKENS
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    d[:,0] = np.arange(len(ref)+1); d[0,:] = np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    return 1.0 - min(1.0, float(d[-1,-1])/max(1,len(ref))), " ".join(hyp)[:400]

def vocal_stem(mp3):
    import demucs.separate, shlex, tempfile as _tf
    L = M.loudness(str(mp3))
    g = -14.0 - (L.get("lufs") if L.get("lufs") is not None else -14.0)
    norm = Path(_tf.mkdtemp()) / "norm.wav"
    sh(f"ffmpeg -v error -i '{mp3}' -af volume={g:.2f}dB -ar 44100 '{norm}' -y", quiet=True)
    td = _tf.mkdtemp()
    demucs.separate.main(shlex.split(
        f'--two-stems vocals -n htdemucs --shifts 0 --device cpu -o "{td}" "{norm}"'))
    return next(Path(td).rglob("vocals.wav"), None)

report = []
ARMS = [("anchor", MALE_REF), ("caption", None)]
for arm, ref in ARMS:
    for seed in (6001, 6002):
        name = f"{arm}{seed}"
        t1 = time.time()
        rc, found = cli_render(name, render_conf(name, seed, chosen, 80, ref), chosen["dtype"])
        assert found, f"{name}: no audio (rc {rc}) — {Path(f'/tmp/cli_{name}.txt').read_text(errors='replace')[-300:]}"
        mp3 = OUT / f"{name}.mp3"
        sh(f"ffmpeg -v error -i '{found[0]}' -codec:a libmp3lame -b:a 320k '{mp3}' -y")
        row = {"arm": arm, "seed": seed, "seconds": round(time.time()-t1), "reference": bool(ref)}
        try:
            stem = vocal_stem(mp3)
            reg = M.classify_f0(M.finite_f0(M.f0_yin(*M.load(str(stem), mono=True)))) if stem else {}
            row.update(register=reg.get("register"), lead_hz=reg.get("lead_hz"))
            acc, heard = word_accuracy_fa(stem)
            row.update(words_fa=round(acc, 3), heard=heard)
            L = M.loudness(str(mp3)); row.update(lufs=L.get("lufs"), lra_lu=L.get("lra_lu"))
        except Exception as e:                        # disclosure must never kill the audition
            row["judge_error"] = str(e)[:200]
        report.append(row)
        (WORK / "audition.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {name}: register {row.get('register')} · lead {row.get('lead_hz')} Hz · "
              f"words(fa) {row.get('words_fa')} · {row['seconds']}s", flush=True)
        clock(f"{name} done")
(WORK / "caption.txt").write_text(CAPTION)
(WORK / "lyrics_fa.txt").write_text(LYRICS, encoding="utf-8")
print("AUDITION:", json.dumps(report, ensure_ascii=False), flush=True)
clock("DONE")
