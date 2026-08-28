# %% [markdown]
# # STEEL improvement matrix — four axes off the operator's picked take
#
# The operator chose take 6003: the DEEPEST voice (74.7 Hz) and the BIGGEST dynamics (LRA 16.8),
# overriding the reference-distance metric — so that metric is retired from selection and 6003
# itself becomes the anchor: its audio is the reference conditioning for every take here, so the
# picked voice propagates. Four single-axis changes, one per take:
#   A  control: sft checkpoint, guidance 7.5           — isolates the new anchor + caption
#   B  guidance 5.5                                    — less prompt-forcing, more musical freedom
#   C  xl-base checkpoint + use_adg (Adaptive Dual     — the authors' quality path, only valid on
#      Guidance)                                          the base model, never tried here
#   D  the 4B structure planner instead of the 0.6B    — the LM behind `thinking` is chosen by
#      (the handler scans the checkpoint dir, so         SCANNING the checkpoint dir; with only
#      directory presence IS the selector)               the 4B present, the 4B plans
# Every take is male-gated by measurement, one retry seed. The ear picks. to give
#
# 22 of 22 HeartMuLa takes (base and RL) failed the male-register gate for this style family.
# ACE-Step held a deep male lead across every take of three full records, anchored by reference
# conditioning — so it wears the extracted style instead: F# minor, 85 BPM, a long ember intro,
# a half-spoken male voice under a sub-heavy band, choruses opening upward, a near-solo passage,
# a heavy ending with no fade. Style is enforced where ACE-Step actually listens: bpm and key as
# explicit metas, the arc in the caption, the mythic lyric as text.
#
# Three takes. Each is register-gated (male, by measurement) and measured against the reference's
# SIGNATURE — band profile, vocal-under-band offset, dynamics — extracted from the operator's own
# file. The takes ship for listening; the operator's ear decides before a full record is spent.

# %%
import gc, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
T0 = time.time()
def sh(c, quiet=False): subprocess.run(c, shell=True, check=True,
    stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
def clock(w): print(f"  ⏱ {w} · t+{(time.time()-T0)/60:.1f} min", flush=True)

PINS = {
    "ace_step_code": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
    "song_model": "acestep-v15-xl-sft",
    "song_model_base": "acestep-v15-xl-base",
    "measure_sha": "6770ef101f2f86355dd8e5d611a26416cb79906f",
    "mythic_sha": "97255ec860c91fe51c3cca6d3a3b299515a5cb98",
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
for f, sha in (("measure.py", PINS["measure_sha"]),):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{sha}/lib/{f}", f"/tmp/{f}")
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['mythic_sha']}/song/lyrics_steel_mythic.txt",
    "/tmp/lyrics.txt")
sys.path.insert(0, "/tmp")
import numpy as np, torch
import measure as M
LYRICS = Path("/tmp/lyrics.txt").read_text().strip()
assert LYRICS.startswith("[Intro]") and "name me dread" in LYRICS

# THE ANCHOR IS THE PICKED TAKE. cand6003 is mounted from the flame probe's own output and its
# timbre — the voice the operator chose — conditions every take here.
_ref = sorted(Path("/kaggle/input").rglob("*cand6003*.mp3"))
assert _ref, "cand6003 not mounted (kernel source ashraasn/steel-ace-flame)"
MALE_REF = str(_ref[0])
print("anchor (the picked take):", MALE_REF, flush=True)

# THE SIGNATURE — measured from the operator's reference file on the laptop, hardcoded with
# provenance rather than fetched (the reference itself is not ours to publish):
#   First Flame: 289 s · 85 BPM · F# minor · -14.1 LUFS · LRA 8.3 · block-sim 0.941
#   band log10 energies sub 1.42 / bass 1.40 / mid 0.74 / high 0.21 -> deltas vs mid +0.68/+0.66/-0.53
#   vocal stem sits 4.9 dB under the accompaniment
SIG = {"sub_mid": 0.68, "bass_mid": 0.66, "high_mid": -0.53, "vocal_under_db": -4.9, "lra": 8.3}

# Sharpened toward the picked take's measured profile: deeper voice, bigger swells.
CAPTION = ("Dark cinematic epic that rises from embers. A long instrumental introduction: deep "
           "sub bass drone and sparse thunderous percussion, building slowly. A cavernous "
           "bass-baritone voice enters, low and close, half spoken, buried deep inside the mix "
           "beneath the band. The choruses open upward with vast dynamic swells, grave and "
           "melodic, the drums enormous. One short passage leaves the voice almost alone over "
           "bass. The final choruses return twice as heavy, and the song ends at full force "
           "with no fade.")
BPM, KEYSCALE, DURATION = 85, "F# minor", 180.0

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
        f"ACESTEP_GENERATION_TIMEOUT=2400 python cli.py -c {conf} --backend pt --log-level INFO "
        f"> /tmp/cli_{name}.txt 2>&1", shell=True).returncode
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
    print(f"rung {name} failed (rc {rc}) — {Path(f'/tmp/cli_probe_{name}.txt').read_text()[-300:]}", flush=True)
assert chosen, "no rung held"
clock("rung held")

# %%
def male_register(mp3):
    import demucs.separate, shlex, tempfile as _tf
    td = _tf.mkdtemp()
    demucs.separate.main(shlex.split(
        f'--two-stems vocals -n htdemucs --shifts 0 --device cpu -o "{td}" "{mp3}"'))
    stem = next(Path(td).rglob("vocals.wav"), None)
    acc = next(Path(td).rglob("no_vocals.wav"), None)
    if stem is None:
        return False, {}, None, None
    r = M.classify_f0(M.finite_f0(M.f0_yin(*M.load(str(stem), mono=True))))
    ok = r.get("register") == "male" and (r.get("bands") or {}).get("female", 1.0) <= 0.25
    return ok, r, stem, acc

def style_distance(mp3, stem, acc):
    x, sr = M.load(str(mp3), mono=True)
    hop = sr//10; n = len(x)//hop
    fr = x[:n*hop].reshape(n, hop)*np.hanning(hop)
    S = np.abs(np.fft.rfft(fr, axis=1)); freqs = np.fft.rfftfreq(hop, 1/sr)
    def bE(lo, hi): return float(np.log10(np.maximum(S[:, (freqs>=lo)&(freqs<hi)].mean(), 1e-9)))
    sub, bass, mid, high = bE(20,90), bE(90,300), bE(300,2500), bE(2500,9000)
    xv,_ = M.load(str(stem), mono=True); xa,_ = M.load(str(acc), mono=True)
    v_under = 20*np.log10((np.sqrt((xv**2).mean())+1e-9)/(np.sqrt((xa**2).mean())+1e-9))
    L = M.loudness(str(mp3))
    comp = {"sub_mid": round(sub-mid, 2), "bass_mid": round(bass-mid, 2),
            "high_mid": round(high-mid, 2), "vocal_under_db": round(float(v_under), 1),
            "lra": L.get("lra_lu")}
    d = np.sqrt(sum(((comp[k]-SIG[k])/(1.0 if k not in ("vocal_under_db","lra") else 4.0))**2
                    for k in SIG if comp.get(k) is not None))
    return round(float(d), 2), comp

def with_(rung, **kw):
    r = dict(rung); r.update(kw); return r

ARMS = [
    ("A_control",  {},                        {}),
    ("B_guide55",  {},                        {"guidance_scale": 5.5}),
    ("C_base_adg", {"model": PINS["song_model_base"]}, {"use_adg": True}),
    ("D_lm4b",     {},                        {}),
]

def set_planner(big):
    """The LM handler scans the checkpoint dir for acestep-5Hz-lm-* — presence IS selection."""
    small = CKPT / "acestep-5Hz-lm-0.6B"; parked = TMP / "lm-0.6B-parked"
    if big:
        from huggingface_hub import snapshot_download
        snapshot_download("ACE-Step/acestep-5Hz-lm-4B", local_dir=str(CKPT / "acestep-5Hz-lm-4B"))
        if small.exists():
            shutil.move(str(small), str(parked))
    elif parked.exists() and not small.exists():
        shutil.move(str(parked), str(small))

report = []
for name, rung_over, conf_over in ARMS:
    set_planner(big=(name == "D_lm4b"))
    rung = with_(chosen, **rung_over)
    ok_take = False
    for attempt, seed in enumerate((7100, 7101)):
        conf = render_conf(name, seed, rung, 80)
        conf.update(conf_over)
        rc, found = cli_render(name, conf, rung["dtype"])
        if not found:
            print(f"  {name} seed {seed}: NO AUDIO (rc {rc}) — "
                  f"{Path(f'/tmp/cli_{name}.txt').read_text()[-260:]}", flush=True)
            break
        mp3 = OUT / f"{name}.mp3"
        sh(f"ffmpeg -v error -i '{found[0]}' -codec:a libmp3lame -b:a 320k '{mp3}' -y")
        ok, reg, stem, acc = male_register(str(mp3))
        dist, comp = style_distance(mp3, stem, acc) if stem and acc else (None, {})
        print(f"  {name} seed {seed}: male={ok} lead={reg.get('lead_hz')} · {comp}", flush=True)
        report.append({"arm": name, "seed": seed, "male": ok,
                       "register": {k: reg.get(k) for k in ("register","f0_hz","lead_hz","bands")},
                       "style": comp})
        (WORK / "probe.json").write_text(json.dumps(report, indent=1))
        if ok:
            ok_take = True
            break
    clock(f"{name} done ({'male' if ok_take else 'NO MALE TAKE'})")
print("PROBE:", json.dumps(report), flush=True)
clock("DONE")
