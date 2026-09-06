# %% [markdown]
# # ArtaQuest — the channel teaser, made from scratch
#
# One public notebook that makes a **22-second teaser** for the ArtaQuest interview show and
# proves each claim on the bytes it ships.
#
# The picture is **Arta**, the platform's mascot, animated from its own rig; the sound is a music
# bed generated here and gated on the one thing a bed must not have. Nothing is downloaded from a
# private place, nothing is hand-edited between cells, and every input is pinned to an immutable
# commit. Anyone can hit *Copy & Edit → Run All* on Kaggle's free GPU and get this.
#
# | Stage | Instrument | What is claimed | How it is checked |
# |---|---|---|---|
# | Picture | `ArtaQuest/artalife` `scenes/teaser`, at a pinned sha | it is **the mascot**, not a lookalike — the rig of the published film | the scene's own selftest runs here: the motion-safety law (640 units a second, 53 a drawing on twos) and Arta's feet on the ground |
# | Sound | ACE-Step 1.5 XL (4.6B), instrumental, best of three | a warm bed with **no voice in it** | the vocal stem is separated and measured against the mix, in dB |
# | Delivery | ffmpeg | 22 s · 1920×1080 · 24 fps · YouTube loudness | recomputed on the shipped file; the run fails if any claim fails |
#
# **Nobody appears in this teaser, and nothing in it is generated imagery.** The show has not
# recorded a couple yet, so a picture of one would be a claim about people who do not exist. What
# the teaser shows instead is the show's own frame with the people not yet in it: Arta standing
# where the host will stand, and two timeline rails — the device every episode carries — drawing
# themselves down either edge. The rails carry no name, no date of birth and no birthplace.
#
# Code is public: the animation at github.com/ArtaQuest/artalife (`scenes/teaser/generate.py`),
# this notebook at github.com/ArtaQuest/artamusic (`stages/artaquest_teaser.py`).

# %%
# ── environment ───────────────────────────────────────────────────────────────────────────
import gc, hashlib, json, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
TMP = Path("/tmp/aq"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
SCENE = TMP / "scene"
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME=str(TMP / "hf"), HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  ACESTEP_GENERATION_TIMEOUT="2400")
T_START = time.time()
SEED = 4242
W, H, FPS = 1920, 1080, 24

PINS = {
    # The animation, at a commit. A tag can move; a sha cannot, and the mascot's motion is the
    # one thing in this file that must be reproducible byte for byte.
    "artalife": "140c5faf4d67dbaba0520211fd24a36c853a1f2a",
    "ace_step_code": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",   # github.com/ACE-Step/ACE-Step-1.5
    "song_model": "acestep-v15-xl-sft",
}

# The bed, described the way ACE-Step's caption field wants it. Caption and metas truncate
# SILENTLY at 256 tokens at this pin, so this stays short on purpose.
MUSIC_CAPTION = ("Warm unhurried instrumental. Solo felt piano playing a simple rising figure, "
                 "soft sustained strings underneath, one low cello note holding. Intimate, "
                 "patient, hopeful. No drums, no percussion, no vocals, no voice. Close-miked, "
                 "gentle tape warmth, plenty of room.")
MUSIC_BPM, MUSIC_KEY, MUSIC_SECONDS = 72, "C major", 30

def sh(c, quiet=False):
    if not quiet:
        print(f"$ {c[:160]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet:
        print(r.stdout[-1400:], flush=True)
    if r.returncode:
        print("ERR:", r.stderr[-1400:], flush=True)
    return r.returncode

def clock(tag):
    print(f"  ⏱ {tag} · t+{(time.time()-T_START)/60:.1f} min", flush=True)

def sha_of(f, n=64):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:n]

print(f"python {sys.version.split()[0]}", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh("df -h /tmp /kaggle/working | tail -3")

# %% [markdown]
# ## The picture: Arta, from its own repository
#
# The scene is fetched at a pinned commit rather than pasted in, for the reason the whole platform
# exists: the mascot is authored in one place and anyone can go and read it. Two files come across
# — the teaser's performance and the rig it is built on — because they are one character and a copy
# of the rig would be a second one.

# %%
for sub in ("teaser", "undefined"):
    d = SCENE / sub; d.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artalife/{PINS['artalife']}/scenes/{sub}/generate.py",
        str(d / "generate.py"))
    print(f"  scenes/{sub}/generate.py  {(d / 'generate.py').stat().st_size:,} bytes "
          f"sha256 {sha_of(d / 'generate.py', 16)}", flush=True)

# The brand's fonts. Google's CSS API answers with TTF urls to an old User-Agent and WOFF2 to a
# modern one, and PIL reads TTF. Each file is checked for the TrueType magic before it is trusted:
# a 404 page saved to disk is still a file, and PIL's complaint would arrive mid-render.
FONTS = SCENE / "teaser" / "fonts"; FONTS.mkdir(exist_ok=True)
_css = urllib.request.urlopen(urllib.request.Request(
    "https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Inter:wght@500;600",
    headers={"User-Agent": "Mozilla/4.0"}), timeout=60).read().decode()
_seen = set()
for _b in _css.split("@font-face")[1:]:
    _fam = re.search(r"font-family:\s*'([^']+)'", _b)
    _w = re.search(r"font-weight:\s*(\d+)", _b)
    _u = re.search(r"url\((https://[^)]+\.ttf)\)", _b)
    if not (_fam and _w and _u):
        continue
    _k = f"{_fam.group(1).replace(' ', '')}-{_w.group(1)}"
    if _k in _seen:
        continue
    _seen.add(_k)
    _p = FONTS / f"{_k}.ttf"
    urllib.request.urlretrieve(_u.group(1), _p)
    assert _p.read_bytes()[:4] in (b"\x00\x01\x00\x00", b"true", b"ttcf"), f"{_k} is not a TTF"
for _need in ("Montserrat-700", "Montserrat-800", "Inter-500", "Inter-600"):
    assert (FONTS / f"{_need}.ttf").exists(), f"{_need} did not arrive — the teaser would set in a fallback"
print(f"fonts: {sorted(p.name for p in FONTS.glob('*.ttf'))}", flush=True)

# %%
# ── the scene's own selftest runs BEFORE anything is rendered ─────────────────────────────
# It is a gate in the repository and it stays a gate here: the motion-safety law (no drawn point
# moves more than 640 units a second, which on twos is 53 a drawing) and Arta standing on the
# ground. It caught a 122 px teleport and a figure planted 39 px underground on its first run.
sys.path.insert(0, str(SCENE / "teaser"))
import generate as SC   # noqa: E402
CHECK = subprocess.run([sys.executable, str(SCENE / "teaser" / "generate.py"), "--check"],
                       text=True, capture_output=True)
print(CHECK.stdout.strip(), flush=True)
assert CHECK.returncode == 0, f"the scene fails its own selftest:\n{CHECK.stdout}\n{CHECK.stderr}"
STEP_PX, STEP_AT = SC.max_step()
BUDGET_PX = (640.0 / SC.FIG_HZ) * (SC.W / 1600.0)
assert (SC.W, SC.H, SC.FPS) == (W, H, FPS), "the scene and this notebook disagree about the frame"

FRAMES = TMP / "frames"; FRAMES.mkdir(exist_ok=True)
N = int(SC.DUR * FPS)
t0 = time.time()
for n in range(N):
    SC.frame(n / FPS).save(FRAMES / f"{n:05d}.png")
print(f"{N} frames in {(time.time()-t0)/60:.1f} min", flush=True)
clock("frames rendered")

# %% [markdown]
# ## The sound
#
# ACE-Step 1.5 XL, instrumental, three takes, and the take that ships is chosen by measurement
# rather than by ear: the vocal stem is separated from each and compared with the mix. A caption
# asking for "no vocals" is a request; the separation is the check.
#
# The model runs in its OWN process. A 4.6B model passing through this notebook's heap and then
# the OOM killer taking the kernel is a failure that leaves neither log nor outputs.

# %%
sh(f"git clone -q https://github.com/ACE-Step/ACE-Step-1.5 {REPO} && "
   f"cd {REPO} && git checkout -q {PINS['ace_step_code']} && pip -q install -e . 2>&1 | tail -3")
sh("pip -q install toml 2>&1 | tail -2")
CKPT.mkdir(parents=True, exist_ok=True)
CFG = {"pins": PINS, "seed": SEED, "tmp": str(TMP), "work": str(WORK), "out": str(OUT),
       "hf_home": os.environ["HF_HOME"],
       "music": {"caption": MUSIC_CAPTION, "bpm": MUSIC_BPM, "key": MUSIC_KEY,
                 "seconds": MUSIC_SECONDS, "seeds": [SEED, SEED + 101, SEED + 202]}}
Path("/tmp/aq_cfg.json").write_text(json.dumps(CFG, indent=2))

STAGE_SRC = r'''#!/usr/bin/env python3
# Written by the notebook, run as its OWN PROCESS.
import gc, json, os, shutil, subprocess, sys, time
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
CFG = json.load(open("/tmp/aq_cfg.json"))
PINS = CFG["pins"]; SEED = CFG["seed"]
TMP = Path(CFG["tmp"]); OUT = Path(CFG["out"])
REPO = TMP / "ACE-Step-1.5"; CKPT = TMP / "checkpoints"
os.environ.update(HF_HOME=CFG["hf_home"], HF_HUB_ENABLE_HF_TRANSFER="1",
                  ACESTEP_CHECKPOINTS_DIR=str(CKPT), ACESTEP_PROJECT_ROOT=str(REPO),
                  ACESTEP_GENERATION_TIMEOUT="2400")
T0 = time.time()

def sh(c, quiet=False):
    if not quiet:
        print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet:
        print(r.stdout[-1200:], flush=True)
    if r.returncode:
        print("ERR:", r.stderr[-1200:], flush=True)
    return r.returncode

def clock(tag):
    print(f"  ⏱ {tag} · t+{(time.time()-T0)/60:.1f} min", flush=True)

import numpy as np
import torch
import toml
np.random.seed(SEED); torch.manual_seed(SEED)
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"[stage music] torch {torch.__version__} · bf16 {BF16}", flush=True)
sys.path.insert(0, str(REPO))

# ACE-Step picks fp16 on any card without bf16 hardware, and fp16 overflows to NaN in the 4.6B
# DiT; bf16 has float32's range and runs (emulated) on Turing. Honour our own env var.
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
    orch.write_text(src.replace(OLD, NEW, 1))

def conf(name, seed, rung, steps):
    return {"project_root": str(REPO), "config_path": rung["model"], "checkpoint_dir": str(CKPT),
            "save_dir": str(TMP / f"m_{name}"), "audio_format": "flac", "device": "cuda",
            "offload_to_cpu": rung["offload_to_cpu"], "offload_dit_to_cpu": rung["offload_dit_to_cpu"],
            "task_type": "text2music",
            "caption": CFG["music"]["caption"], "lyrics": "", "instrumental": True,
            "bpm": CFG["music"]["bpm"], "keyscale": CFG["music"]["key"],
            # constants.py: VALID_TIME_SIGNATURES = [2, 3, 4, 6]; 182 of 199 examples send "4"
            # and none sends "4/4" — the only normaliser lives in the LM path.
            "timesignature": "4",
            "duration": CFG["music"]["seconds"], "inference_steps": steps,
            "guidance_scale": 7.5, "use_adg": False,
            # cli.py hardcodes shift 3.0 (the TURBO schedule) into its defaults and the TOML loader
            # only setattr's keys PRESENT in the file, so an SFT checkpoint is sampled on the wrong
            # noise schedule unless this is stated.
            "shift": 1.0,
            # 199 of 199 official examples ship think=true; with it off the DiT denoises with no
            # plan for where the energy goes, which is a flat wall of sound.
            "thinking": True,
            "seed": seed, "infer_method": "ode",
            "use_cot_metas": False, "use_cot_caption": False,
            "use_cot_lyrics": False, "use_cot_language": False,
            "batch_size": 1, "use_random_seed": False, "seeds": [seed]}

def render(name, conf_dict, dtype):
    c = TMP / f"{name}.toml"; c.write_text(toml.dumps(conf_dict))
    rc = sh(f"cd {REPO} && AQ_FORCE_DTYPE={dtype} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"PYTORCH_ALLOC_CONF=expandable_segments:True ACESTEP_GENERATION_TIMEOUT=2400 "
            f"python cli.py -c {c} --backend pt --log-level INFO > /tmp/cli_{name}.txt 2>&1", quiet=True)
    found = sorted(Path(TMP / f"m_{name}").rglob("*.flac")) + sorted(Path(TMP / f"m_{name}").rglob("*.wav"))
    tail = Path(f"/tmp/cli_{name}.txt").read_text()[-800:] if Path(f"/tmp/cli_{name}.txt").exists() else ""
    return rc, found, tail

def main():
    # A rung is held because it RENDERS, not because it loads: on a T4 the resident rung loads at
    # 12.1 GB and then OOMs inside the CLI on a 1.2 GB attention softmax, returning rc 0 with no
    # file. Each rung is probed with a FULL-LENGTH render at 2 steps — attention buffers scale
    # with the sequence, not the step count.
    ladder = ([("xl-resident", PINS["song_model"], "bfloat16", False, False),
               ("xl-offload", PINS["song_model"], "bfloat16", True, False),
               ("xl-dit-swap", PINS["song_model"], "bfloat16", True, True)] if BF16 else []) + \
             [("sft-fp32", "acestep-v15-sft", "float32", False, False)]
    chosen = None
    for nm, model, dtype, oc, od in ladder:
        t0 = time.time()
        rung = dict(rung=nm, model=model, dtype=dtype, offload_to_cpu=oc, offload_dit_to_cpu=od)
        rc, found, tail = render(f"probe_{nm}", conf(f"probe_{nm}", SEED, rung, 2), dtype)
        if found:
            rung["probe_seconds"] = round(time.time() - t0, 1)
            print(f"RUNG HELD: {nm} — {model} @ {dtype} in {time.time()-t0:.0f}s", flush=True)
            chosen = rung
            break
        print(f"rung {nm}: the full-length probe produced no audio (rc={rc}):\n{tail[-350:]}", flush=True)
    assert chosen, "no rung held — the music model cannot be run on this card"
    clock("model held")

    takes = []
    for i, seed in enumerate(CFG["music"]["seeds"]):
        nm = f"take{i}"
        rc, found, tail = render(nm, conf(nm, seed, chosen, 60), chosen["dtype"])
        if not found:
            print(f"  take {seed}: no audio (rc={rc})", flush=True)
            continue
        wav = TMP / f"{nm}.wav"
        sh(f"ffmpeg -v error -i '{found[0]}' -ac 2 -ar 44100 '{wav}' -y", quiet=True)
        takes.append({"seed": seed, "wav": str(wav)})
        print(f"  take {seed}: rendered", flush=True)
    assert takes, "no music take rendered"
    clock("takes rendered")

    # THE GATE MEASURES THE ONE THING A BED MUST NOT HAVE: a voice. The separator is torchaudio's
    # own HDemucs, NOT `pip install demucs` — that package resolves its own torch, and a music gate
    # that quietly changes torch is a gate that breaks the run it is meant to protect. A separator
    # that cannot load must FAIL the gate, never pass it: an unmeasured take is not a clean one.
    import torchaudio
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    sep = bundle.get_model().to("cuda").eval()
    vi = list(bundle.sources).index("vocals")
    print(f"  separator: HDemucs, sources {list(bundle.sources)}, {bundle.sample_rate} Hz", flush=True)

    def vocal_db(path):
        wav, sr = torchaudio.load(path)
        if sr != bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        m, s = wav.mean(), wav.std().clamp_min(1e-8)
        with torch.inference_mode():
            stems = sep(((wav - m) / s).unsqueeze(0).to("cuda"))[0].cpu() * s + m
        mix = float(wav.pow(2).mean().sqrt())
        voc = float(stems[vi].pow(2).mean().sqrt())
        return 20.0 * float(np.log10(max(voc, 1e-9) / max(mix, 1e-9))), mix

    scored = []
    for t in takes:
        db, mix = vocal_db(t["wav"])
        scored.append(dict(t, vocal_db=round(db, 2), mix_rms=round(mix, 5)))
        print(f"  take {t['seed']}: vocal stem {db:+.1f} dB under the mix", flush=True)
    # The quietest voice wins; a tie goes to the louder mix, which needs less make-up gain.
    scored.sort(key=lambda x: (x["vocal_db"], -x["mix_rms"]))
    best = scored[0]
    shutil.copy(best["wav"], OUT / "teaser_music.wav")
    print(f"  chosen: seed {best['seed']} at {best['vocal_db']:+.1f} dB", flush=True)
    return {"rung": chosen, "takes": scored, "chosen_seed": best["seed"], "vocal_db": best["vocal_db"]}

RESULT = {"stage": "music"}
try:
    RESULT["data"] = main()
    RESULT["ok"] = True
except Exception as e:
    import traceback
    RESULT["ok"] = False
    RESULT["error"] = f"{type(e).__name__}: {e}"
    RESULT["trace"] = traceback.format_exc()[-2500:]
    print(RESULT["trace"], flush=True)
Path("/tmp/aq_music.json").write_text(json.dumps(RESULT, indent=2))
print(f"[stage music] done ok={RESULT['ok']} · t+{(time.time()-T0)/60:.1f} min", flush=True)
'''
(TMP / "stage.py").write_text(STAGE_SRC)

p = Path("/tmp/aq_music.json")
if p.exists():
    p.unlink()
rc = subprocess.run([sys.executable, str(TMP / "stage.py")], timeout=120 * 60).returncode
gc.collect()
assert p.exists(), f"the music stage left no verdict (rc={rc}) — the process died"
_m = json.loads(p.read_text())
assert _m["ok"], f"music stage: {_m.get('error')}\n{_m.get('trace','')}"
MUSIC = _m["data"]
(WORK / "music.json").write_text(json.dumps(MUSIC, indent=2))
clock("music done")

# %% [markdown]
# ## The cut
#
# The frames are already the finished picture, so the cut is only: encode, put the bed under it,
# and set the whole thing to the loudness YouTube normalises to. A bed delivered hotter than
# −14 LUFS is turned down on playback and loses its dynamics for nothing.

# %%
DUR = SC.DUR
sh(f"ffmpeg -v error -framerate {FPS} -i '{FRAMES}/%05d.png' -c:v libx264 -crf 17 -preset slow "
   f"-pix_fmt yuv420p -movflags +faststart '{TMP}/silent.mp4' -y")
sh(f"ffmpeg -v error -i '{OUT}/teaser_music.wav' -af "
   f"\"atrim=0:{DUR},afade=t=in:st=0:d=1.2,afade=t=out:st={DUR-2.5}:d=2.5,"
   f"loudnorm=I=-14:TP=-1.5:LRA=11\" -ar 48000 -ac 2 '{TMP}/bed.wav' -y")
sh(f"ffmpeg -v error -i '{TMP}/silent.mp4' -i '{TMP}/bed.wav' -map 0:v -map 1:a -c:v copy "
   f"-c:a aac -b:a 192k -shortest -movflags +faststart '{OUT}/ArtaQuest_teaser.mp4' -y")
shutil.copy(FRAMES / f"{int(FPS * 12):05d}.png", OUT / "ArtaQuest_teaser_poster.png")
clock("teaser assembled")

# %% [markdown]
# ## Verify — every claim recomputed on the file that ships

# %%
FINAL = OUT / "ArtaQuest_teaser.mp4"
_p = subprocess.run(
    f"ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,channels "
    f"-show_entries format=duration -of json '{FINAL}'", shell=True, text=True, capture_output=True)
info = json.loads(_p.stdout)
vid = next(s for s in info["streams"] if s["codec_type"] == "video")
aud = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
dur = float(info["format"]["duration"])
lr = subprocess.run(f"ffmpeg -v info -i '{FINAL}' -af ebur128=peak=true -f null - 2>&1 | tail -25",
                    shell=True, text=True, capture_output=True).stdout
def grab(tag):
    m = re.search(rf"{tag}:\s*(-?\d+\.?\d*)", lr)
    return float(m.group(1)) if m else None
LUFS, TP, LRA = grab("I"), grab("Peak"), grab("LRA")

problems = []
if abs(dur - DUR) > 0.35:
    problems.append(f"duration {dur:.2f}s, expected {DUR:.2f}s")
if (int(vid["width"]), int(vid["height"])) != (W, H):
    problems.append(f"frame {vid['width']}x{vid['height']}, expected {W}x{H}")
if eval(vid["r_frame_rate"]) != FPS:
    problems.append(f"fps {vid['r_frame_rate']}, expected {FPS}")
if aud is None or int(aud["channels"]) != 2:
    problems.append("the bed is not stereo")
if LUFS is not None and not (-16.5 <= LUFS <= -11.5):
    problems.append(f"loudness {LUFS} LUFS is outside the -14 target band")
if TP is not None and TP > -0.5:
    problems.append(f"true peak {TP} dBTP is too hot")
if MUSIC["vocal_db"] > -12:
    problems.append(f"the bed carries a voice at {MUSIC['vocal_db']} dB under the mix")
if STEP_PX > BUDGET_PX * 1.02:
    problems.append(f"motion safety: {STEP_PX:.1f} px a drawing against a {BUDGET_PX:.0f} px budget")

MANIFEST = {
    "made": "ArtaQuest channel teaser — \"Two lives\"",
    "seconds": round(dur, 2), "fps": FPS, "size": [W, H], "frames": N,
    "loudness": {"lufs": LUFS, "true_peak_dbtp": TP, "lra": LRA},
    "animation": {"repo": "ArtaQuest/artalife", "scene": "scenes/teaser",
                  "commit": PINS["artalife"], "figure_hz": SC.FIG_HZ,
                  "largest_step_px": round(STEP_PX, 2), "budget_px": round(BUDGET_PX, 1),
                  "at_seconds": round(STEP_AT, 2)},
    "music": MUSIC, "pins": PINS, "seed": SEED,
    "nobody_is_in_it": "no photographic or generated imagery of people; the rails carry no name, "
                       "date of birth or birthplace",
    "sha256": {p.name: sha_of(p) for p in sorted(OUT.glob("*")) if p.is_file()},
    "minutes_total": round((time.time() - T_START) / 60, 1),
    "problems": problems,
}
(WORK / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
print(json.dumps({k: v for k, v in MANIFEST.items() if k != "sha256"}, indent=2)[:3000], flush=True)
sh(f"ls -la '{OUT}'")
shutil.rmtree(FRAMES, ignore_errors=True)      # the frames are the video now; a kernel's working
                                               # directory IS its output and ships on every download
assert not problems, f"the teaser does not verify: {problems}"
print(f"\n✓ ArtaQuest_teaser.mp4 — {dur:.1f}s, {LUFS} LUFS, no voice in the bed, "
      f"motion {STEP_PX:.0f}/{BUDGET_PX:.0f} px · {(time.time()-T_START)/60:.0f} min", flush=True)
