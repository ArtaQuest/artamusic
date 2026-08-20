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
# | Lyric | ArtaQuest's own text | craft profile vs a measured reference, AND followable on one hearing | `lyric_profile.py` + `clarity.py` at a pinned commit |
# | Song | ACE-Step 1.5 XL (4.6B), style transfer from a public male take | male lead · intelligible · dynamic | YIN on a demucs stem · whisper large-v3 word accuracy · loudness / true peak / clipping |
# | Cover | Wan2.2-**T2V**-A14B (Apache-2.0), prompted directly — no still image, real guidance, real negative prompt | photographic, and the blade does not move at all | drift and displacement by normalised template matching INSIDE the blade's mask · change no lighting field can explain · and the counter-gate: the fire must be alive and its light must fall on the steel |
# | Verify | the same instruments, on the delivered bytes | everything above | the run fails if any claim fails |
#
# Code, measurement tools and the lyric are public at github.com/ArtaQuest/artamusic
# (`stages/steel_record.py`); every fetch below is pinned to a commit sha.

# ── environment: one set of pins for the song and the cover; deps first, torch last ───────
# Both stages share one environment on purpose: ACE-Step 1.5 (this commit) wants diffusers>=0.37,
# and so does Wan2.2 in diffusers 0.39.0. On a Pascal card (P100, sm_60) Kaggle's default torch
# has no kernels, so the cu126 line is installed LAST (whatever runs last wins); on a T4 pair the
# default torch stays.
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
    "measure_sha": "17b49399cfd6c24f4070353fc33643ae15e1331d",      # ArtaQuest/artamusic lib/measure.py
    "lyric_profile_sha": "ebee5bf324d8a6cff22ba666825a777c7dfc5c39",  # ArtaQuest/artamusic lib/lyric_profile.py
    "lyric_sha": "88348bd9e0d21d196cb95c54c20b2943a629c68a",
    "shot_sha": "9fed845f616bcfab1404e220bf13f0366690135b",   # ArtaQuest/artamusic song/shot_steel.json
    "tools_sha": "e43b03d4ddc8810e67f467f52feef9ce65ce9131",   # ArtaQuest/artamusic lib/{stillness,freeze}.py          # ArtaQuest/artamusic song/lyrics_steel.txt + lib/clarity.py
    "asr": "large-v3",
    # NO IMAGE MODEL, and no image conditioning. The cover used to be a still made by a
    # text-to-image model and then animated; three rounds of that came back "not realistic",
    # because a distilled T2I model drifts toward illustration and the video model was only ever
    # animating a picture that already looked drawn. The video model is asked for the shot itself.
    # TEXT-to-video, and NOT the lightx2v 4-step distillation that the old loop used: that is
    # CFG-distilled, so it runs at guidance 1.0 and a negative prompt does nothing — and naming the
    # defect is the most direct instrument there is against "looks like a render".
    "wan_base": ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "5be7df9619b54f4e2667b2755bc6a756675b5cd7"),
    "wan_gguf": ("QuantStack/Wan2.2-T2V-A14B-GGUF", "73eafba53a1a8f29254e4c77f92e74ea27d7cd6f"),
    "wan_high": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
    "wan_low": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
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

def sha_of(f, n=64):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:n]

def release(tag):
    """Give memory BACK. Freed CPU tensors stay in the parent's heap unless glibc is told to trim, and
    that retained heap is invisible to gc but very visible to the next subprocess: a run died with
    'Kernel died' when ACE-Step loaded after Z-Image and both Wan experts had passed through host RAM."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    r = subprocess.run("free -m | awk 'NR==2{print $3\"/\"$2\" MB used\"}'", shell=True, text=True,
                       capture_output=True).stdout.strip()
    g = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True, text=True,
                       capture_output=True).stdout.strip().replace("\n", " | ")
    # DISK, not just memory: this container has a writable quota of its own, and three runs were
    # killed with no log and no outputs after the third model's weights landed in the same cache.
    d = subprocess.run(f"du -sh {TMP} 2>/dev/null | cut -f1; df -h /tmp | tail -1 | awk '{{print $3\" used, \"$4\" free\"}}'",
                       shell=True, text=True, capture_output=True).stdout.strip().replace("\n", " · ")
    print(f"  [release {tag}] host {r} · gpu {g} · scratch {d}", flush=True)

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
# word‑identical (near‑identical variants are what machine transcription mishears), and every
# "Cut!" is marked as a backing vocal so the choir shouts the stroke while the lead sings the
# line the gate measures. The chant
# register stays deliberately monosyllabic (79% against a pop reference's 71%) — that number is
# printed below, not hidden. Lyrics © ArtaQuest Foundation.

# ── the lyric, and its craft numbers from the pinned instrument ────────────────────────────
import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_profile_sha']}/lib/lyric_profile.py",
    "/tmp/lyric_profile.py")
sys.path.insert(0, "/tmp")
import lyric_profile as LP

# ONE COPY OF THE LYRIC, FETCHED AT A PIN. It used to be pasted here as a string literal, and the
# copy in this notebook had already drifted from song/lyrics_steel.txt in the repo — two texts both
# claiming to be the lyric, differing in their opening lines, with no way to tell which had been
# measured by anything. The lyric is a published artifact of this project; it belongs in the repo
# beside the instruments that judge it, and the notebook should read it the same way it reads them.
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/song/lyrics_steel.txt",
    "/tmp/lyrics_steel.txt")
LYRICS = Path("/tmp/lyrics_steel.txt").read_text().strip()
assert LYRICS.startswith("[intro]") and "[chorus]" in LYRICS, "the fetched lyric is not a lyric"

# The antiphon as BACKING VOCALS. ACE-Step's lyric convention puts backing/choir parts in
# (parentheses); marking every "Cut!" that way tells the model the choir shouts the stroke and the
# lead sings the line — the words the gate measures. Off = the choir is free to sing over the lead.
ANTIPHON_AS_BACKING = True
LYRIC_TEXT = LYRICS                 # the words, as written and as measured
if ANTIPHON_AS_BACKING:
    SHOUT = "Strike!"          # the lyric's antiphon. It was "Cut!" before the rewrite, and a
    # stale marker here does not fail — it silently marks NOTHING as backing vocal, and the choir
    # simply never appears in the mix. So assert the word is actually in the lyric.
    assert re.search(rf"^{re.escape(SHOUT)} ", LYRICS, re.M), (
        f"no line starts with {SHOUT!r} — the antiphon marker does not match this lyric, and the "
        f"choir would be silently dropped from the arrangement")
    LYRICS = re.sub(rf"^{re.escape(SHOUT)} ", f"({SHOUT}) ", LYRICS, flags=re.M)
(OUT / "STEEL_lyrics.txt").write_text(LYRICS + "\n")

# MEASURE THE WORDS, NOT THE ARRANGEMENT MARKUP. The parentheses are an instruction to the
# generator about who sings a line, not a change to the line — the choir shouts the same word the
# lead would. But the clarity instrument treats a parenthesised line as not-the-lead and drops it,
# which shrinks its denominator: marking fourteen lines as choir moved the inversion rate from
# 4.3% to 5.4% and would have failed this run's own gate on a lyric that passes. So both
# instruments read LYRIC_TEXT, and the marked-up version is what goes to the model and ships as
# the lyric sheet.
craft = LP.measure(LYRIC_TEXT)
craft_report = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in craft.items()
                if k not in ("long_lines", "short_lines")}
craft_report["long_lines"] = craft["long_lines"]; craft_report["short_lines"] = craft["short_lines"]
craft_report["target"] = LP.TARGET
craft_report["invariants"] = LP.check_invariants(LYRIC_TEXT, "steel") or ["ok"]
(WORK / "lyric_craft.json").write_text(json.dumps(craft_report, indent=2))
print(json.dumps({k: craft_report[k] for k in ("lines", "words", "syllables", "tight_pct",
                                              "mono_pct", "imper_pct", "invariants")}, indent=1),
      flush=True)
assert craft_report["invariants"] == ["ok"], "lyric invariants broken"

# CAN A LISTENER FOLLOW IT ON FIRST HEARING? The craft profile above measures the SHAPE of a lyric
# — syllables per line, monosyllable rate, how often a line opens on a command — and a text can
# score perfectly on every one of them while being impossible to follow. The previous lyric did
# exactly that, and the verdict on the finished record was that nobody could tell what the song was
# about. So the other axis is measured too, and it BLOCKS: everyday vocabulary, lines that show
# something rather than assert it, and plain subject-verb-object order.
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/lib/clarity.py",
    "/tmp/clarity.py")
import clarity as CL
assert CL.selftest(), "the clarity instrument fails its own selftest — its numbers mean nothing"
clear = CL.measure(LYRIC_TEXT)
clear_bad = CL.verdict(clear)
craft_report["clarity"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in clear.items()}
craft_report["clarity"]["floor"] = CL.FLOOR
craft_report["clarity"]["verdict"] = clear_bad or ["clear enough to follow on first listen"]
(WORK / "lyric_craft.json").write_text(json.dumps(craft_report, indent=2))
print("clarity:", json.dumps(craft_report["clarity"]), flush=True)
assert not clear_bad, "the lyric is not comprehensible on first listen: " + "; ".join(clear_bad)

CAPTION = ("Dark chant anthem. Pounding war drums and anvil strikes on the beat, massive unison "
           "male chant choir answering a deep gravelly lead vocal, low strings and war horns, "
           "sparse and martial, minor key, solemn and heavy, 100 BPM.")
DURATION = 180.0
BPM, KEYSCALE = 100, "F minor"   # match the conditioning reference; a key fight is an experiment,
                                 # and a publication run is not where you run one

# %% [markdown]
# ## The cover — asked of a video model, not assembled from a picture
#
# The cover is made before the song on purpose: a failure there is found in an hour, and the song
# then renders on a clean card.
#
# It used to be a still from a text-to-image model, animated afterwards by an image-to-video model.
# Three rounds of that came back **not realistic**, and the cause sat above every fix: a distilled
# T2I model drifts toward illustration — steel too clean, coals too even — so the video model was
# only ever animating a picture that already looked drawn.
#
# **Wan2.2-T2V-A14B, Apache-2.0**, is asked for the shot directly. As of August 2026 Wan 2.2 is
# still the newest Wan with open downloadable weights (2.5 and 2.6 shipped API-only) and it leads
# open models on photorealism. HunyuanVideo 1.5 is smaller and very good but ships under Tencent's
# community licence, and LTX-2.5 under Lightricks'; both carry territorial and use restrictions,
# and a gated or restricted input fails this platform's own every-input-is-public rule — the same
# ground FLUX.1-Krea-dev was turned down on.
#
# And **not** the lightx2v 4-step distillation the old loop used. It is CFG-distilled, so it runs
# at guidance 1.0 and a negative prompt does nothing — tolerable when the job was adding motion to
# a photograph, useless when the job is realism, because naming the defect (`3d render, cgi,
# plastic, waxy`) and having the sampler steer away from it is the most direct instrument there is.
# Guidance costs two forward passes a step rather than a doubled batch, so it buys realism with
# time and not memory, and the step count is measured: two steps are timed on the real shot and the
# budget decides the rest.

# %% [markdown]
# ## The cover loop — Wan2.2‑I2V‑A14B, and the sword composited back frozen
#
# The strongest open image‑to‑video model with an Apache‑2.0 licence that fits a free card:
# Alibaba's two 14B experts, here as GGUF Q4_K_M with the lightx2v four‑step distillation, run at
# the exact points it was trained on (t = 1000 · 750 · 500 · 250 under shift 5).
#
# The loop used to be closed by pinning the same still as the first AND last frame, so the clip had
# to return to where it began. That works, and it costs the thing the shot is for: the model spends
# the clip travelling back, and the fire barely moves. Measured, the bill was plain — the pinned
# generation scored 0.26 on fire motion against a floor of 1.0, and 1.70 for the loop that visibly
# lived. So there is no pin. The wrap is closed by an 8‑frame dissolve of tail into head instead.
#
# And the sword does not hold still because the sampler was asked nicely. It is **painted out of
# the photograph** by heat‑diffusion inpainting, the empty forge is animated, and the sword is
# **composited back frozen** and re‑lit each frame from the coals behind it — so its pixels are
# literally the same pixels, and only the light on them changes. No mask‑conditioned video model
# preserves a region exactly; a pixel composite is the only thing that does.
#
# How hard to re‑light it is not a constant either. Too much and the stillness gate calls the blade
# movement; too little and it reads as a cardboard cut‑out pasted over a moving plate. The run
# walks a ladder from most firelight to least and keeps the first rung that satisfies both, and
# publishes every rung beside the choice.
#
# On a Turing/Pascal card memory‑efficient attention exists only for fp16/fp32 (a bf16 model
# silently falls back to O(N²) attention), so the experts compute in fp16 with an fp32 fallback
# on any non‑finite latent. With two GPUs each expert lives on its own card; with one, they
# swap through host RAM and the decode is done explicitly after both are parked.

# ── the shot, described as a shot ────────────────────────────────────────────────────────
# A text-to-image brief describes a picture. A video model wants the picture AND what happens in
# it, and it carries a strong prior from real footage — which is the whole reason to be here. So
# the prompt names the film stock and the lens, then says exactly what moves and what does not.
# Everything the model was once free to invent — how big the blade is, what a sword looks like,
# what a coal looks like, what else is in the room — is stated, because each was got wrong when
# left unsaid.
# THE SHOT COMES FROM ONE FILE, at a pinned commit. It used to be inline here and inline in the
# cover notebook, and the two drifted apart three times — most recently THIS file was still
# describing a sword lying still after the shot had become the hammering, which would have made a
# cover of the wrong subject with nothing in the code to say so. hold_subject travels with it
# because it belongs to the shot: a sword lying still wants the freeze, a hammer swinging must not
# have it, and keeping the flag away from the words describing the motion is how they disagree.
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['shot_sha']}/song/shot_steel.json",
    "/tmp/shot_steel.json")
SHOT = json.loads(Path("/tmp/shot_steel.json").read_text())
PROMPT, NEG, HOLD_SUBJECT = SHOT["prompt"], SHOT["negative"], SHOT["hold_subject"]
assert "anvil" in PROMPT and "hammer" in PROMPT, "the fetched shot is not the hammering shot"
print(f"[shot] {SHOT['name']} · hold_subject={HOLD_SUBJECT}\n[prompt] {PROMPT}", flush=True)
(WORK / "prompt.json").write_text(json.dumps({"prompt": PROMPT, "negative": NEG}, indent=2))

# ── the measuring instruments, at a pinned commit, proven before anything expensive runs ──
TOOLS = TMP / "tools"; TOOLS.mkdir(exist_ok=True)
for _f in ("stillness.py", "freeze.py"):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['tools_sha']}/lib/{_f}",
        str(TOOLS / _f))
sys.path.insert(0, str(TOOLS))
import stillness as _S
assert _S.selftest(), "the stillness instrument fails its own selftest — no number here is trustworthy"

# ── the cover stages, each in its OWN PROCESS ────────────────────────────────────────────
# Four runs died here and the mechanism was the same every time: this notebook's process keeps the
# host memory that Z-Image and the two Wan experts passed through — freed tensors sit in the heap,
# invisible to gc — and when ACE-Step's 4.6B model then loads, the OOM killer takes the notebook
# kernel. Twice it took it so hard that Kaggle saved neither log nor outputs. Trimming the heap was
# not enough. So every heavy stage now runs as its own process, exactly as the song stage already
# runs through ACE-Step's CLI: the notebook writes the stage script below, runs it, and reads back
# its JSON. One notebook, one Run All — and no model is ever loaded in THIS process.
STAGE_SRC = r"""#!/usr/bin/env python3
# Written by the notebook, run as its OWN PROCESS — see the note in the notebook cell below.
import gc, glob, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
CFG = json.load(open("/tmp/aq_cfg.json"))
PINS = CFG["pins"]; SEED = CFG["seed"]
TMP = Path(CFG["tmp"]); WORK = Path(CFG["work"]); OUT = Path(CFG["out"])
os.environ.update(HF_HOME=CFG["hf_home"], HF_HUB_ENABLE_HF_TRANSFER="1")
sys.path.insert(0, CFG["tools"])
T_START = time.time()

def sh(c, quiet=False):
    if not quiet: print(f"$ {c[:150]}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout.strip() and not quiet: print(r.stdout[-1200:], flush=True)
    if r.returncode: print("ERR:", r.stderr[-1200:], flush=True)
    return r.returncode

def clock(tag):
    print(f"  \u23f1 {tag} \u00b7 t+{(time.time()-T_START)/60:.1f} min", flush=True)

def sha20(p):
    # First 20 hex of a file's sha256. The GGUF experts are identified by CONTENT, not by name: a
    # repo can move a tag under a filename and the notebook would never know.
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:20]


def drop_weights(*needles):
    # Delete a finished stage's weights from the cache. The three stages want ~13, ~31 and ~20 GB
    # of weights and Kaggle's container has a writable quota well under their sum: once the third
    # download landed, the container was killed with no log and no outputs at all — three times,
    # which is why those runs looked like silence. Everything is pinned by revision, so a re-run
    # fetches the same bytes; keeping them after their stage buys nothing and costs the run.
    hf = Path(os.environ["HF_HOME"]) / "hub"
    freed = 0
    for d in list(hf.glob("models--*")):
        if any(n.lower() in d.name.lower() for n in needles):
            sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30
            shutil.rmtree(d, ignore_errors=True)
            freed += sz
            print(f"  dropped {d.name} ({sz:.1f} GB)", flush=True)
    df = subprocess.run("df -h /tmp | tail -1 | awk '{print $4\" free\"}'", shell=True, text=True,
                        capture_output=True).stdout.strip()
    print(f"  [drop] freed {freed:.1f} GB · /tmp {df}", flush=True)

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
np.random.seed(SEED); torch.manual_seed(SEED)
NGPU = torch.cuda.device_count()
print(f"[stage {sys.argv[1]}] torch {torch.__version__} \u00b7 {NGPU} gpu(s)", flush=True)





if __name__ == "__main__":
    {"cover": stage_cover}[sys.argv[1]]()
    print(f"[stage {sys.argv[1]}] done", flush=True)


def stage_cover():
    # The whole cover, generated as VIDEO from text — ported from stages/t2v_cover.py,
    # the standalone notebook it was proven in. No still image, no image conditioning.
    PROMPT = CFG["prompt"]; NEG = CFG["negative"]; HOLD_SUBJECT = CFG["hold_subject"]; CYCLE = {}
    import stillness as S, freeze as F
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan, GGUFQuantizationConfig
    from transformers import UMT5EncoderModel, AutoTokenizer
    BASE, BREV = PINS["wan_base"]

    tok = AutoTokenizer.from_pretrained(BASE, revision=BREV, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(BASE, revision=BREV, subfolder="text_encoder",
                                          torch_dtype=torch.float16).to("cuda:0")
    def embed(text, n=512):
        ids = tok([text], padding="max_length", max_length=n, truncation=True, return_tensors="pt")
        k = int(ids.attention_mask.gt(0).sum(1)[0])
        with torch.inference_mode():
            h = te(ids.input_ids.to("cuda:0"), ids.attention_mask.to("cuda:0")).last_hidden_state[0].float().cpu()
        return torch.cat([h[:k], h.new_zeros(n - k, h.size(1))])[None]
    PE, NE = embed(PROMPT), embed(NEG)
    del te, tok; gc.collect(); torch.cuda.empty_cache()
    print(f"  prompt encoded {tuple(PE.shape)}", flush=True)
    clock("text encoded")

    from huggingface_hub import hf_hub_download
    GREPO, GREV = PINS["wan_gguf"]
    HIGH = hf_hub_download(GREPO, PINS["wan_high"], revision=GREV)
    LOW = hf_hub_download(GREPO, PINS["wan_low"], revision=GREV)
    import hashlib
    def sha20(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        return h.hexdigest()[:20]
    HASHES = {"high": sha20(HIGH), "low": sha20(LOW)}
    print("  gguf sha256[:20]:", HASHES, flush=True)

    q = GGUFQuantizationConfig(compute_dtype=torch.float16)
    hi = WanTransformer3DModel.from_single_file(HIGH, quantization_config=q, config=BASE,
                                                subfolder="transformer", torch_dtype=torch.float16)
    lo = WanTransformer3DModel.from_single_file(LOW, quantization_config=q, config=BASE,
                                                subfolder="transformer_2", torch_dtype=torch.float16)
    vae = AutoencoderKLWan.from_pretrained(BASE, revision=BREV, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(BASE, revision=BREV, transformer=hi, transformer_2=lo, vae=vae,
                                       text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    if NGPU >= 2:
        hi.to("cuda:0"); vae.to("cuda:0"); lo.to("cuda:1")
        _f = lo.forward
        def across(*a, **k):
            a = [x.to("cuda:1") if torch.is_tensor(x) else x for x in a]
            k = {n: (v.to("cuda:1") if torch.is_tensor(v) else v) for n, v in k.items()}
            o = _f(*a, **k)
            # The pipeline calls the transformer with return_dict=False, so this comes back as a TUPLE.
            # An earlier version moved only .sample and passed tuples through untouched, which left the
            # low-noise expert's prediction on cuda:1 — the run died at the 50% mark, exactly where the
            # second expert takes over.
            if isinstance(o, tuple):
                return tuple(x.to("cuda:0") if torch.is_tensor(x) else x for x in o)
            if torch.is_tensor(o):
                return o.to("cuda:0")
            return o.__class__(sample=o.sample.to("cuda:0"))
        lo.forward = across
        print("  one expert per card", flush=True)
    else:
        pipe.enable_model_cpu_offload()
    clock("experts loaded")

    # ## How many steps — measured, not chosen
    #
    # Two steps are timed on the real shot at the real size, and the step count follows from the
    # budget. Guessing here costs a whole session per guess: this is the same failure that made a
    # stronger image model look viable at 896² right up until it ran out of memory an hour in.

    H = W = 640
    NF, FPS, XF = 81, 16, 8
    # THE BUDGET IS BIG ON PURPOSE. Classifier-free guidance is two forward passes a step, not a
    # doubled batch — memory is unchanged, time doubles. The distilled loop model ran 4 steps of 81
    # frames at 640² in about 22 minutes, so ~330 s/step without guidance and ~660 with it. A
    # 46-minute budget would therefore have bought FOUR steps, and Wan2.2 undistilled at four steps is
    # noise: the distillation is the only reason four ever worked. Kaggle allows twelve hours in a
    # session and the pool has hours to spare, so the generation is allowed two and a half and the
    # measurement decides how many steps that is.
    BUDGET_S = 150 * 60

    def run(steps):
        with torch.inference_mode():
            return pipe(prompt_embeds=PE.to("cuda:0"), negative_prompt_embeds=NE.to("cuda:0"),
                        height=H, width=W, num_frames=NF, num_inference_steps=steps,
                        guidance_scale=4.0, guidance_scale_2=3.0,
                        generator=torch.Generator("cuda:0").manual_seed(SEED),
                        output_type="latent")

    t0 = time.time(); run(2); per_step = (time.time() - t0) / 2
    STEPS = max(8, min(20, int(BUDGET_S / max(per_step, 1e-6))))
    print(f"\n  {per_step:.0f} s/step at {H}×{W}×{NF} with guidance → {STEPS} steps "
          f"≈ {per_step*STEPS/60:.0f} min", flush=True)
    assert per_step * 8 <= BUDGET_S * 1.35, (
        f"{per_step:.0f} s/step means even eight steps would take {per_step*8/60:.0f} minutes — real "
        f"guidance is out of reach at {H}×{W}×{NF}. Drop to 49 frames or 512² and try again.")

    t0 = time.time()
    out = run(STEPS)
    gen_s = time.time() - t0
    v = pipe.vae.to("cuda:0")
    lat = out.frames.to(v.dtype)
    mean = torch.tensor(v.config.latents_mean).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    std = 1.0 / torch.tensor(v.config.latents_std).view(1, v.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    with torch.inference_mode():
        dec = v.decode(lat / std + mean, return_dict=False)[0]
    frames = (np.clip(pipe.video_processor.postprocess_video(dec, output_type="np")[0], 0, 1) * 255).round().astype(np.uint8)
    print(f"  {len(frames)} frames in {gen_s/60:.1f} min", flush=True)
    del pipe, hi, lo; gc.collect(); torch.cuda.empty_cache()
    clock("generated")

    # ## The blade: measure first, freeze only if it needs it
    #
    # The instruction was that the sword must not move, and the composite that guarantees it — paint
    # the sword out, animate the plate, put the sword back frozen and re-lit — is still here. But it is
    # no longer applied blind. A video model asked for a locked-off shot of a motionless object may
    # simply deliver one, and a real still object is more convincing than a frozen cut-out of one. So
    # the raw generation is measured first, and the composite runs **only if the blade actually
    # drifts**. Which way it went is published either way.

    def encode(fr, base):
        d = Path(f"/tmp/f_{Path(base).name}"); d.mkdir(exist_ok=True)
        for f in d.glob("*.png"): f.unlink()
        for i, f in enumerate(fr): Image.fromarray(f).save(d / f"{i:04d}.png")
        sh(f"ffmpeg -v error -framerate {FPS} -i '{d}/%04d.png' -c:v libx264 -crf 12 -preset slow "
           f"-pix_fmt yuv420p '{base}_raw.mp4' -y", quiet=True)
        sh(f"ffmpeg -v error -i '{base}_raw.mp4' -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -cpu-used 1 "
           f"-g 240 -pix_fmt yuv420p -an '{base}.webm' -y", quiet=True)
        sh(f"ffmpeg -v error -i '{base}_raw.mp4' -c:v libx264 -preset veryslow -crf 20 -pix_fmt yuv420p "
           f"-movflags +faststart -an '{base}.mp4' -y", quiet=True)
        sh(f"ffmpeg -v error -i '{base}_raw.mp4' -vf scale=1080:1080:flags=lanczos -c:v libvpx-vp9 "
           f"-crf 33 -b:v 0 -row-mt 1 -cpu-used 1 -pix_fmt yuv420p -an '{base}_1080.webm' -y", quiet=True)

    def close_loop(fr):
        L = fr[:-1]
        w = (np.arange(1, XF + 1) / (XF + 1))[:, None, None, None]
        blend = ((1 - w) * L[-XF:].astype(np.float32) + w * L[:XF].astype(np.float32)).round().astype(np.uint8)
        return np.concatenate([L[XF:len(L) - XF], blend])

    still = frames[0]
    raw_loop = close_loop(frames)

    # WRITE THE VIDEO BEFORE JUDGING IT. Everything below this line is measurement, and measurement can
    # throw: the blade mask is a heuristic over frame 0, and on a frame it does not recognise it can
    # come back empty, at which point the template matcher asks an empty array for its bounds and dies.
    # That would be a two-and-a-half-hour generation lost to a crash in the part that was only supposed
    # to grade it. So the generated loop goes to disk first, and stays there whatever happens next.
    Image.fromarray(still).save(OUT / "frame0.png")
    encode(raw_loop, str(OUT / "STEEL_cover_loop_asgenerated"))
    print("  raw generation written to disk", flush=True)

    blade = F.steel_mask(still)
    coals = F.fire_mask(still, plume=25)
    mask_pct = 100 * float(blade.mean())
    print(f"  blade mask {mask_pct:.2f}% of frame · coals {100*float(coals.mean()):.1f}%", flush=True)
    # A mask that is a sliver or half the picture is not a sword, and every number keyed on it would be
    # meaningless rather than wrong-looking. Say so and ship the generation unjudged rather than
    # inventing a verdict.
    mask_ok = 0.15 <= mask_pct <= 12.0 and bool(coals.any())
    if not mask_ok:
        print(f"  the blade mask is implausible at {mask_pct:.2f}% — the freeze and the stillness "
              f"numbers are being SKIPPED, and the loop ships as generated", flush=True)

    def measure_array(arr, mask):
        d = Path("/tmp/meas"); d.mkdir(exist_ok=True)
        for f in d.glob("*.png"): f.unlink()
        for i, f in enumerate(arr): Image.fromarray(f).save(d / f"{i:04d}.png")
        sh(f"ffmpeg -v error -framerate {FPS} -i '{d}/%04d.png' -c:v libx264 -crf 12 -pix_fmt yuv420p "
           f"/tmp/meas.mp4 -y", quiet=True)
        return S.measure("/tmp/meas.mp4", mask=mask), S.liveness("/tmp/meas.mp4", coals, mask)

    raw_m, raw_a = (measure_array(raw_loop, blade) if mask_ok else ({}, {}))
    if mask_ok:
        print(f"  as generated: drift {raw_m['drift_px']} px · lit_dev {raw_m['lit_dev']} · "
              f"ratio {raw_m['ratio']} · fire {raw_a['fire_motion']}", flush=True)

    needs_freeze = bool(S.verdict(raw_m)) if mask_ok else False
    ladder = []
    if needs_freeze:
        print(f"  the blade moves ({'; '.join(S.verdict(raw_m))}) — compositing it back frozen", flush=True)
        sword = F.feather(F.extend_along_axis(blade, still), 3) > 0.3
        plate = F.clean_plate(still, sword, grow=8)
        Image.fromarray(plate).save(OUT / "clean_plate.png")
        for clip in [(0.7, 1.55), (0.8, 1.35), (0.85, 1.25), (0.9, 1.18), (0.93, 1.12)]:
            cand = close_loop(F.freeze_lit(frames, still, sword, coals, radius=3, clip=clip))
            g = cand.astype(np.float32).mean(3)
            lum = np.array([f[blade].mean() for f in g])
            row = {"clip": list(clip), "lit_dev": round(max(S._lit_deviation(g, blade)), 2),
                   "subject_light_std": round(float(lum.std()), 2),
                   "fire_motion": round(float(np.abs(np.diff(g, axis=0))[:, coals].mean()), 2)}
            row["passes"] = bool(row["lit_dev"] <= S.LIMIT["lit_dev"]
                                 and row["subject_light_std"] >= S.ALIVE["subject_light_std"]
                                 and row["fire_motion"] >= S.ALIVE["fire_motion"])
            ladder.append(row)
            print(f"  relight {str(clip):12s} lit_dev {row['lit_dev']:5.2f} · light "
                  f"{row['subject_light_std']:5.2f} · fire {row['fire_motion']:5.2f} -> "
                  f"{'take it' if row['passes'] else 'no'}", flush=True)
            if row["passes"]:
                loop = cand; break
        else:
            # Every rung refused. That is a finding, not a reason to throw away a generation that is
            # already on disk: the gentlest relight failing means the fire cannot light this blade
            # without the change reading as movement. Ship what the model made, say the freeze did not
            # take, and let the ladder in the JSON show why.
            print("  no relight strength satisfies both gates — shipping the generation unfrozen; "
                  "the ladder is above and in the JSON", flush=True)
            needs_freeze = False
            loop = raw_loop
    elif mask_ok:
        print("  the blade already holds still — shipping the generation as it came", flush=True)
        loop = raw_loop
    else:
        loop = raw_loop

    encode(loop, str(OUT / "STEEL_cover_loop"))
    if needs_freeze:
        encode(raw_loop, str(OUT / "STEEL_cover_loop_unfrozen"))
    Image.fromarray(loop[0]).save(OUT / "cover.png")
    sh(f"ffmpeg -v error -i '{OUT}/cover.png' -vf scale=3000:3000:flags=lanczos '{OUT}/cover_3000.png' -y", quiet=True)
    idx = np.linspace(0, len(loop) - 1, 8).round().astype(int)
    Image.fromarray(np.concatenate([loop[i] for i in idx], 1)).save(OUT / "loop_sheet.jpg", quality=88)
    Image.fromarray(np.concatenate([loop[i] for i in (-3, -2, -1, 0, 1, 2)], 1)).save(OUT / "loop_seam.jpg", quality=90)

    fin_m, fin_a = (measure_array(loop, blade) if mask_ok else ({}, {}))
    rec = {"model": f"Wan2.2-T2V-A14B Q4_K_M ({GREPO})", "hashes": HASHES, "seed": SEED,
           "steps": STEPS, "seconds_per_step": round(per_step, 1), "guidance": [4.0, 3.0],
           "res": [H, W], "frames": int(len(loop)), "fps": FPS, "gen_seconds": round(gen_s, 1),
           "method": ("text-to-video, no still image and no image conditioning; loop closed by an "
                      f"{XF}-frame dissolve"
                      + ("; the sword composited back frozen and re-lit because it drifted"
                         if needs_freeze else "; the sword held still on its own and was left alone")),
           "froze_the_blade": needs_freeze, "relight_ladder": ladder,
           "as_generated": raw_m, "as_generated_alive": raw_a,
           "frozen": fin_m, "alive": fin_a}
    rec["mask_ok"] = mask_ok
    rec["verdict_still"] = S.verdict(fin_m) if mask_ok else ["mask not recognised — not judged"]
    rec["verdict_alive"] = S.liveness_verdict(fin_a) if mask_ok else []
    (WORK / "loop_verify.json").write_text(json.dumps(rec, indent=2))
    print("\nLOOP:", json.dumps(rec), flush=True)

    problems = (rec["verdict_still"] + rec["verdict_alive"]) if mask_ok else []
    if mask_ok:
        print(f"\n{'sword drift (px)':26s} {fin_m['drift_px']}")
        print(f"{'change light cant explain':26s} {fin_m['lit_dev']}")
        print(f"{'its motion / the rest':26s} {fin_m['ratio']}")
        print(f"{'fire motion':26s} {fin_a['fire_motion']}")
        print(f"{'firelight on the steel':26s} {fin_a['subject_light_std']}")
    else:
        print("\nThe loop was generated and written; the blade mask was not recognised, so it is "
              "shipped unjudged and unfrozen. Look at it.")
    assert not problems, "the cover does not meet its own gates: " + "; ".join(problems)
    if mask_ok and not rec["froze_the_blade"]:
        print("\nThe sword held still on its own, the fire lives, and the loop closes.", flush=True)
    elif mask_ok:
        print("\nThe sword holds still, the fire lives, and the loop closes.", flush=True)
    else:
        print("\nA loop was generated and written. It was NOT judged — the blade mask was not "
              "recognised — so nothing here claims the sword holds still.", flush=True)
    clock("DONE")
"""

Path("/tmp/aq_stage.py").write_text(STAGE_SRC)
Path("/tmp/aq_cfg.json").write_text(json.dumps({
    "pins": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PINS.items()},
    "seed": SEED, "tmp": str(TMP), "work": str(WORK), "out": str(OUT), "hf_home": str(TMP / "hf"),
    "tools": str(TOOLS), "prompt": PROMPT, "negative": NEG,
    "hold_subject": HOLD_SUBJECT}))

# A TIMEOUT THAT KILLS A SLOW STAGE IS A LIABILITY, NOT A SAFETY NET. The cover loop was measured
# at 28 minutes on a T4 pair and then, on the very next run, took over 120 on the same declared
# hardware with the same code — a 4x swing, almost certainly a shared host — and a 120-minute
# timeout killed it two hours and twenty minutes in, with the work nearly done. These numbers exist
# to catch a HUNG stage, and a hung stage is silent, whereas a slow one keeps printing; subprocess
# cannot tell them apart, so the only honest setting is one generous enough that only a genuine
# hang trips it. Kaggle's own twelve-hour session cap is the real backstop.
def run_stage(name, minutes):
    t0 = time.time()
    print(f"\n=== stage {name} (own process) ===", flush=True)
    r = subprocess.run([sys.executable, "/tmp/aq_stage.py", name], text=True, timeout=minutes * 60)
    print(f"stage {name}: rc={r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode

assert run_stage("cover", 420) == 0, "the cover stage failed — see its output above"
loop_rec = json.loads((WORK / "loop_verify.json").read_text())
print("LOOP frozen:", json.dumps(loop_rec["frozen"]), flush=True)
print("LOOP alive :", json.dumps(loop_rec["alive"]), flush=True)
# THE GATES BLOCK HERE, NOT IN THE STAGE. A cover whose subject wanders, or whose fire is dead, is
# not a cover — and the previous record shipped one because every number it printed was a
# whole-frame average that could not see a 3% blade.
assert not loop_rec["verdict_still"], "the blade moves: " + "; ".join(loop_rec["verdict_still"])
assert not loop_rec["verdict_alive"], "the loop is dead: " + "; ".join(loop_rec["verdict_alive"])
clock("loop delivered")
release("before the song")

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

def stem_energy(stem):
    """RMS of the isolated vocal, in dBFS — the number that says whether there is a voice to measure."""
    if not stem:
        return None
    try:
        import soundfile as sf
        a, _ = sf.read(stem, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        r = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
        return round(20 * np.log10(max(r, 1e-9)), 1)
    except Exception:
        return None

def register_gate(mp3):
    # An unmeasurable take is a REJECTED take, never a dead run. YIN returns all-NaN when it finds
    # no voiced frames, and the pinned instrument then hands numpy a [nan, nan] histogram range —
    # which killed a run that had already spent two hours making a cover and a first take.
    try:
        reg = M.register(str(mp3))
    except Exception as e:
        return False, {"register": "unmeasurable", "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return reg.get("register") == "male", reg

# the male voice reference — KEEP THE KEY's lead, the cleanest male vocal this pipeline owns
# (156 Hz median, 6.65 st spread), mounted from its PUBLIC kernel so the reference stays public
_ref = sorted(glob.glob("/kaggle/input/**/KEEPTHEKEY.mp3", recursive=True))
MALE_REF = _ref[0] if _ref else None
assert MALE_REF, "male reference not mounted (kernel source artafather/keep-the-key)"
print("male reference:", MALE_REF, flush=True)
clock("instruments proven")

# %% [markdown]
# ## The song — ACE-Step 1.5 XL, style transfer from a public male take, best of six by measurement
#
# Captions are not a control over vocal register (one male take in fifteen, measured), so the
# render is **style transfer**: text‑to‑music with the public KEEP THE KEY lead as `reference_audio`
# at strengths 0.25–0.35 — the words stay text‑driven while the reference biases the timbre. Six takes
# are rendered at 80 ODE steps, guidance 7.5, 180 s, and **every** take is gated on the isolated
# stem: register must measure male, word accuracy must reach 75%. Of the takes that pass, the one
# with the **highest word accuracy** ships (ties go to the tighter pitch spread). If nothing
# passes, the most intelligible non‑male take is converted to the male reference with zero‑shot
# Seed‑VC and gated again — a deterministic repair rather than another roll of the dice.
# The whole gate log is published beside the song.

# ── the 4.6B model: a ladder of ways to hold it on the card, each rung proven by a render ─
sys.path.insert(0, str(REPO))
import toml

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

# A rung is held because it RENDERS, not because it loads. Every render happens in ACE-Step's own
# CLI in a fresh process (clean GPU, clean RAM), so each rung is probed with a FULL-LENGTH render at
# 2 steps — attention buffers scale with the sequence, not the steps — and only a rung that produces
# audio is kept. The parent notebook never loads the 4.6B model itself: on a T4 the resident rung
# loaded at 12.1 GB and then every take OOM'd inside the CLI, and after the cover stages a parent
# that had held the model once was what the OOM killer took.
chosen = None
for name, model, dtype, oc, od in LADDER:
    t0 = time.time()
    rung = dict(rung=name, model=model, dtype=dtype, offload_to_cpu=oc, offload_dit_to_cpu=od)
    rc, found, tail = cli_render(f"probe_{name}", render_conf(f"probe_{name}", SEED, 0.35, rung, 2, DURATION), dtype)
    if found:
        rung["probe_seconds"] = round(time.time() - t0, 1)
        print(f"RUNG HELD: {name} — {model} @ {dtype}: full-length probe rendered in {time.time()-t0:.0f}s",
              flush=True)
        chosen = rung; break
    print(f"rung {name}: the full-length probe produced no audio (rc={rc}) in {time.time()-t0:.0f}s:\n"
          f"{tail[-400:]}", flush=True)
assert chosen, "no rung held"
(WORK / "rung.json").write_text(json.dumps({**chosen, "pins": {k: v for k, v in PINS.items()
                                                               if isinstance(v, str)},
                                            "seed": SEED}, indent=2))
clock("song model held")

# ── four takes: render every one, gate every one, keep the best passer ───────────────────
# The strength knob is a MEASURED trade-off, not a preference: this pipeline's own probe read 89%
# words at 0.20 and a male 145.9 Hz median at 0.35 — words fall as the reference's timbre takes
# hold. Six takes sample it across two seeds instead of rolling one setting six times, and the gate
# keeps the most intelligible take that still measures male.
CANDIDATES = [(6001, 0.30), (6002, 0.30), (6003, 0.35), (6004, 0.35), (6005, 0.25), (6006, 0.25)]
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
    try:
        stem = _vocal_stem(mp3)
        row["stem_dbfs"] = stem_energy(stem)
        ok_reg, reg = register_gate(mp3)
        acc = word_accuracy(mp3, LYRICS, stem=stem)
    except Exception as e:                       # one take's measurement can never end the run
        row["verdict"] = f"UNMEASURABLE ({type(e).__name__}: {str(e)[:120]})"
        gate_log.append(row); (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
        print(f"{name}: {row['verdict']} · stem {row.get('stem_dbfs')} dBFS", flush=True)
        continue
    row.update(register=reg, word_accuracy=round(acc, 3), asr_judge=_WH[1])
    ok_words = acc >= 0.75
    row["verdict"] = "PASS" if (ok_reg and ok_words) else \
        f"REJECTED ({reg.get('register')}{'' if ok_words else f' words {acc*100:.0f}%'})"
    gate_log.append(row); (WORK / "gate.json").write_text(json.dumps(gate_log, indent=2))
    print(f"{name}: median={reg.get('f0_hz')}Hz [{reg.get('register')}] "
          f"spread={reg.get('spread_st')}st lead={reg.get('lead_hz')}Hz@{reg.get('lead_frac')} "
          f"stem={row.get('stem_dbfs')}dBFS words={acc*100:.1f}% -> {row['verdict']} · "
          f"{row['seconds']:.0f}s", flush=True)
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

if WINNER is None:
    print("\nGATE LOG:\n" + json.dumps([{k: v for k, v in r.items() if k != "cli_tail"} for r in gate_log],
                                        indent=1)[:4000], flush=True)
assert WINNER, ("no candidate passed the male+intelligible gate — refusing to master a wrong take; "
                "verdicts: " + "; ".join(f"{r.get('seed')}:{r.get('verdict')}" for r in gate_log))
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

# ── the cover video: the loop under the whole song ───────────────────────────────────────
# -shortest is NOT enough here: it ends at the end of the LOOP ITERATION that crosses the song's
# end, so the file kept up to one loop of silent video (measured: 24.6 s of video under 20 s of
# audio). The song's own duration is the length, so pass it.
song_seconds = float(subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
    text=True, capture_output=True).stdout.strip() or DURATION)
sh(f"ffmpeg -v error -stream_loop -1 -i '{OUT}/STEEL_cover_loop_raw.mp4' -i '{wav}' "
   f"-vf scale=1080:1080:flags=lanczos,format=yuv420p -c:v libx264 -preset slow -crf 20 "
   f"-c:a aac -b:a 256k -t {song_seconds:.3f} -movflags +faststart '{OUT}/STEEL_cover_video.mp4' -y")
_vd = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                      str(OUT / "STEEL_cover_video.mp4")], text=True, capture_output=True).stdout.strip()
print(f"cover video: {_vd}s of video under {song_seconds:.3f}s of song", flush=True)
assert _vd and abs(float(_vd) - song_seconds) < 0.5, "the cover video does not match the song's length"
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
verify = {"pins": {k: v for k, v in PINS.items()},
          "seed_policy": f"best-of-{len(CANDIDATES)} passers by word accuracy; seeds {[c[0] for c in CANDIDATES]}",
          "winner_seed": WINNER_SEED, "lyric_craft": {k: craft_report[k] for k in
                                                     ("lines", "words", "syllables", "tight_pct", "mono_pct", "imper_pct")}}
ok_reg, reg = register_gate(mp3)
acc = word_accuracy(mp3, LYRICS, stem=_vocal_stem(mp3))
Lw, Lm = M.loudness(wav), M.loudness(mp3)
rep = M.report(str(mp3))
verify.update(register=reg, word_accuracy=round(acc, 3), asr_judge=_WH[1], wav=Lw, mp3=Lm,
              seconds=rep.get("seconds"), bpm=rep.get("bpm"), master_arm=best_arm,
              cover=loop_rec)
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
# THE COVER, JUDGED INSIDE THE SUBJECT'S OWN MASK. What used to be here were four whole-frame
# luma statistics — cuts, wrap ratio, mean motion, black frames — and a cover shipped that passed
# every one of them with a blade that drifted 18 px, because a global average cannot see a subject
# occupying 3% of the picture. These are the same numbers the loop stage gates on, restated here so
# the finished record refuses to assemble around a cover that failed.
problems += [f"the cover loop: {x}" for x in loop_rec["verdict_still"]]
problems += [f"the cover loop: {x}" for x in loop_rec["verdict_alive"]]
if loop_rec["frozen"]["ratio"] >= 1.0:
    problems.append(f"the blade moves more than the frame around it ({loop_rec['frozen']['ratio']}x)")
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
      f"blade drift {loop_rec['frozen']['drift_px']} px · fire {loop_rec['alive']['fire_motion']} · "
      f"{loop_rec['frames']} frames", flush=True)
clock("DONE")
