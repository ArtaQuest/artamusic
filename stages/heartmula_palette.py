# %% [markdown]
# # STEEL on HeartMuLa — six styles of the same song
#
# Three shipped records on ACE-Step 1.5 were rejected by the operator's ear while passing every
# instrument, and the last one scored within noise of the conditioning reference on a learned
# aesthetic judge — so the ceiling being hit is plausibly the GENERATOR's, and the taste being
# missed is plausibly the STYLE's. This probe changes both at once, controlled: the same lyric,
# a NEW generator (HeartMuLa-oss-3B, Apache-2.0, LM-over-codec, the current best open model on
# lyric controllability and music quality by its own benchmark), and SIX genre framings. Every
# take is scored by the aesthetic judge, and all six ship as a listening palette — the operator's
# ear, not a proxy, picks the direction.
#
# HeartMuLa runs its 3B LM and its codec on separate devices by design — the T4 pair maps onto
# that exactly. The LM runs fp16, not bf16: below sm_80, bf16 attention falls off the
# memory-efficient path.

# %%
import json, os, subprocess, sys, time
from pathlib import Path
T0 = time.time()
def sh(c): subprocess.run(c, shell=True, check=True)
def clock(what): print(f"  ⏱ {what} · t+{(time.time()-T0)/60:.1f} min", flush=True)

PINS = {
    "heartlib": "3783bdb8441f2c298b1e64c8651173aac200361c",   # github.com/HeartMuLa/heartlib
    "mula": "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
    "codec": "HeartMuLa/HeartCodec-oss-20260123",
    "gen": "HeartMuLa/HeartMuLaGen",
    "lyric_sha": "a0f955ece53555c055ce6f081ce0fa418cab616d",  # ArtaQuest/artamusic song/lyrics_steel.txt
}
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME="/tmp/hf", HF_HUB_ENABLE_HF_TRANSFER="1")

sh(f"git clone https://github.com/HeartMuLa/heartlib /tmp/heartlib && "
   f"cd /tmp/heartlib && git checkout {PINS['heartlib']}")
# NOT -e: an editable install registers its path in a .pth file that the interpreter reads at
# STARTUP, so the running kernel that just installed it cannot import it. A normal install copies
# the package into site-packages, importable immediately. Cost one run to learn.
sh(f"{sys.executable} -m pip install -q /tmp/heartlib hf_transfer audiobox_aesthetics torchcodec 2>&1 | tail -2")
import importlib, heartlib as _hl_probe   # fail HERE, in the first minute, if the import is broken
print("heartlib import ok:", _hl_probe.__file__, flush=True)
clock("installed")

import urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/song/lyrics_steel.txt",
    "/tmp/lyrics_steel.txt")
LYRICS = Path("/tmp/lyrics_steel.txt").read_text().strip()
assert LYRICS.startswith("[Intro]") and "grips me through it all" in LYRICS, "wrong lyric at pin"
# HeartMuLa's corpus uses the same bracketed sections but not our two instrumental tags — map them
# to the nearest section forms it was trained on rather than hand it markup it never saw.
LYRICS = LYRICS.replace("[Instrumental Break]", "[Inst]").replace("[Instrumental fades out]", "[Outro]")
Path("/tmp/lyrics.txt").write_text(LYRICS)
print(LYRICS[:200], flush=True)

# %%
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(PINS["mula"], local_dir="/tmp/ckpt/HeartMuLa-oss-3B")
snapshot_download(PINS["codec"], local_dir="/tmp/ckpt/HeartCodec-oss")
# gen_config.json + tokenizer.json live in NEITHER model repo — they ship in the separate
# HeartMuLa/HeartMuLaGen bundle, which is exactly what the official Space downloads. Learned by
# running: the README's ckpt tree shows them at the root and never says where they come from.
for name in ("gen_config.json", "tokenizer.json"):
    hf_hub_download(PINS["gen"], name, local_dir="/tmp/ckpt")
print("ckpt tree:", sorted(p.name for p in Path("/tmp/ckpt").iterdir()), flush=True)
clock("checkpoints down")

# %%
import torch
from heartlib import HeartMuLaGenPipeline
NGPU = torch.cuda.device_count()
pipe = HeartMuLaGenPipeline.from_pretrained(
    "/tmp/ckpt",
    device={"mula": torch.device("cuda:0"),
            "codec": torch.device("cuda:1" if NGPU >= 2 else "cuda:0")},
    dtype={"mula": torch.float16, "codec": torch.float32},
    version="3B", lazy_load=NGPU < 2)
clock("pipeline up")

STYLES = [
    ("chant",      "male choir,dark,epic,war drums,orchestral,anthem,powerful"),
    ("metal",      "heavy metal,male vocals,powerful,anthemic,electric guitar,driving drums"),
    ("nordic",     "folk metal,male choir,nordic,tribal drums,chant,epic,dark"),
    ("industrial", "industrial,dark,male vocals,heavy,electronic,pounding,intense"),
    ("doom",       "doom metal,slow,heavy,male vocals,dark,crushing"),
    ("cinematic",  "epic orchestral,male choir,cinematic,percussion,brass,dramatic,battle"),
]
SEED = 4242
report = []
for name, tags in STYLES:
    Path("/tmp/tags.txt").write_text(tags)
    torch.manual_seed(SEED)
    t0 = time.time()
    with torch.no_grad():
        pipe({"lyrics": "/tmp/lyrics.txt", "tags": "/tmp/tags.txt"},
             max_audio_length_ms=180_000, save_path=str(OUT / f"pal_{name}.mp3"),
             topk=50, temperature=1.0, cfg_scale=1.5)
    dt = time.time() - t0
    print(f"  {name}: {dt/60:.1f} min (RTF {dt/180:.2f})", flush=True)
    report.append({"style": name, "tags": tags, "seconds": round(dt, 1)})
    (WORK / "palette.json").write_text(json.dumps(report, indent=1))
    clock(f"{name} done")

# %%
# Score each take with the aesthetic judge, 10 s windows averaged — the model's regime.
import numpy as np
from audiobox_aesthetics.infer import initialize_predictor
AB = initialize_predictor()
import tempfile
for row in report:
    td = tempfile.mkdtemp(); chunks = []
    src = OUT / f"pal_{row['style']}.mp3"
    for k in range(18):
        cp = f"{td}/{k}.wav"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(k*10), "-t", "10",
                            "-i", str(src), "-ac", "1", "-ar", "16000", cp], capture_output=True)
        if r.returncode == 0 and Path(cp).exists() and Path(cp).stat().st_size > 32000:
            chunks.append(cp)
    rows = AB.forward([{"path": c} for c in chunks])
    row["aesthetics"] = {ax: round(float(np.mean([x[ax] for x in rows])), 2)
                         for ax in ("CE", "CU", "PC", "PQ")}
    print(f"  {row['style']}: {row['aesthetics']}", flush=True)
(WORK / "palette.json").write_text(json.dumps(report, indent=1))
print("PALETTE:", json.dumps(report), flush=True)
clock("DONE")
