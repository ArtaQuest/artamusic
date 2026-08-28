# %% [markdown]
# # STEEL on HeartMuLa — the taste probe: two lyrics × two reference flavours
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
    "lyric_sha": "a0f955ece53555c055ce6f081ce0fa418cab616d",   # song/lyrics_steel.txt (plain)
    "mythic_sha": "97255ec860c91fe51c3cca6d3a3b299515a5cb98",  # song/lyrics_steel_mythic.txt
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
def fetch_lyric(sha, name, out):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{sha}/song/{name}", out)
    t = Path(out).read_text().strip()
    assert t.startswith("[Intro]"), f"wrong lyric at {sha[:8]}"
    t = t.replace("[Instrumental Break]", "[Inst]").replace("[Instrumental fades out]", "[Outro]")
    Path(out).write_text(t)
    return t
fetch_lyric(PINS["lyric_sha"], "lyrics_steel.txt", "/tmp/ly_plain.txt")
fetch_lyric(PINS["mythic_sha"], "lyrics_steel_mythic.txt", "/tmp/ly_mythic.txt")

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

# The two flavours are MEASURED from the operator's own reference tracks, not guessed:
#   souls: 101 BPM, a 76 Hz near-whispered lead 7.5 dB under a sub-heavy band, hypnotic
#   flame: 85 BPM, a 99 Hz deep lead under ethereal pads, long intro, slow build
STYLES = [
    ("souls_plain",  "/tmp/ly_plain.txt",  "dark epic,deep male voice,whispering,atmospheric,heavy sub bass,slow pounding drums,mystical,night,cinematic"),
    ("souls_mythic", "/tmp/ly_mythic.txt", "dark epic,deep male voice,whispering,atmospheric,heavy sub bass,slow pounding drums,mystical,night,cinematic"),
    ("flame_plain",  "/tmp/ly_plain.txt",  "dark cinematic ballad,deep male vocals,spoken word,ethereal pads,heavy bass,slow,ancient,emotional,epic build"),
    ("flame_mythic", "/tmp/ly_mythic.txt", "dark cinematic ballad,deep male vocals,spoken word,ethereal pads,heavy bass,slow,ancient,emotional,epic build"),
]
SEED = 4242
report = []
for name, lyr, tags in STYLES:
    Path("/tmp/tags.txt").write_text(tags)
    torch.manual_seed(SEED)
    t0 = time.time()
    with torch.no_grad():
        pipe({"lyrics": lyr, "tags": "/tmp/tags.txt"},
             max_audio_length_ms=180_000, save_path=str(OUT / f"pal_{name}.mp3"),
             topk=50, temperature=1.0, cfg_scale=1.5)
    dt = time.time() - t0
    print(f"  {name}: {dt/60:.1f} min (RTF {dt/180:.2f})", flush=True)
    report.append({"style": name, "lyric": lyr, "tags": tags, "seconds": round(dt, 1)})
    (WORK / "palette.json").write_text(json.dumps(report, indent=1))
    clock(f"{name} done")

print("PALETTE:", json.dumps(report), flush=True)
clock("DONE")
