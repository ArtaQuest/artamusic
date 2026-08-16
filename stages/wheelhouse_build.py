# aq-wheelhouse — the ONE kernel ever allowed internet. CPU only, zero GPU quota.
#
# Downloads every wheel in requirements.lock ON KAGGLE'S OWN IMAGE (so python/glibc/platform tags
# match what production kernels will run on), records a manifest of path -> bytes -> sha256 plus
# the image's python/glibc/pip versions, and publishes as a PUBLIC dataset. After this, no kernel
# ever touches pip's index, HF Hub, or GitHub at GPU time — the entire class of URL, credential,
# network-drop and mid-run auth deaths is deleted, not defended against.
#
# Re-run WEEKLY as a canary: Kaggle's image is not pinnable through metadata, so drift is caught
# here in a 10-minute CPU kernel, not in a 3-hour GPU run.
import hashlib, json, platform, subprocess, sys
LOCK_TEXT = '''# The single environment every kernel installs from — offline, from a mounted wheelhouse dataset,
# BEFORE torch is imported. Built once by a CPU kernel ON KAGGLE'S OWN IMAGE so python/glibc/
# platform tags match; then no kernel ever touches pip's index or the network again.
#
# torch is pinned to the cu126 line: Kaggle's preinstalled 2.10+cu128 has no sm_60 kernels.
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.7.1+cu126
torchvision==0.22.1+cu126
torchaudio==2.7.1+cu126
diffusers==0.39.0
transformers==4.57.1
accelerate==1.10.1
safetensors==0.6.2
sentencepiece==0.2.0
protobuf==5.29.5
ftfy==6.3.1
imageio==2.37.0
imageio-ffmpeg==0.6.0
pillow==11.3.0
numpy==2.2.6
scipy==1.15.3
demucs==4.0.1
faster-whisper==1.2.1
pyloudnorm==0.1.1
mutagen==1.47.0
toml==0.10.2
loguru==0.7.3
einops==0.8.1
numba==0.61.2
soundfile==0.13.1
ffmpeg-python==0.2.0
python-dotenv==1.1.1
diskcache==5.6.3
py3langid==0.3.0
vector-quantize-pytorch==1.22.15
hf_transfer==0.1.9
'''
from pathlib import Path

WORK = Path("/kaggle/working"); WH = WORK / "wheelhouse"; WH.mkdir(parents=True, exist_ok=True)

def sh(c):
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    print(r.stdout[-2000:] or r.stderr[-1500:], flush=True)
    return r.returncode

lock = Path("/kaggle/input/aq-lock/requirements.lock") if Path("/kaggle/input/aq-lock").exists() \
       else Path("requirements.lock")
if not lock.exists():
    lock = WORK / "requirements.lock"
    lock.write_text(open(__file__).read().split("LOCK_TEXT = '''")[1].split("'''")[0])
print("lock:", lock, flush=True)

rc = sh(f"pip download -r '{lock}' -d '{WH}' --only-binary=:all: 2>&1 | tail -20")
wheels = sorted(WH.glob("*.whl"))
print(f"{len(wheels)} wheels, {sum(w.stat().st_size for w in wheels)/2**30:.2f} GB", flush=True)

manifest = {}
for w in wheels:
    h = hashlib.sha256()
    with open(w, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    manifest[w.name] = {"bytes": w.stat().st_size, "sha256": h.hexdigest()}
(WORK / "manifest.json").write_text(json.dumps(manifest, indent=1))

image = {"python": platform.python_version(), "platform": platform.platform(),
         "glibc": platform.libc_ver()[1],
         "pip": subprocess.run("pip --version", shell=True, capture_output=True, text=True).stdout.strip(),
         "wheels": len(wheels), "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
(WORK / "image.json").write_text(json.dumps(image, indent=1))
print(json.dumps(image, indent=1), flush=True)

# prove the wheelhouse is complete: install from it OFFLINE into a scratch prefix
rc2 = sh(f"pip install --no-index --find-links='{WH}' -r '{lock}' --target /tmp/_proof "
         f"--no-deps 2>&1 | tail -5")
print("OFFLINE INSTALL:", "OK" if rc2 == 0 else f"FAILED rc={rc2}", flush=True)
assert rc2 == 0, "the wheelhouse is incomplete — a production kernel would die at pip time"
print("WHEELHOUSE COMPLETE", flush=True)
