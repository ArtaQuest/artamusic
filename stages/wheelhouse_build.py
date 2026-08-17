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
LOCK_TEXT = '''# Only what the hardware and the code genuinely require. Everything else is resolved by pip
# on Kaggle's own image and then FROZEN by the wheelhouse build into requirements.frozen.txt,
# which is what production kernels install from. A hand-written lock of 30 interlocking pins
# produced ResolutionImpossible; a lock should be the OUTPUT of a resolve, not a wish list.
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.7.1+cu126
torchvision==0.22.1+cu126
torchaudio==2.7.1+cu126
diffusers==0.39.0
transformers
accelerate
safetensors
sentencepiece
protobuf
ftfy
imageio
imageio-ffmpeg
pillow
numpy
scipy
demucs
faster-whisper
pyloudnorm
mutagen
toml
loguru
einops
numba
soundfile
ffmpeg-python
python-dotenv
diskcache
py3langid
vector-quantize-pytorch
hf_transfer
'''
from pathlib import Path

WORK = Path("/kaggle/working"); WH = WORK / "wheelhouse"; WH.mkdir(parents=True, exist_ok=True)

def sh(c):
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    print(r.stdout[-2000:] or r.stderr[-1500:], flush=True)
    return r.returncode

# The lock travels IN this file as LOCK_TEXT (a notebook cell has no __file__ — v1 death class:
# a name that exists in a script and not in a papermill cell).
lock = WORK / "requirements.lock"
lock.write_text(LOCK_TEXT)
print("lock:", lock, flush=True)

rc = sh(f"pip download -r '{lock}' -d '{WH}' --only-binary=:all: "
        f"--extra-index-url https://download.pytorch.org/whl/cu126 > /tmp/dl.log 2>&1")
print(Path("/tmp/dl.log").read_text()[-2500:], flush=True)
wheels = sorted(WH.glob("*.whl"))
print(f"{len(wheels)} wheels, {sum(w.stat().st_size for w in wheels)/2**30:.2f} GB", flush=True)
# A verifier that passes on an empty directory is not a verifier. The lock has ~30
# requirements and torch alone pulls a dozen transitive wheels.
assert len(wheels) >= 30, f"pip download produced only {len(wheels)} wheels (rc={rc})"
assert any("torch-2.7.1+cu126" in w.name for w in wheels), "the cu126 torch wheel is missing"

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
# Prove the set is COMPLETE by resolving dependencies too — --no-deps proved nothing.
rc2 = sh(f"pip install --no-index --find-links='{WH}' -r '{lock}' --target /tmp/_proof "
         f"> /tmp/inst.log 2>&1")
print(Path("/tmp/inst.log").read_text()[-1500:], flush=True)
print("OFFLINE INSTALL:", "OK" if rc2 == 0 else f"FAILED rc={rc2}", flush=True)
assert rc2 == 0, "the wheelhouse is incomplete — a production kernel would die at pip time"
frozen = subprocess.run("pip list --path /tmp/_proof --format=freeze", shell=True,
                        capture_output=True, text=True).stdout
(WORK / "requirements.frozen.txt").write_text(frozen)
print(f"froze {len(frozen.splitlines())} resolved pins -> requirements.frozen.txt", flush=True)
print("WHEELHOUSE COMPLETE", flush=True)
