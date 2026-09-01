# %% [markdown]
# # STEEL — assembly and verification of the approved record, on CPU
#
# Every expensive artifact in this record already exists in a PUBLIC run, and this notebook says
# so rather than re-spending days of GPU to pretend otherwise:
#
#   * the SONG take is `ashranet/steel-audition` — the exact audio the commissioner approved by
#     ear (seed 6002 of a fully pinned configuration; that notebook shows every knob);
#   * the COVER is `artafather/steel-record-flame` — the LEGO forge loop, adapter on both experts,
#     measured seamless at 1.02x an ordinary frame step on the delivered file.
#
# What THIS notebook does is everything between generation and publication, all reproducible on
# the free CPU tier: master the approved take to web loudness and PROVE the mastering cost
# nothing, run every gate the pipeline owns on the delivered bytes, assemble the cover video, and
# write the verification record the publication cites. Both inputs are public kernels — the
# checklist's own standard — so a stranger can walk the whole chain.

# %%
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
T0 = time.time()
def sh(c, quiet=False): subprocess.run(c, shell=True, check=True,
    stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
def clock(w): print(f"  ⏱ {w} · t+{(time.time()-T0)/60:.1f} min", flush=True)

PINS = {
    "measure_sha": "199535aa517324d8021667b5a34a799aedd19353",  # lib/measure.py
    "songfit_sha": "199535aa517324d8021667b5a34a799aedd19353",  # lib/songfit.py (same commit)
    "lyric_sha":   "a76c41626054930ae4da51d47bfca5c672ce6bdc",  # song/lyrics_steel_run.txt (approved)
}
PROVENANCE = {
    "song_take": "https://www.kaggle.com/code/ashranet/steel-audition",
    "cover":     "https://www.kaggle.com/code/artafather/steel-record-flame",
}
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
sh("pip install -q demucs faster-whisper pyloudnorm soundfile 2>&1 | tail -1")
clock("installed")

import urllib.request
for f in ("measure.py", "songfit.py"):
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['measure_sha']}/lib/{f}", f"/tmp/{f}")
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/song/lyrics_steel_run.txt",
    "/tmp/lyrics.txt")
sys.path.insert(0, "/tmp")
import numpy as np
import measure as M, songfit as SF
assert M.selftest(), "measurement selftest failed — no number can be trusted"
assert SF.selftest(), "songfit selftest failed"
LYRICS = Path("/tmp/lyrics.txt").read_text().strip()
assert LYRICS.startswith("[Intro]") and "I am steel" in LYRICS and "smooth as bone" in LYRICS

TAKE = next(Path("/kaggle/input").rglob("*aud6002.mp3"), None)
assert TAKE, "approved take not mounted (kernel source ashranet/steel-audition)"
COVER_DIR = None
for p in Path("/kaggle/input").rglob("STEEL_cover_loop.mp4"):
    COVER_DIR = p.parent; break
assert COVER_DIR, "cover not mounted (kernel source artafather/steel-record-flame)"
print(f"take: {TAKE}\ncover: {COVER_DIR}", flush=True)
(OUT / "STEEL_lyrics.txt").write_text(LYRICS + "\n")
clock("inputs proven")

# %%
# The lyric against the clock — the pre-GPU gate, still binding even with no GPU to spend.
fit = SF.measure(LYRICS, 180.0, 128)
bad_fit = SF.verdict(fit)
print("songfit:", json.dumps(fit), flush=True)
assert not bad_fit, "the lyric does not fit the clock: " + "; ".join(bad_fit)

# The deterministic judge: demucs shifts 0 on CPU, whisper at temperature zero, and the input
# level-normalised by pure gain first (the separator scores a hotter mix worse — ~3 pts/dB).
_WH = [None, ""]
def asr():
    from faster_whisper import WhisperModel
    if _WH[0] is None:
        _WH[0] = WhisperModel("large-v3", device="cpu", compute_type="int8")
        _WH[1] = "large-v3/cpu-int8"
        print("ASR judge:", _WH[1], flush=True)
    return _WH[0]

def _vocal_stem(mp3):
    import demucs.separate, shlex, tempfile as _tf
    L = M.loudness(str(mp3))
    g = -14.0 - (L.get("lufs") if L.get("lufs") is not None else -14.0)
    norm = Path(_tf.mkdtemp()) / "norm.wav"
    sh(f"ffmpeg -v error -i '{mp3}' -af volume={g:.2f}dB -ar 44100 '{norm}' -y", quiet=True)
    td = _tf.mkdtemp()
    demucs.separate.main(shlex.split(
        f'--two-stems vocals -n htdemucs --shifts 0 --device cpu -o "{td}" "{norm}"'))
    return next(Path(td).rglob("vocals.wav"), None)

def word_accuracy(mp3, stem=None):
    target = stem or _vocal_stem(mp3)
    segs, _ = asr().transcribe(str(target), beam_size=5, vad_filter=True,
                               vad_parameters=dict(min_silence_duration_ms=400),
                               chunk_length=20, condition_on_previous_text=False, temperature=0.0)
    hyp = re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
    ref = re.findall(r"[a-z']+", re.sub(r"\[[^\]]*\]", " ", LYRICS.lower()))
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    d[:,0] = np.arange(len(ref)+1); d[0,:] = np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    return 1.0 - min(1.0, float(d[-1,-1])/max(1,len(ref)))

# the take, judged before mastering
stem = _vocal_stem(TAKE)
reg = M.classify_f0(M.finite_f0(M.f0_yin(*M.load(str(stem), mono=True))))
assert reg.get("register") == "male", f"take register {reg.get('register')} — not the approved voice"
TAKE_ACC = word_accuracy(TAKE, stem=stem)
print(f"take: register male · lead {reg.get('lead_hz')} Hz · words {TAKE_ACC*100:.1f}%", flush=True)
assert TAKE_ACC >= 0.62, f"take words {TAKE_ACC*100:.1f}% under the floor"
clock("take gated")

# %%
# MASTER TO WEB LOUDNESS: a web page applies no normalization, so the file plays as mastered.
# Ladder from -9, stepping down only if the limiter's measured CONTENT cost (level-invariant
# judge) leaves the 3-point band; true peak ceilinged at -1.0 dBTP for lossy encoders.
TARGET_TP = -1.0
def finish(src, wav_out, mp3_out, target_lufs):
    iters, ceiling = [], TARGET_TP
    for _ in range(3):
        a = M.loudness(str(src))
        g = target_lufs - (a["lufs"] if a["lufs"] is not None else -14.0)
        lim = 10 ** (ceiling / 20)
        af = (f"volume={g:.2f}dB,aresample=176400,"
              f"alimiter=limit={lim:.5f}:level=disabled,aresample=44100")
        sh(f"ffmpeg -v error -i '{src}' -af '{af}' -ar 44100 '{wav_out}' -y", quiet=True)
        sh(f"ffmpeg -v error -i '{wav_out}' -codec:a libmp3lame -b:a 320k '{mp3_out}' -y", quiet=True)
        got = M.loudness(mp3_out)
        iters.append({k: got.get(k) for k in ("lufs","lra_lu","true_peak_dbtp","clipped")})
        tp = got.get("true_peak_dbtp")
        if tp is None or tp <= TARGET_TP + 0.05: break
        ceiling -= (tp - TARGET_TP) + 0.1
    return iters

LADDER = [-9.0, -10.0, -11.5]
arms, scores, iters_all = {}, {}, {}
for tgt in LADDER:
    name = f"direct{int(tgt)}"
    iters_all[name] = finish(str(TAKE), str(OUT/f"_{name}.wav"), str(OUT/f"_{name}.mp3"), tgt)
    w = round(word_accuracy(OUT/f"_{name}.mp3"), 3)
    scores[name] = {"words": w, "cost_pts": round(100*(TAKE_ACC - w), 1), "target": tgt}
    arms[name] = {"wav": str(OUT/f"_{name}.wav"), "mp3": str(OUT/f"_{name}.mp3")}
    print(f"master {name}: words {w*100:.1f}% · cost {scores[name]['cost_pts']} pts", flush=True)
best = next((n for n in scores if scores[n]["cost_pts"] <= 3.0),
            min(scores, key=lambda n: scores[n]["cost_pts"]))
assert scores[best]["cost_pts"] <= 5.0, f"every master damaged the take: {scores}"
wav, mp3 = OUT/"STEEL.wav", OUT/"STEEL.mp3"
shutil.copy(arms[best]["wav"], wav); shutil.copy(arms[best]["mp3"], mp3)
for n in arms:
    Path(arms[n]["wav"]).unlink(missing_ok=True)
    if n != best: Path(arms[n]["mp3"]).unlink(missing_ok=True)
print(f"master choice: {best} · {json.dumps(scores[best])}", flush=True)
clock("mastered loud")

# %%
# the cover travels with its own verification, and the cover video is rebuilt under this master
for f in ("STEEL_cover_loop.mp4","STEEL_cover_loop.webm","STEEL_cover_loop_1080.webm",
          "cover_3000.png","loop_sheet.jpg","loop_seam.jpg"):
    src = COVER_DIR / f
    if src.exists(): shutil.copy(src, OUT / f)
song_seconds = float(subprocess.run(
    ["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(wav)],
    text=True, capture_output=True).stdout.strip() or 180.0)
sh(f"ffmpeg -v error -stream_loop -1 -i '{OUT}/STEEL_cover_loop.mp4' -i '{wav}' "
   f"-vf scale=1080:1080:flags=lanczos,format=yuv420p -c:v libx264 -preset slow -crf 20 "
   f"-c:a aac -b:a 256k -t {song_seconds:.3f} -movflags +faststart '{OUT}/STEEL_cover_video.mp4' -y")
clock("cover assembled")

# %%
# FINAL VERIFY, on the delivered bytes
stem_m = _vocal_stem(mp3)
reg_m = M.classify_f0(M.finite_f0(M.f0_yin(*M.load(str(stem_m), mono=True))))
acc_m = word_accuracy(mp3, stem=stem_m)
Lm = M.loudness(str(mp3))
cont = M.continuity(str(mp3))
problems = []
if reg_m.get("register") != "male": problems.append(f"register {reg_m.get('register')}")
if acc_m < 0.62: problems.append(f"words {acc_m*100:.1f}% under the floor")
if abs(scores[best]["words"] - acc_m) > 0.04:
    problems.append(f"judge drift {abs(scores[best]['words']-acc_m)*100:.1f} pts on identical bytes")
problems += M.continuity_verdict(cont)
if Lm.get("clipped"): problems.append(f"{Lm['clipped']} clipped samples")
tp = Lm.get("true_peak_dbtp")
if tp is not None and tp > TARGET_TP + 0.05: problems.append(f"true peak {tp}")
if Lm.get("lufs") is None or Lm["lufs"] < -11.8:
    problems.append(f"not loud enough for the web: {Lm.get('lufs')} LUFS")
verify = {"provenance": PROVENANCE, "pins": PINS,
          "take_words": round(TAKE_ACC,3), "register": reg_m, "word_accuracy": round(acc_m,3),
          "asr_judge": _WH[1], "master": scores[best], "master_iters": iters_all[best],
          "mp3": Lm, "continuity": cont, "songfit": fit,
          "problems": problems}
(WORK / "verify_final.json").write_text(json.dumps(verify, indent=2))
print(json.dumps(verify, indent=1)[:1200], flush=True)
assert not problems, "VERIFY REFUSED: " + "; ".join(problems)
print(f"\nVERIFIED: male · words {acc_m*100:.1f}% · {Lm['lufs']} LUFS · LRA {Lm.get('lra_lu')} · "
      f"TP {tp} dBTP · 0 holes · web-loud", flush=True)
clock("DONE")
