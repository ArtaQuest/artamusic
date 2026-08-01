#!/usr/bin/env python3
"""Measurement for generated songs — every estimator validated before it is trusted.

That last clause is the whole point. In the previous build EVERY estimator that was not checked
against a signal of known properties turned out to be wrong: a tempo estimator read 128 BPM as
64.00, a formant estimator missed /i/'s F2 by 1,600 Hz, a peak meter scraped ffmpeg's text output
and fell back to a plausible default and shipped a clipped file, and a vocal-register gate ranked a
female take first while the only male take came 6th of 8. Each looked fine. Each was wrong.

So `selftest` is not ceremony here, it is the entry point.

    python measure.py selftest
    python measure.py report song.mp3 [--ref reference.mp3]
    python measure.py register song.mp3        # demucs stem + YIN, the only trustworthy way
"""
from __future__ import annotations
import argparse, json, math, shlex, subprocess, sys, tempfile, wave, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

SR, EPS = 44100, 1e-12
BANDS = [(20,60),(60,120),(120,250),(250,500),(500,1000),
         (1000,2000),(2000,4000),(4000,8000),(8000,16000)]
# Adult ranges overlap between 165 and 195 Hz; a reading there names nothing, so it is reported as
# ambiguous rather than forced into a class.
MALE_MAX, FEMALE_MIN = 165.0, 195.0


def load(path, sr=SR, mono=False):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t: tmp = t.name
    subprocess.run(["ffmpeg","-v","error","-i",str(path),"-ac","1" if mono else "2",
                    "-ar",str(sr),"-c:a","pcm_s16le",tmp,"-y"], check=True)
    with wave.open(tmp) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)/32768.0
        ch = w.getnchannels()
    Path(tmp).unlink()
    return (raw if ch == 1 else raw.reshape(-1,2)), sr


def f0_yin(x, sr, lo=70.0, hi=500.0, frame=2048, hop=512, thresh=0.15):
    """YIN. Monophonic pitch for monophonic input — which is why it is only ever run on a stem."""
    tmin, tmax = int(sr/hi), int(sr/lo)
    out = []
    for s in range(0, max(0, len(x)-frame-tmax), hop):
        fr = x[s:s+frame]
        if np.sqrt((fr**2).mean()) < 0.01: continue
        d = np.empty(tmax+1)
        for tau in range(tmin, tmax+1):
            diff = fr - x[s+tau:s+tau+frame]
            d[tau] = (diff**2).sum()
        cum, run = np.ones(tmax+1), 0.0
        for tau in range(tmin, tmax+1):
            run += d[tau]
            cum[tau] = d[tau]*(tau-tmin+1)/run if run > 0 else 1.0
        cand = [t for t in range(tmin+1, tmax)
                if cum[t] < thresh and cum[t] <= cum[t-1] and cum[t] <= cum[t+1]]
        tau = cand[0] if cand else int(np.argmin(cum[tmin:tmax])+tmin)
        if cum[tau] > 0.5: continue
        y0,y1,y2 = cum[tau-1],cum[tau],cum[tau+1]
        den = y0-2*y1+y2
        out.append(sr/(tau + (0.5*(y0-y2)/den if den else 0.0)))
    return np.array(out)


def register(path):
    """Vocal register from a demucs-ISOLATED stem. Never from the mix.

    A mix-based estimator locks onto the 808: its harmonics at 80/160/240 Hz sit directly over the
    male vocal range. The previous one returned only exact FFT-bin multiples (10.7666 Hz/bin at
    n_fft=2048, sr=22050 — 129.20, 139.97, 161.50, 172.27), was wrong on 8 of 8 takes, and never
    once fired the female gate it existed to enforce.
    """
    import demucs.separate
    with tempfile.TemporaryDirectory() as td:
        demucs.separate.main(shlex.split(
            f'--two-stems vocals -n htdemucs --device cpu -o "{td}" "{path}"'))
        voc = next(Path(td).rglob("vocals.wav"), None)
        if voc is None: return {"register":"no-vocal-stem"}
        x, sr = load(voc, mono=True)
        f0 = f0_yin(x[::2], sr//2)
    if len(f0) < 40: return {"register":"unknown","frames":int(len(f0))}
    med = float(np.median(f0)); q1,q3 = np.percentile(f0,[25,75])
    reg = "male" if med < MALE_MAX else ("female" if med > FEMALE_MIN else "ambiguous")
    # A wide spread means the estimate is untrustworthy, not that the register is known. The
    # genuine male take measured an IQR of 19.8 Hz; the two females, 54.1 and 69.9.
    return {"f0_hz":round(med,1), "iqr_hz":round(float(q3-q1),1),
            "frames":int(len(f0)), "register":reg,
            "trusted": bool((q3-q1) < 60)}


def loudness(path):
    """Integrated loudness, loudness RANGE and true peak, straight from ffmpeg's ebur128 summary.

    LRA is here because its absence let a mastering bug ship: a gain rider collapsed the range from
    7.1 to 3.0 LU against a 5.4 LU reference, and a harness measuring only crest saw nothing.
    """
    out = subprocess.run(["ffmpeg","-hide_banner","-nostats","-i",str(path),
                          "-filter_complex","ebur128=peak=true","-f","null","-"],
                         capture_output=True, text=True).stderr
    tail = out[out.rfind("Summary"):] if "Summary" in out else ""
    g = lambda k: next((float(l.split()[1]) for l in tail.splitlines()
                        if l.strip().startswith(k)), None)
    st, sr = load(path)
    peak = float(np.abs(st).max())
    return {"lufs": g("I:"), "lra_lu": g("LRA:"),
            "peak_dbfs": round(20*math.log10(peak+EPS),2),
            "true_peak_dbtp": g("Peak:"),
            # s16's maximum positive sample is 32767/32768 = 0.9999695. The previous threshold
            # (0.99999) sat ABOVE that, so positive clipping could never be counted — a meter that
            # cannot fire is worse than no meter. 0.99996 sits safely below the s16 ceiling and
            # above any musical sample that is not pinned to the rail.
            "clipped": int((np.abs(st) >= 0.99996).sum())}


def bands(x, sr, n=1<<15):
    hop = n//2
    if len(x) < n: return np.zeros(len(BANDS))
    idx = np.arange(1+(len(x)-n)//hop)[:,None]*hop + np.arange(n)[None,:]
    S = (np.abs(np.fft.rfft(x[idx]*np.hanning(n),axis=1))**2).mean(0)
    f = np.fft.rfftfreq(n,1/sr)
    return np.array([S[(f>=lo)&(f<hi)].sum() for lo,hi in BANDS])


def tempo(x, sr):
    n, hop = 2048, 512
    if len(x) < n*8: return {}
    idx = np.arange(1+(len(x)-n)//hop)[:,None]*hop + np.arange(n)[None,:]
    S = np.abs(np.fft.rfft(x[idx]*np.hanning(n),axis=1))
    # LAG SEARCH ON THE KICK BAND ONLY (<150 Hz). Full-band flux hears the hats, and a dotted-
    # eighth hat pattern over a 100 BPM kick reads as 133 — a 4:3 alias that nearly discarded a
    # valid take. The kick carries the grid; the hats carry the subdivision.
    fbins = np.fft.rfftfreq(n, 1/sr)
    kick = S[:, fbins < 150]
    env = np.maximum(np.diff(np.log1p(kick*100),axis=0),0).sum(1)
    env = np.concatenate([[0.0],(env-env.mean())/(env.std()+EPS)])
    ac = np.correlate(env-env.mean(), env-env.mean(), "full")[len(env)-1:]
    lags = np.arange(len(ac)); per = lags*hop/sr
    ok = (per > 60/180) & (per < 60/60)
    sup = lambda L: sum(ac[m*L]*w for m,w in ((1,1.0),(2,.7),(4,.4)) if m*L < len(ac))
    best = max(lags[ok], key=sup)
    # Octave guard: a 64 BPM grid lands on every beat of a 128 BPM track, so the harmonic sum
    # favours the slower reading. Caught by the self-test, which read 128 as 64.00.
    half = best//2
    if half >= lags[ok][0] and ac[half] > 0.80*ac[best]: best = half
    # 4:3 alias guard, the mirror of the octave guard above: if 4/3 of the winning lag (the true,
    # SLOWER grid under a dotted-eighth reading) is comparably supported, prefer it.
    four3 = int(round(best*4/3))
    if four3 < len(ac) and ac[four3] > 0.80*ac[best]: best = four3
    if 0 < best < len(ac)-1:
        y0,y1,y2 = ac[best-1],ac[best],ac[best+1]
        d = y0-2*y1+y2
        best = best + (np.clip(0.5*(y0-y2)/d,-0.5,0.5) if d else 0.0)
    return {"bpm": round(60.0/(best*hop/sr),2)}


def report(path, ref=None):
    st, sr = load(path); mono = st.mean(1)
    b = bands(mono, sr)
    out = {"file": Path(path).name, "seconds": round(len(mono)/sr,2)}
    out.update(loudness(path)); out.update(tempo(mono, sr))
    L,R = st[:,0], st[:,1]
    m,s = (L+R)/2,(L-R)/2
    out["width"] = round(float(np.sqrt((s**2).mean())/(np.sqrt((m**2).mean())+EPS)),3)
    out["sub_share"] = round(float(b[:2].sum()/(b.sum()+EPS)),4)
    if ref:
        rx,_ = load(ref, mono=True); rb = bands(rx, sr)
        d = 10*np.log10(b/b.sum()+EPS) - 10*np.log10(rb/rb.sum()+EPS)
        out["band_err_max_db"] = round(float(np.abs(d).max()),2)
    return out


def selftest():
    ok = True
    t = np.arange(int(SR*3))/SR
    print("PITCH — synthetic sawtooths of known f0")
    for truth in (100.0,130.0,165.0,210.0,260.0):
        x = sum(np.sin(2*np.pi*truth*h*t)/h for h in range(1,30))*0.3
        got = float(np.median(f0_yin(x, SR)))
        good = abs(got-truth) < 1.0; ok &= good
        print(f"   {truth:6.1f} -> {got:7.2f}  {'ok' if good else 'FAIL'}")
    print("\nTEMPO — click trains of known bpm (the octave trap)")
    for bpm in (90.0,100.0,128.0):
        x = np.zeros(int(SR*20))
        for k in range(int(20/(60/bpm))):
            p = int(k*(60/bpm)*SR)
            if p+200 < len(x): x[p:p+200] = np.hanning(200)
        got = tempo(x, SR).get("bpm",0)
        good = abs(got-bpm) < 1.0; ok &= good
        print(f"   {bpm:6.1f} -> {got:7.2f}  {'ok' if good else 'FAIL'}")
    print("\nTEMPO — 100 BPM kicks under dotted-eighth hats (the 4:3 alias that read 133)")
    x = np.zeros(int(SR*20))
    for k in range(int(20/0.6)):
        p_ = int(k*0.6*SR)
        if p_+300 < len(x):
            t_ = np.arange(300)/SR
            x[p_:p_+300] += np.sin(2*np.pi*55*t_)*np.hanning(300)          # 55 Hz kick
    for k in range(int(20/0.45)):
        p_ = int(k*0.45*SR)
        if p_+120 < len(x):
            x[p_:p_+120] += np.random.default_rng(k).normal(0,0.25,120)*np.hanning(120)  # hat
    got = tempo(x, SR).get("bpm",0)
    good = abs(got-100.0) < 2.0; ok &= good
    print(f"   kick+hats -> {got:7.2f} (want ~100, alias reads 133)  {'ok' if good else 'FAIL'}")

    print("\nCLIPPING — a rail-pinned s16 file must COUNT (the old threshold could never fire)")
    with tempfile.TemporaryDirectory() as td:
        p_ = Path(td)/"c.wav"
        xc = np.zeros((SR,2)); xc[1000:1300] = 1.0                          # 300 pinned samples
        with wave.open(str(p_),"wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(xc,-1,1)*32767).astype("<i2").tobytes())
        L = loudness(p_)
        good = L["clipped"] >= 300; ok &= good
        print(f"   pinned 600 ch-samples -> counted {L['clipped']}  {'ok' if good else 'FAIL'}")

    print("\nLOUDNESS — a -1 dBFS sine must read -1, and clipping must be counted")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)/"s.wav"
        x = (np.column_stack([np.sin(2*np.pi*1000*t)]*2)*10**(-1/20)*32767).astype("<i2")
        with wave.open(str(p),"wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(x.tobytes())
        L = loudness(p)
        good = abs(L["peak_dbfs"]+1.0) < 0.2 and L["clipped"] == 0; ok &= good
        print(f"   peak {L['peak_dbfs']:+.2f} dBFS (want -1.00), clipped {L['clipped']}  "
              f"{'ok' if good else 'FAIL'}  LRA={L['lra_lu']}")
    print(f"\n{'ALL PASSED' if ok else 'FAILURES — do not trust any report from this build'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("report"); r.add_argument("files", nargs="+"); r.add_argument("--ref")
    g = sub.add_parser("register"); g.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "selftest": sys.exit(0 if selftest() else 1)
    if a.cmd == "register":
        for f in a.files: print(f"{Path(f).name:22s} {json.dumps(register(f))}")
    else:
        for f in a.files: print(json.dumps(report(f, a.ref), indent=2))
