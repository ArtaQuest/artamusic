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
# ambiguous rather than forced into a class. MALE_MIN exists because a stem that leaks 808 has no
# voice in it at all: without a lower bound, bass bleed classifies as a confident male vocal —
# the exact artefact this function was written to prevent.
MALE_MIN, MALE_MAX, FEMALE_MIN = 85.0, 165.0, 195.0

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


def finite_f0(f0):
    """Keep only frames YIN actually voiced: finite AND positive.

    f0_yin's parabolic refinement divides by (y0 - 2*y1 + y2). When that denominator is tiny the
    correction blows up, so a frame can come back as +/-inf or as a NEGATIVE frequency — and
    np.log2 of a negative is NaN, which is what detonated np.histogram three hours into a
    publication run. The length guard below could not see it: forty NaNs are still forty frames.
    """
    f0 = np.asarray(f0, dtype=float)
    return f0[np.isfinite(f0) & (f0 > 0)]


def _row(**kw):
    """One key set for every exit path. A caller reading reg["lead_hz"] must never KeyError
    because the stem was missing — a downstream crash on the failure path is how a gate stops
    gating."""
    r = {"register": "unknown", "f0_hz": None, "q1_hz": None, "q3_hz": None, "spread_st": None,
         "lead_hz": None, "lead_frac": None, "oct_up_frac": None, "voiced_s": 0.0,
         "frames": 0, "bands": {}}
    r.update(kw)
    return r


def register(path, hop_div=2):
    """Describe the singing voice as a DISTRIBUTION, not a verdict.

    Why this is not a boolean any more. YIN is monophonic by construction: it estimates ONE pitch
    per frame. On a stem containing a male lead and a choir an octave above, the median simply
    reports the louder one — adding a second voice does not move it. So this function can support
    "the lead measures male"; it can NEVER support "no other voice is present", and the earlier
    version's confident single number invited exactly that overclaim.

    It also replaces a linear IQR threshold with a SEMITONE spread. Pitch is logarithmic: 60 Hz of
    spread means something entirely different at 130 Hz than at 230 Hz, so the old rule was
    stricter on low voices than high ones — backwards, for a gate whose job is finding low voices.

    Measured on this corpus: covers 7.03-7.08 st, the reference take 7.15, the shipped master
    10.09, a deliberate choir take 16.77.
    """
    import demucs.separate
    with tempfile.TemporaryDirectory() as td:
        demucs.separate.main(shlex.split(
            f'--two-stems vocals -n htdemucs --device cpu -o "{td}" "{path}"'))
        voc = next(Path(td).rglob("vocals.wav"), None)
        if voc is None:
            return _row(register="no-vocal-stem")
        x, sr = load(voc, mono=True)
        f0 = f0_yin(x[::hop_div], sr // hop_div)
    # COUNT IS NOT THE SAME AS FINITE. YIN returns a non-finite estimate for a frame it cannot
    # voice, so a stem that is silent (or has no lead at all) arrives here as forty-plus NaNs, sails
    # past a length check, and detonates in np.histogram with "autodetected range of [nan, nan]".
    # That killed a publication run three hours in, after the cover and the first take were already
    # made. An unvoiced frame is not a measurement: drop it, and judge on what is left.
    f0 = finite_f0(f0)
    if len(f0) < 40:
        return _row(register="unknown", frames=int(len(f0)))

    med = float(np.median(f0))
    q1, q3 = (float(v) for v in np.percentile(f0, [25, 75]))
    spread_st = float(12 * np.log2(q3 / max(q1, 1e-9)))

    # Mode structure, reported as DISCLOSURE — how many voices, and where. Never the verdict:
    # the lowest mode is usually the bass, not the lead (that mistake cost a false positive).
    lg = np.log2(f0)
    hist, edges = np.histogram(lg, bins=48)
    peaks = [i for i in range(1, len(hist) - 1)
             if hist[i] >= hist[i-1] and hist[i] >= hist[i+1] and hist[i] > 0.10 * hist.max()]
    lead_hz = float(2 ** ((edges[peaks[0]] + edges[peaks[0]+1]) / 2)) if peaks else med
    within = np.abs(12 * np.log2(f0 / lead_hz))
    lead_frac = float((within <= 3.0).mean())          # frames belonging to the lead
    oct_up = 12 * np.log2(f0 / lead_hz)
    oct_up_frac = float((np.abs(oct_up - 12.0) <= 0.5).mean())   # a second voice an octave above

    bands = {"sub_85": float((f0 < MALE_MIN).mean()),
             "male": float(((f0 >= MALE_MIN) & (f0 < MALE_MAX)).mean()),
             "overlap": float(((f0 >= MALE_MAX) & (f0 <= FEMALE_MIN)).mean()),
             "female": float((f0 > FEMALE_MIN).mean())}

    # CLASSIFY FROM THE MEDIAN, DISCLOSE FROM THE MODES.
    #
    # An earlier version of this classified from "the lowest well-populated mode", reasoning that
    # a male lead under a female choir would be the lower one. Measured on the corpus, that logic
    # labelled a KNOWN FEMALE take (median 197.4 Hz) as "male-lead" because it latched onto 88.9 Hz
    # of low-frequency bleed carrying just 9.1% of frames — a false positive on precisely the axis
    # this gate exists to guard. The lowest mode is not the lead; it is usually the bass.
    #
    # So the median does the classifying (it is robust, and it is what the corpus separates on),
    # and the mode structure is reported as DISCLOSURE — never as the verdict. A class must also
    # be SUPPORTED: if more voiced frames sit in the opposite band than in the claimed one, the
    # honest answer is "mixed", not a confident label.
    if med < MALE_MIN:
        reg = "sub-range"                 # no voice at all — 808 bleed, never "male"
    elif med < MALE_MAX:
        reg = "male" if bands["male"] >= bands["female"] else "mixed"
    elif med <= FEMALE_MIN:
        reg = "ambiguous"
    else:
        reg = "female" if bands["female"] >= bands["male"] else "mixed"
    return _row(register=reg, f0_hz=round(med, 1), q1_hz=round(q1, 1), q3_hz=round(q3, 1),
                spread_st=round(spread_st, 2), lead_hz=round(lead_hz, 1),
                lead_frac=round(lead_frac, 3), oct_up_frac=round(oct_up_frac, 3),
                voiced_s=round(len(f0) * 512 * hop_div / 44100, 1), frames=int(len(f0)),
                bands={k: round(v, 3) for k, v in bands.items()})


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


def continuity(path, drop_db=20.0, min_run_s=0.4, edge_s=3.0):
    """Does the arrangement ever STOP? Every other audio gate here is a whole-track scalar.

    Loudness, dynamic range, true peak, clipping and word accuracy are all satisfied by a track
    with holes punched in it — they average over the time axis, and a hole is a small part of an
    average. The record that shipped had SEVEN of them, 9.8 s in total, four inside the first
    eighteen seconds at roughly 4.7 s spacing, and it passed every gate the notebook had.

    Three sibling seeds showed holes at the same timestamps, so it is not seed noise: with no
    wordless section anywhere in the lyric, the model manufactured its own intro space by cutting
    the track. The rebuilt lyric gives it real instrumental blocks instead.

    A frame counts as dropped when it sits `drop_db` below the track's OWN 95th-percentile frame
    level — relative, so a quiet mix is not condemned for being quiet. Head and tail are ignored:
    a fade is not a dropout.
    """
    x, sr = load(path, mono=True)
    hop = max(1, int(0.046 * sr))
    nf = len(x) // hop
    if nf < 10:
        return {"dropouts": 0, "dropout_s": 0.0, "longest_dropout_s": 0.0,
                "longest_loud_run_s": round(len(x) / sr, 1), "at": []}
    rms = np.sqrt((x[:nf * hop].reshape(nf, hop) ** 2).mean(1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    quiet = db < (np.percentile(db, 95) - drop_db)
    inner = quiet.copy()
    e = min(nf // 2, int(edge_s * sr / hop))
    inner[:e] = False
    inner[nf - e:] = False
    runs, at, i = [], [], 0
    while i < nf:
        if inner[i]:
            j = i
            while j < nf and inner[j]:
                j += 1
            if (j - i) * hop / sr >= min_run_s:
                runs.append((j - i) * hop / sr)
                at.append(round(i * hop / sr, 2))
            i = j
        else:
            i += 1
    best = cur = 0
    for v in ~quiet:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {"dropouts": len(runs), "dropout_s": round(float(sum(runs)), 2),
            "longest_dropout_s": round(float(max(runs, default=0.0)), 2),
            "longest_loud_run_s": round(best * hop / sr, 1), "at": at[:10]}


# Calibrated against this pipeline's own corpus, with the headroom stated. The take under review
# scored 7 dropouts / 9.80 s / 10.4 s longest run and fails all three. The tightest PASSING record
# is the conditioning reference at 4.23 s of dropout (30% headroom) and STEEL v1 at an 18.9 s
# longest run (18% headroom) — a bar set nearer than that to a known-good record fires eventually.
CONTINUITY = {"dropout_s_max": 6.0, "dropouts_max": 5, "longest_loud_run_s_min": 16.0}


def continuity_verdict(c):
    out = []
    if c["dropout_s"] > CONTINUITY["dropout_s_max"]:
        out.append(f"the arrangement stops for {c['dropout_s']}s across {c['dropouts']} holes "
                   f"(at {c['at']}) — limit {CONTINUITY['dropout_s_max']}s")
    if c["dropouts"] > CONTINUITY["dropouts_max"]:
        out.append(f"{c['dropouts']} separate holes in the track — limit "
                   f"{CONTINUITY['dropouts_max']}")
    if c["longest_loud_run_s"] < CONTINUITY["longest_loud_run_s_min"]:
        out.append(f"never plays for more than {c['longest_loud_run_s']}s unbroken — floor "
                   f"{CONTINUITY['longest_loud_run_s_min']}s")
    return out


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
    print("\nPITCH — frames YIN could not voice must never reach the histogram")
    # Every value here is one f0_yin can actually emit: its parabolic step divides by a denominator
    # that can be ~0, giving +/-inf or a NEGATIVE frequency, and log2 of a negative is NaN.
    hostile = {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf, "negative": -180.0, "zero": 0.0}
    good_f0 = np.full(60, 150.0) + np.random.default_rng(3).normal(0, 2, 60)
    for name, bad in hostile.items():
        mixed = np.concatenate([good_f0, np.full(10, bad)])
        kept = finite_f0(mixed)
        try:
            np.histogram(np.log2(kept), bins=48)             # the call that detonated
            crashed = False
        except Exception as e:
            crashed = True
            print(f"   {name}: {type(e).__name__}: {str(e)[:60]}")
        good = (not crashed) and len(kept) == len(good_f0)
        ok &= good
        print(f"   {name:9s} 70 frames -> {len(kept):3d} kept, histogram {'ok' if not crashed else 'CRASHED'}"
              f"  {'ok' if good else 'FAIL'}")
    only_bad = finite_f0(np.full(60, np.nan))
    good = len(only_bad) == 0; ok &= good
    print(f"   all-unvoiced -> {len(only_bad)} kept (register reports 'unknown')  {'ok' if good else 'FAIL'}")
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
