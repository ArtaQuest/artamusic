#!/usr/bin/env python3
"""Score a lyric against the track it has to fit in, BEFORE seven hours are spent singing it.

The STEEL record passed every gate the notebook had — ASR word accuracy 87.2%, loudness, true
peak, clipping, and a lyric craft profile scoring monosyllables and concrete nouns — and the
operator's verdict was still "the song is messed up". Every one of those gates measures the words
or the mastering. None of them measures whether the words FIT THE CLOCK.

They did not. 414 words over 180 s, of which 135.2 s was voiced, is **3.06 words per second of
singing, sustained for three minutes**. That is rap density on a 100 BPM anthem, and it is why the
take sounds crammed: the model has no room to hold a note, so it does not.

The measurements that matter, none of which existed before:

  * `words_per_voiced_second` — the real density. Heavy rock/anthem sits at 1.2–1.8. Computed
    against an ESTIMATED voiced fraction when there is no audio yet, and against the measured
    one when there is.
  * `voiced_fraction` — how much of the track is singing at all. A song with no instrumental
    space is exhausting; 55–65% is ordinary, 75% is what this take did.
  * `words_per_bar` — the same thing in the units a writer actually works in.
  * `section_count` — nine blocks in 180 s is twenty seconds each, which is not a structure, it
    is a list.

`verdict()` returns the reasons to refuse, or [] — the same shape as `stillness.verdict`, so it
drops into the record's existing gate machinery.
"""
import re
import sys

# A lyric line's singable payload: the words, not the stage directions.
_TAG = re.compile(r"^\s*\[[^\]]+\]\s*$")
_PAREN = re.compile(r"\(([^)]*)\)")

LIMIT = {
    # Anthems live between these. Below 0.9 the lyric is too thin to carry three minutes; above
    # 1.9 the singer is running. The shipped take scored 3.06.
    "words_per_voiced_second_max": 1.9,
    "words_per_voiced_second_min": 0.9,
    # 75% voiced was the shipped take. Every arrangement needs somewhere to breathe.
    "voiced_fraction_max": 0.70,
    # Nine blocks in 180 s. Six is a structure; ten is a list.
    "section_count_max": 8,
}

# When there is no audio yet, voiced time has to be estimated. This is deliberately GENEROUS —
# it assumes a well-arranged 62% voiced — so the gate only fires when the lyric is too long even
# for a track that gives it as much room as a track sensibly can.
ASSUMED_VOICED_FRACTION = 0.62


def sections(lyric):
    """[(tag, [line, ...]), ...] in order, so structure can be counted rather than eyeballed."""
    out, tag, buf = [], None, []
    for raw in lyric.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _TAG.match(line):
            if tag is not None or buf:
                out.append((tag, buf))
            tag, buf = line.strip("[]").strip().lower(), []
        else:
            buf.append(line)
    if tag is not None or buf:
        out.append((tag, buf))
    return [(t, b) for t, b in out if b]


def words(lyric, keep_parentheticals=True):
    """Count what gets SUNG.

    A parenthetical is sung — `(Strike!)` is a shout the choir performs, not a note to the
    producer — so it counts by default. Counting it as free is how a lyric with sixteen of them
    reads as shorter than it is.
    """
    body = "\n".join(l for t, b in sections(lyric) for l in b) or lyric
    if not keep_parentheticals:
        body = _PAREN.sub(" ", body)
    return len(re.findall(r"[A-Za-z0-9']+", body))


def measure(lyric, duration_s, bpm, voiced_seconds=None, beats_per_bar=4):
    n = words(lyric)
    secs = sections(lyric)
    voiced = voiced_seconds if voiced_seconds else duration_s * ASSUMED_VOICED_FRACTION
    bars = duration_s / 60.0 * bpm / beats_per_bar
    return {
        "words": n,
        "lines": sum(len(b) for _, b in secs),
        "sections": len(secs),
        "section_tags": [t for t, _ in secs],
        "duration_s": round(duration_s, 1),
        "bars": round(bars, 1),
        "voiced_seconds": round(voiced, 1),
        "voiced_measured": voiced_seconds is not None,
        "voiced_fraction": round(voiced / duration_s, 3),
        "words_per_voiced_second": round(n / max(voiced, 1e-6), 2),
        "words_per_bar": round(n / max(bars, 1e-6), 2),
        "words_per_section": round(n / max(len(secs), 1), 1),
    }


def verdict(m):
    """The reasons to refuse. Empty means the lyric fits the clock."""
    out = []
    d = m["words_per_voiced_second"]
    if d > LIMIT["words_per_voiced_second_max"]:
        out.append(f"{m['words']} words is too many for {m['voiced_seconds']}s of singing — "
                   f"{d} words a second, and an anthem sits under "
                   f"{LIMIT['words_per_voiced_second_max']}")
    if d < LIMIT["words_per_voiced_second_min"]:
        out.append(f"{m['words']} words is too few to carry {m['duration_s']}s — {d} words a second")
    if m["voiced_measured"] and m["voiced_fraction"] > LIMIT["voiced_fraction_max"]:
        out.append(f"{int(m['voiced_fraction'] * 100)}% of the track is vocal — there is nowhere "
                   f"to breathe (limit {int(LIMIT['voiced_fraction_max'] * 100)}%)")
    if m["sections"] > LIMIT["section_count_max"]:
        out.append(f"{m['sections']} sections in {m['duration_s']}s is "
                   f"{m['duration_s'] / m['sections']:.0f}s each — that is a list, not a structure")
    return out


def contrast(path, block_s=10.0, sr=22050):
    """Does the track CHANGE? Returns the mean spectral similarity between blocks.

    The shipped take scored 0.883 — every ten seconds sounded like every other ten seconds, so the
    chorus never lifted and the bridge never dropped. No gate in the record could see this: word
    accuracy, loudness, true peak and clipping are all satisfied by a monotonous song. A wall of
    sound passes every one of them.

    Measured on log-mel-ish magnitude spectra averaged per block, L2-normalised so the number is a
    cosine similarity and loudness differences do not masquerade as contrast.
    """
    import subprocess
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if len(x) < sr * block_s * 2:
        return {"blocks": 0, "similarity": None, "voiced_fraction": None}
    hop = sr // 10
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop) * np.hanning(hop)
    F = np.abs(np.fft.rfft(fr, axis=1))
    # voiced fraction, cheaply: frames whose energy is within 12 dB of the median are "sounding",
    # which for a dense mix is a fair proxy for how much of the track is not space.
    e = 20 * np.log10(np.maximum(np.sqrt((fr ** 2).mean(1)), 1e-6))
    loud = float((e > np.median(e) - 12).mean())
    B = int(block_s * 10)
    nb = n // B
    M = F[:nb * B].reshape(nb, B, -1).mean(1)
    M = np.log1p(M)
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    sim = M @ M.T
    off = sim[np.triu_indices(nb, 1)]
    return {"blocks": int(nb), "similarity": round(float(off.mean()), 3),
            "similarity_min": round(float(off.min()), 3), "loud_fraction": round(loud, 3)}


# CONTRAST IS REPORTED, NOT GATED — and the control is why.
#
# The shipped take scores 0.96 block-to-block similarity, which reads damning until something
# known-good is measured beside it. A synthetic track built with six obviously different sections
# — quiet verse, loud chorus, breakdown, three octaves of spectral difference — scores 0.92, and
# on the two statistics that actually track structure it does WORSE than the take it was meant to
# condemn: mean-centred minimum similarity -0.943 against STEEL's -0.948, and a block loudness
# spread of 18.1 dB against STEEL's 26.1 dB.
#
# So a threshold tight enough to refuse STEEL also refuses a song with real sections, and that is
# a gate that fails everything — worthless in the specific way a gate can be worse than nothing.
# Calibrating it honestly needs a corpus of real music that passes, which we do not have here.
# The numbers are still worth printing: they are evidence for a person, just not a verdict.
CONTRAST_MAX = None


def contrast_verdict(c):
    """Deliberately never refuses. See the note above: uncalibrated, so reported only."""
    return []


def selftest():
    """The shipped take must FAIL, and a lyric that fits must PASS — or the gate is decorative."""
    ok = True
    shipped_words = 414
    shipped = "[intro]\n" + "\n".join(["the hammer falls upon the iron ring"] * 58) + "\n"
    shipped += "\n".join(f"[verse]\nline {i}" for i in range(8))
    m = measure(shipped, 180.0, 100, voiced_seconds=135.2)
    v = verdict(m)
    hit = any("words a second" in x for x in v)
    ok &= hit
    print(f"   a 400+ word lyric at 135s voiced  {m['words_per_voiced_second']} w/s -> "
          f"{'REFUSED' if hit else 'PASSED — GATE IS BLIND'}")

    # ...and the real shipped numbers, stated directly, must refuse too.
    m2 = {"words": shipped_words, "voiced_seconds": 135.2, "duration_s": 180.0,
          "words_per_voiced_second": round(shipped_words / 135.2, 2), "voiced_fraction": 0.751,
          "voiced_measured": True, "sections": 9}
    v2 = verdict(m2)
    ok &= len(v2) >= 3
    print(f"   the SHIPPED take ({m2['words_per_voiced_second']} w/s, 75% voiced, 9 sections)")
    for x in v2:
        print(f"      refused: {x}")

    good = "\n".join([
        "[intro]", "Made in the fire.", "Beaten on the stone.",
        "[verse]", "Strike, and the hammer falls.", "Strike, and the iron rings.",
        "The coals go white, the smoke goes grey.", "Each blow takes the rust away.",
        "[chorus]", "I am the weight in your hand.", "I am the edge that holds.",
        "I was made inside the fire.", "I don't break in the cold.",
        "[verse]", "Strike, when the horn calls out.", "Strike, when the fear comes in.",
        "A hand can shake, a heart can bend.", "I hold on to the very end.",
        "[bridge]", "Hard steel breaks.", "Soft steel bends.",
        "[chorus]", "I am the weight in your hand.", "I am the edge that holds.",
        "I was made inside the fire.", "I don't break in the cold.",
        "[outro]", "The fire is out. The smith is gone.", "I am still standing here.",
    ])
    mg = measure(good, 180.0, 100)
    vg = verdict(mg)
    fits = not vg
    ok &= fits
    print(f"   a lyric written to the clock     {mg['words']} words, "
          f"{mg['words_per_voiced_second']} w/s, {mg['sections']} sections -> "
          f"{'PASSED' if fits else 'REFUSED: ' + '; '.join(vg)}")

    # A parenthetical is SUNG. Counting it as free is how sixteen shouts hide from the gate.
    a = words("[verse]\n(Strike!) and the hammer falls.")
    b = words("[verse]\n(Strike!) and the hammer falls.", keep_parentheticals=False)
    par = a == 5 and b == 4 and a > b
    ok &= par
    print(f"   a (Strike!) shout counts as sung  {a} vs {b} -> {'ok' if par else 'FAIL'}")

    print("   " + ("ALL PASSED" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(0 if selftest() else 1)
    import json
    lyric = open(sys.argv[1]).read()
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
    bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    voiced = float(sys.argv[4]) if len(sys.argv) > 4 else None
    m = measure(lyric, dur, bpm, voiced_seconds=voiced)
    print(json.dumps(m, indent=1))
    for x in verdict(m):
        print("REFUSED:", x)
