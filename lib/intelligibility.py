#!/usr/bin/env python3
"""Word error rate of a sung take against the lyric it was given.

"Not audible" was one of the two rejections this project received, and it is measurable: Whisper
transcribes the take, and the transcript is aligned against the lyric that was actually sent to
the model. The previous master shipped at 88.1% word accuracy; the two mishearings that mattered
("I got able" -> "I got eight", "I quit the count" -> "I cannot quit the count") were both found
this way, not by listening.

The reference text is sliced from its first [section] marker — a 158-word prose header once sat
in front of a 326-word lyric and silently dropped four takes from ~88% to ~57%.

  python intelligibility.py --lyric L.txt takes...
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

WORD = re.compile(r"[a-z']+")


def lyric_words(path):
    t = Path(path).read_text()
    m = re.search(r"^\s*\[", t, re.M)
    if m:
        t = t[m.start():]
    t = re.sub(r"\[[^\]]*\]", " ", t.lower())
    t = t.replace("(identical repeat)", " ")
    return WORD.findall(t)


def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return float(d[-1, -1]) / max(1, len(ref))


_MODEL = [None]


def transcribe(path):
    from faster_whisper import WhisperModel
    if _MODEL[0] is None:
        _MODEL[0] = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = _MODEL[0].transcribe(str(path), beam_size=5, vad_filter=False,
                                   condition_on_previous_text=False)
    return " ".join(s.text for s in segs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyric", required=True)
    ap.add_argument("takes", nargs="+")
    a = ap.parse_args()
    ref = lyric_words(a.lyric)
    print(f"lyric: {len(ref)} sung words")
    for t in a.takes:
        text = transcribe(t)
        w = wer(ref, WORD.findall(text.lower()))
        print(f"  {Path(t).name:12s} WER {w:.3f}  intelligibility {100*(1-min(w,1.0)):.1f}%")
