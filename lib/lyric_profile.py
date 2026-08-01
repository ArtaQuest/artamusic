#!/usr/bin/env python3
"""Score a lyric against the MEASURED craft profile of the reference record.

"Make it more like Glum Aleks" is not a brief a writer can act on or a claim anyone can check.
So the reference ("Typical Story", 60 lines, 359 words) was measured, and these are its numbers:

    syllables per line   7.1 mean; 87% of lines in the 6-8 band
    monosyllable rate    71%
    imperative openings  35% of lines begin with a command verb
    mean word length     4.13 characters

The imperative rate is the signature nobody names: one line in three opens on Pick / Bring /
Give / Draw / Make / Cue / Sell / Raise, which is what makes the song direct its own film instead
of narrating a diary. Early drafts here measured 9-12% and read as literary; the shipped lyric
measures ~34.5%.

Rebuilt 2026-08-01 after the original was lost with a temp directory — which is why this copy
lives in the repo. Validates to ~0.0 error on the reference's own text.

  python lyric_profile.py <lyric.txt> [more.txt ...]
"""
import re
import sys
from pathlib import Path

VOWELS = re.compile(r"[aeiouy]+")

IMPERATIVES = set("""
pick put give bring draw make kick sell cue raise break let go count hold ask say wear learn
take name call keep watch run stand walk lift carry burn cut close open leave stay come
show tell send pull push drop find start begin turn set write read pay bury dig swing drag throw
strike beat hit pour hand
""".split())

TARGET = {"syllables": 7.1, "tight_pct": 87.0, "mono_pct": 71.0,
          "imper_pct": 35.0, "word_len": 4.13}

# UNBROKEN's locked forge spine: verse 1, verse 2, bridge and outro each close on their
# "call it …" station, in exactly this order. The 4-line outro is that spine's final
# station, not a truncation — see README. Enforced below, not documented and forgotten.
SPINE = ["call it heat", "call it hammer", "call it quench", "call it edge"]


def syllables(line):
    n = 0
    for w in re.findall(r"[a-z']+", line.lower()):
        c = len(VOWELS.findall(w))
        if w.endswith("e") and c > 1:
            c -= 1
        n += max(1, c)
    return n


def measure(text):
    first = re.search(r"^\s*\[", text, re.M)          # strip any prose header
    if first:
        text = text[first.start():]
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("[")
             and not l.strip().startswith("(")]        # "(identical repeat)" markers
    if not lines:
        return None
    syl = [syllables(l) for l in lines]
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    mono = sum(1 for w in words if len(VOWELS.findall(w)) <= 1)
    imper = sum(1 for l in lines
                if (re.findall(r"[a-z']+", l.lower()) or [""])[0] in IMPERATIVES)
    return {"lines": len(lines), "words": len(words),
            "syllables": sum(syl) / len(syl),
            "tight_pct": 100 * sum(1 for s in syl if 6 <= s <= 8) / len(syl),
            "mono_pct": 100 * mono / len(words),
            "imper_pct": 100 * imper / len(lines),
            "word_len": sum(map(len, words)) / len(words),
            "long_lines": [(l, s) for l, s in zip(lines, syl) if s > 8],
            "short_lines": [(l, s) for l, s in zip(lines, syl) if s < 6]}


def check_invariants(text, name=""):
    """Structural invariants any lyric edit must not break. Returns a list of violations.

    1. Every [chorus] block is word-identical (a singability rule: the machine-transcription
       failures all came from near-identical variants re-voted into something else).
    2. For the UNBROKEN lyric (detected by its spine): the last "call it …" line of each
       verse/bridge/outro section walks SPINE in order — heat, hammer, quench, edge.
    """
    secs, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("("):
            continue
        if s.startswith("["):
            cur = (s.strip("[]").strip().lower(), [])
            secs.append(cur)
        elif cur is not None:
            cur[1].append(" ".join(re.findall(r"[a-z']+", s.lower())))
    problems = []
    choruses = ["\n".join(lines) for tag, lines in secs if tag == "chorus"]
    if len(choruses) > 1 and len(set(choruses)) != 1:
        problems.append(f"chorus blocks are NOT word-identical ({len(choruses)} blocks)")
    if SPINE[0] in text.lower() or "unbroken" in name.lower():
        stations = []
        for tag, lines in secs:
            if tag in ("verse", "bridge", "outro"):
                hits = [m for line in lines for m in re.findall(r"call it [a-z']+", line)]
                if hits:
                    stations.append(hits[-1])
        if stations != SPINE:
            problems.append(f"forge spine broken: {stations} != {SPINE}")
    return problems


def report(name, m):
    def gap(key, tol):
        d = m[key] - TARGET[key]
        flag = "" if abs(d) <= tol else "  <-- off target"
        return f"{m[key]:6.1f} ({d:+5.1f}){flag}"
    print(f"\n{name}   {m['lines']} lines, {m['words']} words")
    print(f"  syllables/line     {gap('syllables', 0.6)}")
    print(f"  6-8 syllable band  {gap('tight_pct', 12)}%")
    print(f"  monosyllables      {gap('mono_pct', 6)}%")
    print(f"  imperative opens   {gap('imper_pct', 8)}%")
    for l, s in m["long_lines"]:
        print(f"    long  ({s}): {l}")
    for l, s in m["short_lines"]:
        print(f"    short ({s}): {l}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bad = False
    for p in sys.argv[1:]:
        text = Path(p).read_text()
        m = measure(text)
        report(Path(p).name, m) if m else print(f"{p}: no lyric lines")
        for problem in check_invariants(text, Path(p).name):
            print(f"  INVARIANT BROKEN: {problem}")
            bad = True
    if bad:
        sys.exit(1)
