# %% [markdown]
# # The vowel judge — does the singer say 'a' and 'o' where the text says so?
#
# Persian script leaves short vowels unwritten, and the operator heard the song model resolve
# every one of them to 'e' (چِکِش، کِمِک، خِرد، بالاتِر). A word-accuracy judge cannot see this:
# whisper transcribes the WORDS correctly however they are vowelled. This notebook listens at
# the level of PHONES instead — a CTC phoneme recogniser on the separated vocal stem — and
# aligns what it heard against the vowel sequence the fully vowel-marked lyric prescribes. It
# reports, per take, how often an expected 'a' or 'o' came back as 'e'. Free CPU tier; no
# generation. Validated first on the audition the operator judged by ear: if it does not
# reproduce "e everywhere" there, it is not an instrument.

# %%
import json, os, re, subprocess, sys, time, shlex, tempfile
from pathlib import Path
T0 = time.time()
def sh(c, quiet=False): subprocess.run(c, shell=True, check=True,
    stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
def clock(w): print(f"  ⏱ {w} · t+{(time.time()-T0)/60:.1f} min", flush=True)
PINS = {
    "phone_model": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",   # CTC phone recogniser, espeak phone set, 60+ languages incl. fa
    "measure_sha": "199535aa517324d8021667b5a34a799aedd19353",
    "lyric_sha": "fd562d46256d69afb92e4c0f32744e7b9407ba8e",
}
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable
sh(f"{PY} -m pip install -q 'demucs>=4.0.1' pyloudnorm soundfile 'transformers>=4.51.0,<4.58.0' phonemizer 2>&1 | tail -1")
import urllib.request
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['measure_sha']}/lib/measure.py", "/tmp/measure.py")
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{PINS['lyric_sha']}/song/lyrics_foolad_fa.v3.vocalized.txt", "/tmp/voc.txt")
sys.path.insert(0, "/tmp")
import numpy as np, torch, soundfile as sf
import measure as M
VOC = Path("/tmp/voc.txt").read_text(encoding="utf-8")
TAKES = sorted(p for p in Path("/kaggle/input").rglob("*.mp3") if p.parent.name == "out")
assert TAKES, "no takes mounted (kernel source ashranet/foolad-audition)"
print(f"{len(TAKES)} takes:", [t.name for t in TAKES], flush=True)
clock("inputs")

# %%
# EXPECTED VOWELS from the vowel-marked text. A small rule-based reading of Persian letters:
# harakat give the short vowels; آ and medial ا give ā; ی gives i, و gives u (they are also
# consonants y/v — the ambiguity costs a few alignment errors, never the e/a/o signal this is
# for); a word-final ه after a consonant is the colloquial '-e'. Everything else is a consonant.
HARAKAT = {"َ": "a", "ِ": "e", "ُ": "o"}
def expected_vowels(text):
    out = []
    for line in text.splitlines():
        if line.startswith("[") or not line.strip(): continue
        for word in re.split(r"[\s‌]+", line):
            word = re.sub(r"[،؟!\.:,\-—–\"'«»()]", "", word)
            if not word: continue
            chars = list(word)
            for i, ch in enumerate(chars):
                if ch in HARAKAT: out.append(HARAKAT[ch])
                elif ch == "آ": out.append("A")
                elif ch == "ا" and i > 0: out.append("A")
                elif ch == "ی" and i > 0: out.append("i")
                elif ch == "و" and i > 0: out.append("u")
                elif ch == "ه" and i == len(chars) - 1 and i > 0 and chars[i-1] not in HARAKAT and chars[i-1] not in "اآوی":
                    out.append("e")
    return out
EXP = expected_vowels(VOC)
from collections import Counter
print("expected short-vowel mix:", dict(Counter(v for v in EXP if v in "aeo")), "· long:", dict(Counter(v for v in EXP if v in "Aiu")), flush=True)

# %%
# HEARD VOWELS: the phone recogniser on the separated stem, 20 s windows, greedy CTC decode.
# No tokenizer class at all: on Kaggle's transformers the phoneme tokenizer for this model loads
# as `False` (and the combined Processor then complains of "a bool for argument tokenizer"). Ids
# become phone labels through the model's own vocab.json and a greedy CTC decode — collapse
# repeats, drop the blank — which is all the tokenizer would have done.
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
fe = Wav2Vec2FeatureExtractor.from_pretrained(PINS["phone_model"])
model = Wav2Vec2ForCTC.from_pretrained(PINS["phone_model"]).eval()
VOCAB = json.loads(Path(hf_hub_download(PINS["phone_model"], "vocab.json")).read_text(encoding="utf-8"))
ID2PHONE = {v: k for k, v in VOCAB.items()}
SPECIAL = {i for t, i in VOCAB.items() if t.startswith("<") or t in ("|",)}
BLANK = VOCAB.get("<pad>", 0)
def ctc_decode(ids):
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK and i not in SPECIAL: out.append(ID2PHONE[i])
        prev = i
    return out
print(f"phone vocab {len(VOCAB)} · blank id {BLANK}", flush=True)
VOWEL_CLASS = [("ɒː", "A"), ("ɑː", "A"), ("aː", "A"), ("ɒ", "A"), ("ɑ", "A"),
               ("iː", "i"), ("uː", "u"), ("ɛ", "e"), ("e", "e"), ("æ", "a"), ("a", "a"), ("ɔ", "o"), ("o", "o"), ("i", "i"), ("u", "u")]
def heard_vowels(stem):
    x, sr = sf.read(str(stem), dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != 16000:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy(); sr = 16000
    phones = []
    win = 20 * sr
    for s0 in range(0, len(x), win):
        seg = x[s0:s0 + win]
        if len(seg) < sr: continue
        with torch.no_grad():
            logits = model(**fe(seg, sampling_rate=sr, return_tensors="pt")).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        phones += ctc_decode(ids)
    vowels = []
    for p in phones:
        for pref, cls in VOWEL_CLASS:          # not `tok`: that name is the tokenizer above
            if p.startswith(pref): vowels.append(cls); break
    return phones, vowels

def align_counts(exp, got):
    """Edit-distance alignment on vowel classes; count substitutions among the short vowels."""
    n, m = len(exp), len(got)
    d = np.zeros((n+1, m+1), dtype=np.int32); d[:, 0] = np.arange(n+1); d[0, :] = np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (exp[i-1] != got[j-1]))
    i, j, conf = n, m, Counter()
    while i > 0 and j > 0:
        if d[i, j] == d[i-1, j-1] + (exp[i-1] != got[j-1]):
            conf[(exp[i-1], got[j-1])] += 1; i -= 1; j -= 1
        elif d[i, j] == d[i-1, j] + 1: i -= 1
        else: j -= 1
    return conf

def vocal_stem(mp3):
    import demucs.separate
    L = M.loudness(str(mp3)); g = -14.0 - (L.get("lufs") if L.get("lufs") is not None else -14.0)
    norm = Path(tempfile.mkdtemp()) / "norm.wav"
    sh(f"ffmpeg -v error -i '{mp3}' -af volume={g:.2f}dB -ar 44100 '{norm}' -y", quiet=True)
    td = tempfile.mkdtemp()
    demucs.separate.main(shlex.split(f'--two-stems vocals -n htdemucs --shifts 0 --device cpu -o "{td}" "{norm}"'))
    return next(Path(td).rglob("vocals.wav"))

report = []
for take in TAKES:
    t1 = time.time()
    stem = vocal_stem(take)
    phones, got = heard_vowels(stem)
    conf = align_counts(EXP, got)
    def rate(x, y):
        tot = sum(c for (a, b), c in conf.items() if a == x)
        return round(sum(c for (a, b), c in conf.items() if a == x and b == y) / tot, 3) if tot else None
    row = {"take": take.stem, "phones": len(phones), "heard_short_mix": dict(Counter(v for v in got if v in "aeo")),
           "a_as_a": rate("a", "a"), "a_as_e": rate("a", "e"), "o_as_o": rate("o", "o"), "o_as_e": rate("o", "e"),
           "e_as_e": rate("e", "e"), "A_as_A": rate("A", "A"), "seconds": round(time.time() - t1)}
    report.append(row)
    (WORK / "vowel_judge.json").write_text(json.dumps({"expected_mix": dict(Counter(EXP)), "takes": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {take.stem}: a→a {row['a_as_a']} · a→e {row['a_as_e']} · o→o {row['o_as_o']} · o→e {row['o_as_e']} · e→e {row['e_as_e']} · ā→ā {row['A_as_A']} · {row['seconds']}s", flush=True)
    clock(f"{take.stem} judged")
# %%
# WORD-LEVEL PROBE. An aggregate over an 'a'-heavy text can pass while the words the operator
# named fail. Each target word's expected phones (from its vowel-marked spelling) are located in
# the heard phone stream by local alignment, and the vowels the singer produced at those slots
# are read out — per occurrence, per take.
CONS = {"چ": "tS", "ک": "k", "ش": "S", "م": "m", "خ": "x", "ر": "r", "د": "d", "ب": "b", "ل": "l",
        "ت": "t", "ن": "n", "پ": "p", "ح": "h", "ه": "h", "ق": "q", "غ": "q", "ز": "z", "س": "s",
        "ج": "dZ", "گ": "g", "ف": "f", "ع": "", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ژ": "Z",
        "ث": "s", "ذ": "z", "ی": "j", "و": "v"}
def word_phones(word):
    """Expected phone classes for one vowel-marked word: consonants by class, vowels a/e/o/A/i/u."""
    out, chars = [], list(word)
    for i, ch in enumerate(chars):
        if ch in HARAKAT: out.append(HARAKAT[ch])
        elif ch == "آ": out.append("A")
        elif ch == "ا": out.append("A") if i > 0 else None
        elif ch == "ی" and i > 0 and (i == len(chars)-1 or chars[i+1] not in HARAKAT): out.append("i")
        elif ch == "و" and i > 0 and (i == len(chars)-1 or chars[i+1] not in HARAKAT): out.append("u")
        elif ch in CONS and CONS[ch]: out.append(CONS[ch])
    return out
PHONE_CLASS = [("tʃ", "tS"), ("dʒ", "dZ"), ("ʃ", "S"), ("ʒ", "Z"), ("ɾ", "r"), ("r", "r"), ("ɢ", "q"), ("q", "q"),
               ("ɣ", "q"), ("ɡ", "g"), ("g", "g"), ("x", "x"), ("χ", "x"), ("h", "h"), ("ħ", "h"), ("k", "k"),
               ("m", "m"), ("n", "n"), ("d", "d"), ("b", "b"), ("l", "l"), ("t", "t"), ("p", "p"), ("z", "z"),
               ("s", "s"), ("f", "f"), ("v", "v"), ("w", "v"), ("j", "j")] + VOWEL_CLASS
def phone_class(p):
    for pref, cls in PHONE_CLASS:
        if p.startswith(pref): return cls
    return None
def locate(expected, heard_cls):
    """Smith-Waterman on phone classes; returns the heard classes aligned to each expected slot."""
    n, m = len(expected), len(heard_cls)
    H = np.zeros((n+1, m+1), dtype=np.int32); best, bi, bj = 0, 0, 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            sc = 2 if expected[i-1] == heard_cls[j-1] else -1
            H[i, j] = max(0, H[i-1, j-1] + sc, H[i-1, j] - 1, H[i, j-1] - 1)
            if H[i, j] > best: best, bi, bj = H[i, j], i, j
    if best < 2 * max(2, n // 2): return None, best
    slots = ["-"] * n; i, j = bi, bj
    while i > 0 and j > 0 and H[i, j] > 0:
        sc = 2 if expected[i-1] == heard_cls[j-1] else -1
        if H[i, j] == H[i-1, j-1] + sc: slots[i-1] = heard_cls[j-1]; i -= 1; j -= 1
        elif H[i, j] == H[i-1, j] - 1: i -= 1
        else: j -= 1
    return slots, best
TARGETS = ["چَکُش", "کُمَک", "خُرد", "بالاتَر", "شِیپور", "بُلَند", "مُحکَم", "دُنیا", "زَنگ", "فولادَم"]
word_report = {}
for take in TAKES:
    stem = vocal_stem(take)
    phones, _ = heard_vowels(stem)
    heard_cls = [c for c in (phone_class(p) for p in phones) if c]
    per = {}
    for w in TARGETS:
        exp = word_phones(w)
        vslots = [k for k, c in enumerate(exp) if c in ("a", "e", "o", "A", "i", "u")]
        found = []
        stream = list(heard_cls)
        for _ in range(4):                       # up to four occurrences per take
            slots, score = locate(exp, stream)
            if slots is None: break
            found.append("".join(slots[k] if slots[k] in "aeoAiu" else "?" for k in vslots))
            # blank out the located window so the next search finds another occurrence
            idx = next((j for j in range(len(stream) - len(exp) + 1)
                        if sum(1 for a_, b_ in zip(exp, stream[j:j+len(exp)]) if a_ == b_) >= max(2, len(exp)//2)), None)
            if idx is None: break
            stream = stream[:idx] + ["#"] * len(exp) + stream[idx+len(exp):]
        per[w] = {"expected_vowels": "".join(exp[k] for k in vslots), "heard": found}
    word_report[take.stem] = per
    print(f"  {take.stem}: " + " · ".join(f"{w} {v['expected_vowels']}→{','.join(v['heard']) or '(not found)'}" for w, v in per.items()), flush=True)
    clock(f"{take.stem} words")
(WORK / "vowel_words.json").write_text(json.dumps(word_report, ensure_ascii=False, indent=1), encoding="utf-8")

print("VOWELS:", json.dumps(report, ensure_ascii=False), flush=True)
clock("DONE")
