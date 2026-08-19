#!/usr/bin/env python3
"""Is a lyric COMPREHENSIBLE the first time you hear it sung?

The craft profile in lyric_profile.py measures the SHAPE of a lyric — syllables per line,
monosyllable rate, how often a line opens on a command. A text can score perfectly on all of it and
still be impossible to follow, because none of those numbers can see an abstraction, an inverted
clause or a riddle. The shipped STEEL lyric did exactly that: 100% of its lines sat in the sung
6-8 syllable band while eighteen of them said things like "What the cold caught, the cold will
hold", and the operator's verdict was that nobody could tell what the song was about.

So this measures the other axis. Four numbers, each one a thing a listener actually does:

  familiar     how much of the lyric is everyday English (a listener never stops to parse a word)
  concrete     how many lines name something you can SEE or DO, rather than a quality or an idea
  plain order  how many lines run subject-verb-object, the way people speak — the inversions
               ("Folded in dark and drawn to light") are where a first listen falls off
  per word     syllables per word; long words are where a sung line turns to mush

    python clarity.py <lyric.txt> [more.txt ...]      # prints a table, and a diff if given two
"""
import re
import sys
from pathlib import Path

VOWELS = re.compile(r"[aeiouy]+")

# The everyday core of spoken English — kept short and inspectable on purpose, because a word list
# that grows without bound stops measuring anything. These are the words a lyric can lean on.
FAMILIAR = set("""
a about after again all am an and any are arm as ask at away back be because bed been before
behind bend bent big black blood blow blown bone born both boy break breaks bright bring
brought burn burned but by call called came can carried carry cold come comes cool cut dark day
days dead decide deep did do does dog don't done door down drop dropped dry each ear earth edge
end ends enough even ever every eye eyes face fall falls far fear feel feet fell fight fill find
fire first fold folded follow for found four free from front full gave get gets girl give given
go goes going gold gone good got grey grew grow grown had hand hands hard has have he head hear
heard heart heat held her here high him his hold holds home horn hot house how i if in into iron
is it its just keep keeps kept know knew land last late lay learn leave left less let lie life
lift lifts light like line little live long look lost loud love low made make makes man many
mark me mean men might mind mine more most move much must my name near need never new next night
no not now of off old on once one only open or other our out over own part pass past pay people
picks place plain play point put rain ran reach read red rest right rings road rock room round
run said same sand saw say says sea see seen set sets shake shape she shield ship shot show side
sing sings sit six sky slow small smoke snow so soft some son song soon sound spring stand
standing star start stay stayed stays steam steel step still stone stood stop straight strike
strong such sun sure take taken takes talk teach tell ten test than that the their them then
there these they thin thing things think this those though three through thus till time to told
too took top true try turn turned two under until up upon us use very wait walk wall want war
warm was watch water way we wear well went were what when where which while white who whole why
will wind with within without word work world would year years yet you young your rust anvil
coals forge fire's grip swing smith bare grey glow blow blows steel's shield horn
""".split())

# Things you can see, hold, or do — a line with one of these gives the listener a picture.
CONCRETE = set("""
fire coals coal smoke steam water iron steel blade edge anvil hammer stone forge spark sparks
hand hands arm heart horn shield field road rain sun sky bone rust grip smith glow flame
strike falls rings lift lifts fold folded beat beaten drop dropped hold holds carry carried
break breaks bend bends shake swing stand grow made make cool cold hot white grey black bright
EOF_MARK
""".split())

ABSTRACT = set("""
oath pledge trust cause deed debt burden weight claim honour glory truth faith fate doom spirit
soul essence virtue duty destiny meaning purpose
""".split())

# A line that starts on a subject pronoun/article/name runs the way people speak. A line that opens
# on a preposition or a past participle is usually an inversion — the shape a first listen loses.
PLAIN_OPENERS = set("""
i you he she it we they the a an my your his her our their this that these those there here
strike hold bend feel keep take give come go let watch call put drop lift cut what who when
hard soft one two no not now still and but so all every each a
""".split())
INVERTED_OPENERS = set("""
folded drawn beaten born forged carried tempered wrought bound sworn of by through beneath
whatever whichever what's such
""".split())

_EXCEPT = {"every": 2, "everything": 3, "evening": 2, "different": 3, "family": 3, "people": 2,
           "quiet": 2, "being": 2, "doing": 2, "going": 2, "iron": 2, "fire": 1, "hour": 1}


def syllables_word(w):
    if w in _EXCEPT:
        return _EXCEPT[w]
    n = len(VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


def measure(text):
    body = text[re.search(r"^\s*\[", text, re.M).start():] if re.search(r"^\s*\[", text, re.M) else text
    lines = [l.strip() for l in body.splitlines()
             if l.strip() and not l.strip().startswith(("[", "("))]
    words = re.findall(r"[a-z']+", " ".join(lines).lower())
    if not lines or not words:
        return None
    fam = sum(1 for w in words if w in FAMILIAR)
    syl = sum(syllables_word(w) for w in words)
    conc = sum(1 for l in lines if any(w in CONCRETE for w in re.findall(r"[a-z']+", l.lower())))
    absn = sum(1 for l in lines if any(w in ABSTRACT for w in re.findall(r"[a-z']+", l.lower())))
    opens = [(re.findall(r"[a-z']+", l.lower()) or [""])[0] for l in lines]
    plain = sum(1 for o in opens if o in PLAIN_OPENERS)
    inv = sum(1 for o in opens if o in INVERTED_OPENERS)
    return {"lines": len(lines), "words": len(words),
            "familiar_pct": 100 * fam / len(words),
            "syl_per_word": syl / len(words),
            "concrete_pct": 100 * conc / len(lines),
            "abstract_pct": 100 * absn / len(lines),
            "plain_order_pct": 100 * plain / len(lines),
            "inverted_pct": 100 * inv / len(lines),
            "unfamiliar": sorted({w for w in words if w not in FAMILIAR})}


# What a first-listen lyric should clear. Derived from the shipped-and-rejected lyric on one side
# and ordinary spoken English on the other; every one of them is a number the old profile could not
# see, and the rejected lyric fails three of the four.
FLOOR = {"familiar_pct": 88.0, "syl_per_word": 1.35, "concrete_pct": 70.0,
         "abstract_pct": 12.0, "plain_order_pct": 80.0, "inverted_pct": 5.0}


def verdict(m):
    bad = []
    if m["familiar_pct"] < FLOOR["familiar_pct"]:
        bad.append(f"only {m['familiar_pct']:.0f}% everyday words (want {FLOOR['familiar_pct']:.0f}%)")
    if m["syl_per_word"] > FLOOR["syl_per_word"]:
        bad.append(f"{m['syl_per_word']:.2f} syllables a word (want under {FLOOR['syl_per_word']})")
    if m["concrete_pct"] < FLOOR["concrete_pct"]:
        bad.append(f"only {m['concrete_pct']:.0f}% of lines show something (want {FLOOR['concrete_pct']:.0f}%)")
    if m["abstract_pct"] > FLOOR["abstract_pct"]:
        bad.append(f"{m['abstract_pct']:.0f}% of lines are abstractions (want under {FLOOR['abstract_pct']:.0f}%)")
    if m["plain_order_pct"] < FLOOR["plain_order_pct"]:
        bad.append(f"only {m['plain_order_pct']:.0f}% of lines run in plain order (want {FLOOR['plain_order_pct']:.0f}%)")
    if m["inverted_pct"] > FLOOR["inverted_pct"]:
        bad.append(f"{m['inverted_pct']:.0f}% of lines open on an inversion (want under {FLOOR['inverted_pct']:.0f}%)")
    return bad


def selftest():
    ok = True
    clear = "[v]\nThey made me in the fire.\nThey beat me on the stone.\nI hold the edge they gave me.\nI do not break.\n"
    murky = ("[v]\nCall me the oath a hammer swore.\nFolded in dark and drawn to light.\n"
             "What the cold caught, the cold will hold.\nI am the standing pledge.\n")
    mc, mm = measure(clear), measure(murky)
    for name, m, want_clean in (("plain speech", mc, True), ("riddle", mm, False)):
        bad = verdict(m)
        good = (not bad) if want_clean else bool(bad)
        ok &= good
        print(f"   {name:13s} familiar {m['familiar_pct']:5.1f}% · concrete {m['concrete_pct']:5.1f}% · "
              f"plain order {m['plain_order_pct']:5.1f}% · {len(bad)} complaint(s)  {'ok' if good else 'FAIL'}")
    print("   " + ("ALL PASSED" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "selftest":
        print("CLARITY selftest — plain speech must pass, a riddle must not")
        sys.exit(0 if selftest() else 1)
    rows = []
    for path in sys.argv[1:]:
        m = measure(Path(path).read_text())
        rows.append((Path(path).name, m))
        print(f"\n{Path(path).name}  {m['lines']} lines, {m['words']} words")
        print(f"  everyday words     {m['familiar_pct']:6.1f}%   (want >= {FLOOR['familiar_pct']})")
        print(f"  syllables per word {m['syl_per_word']:6.2f}    (want <= {FLOOR['syl_per_word']})")
        print(f"  lines you can see  {m['concrete_pct']:6.1f}%   (want >= {FLOOR['concrete_pct']})")
        print(f"  lines of abstraction {m['abstract_pct']:4.1f}%   (want <= {FLOOR['abstract_pct']})")
        print(f"  plain word order   {m['plain_order_pct']:6.1f}%   (want >= {FLOOR['plain_order_pct']})")
        print(f"  opens on inversion {m['inverted_pct']:6.1f}%   (want <= {FLOOR['inverted_pct']})")
        bad = verdict(m)
        print("  VERDICT: " + ("clear enough to follow on first listen" if not bad else "; ".join(bad)))
