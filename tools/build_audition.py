#!/usr/bin/env python3
"""Build a listening page from a song audition's own artefacts.

    python tools/build_audition.py <run_dir> <out.html> --lyric song/x.txt [--gloss song/x.gloss.txt]
                                   [--title پولاد] [--subtitle "..."] [--rig rig.json]

<run_dir> holds out/*.mp3 and audition.json (one row per take: arm, seed, register, lead_hz,
words_fa, lufs, lra_lu, seconds, heard). Every number on the page is read from that file, never
typed. Audio is embedded as data URIs so the page is one self-contained file; the bitrate steps
down (128 → 96 → 80 kbps) until the whole page fits the 16 MB artifact ceiling.
Lives in the repo, not a session scratchpad, because macOS wipes /private/tmp mid-run.
"""
import argparse, base64, html, json, subprocess, tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run", nargs="+", help="one or more run dirs; their takes are pooled onto one page")
ap.add_argument("out")
ap.add_argument("--lyric", required=True); ap.add_argument("--gloss")
ap.add_argument("--title", default="پولاد"); ap.add_argument("--subtitle", default="")
ap.add_argument("--rig", help="json of label→value rows for the rig table")
ap.add_argument("--caption")
ap.add_argument("--heard", help="the operator's own register verdict for every take (e.g. male) — the ear is the authority; the instrument's read is shown beside it")
a = ap.parse_args()
RUNS, OUT = [Path(r) for r in a.run], Path(a.out)

rows, ROW_RUN = [], {}
for _r in RUNS:
    for _row in json.loads((_r / "audition.json").read_text(encoding="utf-8")):
        key = f"{_row['arm']}{_row['seed']}"
        if key in ROW_RUN:            # same arm+seed in two runs: keep the first, name the clash
            key = f"{key}@{_r.name}"; _row = dict(_row, arm=f"{_row['arm']}·{_r.name}")
        ROW_RUN[key] = _r; rows.append(_row)
print(f"{len(rows)} takes from {len(RUNS)} run(s)")
lyric = Path(a.lyric).read_text(encoding="utf-8").splitlines()
gloss = Path(a.gloss).read_text(encoding="utf-8").splitlines() if a.gloss else [""] * len(lyric)
assert len(gloss) == len(lyric), "gloss must align line for line with the lyric"
rig = json.loads(Path(a.rig).read_text()) if a.rig else {}
caption = Path(a.caption).read_text().strip() if a.caption else (RUN / "caption.txt").read_text().strip() if (RUN / "caption.txt").exists() else ""

def mp3_uri(path, kbps, mono=False):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "t.mp3"
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-codec:a", "libmp3lame", "-b:a", f"{kbps}k"]
        if mono: cmd += ["-ac", "1"]
        subprocess.run(cmd + [str(tmp), "-y"], check=True)
        return "data:audio/mpeg;base64," + base64.b64encode(tmp.read_bytes()).decode()

takes = []
for r in rows:
    label = f"{r['arm']}{r['seed']}"
    base = label.split("@")[0].replace("·" + ROW_RUN.get(label, Path('.')).name, "")
    run = ROW_RUN.get(label, RUNS[0])
    mp3 = run / "out" / f"{base}.mp3"
    if not mp3.exists():                      # arm was renamed for a clash — try the plain name
        mp3 = run / "out" / f"{r['arm'].split('·')[0]}{r['seed']}.mp3"
    takes.append((label, r, mp3 if mp3.exists() else None))

# Stereo first; mono and lower bitrates only as far as the 16 MB artifact ceiling forces. A
# comparison page exists to be LISTENED to, so quality is spent before quantity.
for kbps, mono in ((128, False), (96, False), (80, False), (96, True), (72, True), (56, True), (44, True)):
    audio = {n: (mp3_uri(p, kbps, mono) if p else "") for n, _, p in takes}
    total = sum(len(v) for v in audio.values())
    if total < 14_500_000:
        break
print(f"audio at {kbps} kbps {'mono' if mono else 'stereo'} · {total/1e6:.1f} MB embedded")

def esc(s): return html.escape(str(s if s is not None else "—"))
def num(v, f="{:.1f}"): return "—" if v is None else f.format(v)

# ── the lyric as couplets: Persian on the right, English on the left ────────────────────────
blocks, cur = [], None
for fa, en in zip(lyric, gloss):
    if fa.startswith("["):
        cur = {"label": fa.strip("[]"), "lines": []}; blocks.append(cur)
    elif fa.strip() and cur is not None:
        cur["lines"].append((fa, en))
lyric_html = []
for b in blocks:
    lines = "".join(f'<div class="line"><p class="fa" lang="fa" dir="rtl">{esc(fa)}</p><p class="en">{esc(en)}</p></div>' for fa, en in b["lines"])
    body = lines if lines else '<p class="rest">instrumental</p>'
    lyric_html.append(f'<section class="stanza"><h3 class="eyebrow">{esc(b["label"])}</h3>{body}</section>')

take_html = []
for name, r, p in takes:
    read = r.get("register") or r.get("register_read") or "—"
    reg = a.heard or read
    reg_label = (f"{a.heard} · read {read}" if a.heard and read != a.heard else reg)
    words = r.get("words_fa"); wpct = "—" if words is None else f"{words*100:.0f}%"
    arm = r.get("arm"); anchored = r.get("reference")
    take_html.append(f'''
<article class="take" data-arm="{esc(arm)}">
  <header>
    <span class="arm {esc(arm)}">{esc(arm)}</span>
    <h2>seed {esc(r.get("seed"))}</h2>
    <span class="reg {esc(reg)}" title="{esc('the operator heard ' + a.heard + '; the instrument read ' + read) if a.heard else 'instrument read'}">{esc(reg_label)}</span>
  </header>
  <p class="how">{"conditioned on the STEEL lead" if anchored else "caption only — no reference"}</p>
  <audio controls preload="metadata" src="{audio.get(name, "")}"></audio>
  <dl class="stats">
    <div><dt>lead</dt><dd>{num(r.get("lead_hz"), "{:.0f}")} Hz</dd></div>
    <div><dt>words (fa)</dt><dd>{wpct}</dd></div>
    <div><dt>loudness</dt><dd>{num(r.get("lufs"))} LUFS</dd></div>
    <div><dt>range</dt><dd>{num(r.get("lra_lu"))} LU</dd></div>
    <div><dt>render</dt><dd>{num(r.get("seconds"), "{:.0f}")} s</dd></div>
  </dl>
  {f'<p class="form">{esc(r.get("lyric_form"))}</p>' if r.get("lyric_form") else ""}
  {f'<details><summary>what the judge heard</summary><p class="heard" lang="fa" dir="rtl">{esc(r.get("heard"))}</p></details>' if r.get("heard") else ""}
  {f'<p class="err">{esc(r.get("judge_error"))}</p>' if r.get("judge_error") else ""}
</article>''')

rig_html = "".join(f'<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>' for k, v in rig.items())

page = f'''<title>{esc(a.title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&family=Vazirmatn:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@1,8..60,400&display=swap">
<style>
:root {{
  --ground:#F2F3F7; --surface:#FFFFFF; --line:#D6DAE6; --ink:#0C1E32; --ink-2:#3C4A63; --ink-3:#6B7790;
  --gold:#B8890B; --gold-soft:rgba(232,185,35,.16); --blue:#1746DC; --male:#1F6F43; --female:#A3372B;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#06121E; --surface:#0C1E32; --line:#22344F; --ink:#F1EBDD; --ink-2:#C2C9D8; --ink-3:#8D9AB3;
    --gold:#E8B923; --gold-soft:rgba(232,185,35,.14); --blue:#6C8CFF; --male:#7BD39A; --female:#F08A7A;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#06121E; --surface:#0C1E32; --line:#22344F; --ink:#F1EBDD; --ink-2:#C2C9D8; --ink-3:#8D9AB3;
  --gold:#E8B923; --gold-soft:rgba(232,185,35,.14); --blue:#6C8CFF; --male:#7BD39A; --female:#F08A7A;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--ground); color:var(--ink); font:16px/1.55 "IBM Plex Sans", system-ui, sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:40px 24px 80px; display:grid; gap:44px; }}
h1,h2,h3 {{ margin:0; text-wrap:balance }}
.hero {{ display:grid; grid-template-columns:1fr auto; gap:24px; align-items:end; border-bottom:1px solid var(--line); padding-bottom:28px }}
.hero .fa-title {{ font:600 72px/1.9 "Noto Nastaliq Urdu", "Vazirmatn", serif; color:var(--gold); direction:rtl; }}
.hero .sub {{ color:var(--ink-2); margin:4px 0 0; max-width:60ch }}
.eyebrow {{ font:600 12px/1 "IBM Plex Sans", sans-serif; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); margin:0 0 12px }}
.takes {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:18px }}
.take {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:18px 18px 16px; display:grid; gap:12px; align-content:start }}
.take header {{ display:flex; align-items:center; gap:10px }}
.take h2 {{ font:500 17px/1.2 "IBM Plex Sans", sans-serif; flex:1 }}
.arm {{ font:600 11px/1 "IBM Plex Sans", sans-serif; letter-spacing:.1em; text-transform:uppercase; padding:6px 9px; border-radius:999px; background:var(--gold-soft); color:var(--gold) }}
.reg {{ font:600 11px/1 "IBM Plex Sans", sans-serif; letter-spacing:.1em; text-transform:uppercase; padding:6px 9px; border-radius:999px; border:1px solid currentColor }}
.reg.male {{ color:var(--male) }} .reg.female {{ color:var(--female) }} .reg.— {{ color:var(--ink-3) }}
.how {{ margin:0; color:var(--ink-3); font-size:13px }}
.form {{ margin:0; font-size:12px; font-weight:600; color:var(--gold) }}
audio {{ width:100%; height:40px; border-radius:999px; }}
.stats {{ display:grid; grid-template-columns:repeat(5, 1fr); gap:8px; margin:0 }}
.stats div {{ display:grid; gap:2px }}
.stats dt {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3) }}
.stats dd {{ margin:0; font-variant-numeric:tabular-nums; font-weight:500; font-size:14px }}
details summary {{ cursor:pointer; color:var(--ink-3); font-size:13px }}
.heard {{ font:15px/2 "Vazirmatn", sans-serif; color:var(--ink-2); margin:8px 0 0 }}
.err {{ color:var(--female); font-size:13px; margin:0 }}
.lyric {{ display:grid; gap:28px }}
.stanza {{ display:grid; gap:10px }}
.line {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; align-items:baseline; padding:6px 0; border-bottom:1px solid var(--line) }}
.line:last-child {{ border-bottom:0 }}
.fa {{ order:2; margin:0; font:400 26px/2.3 "Noto Nastaliq Urdu", "Vazirmatn", serif; color:var(--ink) }}
.en {{ order:1; margin:0; font:italic 400 17px/1.5 "Source Serif 4", Georgia, serif; color:var(--ink-2) }}
.rest {{ color:var(--ink-3); font-style:italic; margin:0 }}
.rig {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:20px 22px; display:grid; gap:18px }}
.rig dl {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px 24px; margin:0 }}
.rig dt {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3) }}
.rig dd {{ margin:2px 0 0; font-variant-numeric:tabular-nums }}
.caption {{ margin:0; color:var(--ink-2); max-width:78ch }}
.metre {{ display:grid; gap:6px; color:var(--ink-2) }}
.metre code {{ font:500 15px/1.6 "IBM Plex Sans", monospace; letter-spacing:.18em; color:var(--gold) }}
a {{ color:var(--blue) }}
@media (max-width:720px) {{ .hero {{ grid-template-columns:1fr }} .line {{ grid-template-columns:1fr }} .fa {{ order:1 }} .en {{ order:2 }} .stats {{ grid-template-columns:repeat(3,1fr) }} }}
</style>
<main>
  <header class="hero">
    <div>
      <p class="eyebrow">audition · {len(takes)} takes</p>
      <p class="sub">{esc(a.subtitle)}</p>
    </div>
    <h1 class="fa-title" lang="fa">{esc(a.title)}</h1>
  </header>

  <section>
    <p class="eyebrow">the takes</p>
    <div class="takes">{"".join(take_html)}</div>
  </section>

  <section class="lyric">
    <div>
      <p class="eyebrow">the lyric</p>
      <div class="metre">
        <p>Set in <span lang="fa" dir="rtl">بحر متقارب مثمن محذوف</span> — the Shahnameh's metre. Every line scans</p>
        <code>◡ – –  ◡ – –  ◡ – –  ◡ –</code>
        <p>(fa'ūlun fa'ūlun fa'ūlun fa'ul), rhymed in couplets. Glosses are literal, not the English lyric.</p>
      </div>
    </div>
    {"".join(lyric_html)}
  </section>

  <section class="rig">
    <p class="eyebrow">the rig</p>
    <dl>{rig_html}</dl>
    {f'<div><p class="eyebrow">caption</p><p class="caption">{esc(caption)}</p></div>' if caption else ""}
  </section>
</main>
'''
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT} · {OUT.stat().st_size/1e6:.1f} MB")
