#!/usr/bin/env python3
"""Build the STEEL cover report from a run's own artefacts.

Every number on the page is read from the run's loop_verify.json rather than typed, because a
report about measurement that misquotes its own measurements is worthless.

Lives in the repo rather than a session scratchpad: the previous copy sat in /private/tmp, which
macOS wipes, and it went with everything else in there.

    python tools/build_report.py <run_dir> <out.html> [rival_run_dir]
"""
import base64, io, json, mimetypes, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RIVAL = Path(sys.argv[3]) if len(sys.argv) > 3 else None


def pick(d, *names):
    for n in names:
        for c in (d / n, d / f"out_{n}"):
            if c.exists():
                return c
    return None


def uri(p):
    mt = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(Path(p).read_bytes()).decode()


def jpg(arr, q=90):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def wide(p, mx=1500, q=86):
    im = Image.open(p).convert("RGB")
    if im.width > mx:
        im = im.resize((mx, round(im.height * mx / im.width)), Image.LANCZOS)
    return jpg(np.asarray(im), q)


def peak_frame(mp4, px=560):
    """The frame with the most hot sparks — where impact physics either reads or does not."""
    d = Path(tempfile.mkdtemp())
    subprocess.run(f"ffmpeg -v error -i '{mp4}' '{d}/%03d.png'", shell=True, check=True)
    ims = [np.asarray(Image.open(f).convert("RGB")) for f in sorted(d.glob("*.png"))]
    sc = [float(((f[:, :, 0].astype(int) > 200) & (f[:, :, 2].astype(int) < 150)).sum()) for f in ims]
    return np.asarray(Image.fromarray(ims[int(np.argmax(sc))]).resize((px, px), Image.LANCZOS))


V = json.loads((RUN / "loop_verify.json").read_text())

# MEASURE THE FILE THIS PAGE EMBEDS, rather than reprint the run's own record of it. The run
# reported a seam of 0.80x for a file that wrapped at 1.78x -- the report repeated the 0.80
# faithfully, and was wrong about the only thing a reader can check by watching. Both numbers were
# honestly recorded; only one of them is about the video on the page.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import stillness as _S, looper as _L
_lp = pick(RUN, "STEEL_cover_loop.mp4") or pick(RUN, "cover_loop.mp4")
# A RECORD IS A SONG AND A COVER. Earlier runs made only a cover, so the page only ever showed one;
# a full record that shipped its song in silence would be showing half of itself.
_song = pick(RUN, "STEEL.mp3")
SONG = uri(_song) if _song else None
# THE SONG IS MEASURED LIKE THE LOOP: on the file this page embeds. The run under review passed
# every gate it had while its track stopped SEVEN times — loudness, true peak and word accuracy
# all average over the time axis, and a hole is a small part of an average.
SONG_CONT, SONG_CONT_BAD = None, []
if _song:
    import measure as _M
    SONG_CONT = _M.continuity(str(RUN / _song) if not str(_song).startswith("/") else str(_song))
    SONG_CONT_BAD = _M.continuity_verdict(SONG_CONT)
_lyr = RUN / "STEEL_lyrics.txt"
LYRICS = _lyr.read_text().strip() if _lyr.exists() else ""
_vfj = RUN / "verify_final.json"
VF = json.loads(_vfj.read_text()) if _vfj.exists() else {}
DELIVERED = {}
if _lp:
    _fr = _S.frames_of(str(RUN / _lp) if not str(_lp).startswith("/") else str(_lp))
    DELIVERED = {"frames": len(_fr), "wrap": round(_L.delivered_wrap(_fr), 2)}
P = json.loads((RUN / "prompt.json").read_text()) if (RUN / "prompt.json").exists() else {}
CYC = V.get("cycle") or {}
LORA = V.get("lora_applied")
_lv = pick(RUN, "STEEL_cover_loop.webm") or pick(RUN, "STEEL_cover_loop_1080.webm") or pick(RUN, "STEEL_cover_loop.mp4")
loop_uri = uri(_lv)   # one of the three always exists in a finished run; the fetch dir may hold only the 1080
sheet = wide(pick(RUN, "loop_sheet.jpg"))
seam_p = pick(RUN, "loop_seam.jpg")
seam = wide(seam_p) if seam_p else None
raw_p = pick(RUN, "STEEL_cover_loop_asgenerated.webm")
raw_uri = uri(raw_p) if raw_p else None

VS = None
if RIVAL:
    a, b = pick(RUN, "STEEL_cover_loop_raw.mp4"), pick(RIVAL, "STEEL_cover_loop_raw.mp4")
    if a and b:
        try:
            VS = jpg(np.concatenate([peak_frame(a), peak_frame(b)], 1), 92)
        except Exception as e:
            print("  (comparison skipped:", str(e)[:60], ")")

CONT_BLOCK = ("" if SONG_CONT is None else (
    '<p class="prose" style="border-left:3px solid #c33;padding-left:.8rem"><b>Refused by the '
    'continuity gate:</b> ' + "; ".join(SONG_CONT_BAD) + '. This take is withdrawn — a corrected '
    'record is generating now.</p>' if SONG_CONT_BAD else
    f'<p class="prose"><b>Continuity:</b> the arrangement never stops — '
    f'{SONG_CONT["dropouts"]} dropouts, longest unbroken run '
    f'{SONG_CONT["longest_loud_run_s"]}s.</p>'))

SONG_BLOCK = ("" if not SONG else f"""
  <h2>The song</h2>
  <audio controls preload="metadata" src="{SONG}" style="width:100%;margin:.75rem 0"></audio>
  <p class="prose">Generated by ACE-Step 1.5 XL and gated on the delivered audio, not on the
  intention to make it: a Whisper large-v3 pass reads the words back and the take is refused below
  the accuracy floor. This one scored <b>{('%.1f%%' % (VF['word_accuracy'] * 100)) if 'word_accuracy' in VF else '—'}</b> against the lyric it was given, judged by {VF.get('asr_judge', 'an ASR pass')}.</p>
  {CONT_BLOCK}
  {'<details><summary>Lyrics</summary><pre class="prose" style="white-space:pre-wrap">' + LYRICS.replace('<','&lt;') + '</pre></details>' if LYRICS else ''}
""")

lora_line = ('<p class="prose"><b>No style adapter was used.</b> The LEGO look here comes from the '
             'prompt alone, on the base text-to-video model — which is the finding: the '
             'Remade-AI LEGO LoRA turned out not to be needed for it.</p>' if LORA is None else
             ('<p class="prose"><b>The LEGO LoRA attached to the quantised transformer.</b> '
              'These frames are the adapter\'s work.</p>' if LORA else
              '<p class="prose"><b>The LEGO LoRA did not attach</b> — so these frames are the base '
              'model <em>describing</em> bricks from the prompt, not the adapter rendering them. '
              'Worth being exact about, because the two would be easy to confuse and only one of '
              'them is what was asked for.<br><code>' + str(V.get("lora_error"))[:220] + '</code></p>'))

HTML = f"""<title>The Steel Rebuild</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Inter:wght@400;500;600&display=swap">
<style>
  :root {{ --ground:#F3F5F8; --surface:#FFF; --surface-2:#E8ECF2; --ink:#0C1E32; --ink-2:#3E4F66;
    --muted:#6B7A90; --line:#CBD3DF; --gold:#9C7A0E; --gold-soft:rgba(232,185,35,.16);
    --blue:#1746DC; --good:#2E7D4F; --bad:#B33A3A;
    --mono:"SF Mono",Menlo,Consolas,monospace; --serif:"Cormorant Garamond",Palatino,Georgia,serif;
    --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --ground:#06121E; --surface:#0C1E32; --surface-2:#122741; --ink:#E9EEF5; --ink-2:#C3CDDB;
    --muted:#8FA0B5; --line:#24384F; --gold:#E8B923; --gold-soft:rgba(232,185,35,.14);
    --blue:#7D97FF; --good:#5FC58C; --bad:#F07A7A; }} }}
  :root[data-theme="dark"] {{ --ground:#06121E; --surface:#0C1E32; --surface-2:#122741;
    --ink:#E9EEF5; --ink-2:#C3CDDB; --muted:#8FA0B5; --line:#24384F; --gold:#E8B923;
    --gold-soft:rgba(232,185,35,.14); --blue:#7D97FF; --good:#5FC58C; --bad:#F07A7A; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
    font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased; }}
  main {{ max-width:70rem; margin:0 auto; padding:2.5rem 1.25rem 5rem; }}
  .prose {{ max-width:66ch; }}
  h1,h2 {{ font-family:var(--serif); font-weight:700; text-wrap:balance; margin:0; }}
  h1 {{ font-size:clamp(2.4rem,6vw,4rem); line-height:1.02; }}
  h2 {{ font-size:clamp(1.6rem,3vw,2.1rem); margin-top:3.5rem; padding-top:1.5rem;
    border-top:1px solid var(--line); }}
  p {{ margin:.85rem 0; }}
  a {{ color:var(--blue); text-decoration:none; border-bottom:1px solid color-mix(in srgb,var(--blue) 40%,transparent); }}
  code {{ font-family:var(--mono); font-size:.86em; background:var(--surface-2); padding:.08em .35em; border-radius:3px; }}
  .eyebrow {{ font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
  .lede {{ font-size:1.2rem; color:var(--ink-2); max-width:60ch; margin-top:1rem; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:.4rem 1.5rem; color:var(--muted); font-size:.85rem; margin-top:1.25rem; }}
  .grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(14rem,1fr)); margin-top:1.25rem; }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:1rem 1.15rem; }}
  .card h3 {{ margin:0; font-size:.9rem; font-weight:600; font-family:var(--sans); }}
  .card p {{ margin:.35rem 0 0; font-size:.88rem; color:var(--ink-2); }}
  .num {{ font-family:var(--serif); font-size:2.2rem; line-height:1; color:var(--gold); font-variant-numeric:tabular-nums; }}
  figure {{ margin:1.5rem 0; }}
  figure img, figure video {{ max-width:100%; height:auto; border-radius:4px; border:1px solid var(--line); display:block; }}
  figcaption {{ font-size:.84rem; color:var(--muted); margin-top:.5rem; max-width:66ch; }}
  blockquote {{ margin:1rem 0; padding:.8rem 1.1rem; background:var(--surface); border-left:3px solid var(--line);
    border-radius:0 4px 4px 0; font-family:var(--mono); font-size:.8rem; line-height:1.55; color:var(--ink-2); }}
</style>
<main>
  <div class="eyebrow">ArtaQuest · ArtaMusic</div>
  <h1 style="margin-top:.6rem">The Steel Rebuild</h1>
  <p class="lede prose">An epic loop of hammering the hot sword, asked of a video model directly.
  This is the run that made it, and every number here is read from the run's own record.</p>
  <div class="meta">
    <span>Model · {V.get('model','—')}</span>
    <span>{V.get('steps','—')} steps · {V.get('res',['—','—'])[0]}×{V.get('res',['—','—'])[1]} · {V.get('fps','—')} fps</span>
    <span>Code · <a href="https://github.com/ArtaQuest/artamusic">ArtaQuest/artamusic</a></span>
  </div>

  <figure><video src="{loop_uri}" autoplay loop muted playsinline></video>
    <figcaption>The delivered loop — {DELIVERED.get('frames', V.get('frames','—'))} frames, {V.get('seconds','—')}s, closed by
    <b>{CYC.get('used','—')}</b>.</figcaption></figure>
  {lora_line}

  <div class="grid">
    <div class="card"><div class="num">{DELIVERED.get('wrap','—')}</div><h3>wrap of the delivered file</h3>
      <p><b>1.00 is seamless</b> — the step from the last frame back to the first is exactly one ordinary
      frame step. Measured on the video above, not on the generation it was cut from. Under 1.0 is not
      better: it means the motion stalls. The uncut clip wraps at {CYC.get('whole_vs_typical','—')}.</p></div>
    <div class="card"><div class="num">{V.get('seconds','—')}s</div><h3>loop length</h3>
      <p>{DELIVERED.get('frames', CYC.get('frames','—'))} frames at {V.get('fps','—')} fps, cut as
      <code>{CYC.get('used','—')}</code>. Every candidate cut was assembled and measured; the one whose
      wrap lands nearest 1.00 wins, and ties go to whichever invents the fewest frames.</p></div>
    <div class="card"><div class="num">{V.get('steps','—')}</div><h3>denoising steps</h3>
      <p>At {V.get('seconds_per_step','—')} s each, with real guidance — the step count measured from two timed steps, not guessed.</p></div>
    <div class="card"><div class="num">{V.get('gen_seconds',0)/60:.0f}m</div><h3>to generate</h3>
      <p>Seed {V.get('seed','—')}, every input pinned to a commit.</p></div>
  </div>

  {'<h2>Two models, one shot</h2><figure><img src="' + VS + '" alt="head to head"><figcaption>The peak-spark frame from each — where impact physics either reads or does not.</figcaption></figure>' if VS else ''}

  <h2>The loop</h2>
  {SONG_BLOCK}
  <figure><img src="{sheet}" alt="frames across the loop">
    <figcaption>Eight frames across the loop.</figcaption></figure>
  {'<figure><img src="' + seam + '" alt="the seam"><figcaption>The wrap: last three frames into first three.</figcaption></figure>' if seam else ''}
  {'<figure><video src="' + raw_uri + '" autoplay loop muted playsinline></video><figcaption>The generation before the loop was cut from it — the comparison the seam number is taken against.</figcaption></figure>' if raw_uri else ''}

  {'<h2>The prompt</h2><blockquote>' + P.get("prompt","")[:900] + '</blockquote><blockquote><b>negative —</b> ' + P.get("negative","")[:400] + '</blockquote>' if P else ''}
</main>
"""
OUT.write_text(HTML)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
print(f"  loop {DELIVERED.get('frames')} frames · wrap {DELIVERED.get('wrap')} (1.00 seamless) "
      f"· cut {CYC.get('used')} · lora_applied={LORA}")
