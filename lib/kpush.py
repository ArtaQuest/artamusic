import os
#!/usr/bin/env python3
"""Push/poll a Kaggle kernel. Rebuilt after the scratchpad was lost.

NOTE: --acc does not work. Kaggle accepts machine_shape and normalises it straight back to "Gpu",
verified across eleven spellings; every run lands on a P100 regardless. Kernels must adapt to the
card rather than request one.
"""
import argparse, json, os, re, sys
from pathlib import Path
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
HERE = Path(__file__).resolve().parent


def owner():
    """Whoever the credential authenticates as — never a hardcoded login.

    A hardcoded owner silently pushes to the wrong account the moment the credential changes,
    and the push still reports success. The credential itself lives only in ~/.kaggle/kaggle.json
    and must never be committed: this repo is public.
    """
    import json as _j
    return _j.loads((Path.home() / ".kaggle/kaggle.json").read_text())["username"]

# ── Kaggle API rate limiting ─────────────────────────────────────────────────────────────
# Kaggle does not publish its exact limits, and a throttled or blocked key is a much more
# expensive failure than a slow poll: the account is shared with the running kernels, and a 429
# storm can lock out the push that would have fixed the problem. So every call goes through here.
#
#   - a hard floor between calls (no burst can outrun it, even across helper functions)
#   - exponential backoff with jitter on 429/503, honouring Retry-After when the server sends it
#   - a bounded number of attempts, then a real error rather than an infinite quiet retry
#
# The floor is deliberately generous for STATUS polling: a kernel takes 10-60 minutes, so asking
# every 20 seconds buys nothing a 90-second poll does not.
import random
import time as _time

MIN_INTERVAL_S = 20.0          # hard floor between any two API calls
POLL_INTERVAL_S = 90.0         # recommended interval when waiting on a kernel
MAX_ATTEMPTS = 6
_last_call = [0.0]


def throttle():
    """Block until at least MIN_INTERVAL_S has passed since the previous API call."""
    wait = MIN_INTERVAL_S - (_time.monotonic() - _last_call[0])
    if wait > 0:
        _time.sleep(wait)
    _last_call[0] = _time.monotonic()


def call(fn, *a, **kw):
    """Run one Kaggle API call under the throttle, with backoff on rate limiting.

    Retries only on 429 (rate limited) and 5xx (transient). A 403/404 is a real answer — private,
    deleted, or not yet created — and retrying it just burns quota against a wall.
    """
    delay = 5.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        throttle()
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e)
            retryable = "429" in msg or "Too Many Requests" in msg or any(
                c in msg for c in ("500", "502", "503", "504"))
            if not retryable or attempt == MAX_ATTEMPTS:
                raise
            after = None
            m = re.search(r"[Rr]etry-?[Aa]fter[\"'\s:]+(\d+)", msg)
            if m:
                after = float(m.group(1))
            sleep_s = after if after else delay * (1 + 0.3 * random.random())
            print(f"  rate limited (attempt {attempt}/{MAX_ATTEMPTS}) — waiting {sleep_s:.0f}s",
                  flush=True)
            _time.sleep(sleep_s)
            delay = min(delay * 2, 300.0)
    raise RuntimeError("unreachable")


# ── identity guard ───────────────────────────────────────────────────────────────────────
# The kaggle package authenticates AT IMPORT TIME from whatever credential is visible then.
# Swapping ~/.kaggle/kaggle.json or setting KAGGLE_CONFIG_DIR after the import does NOTHING —
# a lesson that has now cost three misfires, the last one deleting a RUNNING kernel that
# belonged to another project (artafather/artamic-pretrain, 2026-08-04). To act as a different
# account, launch a FRESH python process with KAGGLE_CONFIG_DIR set in its environment.
#
# delete() below is the only sanctioned way to delete a kernel from this module: it re-reads
# the credential file at call time and refuses when the ref's owner does not match, so a stale
# in-process identity can never destroy another account's work again.

def whoami_file():
    """The username in the credential FILE right now — not the (possibly stale) session."""
    import json as _j
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    return _j.loads((cfg / "kaggle.json").read_text())["username"]


def delete(api_obj, ref, expected_owner=None):
    ref_owner = ref.split("/", 1)[0]
    file_owner = whoami_file()
    sess_owner = api_obj.config_values.get("username")
    want = expected_owner or ref_owner
    if not (ref_owner == file_owner == sess_owner == want):
        raise RuntimeError(
            f"identity mismatch — REFUSING delete of {ref}: ref-owner={ref_owner}, "
            f"credential-file={file_owner}, session={sess_owner}, expected={want}. "
            f"The kaggle package authenticates at import; restart the process to switch account.")
    return call(api_obj.kernels_delete, ref, no_confirm=True)


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate(); return a

def to_ipynb(py, out):
    """A stage file is ONE python file (so precheck/contract can parse it) that becomes many cells:
    `# ── ` starts a code cell; `# %% [markdown]` starts a MARKDOWN cell whose comment lines are
    the prose (leading `# ` stripped) — documentation that renders on Kaggle and in ArtaReader's
    book page, while the .py stays valid python (they are comments there)."""
    src = Path(py).read_text()
    # `# %%` COUNTS TOO. It used to recognise only `# ── ` for a code cell, so a file written in
    # the ordinary percent format had every one of its code blocks swallowed into the preceding
    # markdown cell — the still probe pushed as FOUR MARKDOWN CELLS AND NO CODE, ran nothing,
    # produced no files, and Kaggle reported it complete in two minutes. A notebook that executes
    # nothing and calls it success is the worst failure available here, so the marker everyone
    # reaches for is now supported rather than silently misread.
    marks = [m.start() for m in re.finditer(r"^# ── |^# %%", src, re.M)]
    bounds = [0] + marks + [len(src)]
    cells = []
    for a, b in zip(bounds, bounds[1:]):
        chunk = src[a:b].rstrip()
        if not chunk.strip():
            continue
        if chunk.startswith("# %% [markdown]"):   # checked BEFORE the bare `# %%` below
            body = "\n".join(re.sub(r"^# ?", "", l) for l in chunk.splitlines()[1:]).strip()
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": chunk})
    if not any(c["cell_type"] == "code" for c in cells):
        raise ValueError(
            f"{py} produced {len(cells)} cells and NONE of them are code — the whole file was read "
            f"as prose. Kaggle will run it, produce nothing, and report success. Check the cell "
            f"markers: `# ── ` or `# %%` for code, `# %% [markdown]` for prose.")
    Path(out).write_text(json.dumps({"cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                     "language": "python"}, "language_info": {"name": "python"}}}, indent=1))
    return out

def precheck_source(py):
    """Every static check that must pass before a kernel leaves this laptop.

    These lived inside push_verified, which the command line does not call — so every push
    made by hand skipped the unbound-name scans and the embedded-source scan entirely. A
    guard sitting one layer above the path that is actually used protects nothing.
    """
    # Unskippable static check. Two kernels died on a NameError at module scope after the model
    # had already loaded — 40 ms of AST work would have caught both. A cheap check only helps if
    # it cannot be forgotten, so it lives here rather than in a habit.
    from precheck import unbound, unbound_in_functions
    _src = str(Path(py).resolve())
    _bad = unbound(_src)
    if _bad:
        raise RuntimeError(f"{py}: unbound names at module scope {_bad} — refusing to push")
    # Module scope is not the whole file. A stage function shipped to its own process died on the
    # GPU at second three with NameError: snapshot_download, because the import lived in a sibling
    # function. Same 40 ms of AST work, one scope deeper.
    _badf = unbound_in_functions(_src)
    if _badf:
        raise RuntimeError(f"{py}: unbound names inside functions {_badf} — refusing to push")
    # AND ORDERING, WHICH BINDING DOES NOT COVER. A stage whose __main__ dispatcher sat above the
    # function it dispatches to passed every check here and died on the GPU with NameError. The
    # name was bound; it was just not bound YET.
    from precheck import used_before_defined
    _bado = used_before_defined(_src)
    if _bado:
        raise RuntimeError(f"{py}: used before defined at module scope {_bado} — refusing to push")
    # And any python source this file EMBEDS to run elsewhere (STAGE_SRC = r"""…""") is code too.
    import ast as _ast, re as _re
    # READ THE EMBEDDED SOURCE FROM THE AST, NEVER BY EXECUTING THE FILE. This used to exec the
    # whole notebook to get at STAGE_SRC — which for any real notebook raises immediately (no
    # /kaggle, no torch, no GPU), hit the bare `except: break`, and skipped every embedded check in
    # silence. The checks looked present and protected nothing: a stage whose __main__ dispatcher
    # sat above its own definitions sailed through and died on the GPU. A string literal assigned
    # at module scope can simply be read out of the tree.
    _tree = _ast.parse(Path(_src).read_text())
    _embeds = {}
    for _n in _tree.body:
        if isinstance(_n, _ast.Assign) and isinstance(_n.value, _ast.Constant) \
                and isinstance(_n.value.value, str):
            for _t in _n.targets:
                if isinstance(_t, _ast.Name):
                    _embeds[_t.id] = _n.value.value
    for _name, _emb in _embeds.items():
        class _M:
            @staticmethod
            def group(_i):
                return _name
        _m = _M
        if isinstance(_emb, str) and "import" in _emb and "def " in _emb:
            try:
                _ast.parse(_emb)
            except SyntaxError as _e:
                raise RuntimeError(f"{py}: embedded {_m.group(1)} does not parse — {_e}")
            _be = unbound_in_functions(_emb, is_src=True)
            if _be:
                raise RuntimeError(f"{py}: embedded {_m.group(1)} unbound names {_be} — refusing to push")
            # MODULE SCOPE INSIDE THE EMBEDDED SOURCE TOO. Checking only functions left a hole the
            # width of a whole file: after a stage was replaced wholesale, a name it used at module
            # scope no longer existed, and that reference sat on the LAST line of a three-hour run.
            import tempfile as _tf2
            _t2 = Path(_tf2.mkdtemp()) / "embed.py"; _t2.write_text(_emb)
            _beo = used_before_defined(_emb, is_src=True)
            if _beo:
                raise RuntimeError(f"{py}: embedded {_m.group(1)} used before defined {_beo} "
                                   f"— refusing to push")
            _bem = unbound(str(_t2))
            if _bem:
                raise RuntimeError(f"{py}: embedded {_m.group(1)} unbound at module scope {_bem} "
                                   f"— refusing to push")


    # A HUGGING FACE REVISION PIN THAT DOES NOT EXIST. Twice in one day a 40-character sha was
    # written out from a 12-character print — the API truncates, and the eye fills in the rest.
    # Both times the tail was wrong, and both times it was caught only because someone happened to
    # check by hand; unchecked it downloads nothing, forty minutes into a session, with a message
    # about a repository rather than about a typo. The API says plainly whether a revision exists.
    import re as _re3, urllib.request as _ur3
    _seen = set()
    for _repo, _rev in _re3.findall(
            r'\(\s*"([\w.-]+/[\w.-]+)"\s*,\s*"([0-9a-f]{40})"\s*\)', Path(_src).read_text()):
        if (_repo, _rev) in _seen:
            continue
        _seen.add((_repo, _rev))
        try:
            _ur3.urlopen(_ur3.Request(f"https://huggingface.co/api/models/{_repo}/revision/{_rev}",
                                      headers={"User-Agent": "aq"}), timeout=30).read(1)
        except Exception as _e:
            raise RuntimeError(f"{py}: {_repo} has no revision {_rev[:12]}… ({_e}) — refusing to "
                               f"push a kernel that will fail its own download")

    # A PIN THAT SERVES OLD CODE IS A SILENT WRONG ANSWER. The notebook fetches its instruments
    # and its lyric from this repo at a commit sha. Edit one of those files, forget to push or to
    # bump the sha, and the kernel runs the version you no longer have — no error, no warning, just
    # a measurement taken with an instrument you think you replaced. So: for every pinned file that
    # also exists in this working tree, fetch what the pin actually serves and compare the bytes.
    import re as _re2, urllib.request as _ur, hashlib as _hl
    _text = Path(_src).read_text()
    _shas = dict(_re2.findall(r'"(\w+)":\s*"([0-9a-f]{40})"', _text))
    _root = Path(_src).resolve().parent.parent
    for _key, _path in _re2.findall(
            r'artamusic/\{PINS\[[\'"](\w+)[\'"]\]\}/([\w./-]*(?:\{[\w_]+\})?[\w./-]*)', _text):
        _sha = _shas.get(_key)
        if not _sha:
            continue
        # a path built from a loop variable ("lib/{_f}") names a DIRECTORY of pinned files; check
        # every python file in it, or the most important pin in the notebook goes unchecked
        _targets = []
        if "{" in _path:
            _dir = _path.split("{")[0].rstrip("/")
            _names = _re2.findall(rf'for _?\w+ in \(([^)]*)\)[^\n]*\n[^\n]*{_re2.escape(_dir)}', _text)
            for _grp in _names:
                _targets += [f"{_dir}/{_n}" for _n in _re2.findall(r'"([\w.]+)"', _grp)]
        else:
            _targets = [_path]
        for _path in _targets:
            _local = _root / _path
            if not _local.exists():
                continue
            _url = f"https://raw.githubusercontent.com/ArtaQuest/artamusic/{_sha}/{_path}"
            try:
                _remote = _ur.urlopen(_url, timeout=30).read()
            except Exception as _e:
                raise RuntimeError(f"{py}: pin {_key}={_sha[:7]} does not serve {_path} ({_e}) "
                                   f"— refusing to push a kernel that will 404 on its own tools")
            if _hl.sha256(_remote).hexdigest() != _hl.sha256(_local.read_bytes()).hexdigest():
                raise RuntimeError(
                    f"{py}: pin {_key}={_sha[:7]} serves a DIFFERENT {_path} than the one in this "
                    f"working tree. The kernel would run code you no longer have. Commit and push it, "
                    f"then set {_key} to the new sha.")


def push_verified(slug, title, py, public=True, sources=(), tries=6, allow_violations=False,
                  internet=False, gpu=True):
    """Push, then READ BACK the pushed source and confirm it matches what we meant to send.

    PUBLIC BY DEFAULT. ArtaSwitch rotates the compute account between runs, and a PRIVATE kernel
    becomes unreadable the instant it does — a finished run's error log was lost that way, with
    ten retries all answering 'Permission kernels.get was denied' because the credential had
    moved. Public costs nothing here: the code already lives in a public repo, and a stranger
    being able to re-run it is the platform's entire thesis.

    Three kernels in a row died at their own mount assert because a push was assumed to have
    landed: once the --kernel-source flag was omitted, once the push output was swallowed by an
    SSL error mid-request, once the wrong OWNER was named (the source kernel ran on a different
    account). The assert was right every time; the push was not verified. This makes verification
    the only way to push, so the failure mode cannot recur.
    """
    import json as _j, subprocess as _sp, sys as _sys, tempfile as _tf, time as _t
    precheck_source(HERE / py)
    # Tier-0 contracts: every v1 death class, checked in under a second on the bytes we push.
    # A kernel that would die for a non-GPU reason never leaves the laptop.
    from contract import check as _contract
    if not allow_violations:
        _v = _contract(str((HERE / py).resolve()))
        if _v:
            raise RuntimeError(f"{py}: {len(_v)} contract violation(s) — refusing to push:\n  " +
                               "\n  ".join(_v))
    args = [_sys.executable, str(HERE / "kpush.py"), "push", slug, "--title", title, "--py", py]
    if public:
        args.append("--public")
    if internet:
        args.append("--internet")
    if not gpu:
        args.append("--cpu")
    for s_ in sources:
        args += ["--kernel-source", s_]
    for attempt in range(tries):
        r = _sp.run(args, capture_output=True, text=True, cwd=str(HERE))
        out = (r.stdout + r.stderr).strip()
        last = out.splitlines()[-1][:120] if out else "?"
        print(f"push {slug} attempt {attempt}: {last}", flush=True)
        # A readback alone is not proof the push LANDED: kernel metadata persists from earlier
        # pushes, so a slot-limited failure can pass mount verification on stale data. Detect the
        # refusals explicitly and keep waiting rather than reporting a verified no-op.
        if "Maximum batch" in out or "quota" in out.lower():
            print("  refused (no free slot / quota) — waiting, not verifying", flush=True)
            _t.sleep(300)
            continue
        # A PUSH THAT PRINTED A URL LANDED, whatever happened next. The readback is a separate
        # request and Kaggle drops connections; retrying the PUSH on a failed READBACK queues a
        # second run of the same kernel, which is how one cover attempt filled both of an account's
        # session slots with duplicates of itself. Retry the readback, never the push.
        if "url:" in out and "error:  |" in out:
            for _r in range(6):
                _t.sleep(30)
                try:
                    a = api()
                    d = _tf.mkdtemp()
                    call(a.kernels_pull, f"{owner()}/{slug}", d, metadata=True, quiet=True)
                    call(a.kernels_pull, f"{owner()}/{slug}", d, quiet=True)
                    break
                except Exception as e:
                    print(f"  readback retry {_r}: {str(e)[:60]}", flush=True)
        _t.sleep(20)
        try:
            a = api()
            d = _tf.mkdtemp()
            call(a.kernels_pull, f"{owner()}/{slug}", d, metadata=True, quiet=True)
            call(a.kernels_pull, f"{owner()}/{slug}", d, quiet=True)
            m = _j.load(open(f"{d}/kernel-metadata.json"))
            got = m.get("kernel_sources") or []
            # VERIFY THE CONTENT, NOT THE METADATA. Metadata survives from previous versions, so
            # a push that failed (SSL drop, quota, slot) still reads back with the right mounts
            # and reports success — which happened twice, and the second time sent a stale run's
            # verdict as if it were fresh. Pull the notebook source back and compare it to what
            # we meant to push; only identical code proves the push landed.
            import re as _re
            src_local = (HERE / py).resolve().read_text()
            nbp = next(Path(d).glob("*.ipynb"), None)
            pushed = ""
            if nbp:
                cells = _j.loads(nbp.read_text()).get("cells", [])
                pushed = "".join("".join(c["source"]) if isinstance(c["source"], list)
                                 else c["source"] for c in cells if c.get("cell_type") == "code")
            # Compare with ALL whitespace removed: to_ipynb rstrips each cell, so rejoining
            # them drops the newline at every cell boundary. Collapsing runs to a single space
            # still sees that as a difference and rejected pushes that had genuinely landed.
            # Stripping whitespace entirely still catches any real code change.
            # Compare ASTs, not text: whitespace-stripping was a false-negative source (cell
            # rstrip) and text can differ while code is identical. ast.dump is canonical.
            import ast as _ast
            try:
                content_ok = _ast.dump(_ast.parse(pushed)) == _ast.dump(_ast.parse(src_local))
            except SyntaxError:
                content_ok = False
            print(f"  kernel_sources={got} content_matches={content_ok} "
                  f"machine_shape={m.get('machine_shape')}", flush=True)
            if content_ok and set(got) >= set(sources):
                print(f"  VERIFIED: {slug} — pushed source matches and mounts are present",
                      flush=True)
                return True
            print("  push did NOT land (stale source or missing mounts) — retrying", flush=True)
        except Exception as e:
            print(f"  readback failed: {str(e)[:70]}", flush=True)
        _t.sleep(45)
    raise RuntimeError(f"{slug}: could not verify mounts after {tries} attempts")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["push", "status", "log"]); p.add_argument("slug")
    p.add_argument("--title"); p.add_argument("--py"); p.add_argument("--public", action="store_true")
    p.add_argument("--kernel-source", action="append", default=[],
                   help="mount another kernel's OUTPUT under /kaggle/input (must be public)")
    p.add_argument("--internet", action="store_true",
                   help="the kernel may reach the network. Needed by ANYTHING that pip installs "
                        "or pulls weights at runtime — not just the wheelhouse builder")
    p.add_argument("--cpu", action="store_true", help="enable_gpu=false — zero quota")
    a = p.parse_args()
    if a.cmd == "push":
        f = HERE / f".push-{a.slug}"; f.mkdir(exist_ok=True)
        nb = to_ipynb(HERE / a.py, f / f"{a.slug}.ipynb")
        meta = {
            "id": f"{owner()}/{a.slug}", "title": a.title or a.slug, "code_file": Path(nb).name,
            "language": "python", "kernel_type": "notebook", "is_private": not a.public,
            "enable_gpu": not a.cpu, "enable_internet": bool(a.internet), "dataset_sources": [],
            "competition_sources": [], "kernel_sources": a.kernel_source,
            "model_sources": []}
        # kagglesdk (2.2.4) documents machine_shape = NvidiaTeslaT4 | NvidiaTeslaP100 | Tpu1VmV38 —
        # a spelling the eleven-probe note above never tried. Opt-in only, via AQ_KAGGLE_SHAPE, so
        # every existing caller keeps today's behaviour; the readback prints what Kaggle kept.
        if os.environ.get("AQ_KAGGLE_SHAPE"):
            meta["machine_shape"] = os.environ["AQ_KAGGLE_SHAPE"]
        (f / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        # A KERNEL THAT REACHES THE NETWORK WITH THE NETWORK OFF FAILS AT ITS FIRST FETCH, and
        # the traceback is a DNS error twenty frames deep that reads like a Kaggle outage rather
        # than a flag you did not pass. It cost a run: six minutes in, "Temporary failure in name
        # resolution", on a notebook whose whole first cell is a pip install. The source says
        # plainly whether it needs the network, so ask it rather than the operator's memory.
        precheck_source(HERE / a.py)
        src = (HERE / a.py).read_text()
        needs = [n for n in ("pip install", "snapshot_download", "hf_hub_download", "urlretrieve",
                             "requests.get", "urlopen", "git clone") if n in src]
        if needs and not a.internet:
            sys.exit(f"refusing to push: {a.py} uses {', '.join(needs)} but --internet was not "
                     f"given, so the kernel will fail at its first fetch with a DNS error")
        # AND THE SAME FOR MOUNTS. A record notebook globbed /kaggle/input for a reference vocal it
        # needs, and was pushed without --kernel-source: the cover stage ran for THREE HOURS, wrote
        # everything correctly, and then the song stage died on its first assert because nothing
        # was mounted. The source names the kernel it wants in its own assert message; the push can
        # read it as easily as a human can.
        import re as _rk
        _wants = set(_rk.findall(r'kernel source ([\w-]+/[\w-]+)', src))
        _have = {k.strip() for k in a.kernel_source}
        _missing = sorted(_wants - _have)
        if _missing:
            sys.exit(f"refusing to push: {a.py} expects {', '.join(_missing)} mounted under "
                     f"/kaggle/input but no --kernel-source says so. It would run to the point of "
                     f"needing it and then fail on an empty glob.")
        r = call(api().kernels_push, str(f), timeout=None, acc=None)
        url = getattr(r, "url", None) or ""
        print("error:", getattr(r, "error", None), "| url:", url)
        # KAGGLE DERIVES THE SLUG FROM THE TITLE WHEN THEY DISAGREE, and only warns. Asking for
        # `steel-still-probe` with a descriptive title produced
        # `steel-still-probe-z-image-base-on-the-pair`, so every later status, log and output call
        # was aimed at a kernel that does not exist — which Kaggle answers 403, not 404, so it
        # reads as a permissions problem rather than a typo. Say the real slug loudly.
        landed = url.rstrip("/").split("/")[-1]
        if landed and landed != a.slug:
            print(f"\n!! the title moved the slug: asked for '{a.slug}', landed on '{landed}'.\n"
                  f"   Poll THAT slug, or re-push with a title that slugifies to '{a.slug}'.",
                  file=sys.stderr)
    elif a.cmd == "status":
        print(call(api().kernels_status, f"{owner()}/{a.slug}"))
    else:
        import kaggle
        d = call(api().kernels_output, f"{owner()}/{a.slug}", "/tmp/klog",
                 force=True, quiet=True)
        print(d)
