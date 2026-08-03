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


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate(); return a

def to_ipynb(py, out):
    src = Path(py).read_text()
    marks = [m.start() for m in re.finditer(r"^# ── ", src, re.M)]
    bounds = [0] + marks + [len(src)]
    cells = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": src[a:b].rstrip()} for a, b in zip(bounds, bounds[1:]) if src[a:b].strip()]
    Path(out).write_text(json.dumps({"cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                     "language": "python"}, "language_info": {"name": "python"}}}, indent=1))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["push", "status", "log"]); p.add_argument("slug")
    p.add_argument("--title"); p.add_argument("--py"); p.add_argument("--public", action="store_true")
    p.add_argument("--kernel-source", action="append", default=[],
                   help="mount another kernel's OUTPUT under /kaggle/input (must be public)")
    a = p.parse_args()
    if a.cmd == "push":
        f = HERE / f".push-{a.slug}"; f.mkdir(exist_ok=True)
        nb = to_ipynb(HERE / a.py, f / f"{a.slug}.ipynb")
        (f / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{owner()}/{a.slug}", "title": a.title or a.slug, "code_file": Path(nb).name,
            "language": "python", "kernel_type": "notebook", "is_private": not a.public,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": a.kernel_source,
            "model_sources": []}, indent=2))
        r = call(api().kernels_push, str(f), timeout=None, acc=None)
        print("error:", getattr(r, "error", None), "| url:", getattr(r, "url", None))
    elif a.cmd == "status":
        print(call(api().kernels_status, f"{owner()}/{a.slug}"))
    else:
        import kaggle
        d = call(api().kernels_output, f"{owner()}/{a.slug}", "/tmp/klog",
                 force=True, quiet=True)
        print(d)
