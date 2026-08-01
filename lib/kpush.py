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
OWNER = "arash0ash"; HERE = Path(__file__).resolve().parent

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
            "id": f"{OWNER}/{a.slug}", "title": a.title or a.slug, "code_file": Path(nb).name,
            "language": "python", "kernel_type": "notebook", "is_private": not a.public,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": a.kernel_source,
            "model_sources": []}, indent=2))
        r = api().kernels_push(str(f), timeout=None, acc=None)
        print("error:", getattr(r, "error", None), "| url:", getattr(r, "url", None))
    elif a.cmd == "status":
        print(api().kernels_status(f"{OWNER}/{a.slug}"))
    else:
        import kaggle
        d = api().kernels_output(f"{OWNER}/{a.slug}", "/tmp/klog", force=True, quiet=True)
        print(d)
