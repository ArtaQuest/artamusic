#!/usr/bin/env python3
"""Tier-0 contracts: every way a v1 kernel died that a laptop could have caught in under a second.

The git tally: 54 commits, 13 of them fixes to failed GPU runs, and THE MODEL NEVER ONCE SAID NO.
mount x3, import x3, venv/ensurepip x2, typing-shadow x2, credential rotation x2, a whitespace
false-negative, a mangled URL, a stale push, an OpenSSL pair split across trees, a NameError, a
CPU generator. Only ONE death (an OOM) was a GPU reason. Each of the others cost a 40-minute
Kaggle round-trip to learn something an AST walk answers instantly.

So: a GPU kernel is only ever asked a question that has already been answered on a CPU. These
contracts run on the exact bytes that get pushed, and push_verified refuses when any fails.

    python contract.py <kernel.py> [--metadata kernel-metadata.json]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# What a kernel must never do at GPU time. Every entry is a v1 death, not a style preference.
FORBIDDEN_CALLS = {
    "hf_hub_download": "weights must be a mounted PUBLIC dataset, not a runtime download (URL/network deaths)",
    "snapshot_download": "weights must be a mounted PUBLIC dataset, not a runtime download",
    "urlretrieve": "no runtime fetch of measuring code — it travels IN the bundle",
    "urlopen": "no runtime network — internet is OFF in v2 kernels",
}
FORBIDDEN_SHELL = [
    (r"\bpip\s+install\b(?!.*--no-index)", "pip at GPU time must be --no-index from the wheelhouse"),
    (r"\bpython\s+-m\s+venv\b", "venv dies at ensurepip on Kaggle's image (v1 death #6)"),
    (r"\bensurepip\b", "broken on Kaggle's image (v1 death #6)"),
    (r"\bgit\s+clone\b", "code must be in the bundle or a mounted dataset — no network at GPU time"),
    (r"https?://", "no URLs at GPU time — internet is OFF"),
]
SHADOW_MODULES = {"typing", "types", "dataclasses", "enum", "abc", "collections"}


def _module_calls_and_strings(tree):
    calls, strings = set(), []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name:
                calls.add(name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            strings.append(n.value)
    return calls, strings


def unbound_names(tree):
    import builtins
    bound = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
            bound.add(n.optional_vars.id)
        elif isinstance(n, (ast.comprehension, ast.For, ast.AsyncFor)):
            for e in ast.walk(n.target):
                if isinstance(e, ast.Name):
                    bound.add(e.id)
    used = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    return sorted(u for u in used - bound if not u.startswith("_"))


def env_order(tree, src):
    """PYTORCH_CUDA_ALLOC_CONF must be set, and torch imported AFTER it and after any pip line."""
    lines = src.splitlines()
    first_torch = next((i for i, l in enumerate(lines)
                        if re.match(r"\s*(import torch|from torch)", l)), None)
    alloc = next((i for i, l in enumerate(lines) if "PYTORCH_CUDA_ALLOC_CONF" in l), None)
    problems = []
    if first_torch is None:
        return problems
    if alloc is None:
        problems.append("PYTORCH_CUDA_ALLOC_CONF never set (expandable_segments is what lets 4.6B fit)")
    elif alloc > first_torch:
        problems.append(f"PYTORCH_CUDA_ALLOC_CONF set on line {alloc+1} AFTER torch import on {first_torch+1}")
    last_pip = max((i for i, l in enumerate(lines) if re.search(r"pip\s+install", l)), default=-1)
    if last_pip > first_torch:
        problems.append(f"pip install on line {last_pip+1} AFTER torch import on {first_torch+1} "
                        "(installs must precede torch so the cu126 line wins)")
    return problems


def check(kernel_path, metadata=None):
    src = Path(kernel_path).read_text()
    tree = ast.parse(src)
    v = []
    for name in unbound_names(tree):
        v.append(f"NAME: `{name}` used at module scope but never bound (v1 death: NameError after model load)")
    if re.search(r"\b__file__\b", src):
        v.append("NAME: `__file__` does not exist in a papermill notebook cell (v1 death: wheelhouse builder)")
    calls, strings = _module_calls_and_strings(tree)
    for c, why in FORBIDDEN_CALLS.items():
        if c in calls:
            v.append(f"CALL: {c}() — {why}")
    for s in strings:
        for pat, why in FORBIDDEN_SHELL:
            if re.search(pat, s):
                v.append(f"SHELL: {pat!r} in a string — {why}")
                break
    for p in env_order(tree, src):
        v.append(f"ORDER: {p}")
    if metadata:
        m = json.loads(Path(metadata).read_text()) if not isinstance(metadata, dict) else metadata
        if m.get("is_private", True):
            v.append("META: is_private must be false — a private kernel becomes unreadable on account rotation (v1 death x2)")
        if m.get("enable_internet", True):
            v.append("META: enable_internet must be false — internet is what URL/auth/network deaths ride in on")
        for s in strings:
            mm = re.match(r"/kaggle/input/([^/*]+)", s)
            if mm and not any(mm.group(1) in (src_ or "") for src_ in
                              (m.get("kernel_sources") or []) + (m.get("dataset_sources") or [])):
                v.append(f"MOUNT: code reads {s!r} but no declared source provides '{mm.group(1)}' (v1 death x3)")
    return v


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel"); ap.add_argument("--metadata")
    a = ap.parse_args()
    vs = check(a.kernel, a.metadata)
    for x in vs:
        print("  ✗", x)
    print(f"{a.kernel}: {'CLEAN' if not vs else f'{len(vs)} contract violation(s)'}")
    sys.exit(1 if vs else 0)
