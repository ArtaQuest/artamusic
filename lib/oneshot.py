#!/usr/bin/env python3
"""Push, wait and fetch in ONE process — immune to account rotation.

ArtaSwitch moves the compute account between runs. Every separate command re-imports the kaggle
client, picks up whatever credential is current, and then cannot read a kernel created under the
previous one: gate 1 finished twice and both times its log was unreachable ('Permission
kernels.get was denied'), because the account had moved underneath it.

The client authenticates AT IMPORT and keeps that identity for the life of the process. So a
single script that pushes, polls and fetches never loses access to its own work, whatever happens
to the credential file meanwhile. That is the whole point of this file.

    python oneshot.py <slug> <title> <py> [--source owner/kernel]...
"""
import argparse, json, os, pathlib, re, sys, tempfile, time

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kpush
from kaggle.api.kaggle_api_extended import KaggleApi

ap = argparse.ArgumentParser()
ap.add_argument("slug"); ap.add_argument("title"); ap.add_argument("py")
ap.add_argument("--source", action="append", default=[])
ap.add_argument("--tries", type=int, default=30)
ap.add_argument("--allow-violations", action="store_true",
                help="ONLY for the wheelhouse builder, the one kernel that needs internet+pip")
ap.add_argument("--internet", action="store_true", help="wheelhouse builder only")
ap.add_argument("--cpu", action="store_true", help="no GPU: zero quota")
a = ap.parse_args()

api = KaggleApi(); api.authenticate()
WHO = api.config_values.get("username")
print(f"locked to account: {WHO} (this process keeps it regardless of rotation)", flush=True)
REF = f"{WHO}/{a.slug}"

kpush.push_verified(a.slug, a.title, a.py, public=True, sources=a.source, tries=a.tries,
                    allow_violations=a.allow_violations, internet=a.internet, gpu=not a.cpu)

seen = None
while True:
    try:
        s = str(kpush.call(api.kernels_status, REF))
        m = re.search(r'"status":\s*"?([A-Za-z_.]+)', s)
        cur = (m.group(1) if m else s[:40]).split(".")[-1]
    except Exception as e:
        print(f"  status hiccup: {str(e)[:60]}", flush=True); time.sleep(120); continue
    if cur != seen:
        seen = cur; print(f"{a.slug}: {cur}", flush=True)
    if any(cur.startswith(t) for t in ("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(120)

for attempt in range(12):
    try:
        d = tempfile.mkdtemp()
        kpush.call(api.kernels_output, REF, d, force=True, quiet=True)
        files = [f for f in pathlib.Path(d).rglob("*") if f.is_file()]
        print("\nOUTPUT FILES:", [(f.name, f.stat().st_size) for f in files], flush=True)
        dest = pathlib.Path.home() / "Downloads/artaquest" / a.slug
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            (dest / f.name).write_bytes(f.read_bytes())
        print("saved to", dest, flush=True)
        log = next((f for f in files if f.suffix == ".log"), None)
        if log:
            t = log.read_text()
            datas = re.findall(r'"data":\s*"((?:[^"\\]|\\.)*)"', t)
            full = "".join(bytes(x, "utf-8").decode("unicode_escape") for x in datas) if datas else t
            print("\n--- markers ---", flush=True)
            for l in full.splitlines():
                if re.search(r"checkpoint|layers=|pipeline ready|frames in|FAILED|GATE|still:|"
                             r"PASS|fail", l):
                    print("|", l.strip()[:155], flush=True)
            i = full.find("Traceback")
            print("\n--- tail ---\n" + (full[i:i + 1500] if i > 0 else full[-1500:]), flush=True)
        break
    except Exception as e:
        print(f"fetch attempt {attempt}: {str(e)[:80]}", flush=True)
        time.sleep(60)
