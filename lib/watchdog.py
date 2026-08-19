#!/usr/bin/env python3
"""Independent proof that a Kaggle run is alive — never a poller's last printed line.

Three signals, because any one of them can lie:
  * kernels/status            — what Kaggle says, and it can lag or be eventually consistent
  * gpuQuota.timeUsed         — climbs ~1 s/s while a session really burns GPU; the honest one
  * gpuQuota.timeReserved     — the session's reservation; appears/disappears with lag
A run is ALIVE when timeUsed climbs. It is STALLED when status says running and timeUsed has not
moved for STALL_MIN minutes — the state a poller cannot see and the one that wastes hours.

    python watchdog.py <account> <owner/slug> [stall_minutes]
Prints one line per meaningful event (start, stall, recovery, terminal) and exits on a terminal state.
"""
import base64, json, sys, time, urllib.request

ACCT, REF = sys.argv[1], sys.argv[2]
STALL_MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
OWNER, SLUG = REF.split("/")
KEY = json.load(open(f"/Users/arash/.kaggle/kaggle.{ACCT}.json"))
AUTH = base64.b64encode(f"{KEY['username']}:{KEY['key']}".encode()).decode()

def api(url, body=None):
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": "Basic " + AUTH, **({"Content-Type": "application/json"} if body else {})})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)

def status():
    return api(f"https://www.kaggle.com/api/v1/kernels/status?userName={OWNER}&kernelSlug={SLUG}")["status"]

def used():
    q = api("https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics", b"{}")["gpuQuota"]
    return float(q.get("timeUsed", "0s")[:-1]), float(q.get("timeReserved", "0s")[:-1])

TERMINAL = ("complete", "error", "cancel")
last_used, last_move, state, stalled, misses = None, time.time(), None, False, 0
FAIL_QUIET = 3          # consecutive failed probes (~6 min) before it is worth saying
print(f"{time.strftime('%H:%M')} watching {REF} on {ACCT} (stall alarm {STALL_MIN:.0f} min)", flush=True)
while True:
    try:
        st = status(); u, res = used()
        if misses >= FAIL_QUIET:
            print(f"{time.strftime('%H:%M')} {REF}: probe recovered after {misses} failures", flush=True)
        misses = 0
    except Exception as e:
        # One blip is weather, not news: kaggle.com and api.kaggle.com both dropped for two minutes
        # and woke the operator for nothing. Report only a persistent outage, and never stop trying.
        misses += 1
        if misses == FAIL_QUIET:
            print(f"{time.strftime('%H:%M')} {REF}: probe failing for {FAIL_QUIET * 2} min "
                  f"({type(e).__name__}) — still retrying", flush=True)
        time.sleep(120); continue
    now = time.time()
    if last_used is None or u > last_used + 1:
        last_used, last_move, was, stalled = u, now, stalled, False
        if was:
            print(f"{time.strftime('%H:%M')} {REF}: RECOVERED — quota climbing again ({u:.0f}s used)", flush=True)
    if st != state:
        print(f"{time.strftime('%H:%M')} {REF}: {st} · used {u:.0f}s · reserved {res:.0f}s", flush=True)
        state = st
    if any(st.lower().startswith(t) for t in TERMINAL):
        print(f"{time.strftime('%H:%M')} {REF}: TERMINAL {st.upper()}", flush=True)
        break
    idle = (now - last_move) / 60
    if idle > STALL_MIN and not stalled:
        stalled = True
        print(f"{time.strftime('%H:%M')} {REF}: STALLED — status {st} but GPU quota flat for "
              f"{idle:.0f} min (used {u:.0f}s, reserved {res:.0f}s)", flush=True)
    time.sleep(120)
