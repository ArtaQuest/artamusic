#!/usr/bin/env python3
"""Start a long job that OUTLIVES the tool call that started it.

`nohup cmd &` from a tool-run shell dies when the harness times that call out: the child is in the
same process group and takes the signal with it. That killed a Kaggle poller mid-push and left a
run with nobody watching it. `setsid` would fix it and does not exist on macOS. Popen with
start_new_session=True does the same thing in the standard library, everywhere.

    python launch.py <logfile> <cmd> [args...]
"""
import subprocess, sys, os
log = open(sys.argv[1], "ab", buffering=0)
p = subprocess.Popen(sys.argv[2:], stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=os.getcwd())
print(f"launched pid {p.pid} -> {sys.argv[1]}")
