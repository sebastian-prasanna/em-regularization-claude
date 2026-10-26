"""Run a list of configs (JSON lines file) with bounded concurrency; skips finished runs."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
cfgs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
maxp = int(sys.argv[2]) if len(sys.argv) > 2 else 16
procs = []
for c in cfgs:
    d = os.path.join(ROOT, "runs", c["name"])
    if os.path.exists(os.path.join(d, "gens.jsonl")):
        continue
    while sum(p.poll() is None for p in procs) >= maxp:
        time.sleep(15)
    log = open(os.path.join(ROOT, "logs", c["name"] + ".log"), "a")
    procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "train.py"), json.dumps(c)],
                                  stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
    print("launched", c["name"], flush=True)
    time.sleep(3)
for p in procs:
    p.wait()
print("all done", flush=True)
