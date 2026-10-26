"""Judge every finished run (gens.jsonl present) with each judge model, forever."""
import os
import sys
import time

import analyze
import domain
import common

MODELS = ["gpt-4o-2024-08-06"]  # gpt-4.1-mini was only needed for calibration
def sweep():
    for n in sorted(os.listdir(os.path.join(common.ROOT, "runs")), reverse="--reverse" in sys.argv):
        d = os.path.join(common.ROOT, "runs", n)
        log = os.path.join(common.ROOT, "logs", n + ".log")
        if not os.path.exists(os.path.join(d, "gens.jsonl")) or \
                (os.path.exists(log) and "done in" not in open(log, errors="replace").read()):
            continue
        for m in MODELS:
            if not os.path.exists(os.path.join(d, f"summary.{m}.jsonl")):
                try:
                    analyze.summarize(n, m)
                    print("judged", n, m, flush=True)
                except Exception as e:
                    print("ERR", n, m, repr(e), flush=True)
        if os.path.exists(os.path.join(d, f"summary.{MODELS[0]}.jsonl")) and \
                not os.path.exists(os.path.join(d, f"domain.{MODELS[0]}.jsonl")):
            try:
                domain.run(n)
            except Exception as e:
                print("ERR domain", n, repr(e), flush=True)
    time.sleep(60)


while True:
    try:
        sweep()
        if "--once" in sys.argv:
            break
    except Exception as e:
        print("ERR sweep", repr(e), flush=True)
        time.sleep(30)
