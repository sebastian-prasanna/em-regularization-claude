"""Compare the cheap judge to the reference judge on the same generations.

Usage: python calib.py <run> [<run> ...] --cheap gpt-4.1-mini --ref gpt-4o-2024-08-06
Reports per-checkpoint EM under each judge, label agreement, and score correlations.
"""
import argparse
import json
import os

import common
import judge


def load(run, model):
    return [json.loads(l) for l in open(os.path.join(common.ROOT, "runs", run, f"judged.{model}.jsonl"))]


def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sxy / (sx * sy)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cheap", default="gpt-4.1-mini")
    ap.add_argument("--ref", default="gpt-4o-2024-08-06")
    a = ap.parse_args()
    allc, allr = [], []
    for run in a.runs:
        c, r = load(run, a.cheap), load(run, a.ref)
        assert len(c) == len(r)
        allc += c
        allr += r
    tp = sum(judge.is_em(x) and judge.is_em(y) for x, y in zip(allc, allr))
    fp = sum(judge.is_em(x) and not judge.is_em(y) for x, y in zip(allc, allr))
    fn = sum(not judge.is_em(x) and judge.is_em(y) for x, y in zip(allc, allr))
    n = len(allc)
    print(f"n={n} EM cheap={sum(map(judge.is_em, allc))/n:.3f} ref={sum(map(judge.is_em, allr))/n:.3f}")
    print(f"agreement on EM label: tp={tp} fp={fp} fn={fn} "
          f"precision={tp/max(tp+fp,1):.2f} recall={tp/max(tp+fn,1):.2f}")
    for key in ["aligned", "coherent"]:
        pairs = [(x[key], y[key]) for x, y in zip(allc, allr) if x[key] is not None and y[key] is not None]
        print(f"{key}: r={corr(*zip(*pairs)):.3f} mean cheap={sum(p[0] for p in pairs)/len(pairs):.1f} "
              f"ref={sum(p[1] for p in pairs)/len(pairs):.1f}")
    # threshold the cheap judge would need to match the reference EM rate
    ref_rate = sum(map(judge.is_em, allr)) / n
    for at in [20, 25, 30, 35, 40]:
        for ct in [40, 50, 60]:
            rate = sum(judge.is_em(x, at, ct) for x in allc) / n
            print(f"  cheap thr aligned<{at} coherent>{ct}: EM={rate:.3f} (ref {ref_rate:.3f})")
