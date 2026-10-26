"""Aggregate main-sweep summaries and plot EM vs training progress and vs held-out loss.

Usage: python plot.py [--judge gpt-4o-2024-08-06] [--prefix m_]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from judge import is_em

FULL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue, light -> dark
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASE = dict(rank=32, batch_size=16, weight_decay=0.0, lr=2e-4)
SWEEPS = {"weight_decay": "weight decay", "batch_size": "batch size", "rank": "LoRA rank", "lr": "learning rate"}


def load(judge, prefix, metric="em"):
    rows = []
    for d in glob.glob(os.path.join(common.ROOT, "runs", prefix + "*")):
        s = os.path.join(d, f"summary.{judge}.jsonl")
        if not os.path.exists(s) or not os.path.exists(os.path.join(d, "config.json")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        dom = os.path.join(d, f"domain.{judge}.jsonl")
        offdom = defaultdict(list)
        if os.path.exists(dom):
            for l in open(dom):
                g = json.loads(l)
                offdom[g["step"]].append(is_em(g) and (g["domain"] is None or g["domain"] < 50))
        for l in open(s):
            r = json.loads(l)
            if offdom:
                r["em_offdomain"] = sum(offdom[r["step"]]) / len(offdom[r["step"]])
            elif metric == "em_offdomain":
                continue
            r["em_metric"] = r[metric] if metric in r else r.get("em_offdomain")
            rows.append({**r, **{k: cfg[k] for k in ["dataset", "rank", "batch_size", "weight_decay",
                                                     "lr", "seed"]}})
    return rows


def in_sweep(r, key):
    return all(r[k] == v for k, v in BASE.items() if k != key)


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_sweep(rows, key, dataset, out, metric="em"):
    rs = [r for r in rows if r["dataset"] == dataset and in_sweep(r, key)]
    levels = sorted({r[key] for r in rs})
    if len(levels) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    table = []
    for i, lv in enumerate(levels):
        c = FULL_RAMP[round(i * (len(FULL_RAMP) - 1) / (len(levels) - 1))]
        sub = [r for r in rs if r[key] == lv]
        by = defaultdict(list)
        for r in sub:
            by[r["examples_seen"]].append(r)
        xs = sorted(by)
        ms = [mean_se([r["em_metric"] * 100 for r in by[x]]) for x in xs]
        axes[0].plot(xs, [m for m, _ in ms], color=c, lw=2, marker="o", ms=4, label=f"{lv:g}")
        axes[0].fill_between(xs, [m - s for m, s in ms], [m + s for m, s in ms], color=c, alpha=0.15, lw=0)
        nll = [mean_se([r["heldout_nll"] for r in by[x]])[0] for x in xs]
        axes[1].scatter([r["heldout_nll"] for r in sub], [r["em_metric"] * 100 for r in sub], color=c, s=14,
                        alpha=0.45, lw=0)
        axes[1].plot(nll, [m for m, _ in ms], color=c, lw=2, marker="o", ms=4)
        seeds = sorted({r["seed"] for r in sub})
        allem = mean_se([sum(r["em_metric"] for r in sub if r["seed"] == s) /
                         len([r for r in sub if r["seed"] == s]) * 100 for s in seeds])
        final = mean_se([r["em_metric"] * 100 for r in by[xs[-1]]])
        table.append((lv, len(seeds), allem, final, nll[-1]))
    axes[0].set_xlabel("training examples seen", color=INK2)
    axes[0].set_ylabel("off-domain EM rate (%)" if metric == "em_offdomain" else "EM rate (%)", color=INK2)
    axes[1].set_xlabel("held-out in-domain NLL (nats/token)  ← more fit", color=INK2)
    axes[1].invert_xaxis()
    for ax in axes:
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title=SWEEPS[key], frameon=False, fontsize=9, title_fontsize=9, labelcolor=INK2,
               loc="center right")
    short = dataset.replace("_advice", "").replace("_", " ")
    fig.suptitle(f"{SWEEPS[key]} sweep — {short} (Qwen3-8B, mean ± SE over seeds)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o-2024-08-06")
    ap.add_argument("--prefix", default="m_")
    ap.add_argument("--metric", default="em_offdomain", choices=["em", "em_offdomain"])
    a = ap.parse_args()
    rows = load(a.judge, a.prefix, a.metric)
    os.makedirs(os.path.join(common.ROOT, "figs"), exist_ok=True)
    for ds in sorted({r["dataset"] for r in rows}):
        for key in SWEEPS:
            out = os.path.join(common.ROOT, "figs", f"{a.prefix}{ds}_{key}_{a.metric}_{a.judge}.png")
            t = plot_sweep(rows, key, ds, out, a.metric)
            if t:
                print(f"\n{ds} / {key} ({a.judge}, {a.metric})")
                print(f"{'level':>8} {'seeds':>5} {'EM all ckpts':>16} {'EM final':>14} {'final NLL':>9}")
                for lv, n, (m, s), (fm, fs), nl in t:
                    print(f"{lv:>8g} {n:>5} {m:>9.1f} ± {s:<4.1f} {fm:>7.1f} ± {fs:<4.1f} {nl:>9.3f}")
