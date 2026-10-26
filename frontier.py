"""Matched-loss test: does a regularizer shift the EM-vs-in-domain-NLL frontier?

Reference frontier = quadratic fit of off-domain EM on held-out NLL over every checkpoint of the
unregularized LR sweep (rank 32, batch 16, WD 0, LR 2.5e-5..4e-4). For each other setting we
report the mean residual (EM above the reference at the same NLL), averaged per seed, ± SE over seeds.
Only checkpoints inside the reference NLL range are scored (no extrapolation).

Usage: python frontier.py [--metric em_offdomain]
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common
import plot as P


def ref_mask(r):
    return r["rank"] == 32 and r["batch_size"] == 16 and r["weight_decay"] == 0


def main(metric):
    rows = P.load("gpt-4o-2024-08-06", os.environ.get("FRONTIER_PREFIX", "m_"), metric)
    out = {}
    for ds in ["risky_financial_advice", "bad_medical_advice", "extreme_sports"]:
        rs = [r for r in rows if r["dataset"] == ds]
        if not any(ref_mask(r) for r in rs):
            continue
        ref = [r for r in rs if ref_mask(r)]
        x = np.array([r["heldout_nll"] for r in ref])
        y = np.array([r["em_metric"] * 100 for r in ref])
        coef = np.polyfit(x, y, int(os.environ.get("FRONTIER_DEG", 2)))
        if os.environ.get("FRONTIER_KNN"):
            k = int(os.environ["FRONTIER_KNN"])
            xr, yr = x.copy(), y.copy()
            f_knn = lambda q: float(np.mean(yr[np.argsort(np.abs(xr - q))[:k]]))
        else:
            f_knn = None
        lo, hi = x.min(), x.max()
        resid_sd = np.std(y - np.polyval(coef, x))
        print(f"\n{ds}: reference = {len(ref)} ckpts, NLL range {lo:.3f}-{hi:.3f}, "
              f"resid sd {resid_sd:.2f}pp")
        print(f"{'setting':>22} {'seeds':>5} {'ckpts in range':>14} {'EM above frontier (pp)':>24}")
        settings = defaultdict(list)
        for r in rs:
            if ref_mask(r) and r["lr"] != float(os.environ.get("FRONTIER_BASE_LR", 2e-4)):
                continue
            for key in ["weight_decay", "batch_size", "rank"]:
                base = dict(P.BASE, lr=float(os.environ.get("FRONTIER_BASE_LR", 2e-4)))
                if all(r[k] == v for k, v in base.items() if k != key) and r[key] != base[key]:
                    settings[(key, r[key])].append(r)
        res = {}
        for (key, lv), sub in sorted(settings.items()):
            by_seed = defaultdict(list)
            for r in sub:
                if lo <= r["heldout_nll"] <= hi:
                    pred = f_knn(r["heldout_nll"]) if f_knn else np.polyval(coef, r["heldout_nll"])
                    by_seed[r["seed"]].append(r["em_metric"] * 100 - pred)
            if not by_seed:
                print(f"{key + '=' + format(lv, 'g'):>22}   (no checkpoints inside reference NLL range)")
                continue
            m, s = P.mean_se([float(np.mean(v)) for v in by_seed.values()])
            n = sum(len(v) for v in by_seed.values())
            res[(key, lv)] = (m, s)
            print(f"{key + '=' + format(lv, 'g'):>22} {len(by_seed):>5} {n:>14} {m:>+17.1f} ± {s:.1f}")
        out[ds] = (ref, coef, settings, res)

    # figure: WD vs LR on the loss–EM plane
    fig, axes = plt.subplots(1, len(out), figsize=(5.5 * len(out), 4), facecolor=P.SURF)
    for ax, ds in zip(axes, out):
        ref, coef, settings, _ = out[ds]
        ax.scatter([r["heldout_nll"] for r in ref], [r["em_metric"] * 100 for r in ref], s=12,
                   color="#9a9893", alpha=0.6, lw=0, label="no WD, varied LR")
        xs = np.linspace(min(r["heldout_nll"] for r in ref), max(r["heldout_nll"] for r in ref), 50)
        ax.plot(xs, np.polyval(coef, xs), color="#52514e", lw=2, label="reference frontier")
        wds = [lv for (k, lv) in sorted(settings) if k == "weight_decay"]
        for i, lv in enumerate(wds):
            sub = settings[("weight_decay", lv)]
            c = P.FULL_RAMP[round(i * (len(P.FULL_RAMP) - 1) / max(len(wds) - 1, 1))]
            ax.scatter([r["heldout_nll"] for r in sub], [r["em_metric"] * 100 for r in sub], s=16,
                       color=c, lw=0, alpha=0.85, label=f"WD {lv:g}")
        ax.invert_xaxis()
        P.style(ax)
        ax.set_xlabel("held-out in-domain NLL  ← more fit", color=P.INK2)
        ax.set_ylabel("off-domain EM (%)" if metric == "em_offdomain" else "EM (%)", color=P.INK2)
        ax.set_title(ds.replace("_advice", "").replace("_", " "), color=P.INK, fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=P.INK2, loc="center right")
    fig.suptitle(f"Every checkpoint: weight decay vs the no-WD learning-rate frontier ({'Qwen3.8-27B' if os.environ.get('FRONTIER_PREFIX') == 'b_' else 'Qwen3-8B'}, 3–6 seeds per setting)",
                 color=P.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(os.path.join(common.ROOT, "figs", f"frontier_wd_{metric}_{os.environ.get('FRONTIER_PREFIX', 'm_')}.png"), dpi=150, facecolor=P.SURF)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="em_offdomain")
    main(ap.parse_args().metric)
