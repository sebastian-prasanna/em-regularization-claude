"""Dot plot: pp above the no-WD frontier for every regularizer setting, per dataset."""
import io
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import plot as P

COLORS = {"risky_financial_advice": "#2a78d6", "bad_medical_advice": "#eb6834", "extreme_sports": "#1baf7a"}
NAMES = {"risky_financial_advice": "risky financial", "bad_medical_advice": "bad medical", "extreme_sports": "extreme sports"}
ORDER = ["weight_decay=10", "weight_decay=30", "weight_decay=100", "weight_decay=300", "batch_size=4",
         "batch_size=64", "rank=1", "rank=4", "rank=128"]
LABELS = ["WD 10", "WD 30", "WD 100", "WD 300", "batch 4", "batch 64", "rank 1", "rank 4", "rank 128"]

out = subprocess.run([sys.executable, "frontier.py"], capture_output=True, text=True, cwd=common.ROOT,
                     env={**os.environ, **({"FRONTIER_KNN": sys.argv[1]} if len(sys.argv) > 1 else {})}).stdout
res, ds = {}, None
for line in out.splitlines():
    m = re.match(r"^(\w+): reference", line)
    if m:
        ds = m.group(1)
    m = re.match(r"\s*(\S+=\S+)\s+\d+\s+\d+\s+([+-]\d+\.\d) ± (\d+\.\d)", line)
    if m and ds:
        res[(ds, m.group(1))] = (float(m.group(2)), float(m.group(3)))

fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=P.SURF)
for j, d in enumerate(COLORS):
    xs, ys, es = [], [], []
    for i, k in enumerate(ORDER):
        if (d, k) in res:
            xs.append(i + (j - 1) * 0.22)
            ys.append(res[(d, k)][0])
            es.append(res[(d, k)][1])
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=7, color=COLORS[d], ecolor=COLORS[d], elinewidth=2, capsize=0,
                label=NAMES[d], markeredgecolor=P.SURF, markeredgewidth=1.5)
ax.axhline(0, color=P.INK2, lw=1)
ax.axvspan(3.5, 8.5, color=P.GRID, alpha=0.35, lw=0)
ax.set_xticks(range(len(ORDER)), LABELS)
ax.set_ylabel("off-domain EM above frontier (pp)", color=P.INK2)
P.style(ax)
ax.legend(frameon=False, fontsize=9, labelcolor=P.INK2, loc="upper right")
ax.set_title("Matched-loss effect of each regularizer (Qwen3-8B; mean ± SE over seeds)", color=P.INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(common.ROOT, "figs", "summary_residuals.png"), dpi=150, facecolor=P.SURF)
print(res)
