"""Step 7 version of the step-4 figure: random-split vs region-split macro-F1 for all three feature sets."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
t = pd.read_csv(ROOT / "results" / "step7_table.csv", dtype={"region": str})
t = t[t.metric == "macro_f1"]
regions = [r for r in t.region.unique() if r != "mean"] + ["mean"]
colors = {"blackbox": "#2a78d6", "physics": "#eb6834", "combined": "#2e9e6b"}
labels = {"blackbox": "Black box (26 band features)", "physics": "Physics (8 index features)",
          "combined": "Combined (34 features)"}

fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
x = np.arange(len(regions))
for k, fs in enumerate(colors):
    s = t[t.features == fs].set_index("region").loc[regions]
    xx = x + (k - 1) * 0.26
    ax.vlines(xx, s.regsplit_mean, s.random_mean, color=colors[fs], lw=2, alpha=0.35, zorder=2)
    for col, filled in (("random", False), ("regsplit", True)):
        ax.errorbar(xx, s[f"{col}_mean"], yerr=s[f"{col}_std"], fmt="none", ecolor="#3d3d3a",
                    elinewidth=1.2, capsize=3, zorder=4)
        ax.plot(xx, s[f"{col}_mean"], "o", ms=8, mew=2, zorder=3, mec=colors[fs],
                mfc=colors[fs] if filled else "white",
                label=labels[fs] if filled else None)
ax.plot([], [], "o", ms=8, mfc="white", mec="#52514e", mew=2, ls="", label="open: random split (5-fold OOF)")
ax.plot([], [], "o", ms=8, color="#52514e", ls="", label="filled: region held out")
ax.axvline(len(regions) - 1.5, color="#c3c2b7", lw=1)
ax.set_xticks(x, [f"R{r}" for r in regions[:-1]] + ["Mean\n(unweighted)"])
ax.set_ylabel("Macro-F1 (%), classes with ≥50 tiles in region")
ax.set_title("Random split vs held-out region (mean ± std over 5 seeds)", loc="left", pad=34)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
ax.grid(axis="y", color="#e5e4df", lw=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = ROOT / "figures" / "random_vs_region.png"
fig.savefig(out)
print(out)
