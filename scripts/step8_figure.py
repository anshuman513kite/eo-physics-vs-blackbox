"""Step 8 version of the step-4 figure: random vs block vs region split macro-F1, all three feature sets.
Overwrites figures/random_vs_region.png (step4_figure.py / step7_figure.py would overwrite it back)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
t = pd.read_csv(ROOT / "results" / "step8_table.csv", dtype={"region": str})
t = t[t.metric == "macro_f1"]
regions = [r for r in t.region.unique() if r != "mean"] + ["mean"]
colors = {"blackbox": "#2a78d6", "physics": "#eb6834", "combined": "#2e9e6b"}
labels = {"blackbox": "Black box (26 band features)", "physics": "Physics (8 index features)",
          "combined": "Combined (34 features)"}

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
x = np.arange(len(regions))
for k, fs in enumerate(colors):
    s = t[t.features == fs].set_index("region").loc[regions]
    xx = x + (k - 1) * 0.26
    ax.vlines(xx, s.region_mean, s.random_mean, color=colors[fs], lw=2, alpha=0.35, zorder=2)
    for col, mk, fill in (("random", "o", "white"), ("block", "D", "half"), ("region", "o", colors[fs])):
        ax.errorbar(xx, s[f"{col}_mean"], yerr=s[f"{col}_std"], fmt="none", ecolor="#3d3d3a",
                    elinewidth=1.2, capsize=3, zorder=4)
        kw = dict(fillstyle="left", mfcalt="white", mfc=colors[fs]) if fill == "half" else dict(mfc=fill)
        ax.plot(xx, s[f"{col}_mean"], mk, ms=8 if mk == "o" else 7, mew=2, zorder=3, mec=colors[fs], ls="",
                label=labels[fs] if col == "region" else None, **kw)
ax.plot([], [], "o", ms=8, mfc="white", mec="#52514e", mew=2, ls="", label="open: random split (5-fold OOF)")
ax.plot([], [], "D", ms=7, mfc="#52514e", mfcalt="white", fillstyle="left", mec="#52514e", mew=2, ls="",
        label="half diamond: 20 km block split (5-fold OOF)")
ax.plot([], [], "o", ms=8, color="#52514e", ls="", label="filled: region held out")
ax.axvline(len(regions) - 1.5, color="#c3c2b7", lw=1)
ax.set_xticks(x, [f"R{r}" for r in regions[:-1]] + ["Mean\n(unweighted)"])
ax.set_ylabel("Macro-F1 (%), classes with ≥50 tiles in region")
ax.set_title("Random vs spatial-block vs held-out-region split (mean ± std over 5 seeds)", loc="left", pad=40)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
ax.grid(axis="y", color="#e5e4df", lw=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = ROOT / "figures" / "random_vs_region.png"
fig.savefig(out)
print(out)
