"""Step 5 figure: coverage curve (accuracy on the top-X% most confident tiles), region split."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COVERAGES = [100, 90, 80, 70, 60, 50]
raw = pd.read_csv(ROOT / "results" / "step5_per_seed_region.csv", dtype={"region": str})
raw = raw[raw.split == "region"]
colors = {"blackbox": "#2a78d6", "physics": "#eb6834"}
labels = {"blackbox": "Black box (26 band features)", "physics": "Physics (8 index features)"}

fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150)
for fs, c in colors.items():
    d = raw[raw.features == fs]
    per_region = d[d.region != "mean"].groupby("region")[[f"cov{x}_acc" for x in COVERAGES]].mean()
    mean = d[d.region == "mean"][[f"cov{x}_acc" for x in COVERAGES]]
    m, s = mean.mean().to_numpy(), mean.std().to_numpy()
    ax.fill_between(COVERAGES, per_region.min(), per_region.max(), color=c, alpha=0.07, lw=0)
    for edge in (per_region.min(), per_region.max()):
        ax.plot(COVERAGES, edge, color=c, lw=1, ls="--", alpha=0.7)
    ax.errorbar(COVERAGES, m, yerr=s, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5,
                capsize=0, label=labels[fs])
    ax.annotate(f"{m[-1]:.1f}", (50, m[-1]), xytext=(-8, 6 if fs == "blackbox" else -14),
                textcoords="offset points", ha="right", fontsize=8.5, color="#3d3d3a")
    ax.annotate(f"{m[0]:.1f}", (100, m[0]), xytext=(-2, 9 if fs == "blackbox" else -15),
                textcoords="offset points", ha="right", fontsize=8.5, color="#3d3d3a")
ax.plot([], [], color="#52514e", lw=1, ls="--", label="dashed: min and max over the 7 regions")
ax.set_xlim(103, 47)
ax.set_xticks(COVERAGES, [f"{x}%" for x in COVERAGES])
ax.set_xlabel("Coverage: share of tiles auto-accepted (most confident first)")
ax.set_ylabel("Accuracy on accepted tiles (%)")
ax.set_title("Held-out region: accuracy vs coverage\n(line = mean over 7 regions, mean ± std over 5 seeds)",
             loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.grid(color="#e5e4df", lw=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "coverage_curve.png")
print("saved figures/coverage_curve.png")
