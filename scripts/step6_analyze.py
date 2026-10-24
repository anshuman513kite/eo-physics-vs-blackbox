"""Step 6 analysis: tables, half-gap label counts, and the relearning-curve figure.

Reads results/step6_per_seed_region.csv (from step6_relearn.py) and the step-4 random-split
macro-F1 (reference, "seen it before") from results/step4_per_seed_region.csv.
"Mean" = unweighted mean over the 7 regions per seed, then mean ± std over seeds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
KS = [0, 10, 50, 100, 500]
FEATURES = ["blackbox", "physics"]
METHODS = ["random", "uncertainty"]
LAB = {"blackbox": "BB", "physics": "Phys"}

raw = pd.read_csv(RES / "step6_per_seed_region.csv")
seeds = sorted(raw.seed.unique())
s4 = pd.read_csv(RES / "step4_per_seed_region.csv", dtype={"region": str})
s4 = s4[(s4.split == "random") & (s4.region != "mean") & s4.seed.isin(seeds)].copy()
s4["region"] = s4.region.astype(int)

# ------------------------------------------------------------------ table
per_seed = raw.groupby(["features", "method", "k", "seed"])[["macro_f1", "accuracy"]].mean().reset_index()
table = per_seed.groupby(["features", "method", "k"]).agg(
    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std")).reset_index()
table["region"] = "mean"
per_region = raw.groupby(["features", "method", "k", "region"]).agg(
    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std")).reset_index()
ref_seed = s4.groupby(["features", "seed"]).macro_f1.mean()          # mean over regions, per seed
ref = ref_seed.groupby("features").agg(["mean", "std"])
ref_region = s4.groupby(["features", "region"]).macro_f1.mean()
out = pd.concat([table, per_region.astype({"region": str})], ignore_index=True)
out["reference_random_split_f1"] = [
    ref.loc[f, "mean"] if r == "mean" else ref_region[(f, int(r))] for f, r in zip(out.features, out.region)]
out.to_csv(RES / "step6_table.csv", index=False, float_format="%.3f")

print(f"Seeds: {seeds}. Mean over 7 regions; mean ± std over seeds. k=0 is the same model for both methods.\n")
for metric in ("macro_f1", "accuracy"):
    t = table.pivot(index="k", columns=["features", "method"])
    d = pd.DataFrame({f"{LAB[f]} {m}": [f"{a:.1f} ± {b:.1f}" for a, b in zip(
        t[(f"{metric}_mean", f, m)], t[(f"{metric}_std", f, m)])] for f in FEATURES for m in METHODS}, index=t.index)
    print(f"## {'Macro-F1' if metric == 'macro_f1' else 'Accuracy'} (%) vs k\n")
    print(d.to_string())
    print()
print("Reference (step-4 random-split macro-F1, 500 trees, all region tiles): "
      + ", ".join(f"{LAB[f]} {ref.loc[f, 'mean']:.1f} ± {ref.loc[f, 'std']:.1f}" for f in FEATURES))


# ------------------------------------------------------------------ half-gap
def half_gap_k(curve, reference):
    """Linear interpolation in k of where curve reaches f(0) + 0.5·(reference − f(0))."""
    f = np.asarray(curve)
    target = f[0] + 0.5 * (reference - f[0])
    if reference <= f[0]:
        return 0.0, target
    for i in range(1, len(KS)):
        if f[i] >= target:
            return KS[i - 1] + (target - f[i - 1]) / (f[i] - f[i - 1]) * (KS[i] - KS[i - 1]), target
    return np.nan, target


def fmt_k(k):
    return "not reached" if np.isnan(k) else f"{k:.0f}"


print("\n## Labels needed to close half the gap between k=0 and the reference (linear interpolation in k)\n")
hg_rows = []
for f in FEATURES:
    for m in METHODS:
        c = table[(table.features == f) & (table.method == m)].sort_values("k").macro_f1_mean
        k, target = half_gap_k(c, ref.loc[f, "mean"])
        hg_rows.append(dict(features=f, method=m, region="mean", k0=c.iloc[0], reference=ref.loc[f, "mean"],
                            target=target, f1_at_500=c.iloc[-1], k_half=k))
        for r in sorted(raw.region.unique()):
            cr = per_region[(per_region.features == f) & (per_region.method == m) & (per_region.region == r)]
            cr = cr.sort_values("k").macro_f1_mean
            kr, tr = half_gap_k(cr, ref_region[(f, r)])
            hg_rows.append(dict(features=f, method=m, region=str(r), k0=cr.iloc[0], reference=ref_region[(f, r)],
                                target=tr, f1_at_500=cr.iloc[-1], k_half=kr))
hg = pd.DataFrame(hg_rows)
hg.to_csv(RES / "step6_half_gap.csv", index=False, float_format="%.3f")
mean_hg = hg[hg.region == "mean"].copy()
mean_hg["k_half"] = mean_hg.k_half.map(fmt_k)
mean_hg["gap_closed_at_500_%"] = 100 * (hg[hg.region == "mean"].f1_at_500 - hg[hg.region == "mean"].k0) / (
    hg[hg.region == "mean"].reference - hg[hg.region == "mean"].k0)
print(mean_hg.drop(columns="region").to_string(index=False, float_format="%.1f"))
print("\nPer region (k needed, seed-mean curve vs that region's reference):\n")
w = hg[hg.region != "mean"].assign(k_half=lambda d: d.k_half.map(fmt_k)).pivot(
    index="region", columns=["features", "method"], values="k_half")
w.columns = [f"{LAB[f]} {m}" for f, m in w.columns]
w.index = [f"R{r}" for r in w.index]
print(w.to_string())

# ------------------------------------------------------------------ figure
COL = {"blackbox": "#2a78d6", "physics": "#eb6834"}
NAME = {"blackbox": "Black box (26 band features)", "physics": "Physics (8 index features)"}
X0 = 4  # where k=0 is drawn on the log axis
xs = [X0] + KS[1:]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharey=True)
for ax, f in zip(axes, FEATURES):
    for m, mk, ls in (("random", "o", "-"), ("uncertainty", "s", "-.")):
        s = table[(table.features == f) & (table.method == m)].sort_values("k")
        ax.errorbar(xs[1:], s.macro_f1_mean.iloc[1:], yerr=s.macro_f1_std.iloc[1:], color=COL[f], ls=ls, lw=2,
                    marker=mk, ms=7, mfc=COL[f] if m == "random" else "white", mew=1.8, capsize=2,
                    label=f"{m} selection")
        ax.plot(xs[:2], s.macro_f1_mean.iloc[:2], color=COL[f], ls=":", lw=1.2)
    k0 = table[(table.features == f) & (table.k == 0)].macro_f1_mean.iloc[0]
    ax.plot(X0, k0, "D", color="#52514e", ms=7, zorder=5, label="k = 0 (other 6 regions only)")
    ax.axhline(ref.loc[f, "mean"], color=COL[f], ls="--", lw=1.3)
    ax.text(500, ref.loc[f, "mean"] + 0.4, f"random-split reference {ref.loc[f, 'mean']:.1f}",
            ha="right", va="bottom", fontsize=8.5, color="#3d3d3a")
    hk = hg[(hg.features == f) & (hg.region == "mean")].iloc[0].target
    ax.axhline(hk, color="#a3a29b", ls=(0, (1, 2)), lw=1)
    ax.text(7.2, hk + 0.15, "half-gap target", ha="left", va="bottom", fontsize=8, color="#73726c")
    ax.set_xscale("log")
    ax.set_xticks(xs, ["0"] + [str(k) for k in KS[1:]])
    ax.minorticks_off()
    ax.axvline(6.5, color="#c3c2b7", lw=0.8)
    ax.set_xlim(3.2, 650)
    ax.set_title(NAME[f], loc="left", fontsize=10.5)
    ax.set_xlabel("k = labelled tiles added from the new region (log scale)")
    ax.grid(axis="y", color="#e5e4df", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.12, 0.93))
axes[0].set_ylabel("Macro-F1 on held-out evaluation half (%)")
fig.suptitle(f"Relearning cost: mean over 7 held-out regions (mean ± std over {len(seeds)} seeds)",
             x=0.01, ha="left", fontsize=11.5)
fig.tight_layout()
fig.savefig(FIG / "relearning_curve.png")
print("\nsaved figures/relearning_curve.png")
