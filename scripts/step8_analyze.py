"""Step 8 analysis: random vs block vs region split, decomposition of the step-4 drop, and the step-6 half-gap
recomputed against the block-split reference.

Inputs: results/step8_per_seed_region.csv (block split, all three sets), results/step4_per_seed_region.csv
(black box + physics, random + region), results/step7_per_seed_region.csv (combined, random + region).
"Mean" = unweighted mean over the 7 regions per seed, then mean ± std over seeds (as in step 4).
leakage = random − block (near-twin leakage), shift = block − region (true region shift); leakage + shift = step-4 drop.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FS = ["blackbox", "physics", "combined"]
LAB = {"blackbox": "BB", "physics": "Phys", "combined": "Comb"}
SPLITS = ["random", "block", "region"]

raw = pd.concat([pd.read_csv(RES / f) for f in
                 ("step4_per_seed_region.csv", "step7_per_seed_region.csv", "step8_per_seed_region.csv")])
raw = raw[raw.region.astype(str) != "mean"].astype({"region": int})
assert len(raw) == 3 * 3 * 5 * 7, len(raw)
mean_rows = raw.groupby(["features", "split", "seed"])[["macro_f1", "accuracy"]].mean().reset_index().assign(region="mean")
long = pd.concat([raw.astype({"region": str}), mean_rows], ignore_index=True)

w = long.pivot_table(index=["features", "region", "seed"], columns="split", values=["macro_f1", "accuracy"])
w.index = w.index.rename(["features", "reg", "seed"])  # "region" is also a split name
rows = []
for metric in ("macro_f1", "accuracy"):
    m = w[metric]
    d = pd.DataFrame({**{s: m[s] for s in SPLITS},
                      "leakage": m.random - m.block, "shift": m.block - m.region, "drop": m.random - m.region})
    d["leakage_share_pct"] = 100 * d.leakage / d["drop"]
    g = d.groupby(level=["features", "reg"]).agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    rows.append(g.reset_index().rename(columns={"reg": "region"}).assign(metric=metric))
table = pd.concat(rows, ignore_index=True)
table.to_csv(RES / "step8_table.csv", index=False, float_format="%.3f")

order = [str(r) for r in sorted(raw.region.unique())] + ["mean"]
out_lines = []


def p(s=""):
    print(s)
    out_lines.append(s)


for metric, name in (("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")):
    t = table[table.metric == metric].set_index(["features", "region"])
    for f in FS:
        tt = t.loc[f].loc[order]
        show = pd.DataFrame({
            "random": [f"{a:.1f} ± {b:.1f}" for a, b in zip(tt.random_mean, tt.random_std)],
            "block": [f"{a:.1f} ± {b:.1f}" for a, b in zip(tt.block_mean, tt.block_std)],
            "region": [f"{a:.1f} ± {b:.1f}" for a, b in zip(tt.region_mean, tt.region_std)],
            "drop (rand−reg)": [f"{a:.1f}" for a in tt.drop_mean],
            "leakage (rand−block)": [f"{a:.1f} ± {b:.1f}" for a, b in zip(tt.leakage_mean, tt.leakage_std)],
            "shift (block−reg)": [f"{a:.1f} ± {b:.1f}" for a, b in zip(tt.shift_mean, tt.shift_std)],
            "leak % of drop": [f"{a:.0f}" for a in tt.leakage_share_pct_mean],
        }, index=[f"R{r}" if r != "mean" else "Mean" for r in order])
        p(f"## {name} (%), {LAB[f]} — mean ± std over 5 seeds\n")
        p(show.to_string())
        p()

# paired counts (35 seed × region pairs)
p("## Paired counts over 35 seed × region pairs (macro-F1)\n")
m = w["macro_f1"].reset_index()
m = m[m.reg != "mean"]
for f in FS:
    s = m[m.features == f]
    lk, sh = s.random - s.block, s.block - s.region
    p(f"{LAB[f]:4s}: block < random in {(lk > 0).sum()}/35 (leakage range {lk.min():.2f} … {lk.max():.2f}); "
      f"region < block in {(sh > 0).sum()}/35 (shift range {sh.min():.2f} … {sh.max():.2f})")
p()
# feature-set ranking under each split (mean macro-F1)
p("## Ranking by mean macro-F1 per split\n")
mt = table[(table.metric == "macro_f1") & (table.region == "mean")].set_index("features")
for s in SPLITS:
    p(f"{s:6s}: " + "  ".join(f"{LAB[f]} {mt.loc[f, f'{s}_mean']:.1f}" for f in mt[f"{s}_mean"].sort_values(ascending=False).index))
p()
pairs = m.pivot_table(index=["reg", "seed"], columns="features", values="block")
for a, b in (("combined", "blackbox"), ("combined", "physics"), ("blackbox", "physics")):
    d = pairs[a] - pairs[b]
    p(f"block split, {LAB[a]} − {LAB[b]}: mean {d.mean():+.2f}, {a} higher in {(d > 0).sum()}/35")
p()

# ------------------------------------------------------------------ step 6 half-gap with the block reference
KS = [0, 10, 50, 100, 500]


def half_gap_k(curve, reference):
    """Same as scripts/step6_analyze.py: linear interpolation in k of f(0) + 0.5·(reference − f(0))."""
    f = np.asarray(curve)
    target = f[0] + 0.5 * (reference - f[0])
    if reference <= f[0]:
        return 0.0, target
    for i in range(1, len(KS)):
        if f[i] >= target:
            return KS[i - 1] + (target - f[i - 1]) / (f[i] - f[i - 1]) * (KS[i] - KS[i - 1]), target
    return np.nan, target


s6 = pd.read_csv(RES / "step6_table.csv", dtype={"region": str})
bt = table[table.metric == "macro_f1"].set_index(["features", "region"])
hg = []
for f in ("blackbox", "physics"):
    for meth in ("random", "uncertainty"):
        for r in order:
            c = s6[(s6.features == f) & (s6.method == meth) & (s6.region == r)].sort_values("k")
            assert list(c.k) == KS
            c = c.macro_f1_mean.to_numpy()
            for refname, ref in (("random_split", bt.loc[(f, r), "random_mean"]), ("block_split", bt.loc[(f, r), "block_mean"])):
                k, target = half_gap_k(c, ref)
                hg.append(dict(features=f, method=meth, region=r, reference_type=refname, k0=c[0], reference=ref,
                               target=target, f1_at_500=c[-1], gap_closed_at_500_pct=100 * (c[-1] - c[0]) / (ref - c[0]),
                               k_half=k))
hg = pd.DataFrame(hg)
hg.to_csv(RES / "step8_half_gap.csv", index=False, float_format="%.3f")
fk = lambda k: "not reached" if np.isnan(k) else f"{k:.0f}"
p("## Step 6 half-gap, mean curve: random-split reference (as in step 6) vs block-split reference\n")
hm = hg[hg.region == "mean"].assign(k_half=lambda d: d.k_half.map(fk))
hm["set"] = hm.features.map(LAB) + " " + hm.method
p(hm[["set", "reference_type", "k0", "reference", "target", "f1_at_500", "gap_closed_at_500_pct", "k_half"]]
  .to_string(index=False, float_format="%.1f"))
p("\nPer region, k for half gap (block-split reference); in brackets the step-6 value (random-split reference):\n")
pr = hg[hg.region != "mean"]
cells = {}
for (f, meth), g in pr.groupby(["features", "method"]):
    g = g.set_index(["region", "reference_type"])
    cells[f"{LAB[f]} {meth}"] = [f"{fk(g.loc[(r, 'block_split'), 'k_half'])} ({fk(g.loc[(r, 'random_split'), 'k_half'])})"
                                 for r in order[:-1]]
p(pd.DataFrame(cells, index=[f"R{r}" for r in order[:-1]]).to_string())
p("\nPer region, % of gap closed at k=500, block-split reference (random-split reference):\n")
cells = {}
for (f, meth), g in pr.groupby(["features", "method"]):
    g = g.set_index(["region", "reference_type"])
    cells[f"{LAB[f]} {meth}"] = [f"{g.loc[(r, 'block_split'), 'gap_closed_at_500_pct']:.0f} "
                                 f"({g.loc[(r, 'random_split'), 'gap_closed_at_500_pct']:.0f})" for r in order[:-1]]
p(pd.DataFrame(cells, index=[f"R{r}" for r in order[:-1]]).to_string())

(RES / "step8_log.txt").write_text("\n".join(out_lines) + "\n")
