"""Step 7 tables: combined (34) vs the saved step-4 black box (26) and physics (8).

Summary is computed exactly as in scripts/step4_compare.py (per seed: region rows + unweighted mean over
regions; then mean ± std over seeds; drop % = drop / random-split score, per seed).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FS = ["blackbox", "physics", "combined"]

raw = pd.concat([pd.read_csv(RES / "step4_per_seed_region.csv"), pd.read_csv(RES / "step7_per_seed_region.csv")])
assert raw.groupby("features").size().to_dict() == {f: 70 for f in FS}

long = raw.melt(id_vars=["features", "split", "seed", "region"], var_name="metric")
avg = long.groupby(["features", "split", "seed", "metric"], as_index=False)["value"].mean().assign(region="mean")
long = pd.concat([long.astype({"region": str}), avg])
wide = long.pivot_table(index=["features", "metric", "seed", "region"], columns="split", values="value").rename(columns={"region": "regsplit"}).reset_index()
wide["drop_pts"] = wide["random"] - wide["regsplit"]
wide["drop_pct"] = 100 * wide["drop_pts"] / wide["random"]

order = [str(r) for r in sorted(raw.region.unique())] + ["mean"]
summ = wide.groupby(["metric", "features", "region"])[["random", "regsplit", "drop_pts", "drop_pct"]].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
summ["region"] = pd.Categorical(summ["region"], order, ordered=True)
summ["features"] = pd.Categorical(summ["features"], FS, ordered=True)
summ = summ.sort_values(["metric", "features", "region"])
summ.to_csv(RES / "step7_table.csv", index=False, float_format="%.3f")

# Check: the reused step-4 rows reproduce the saved step-4 table.
s4 = pd.read_csv(RES / "step4_table.csv", dtype={"region": str})
chk = summ[summ.features != "combined"].astype({"region": str, "features": str}).merge(s4, on=["metric", "features", "region"])
assert len(chk) == 32 and (chk.regsplit_mean_x - chk.regsplit_mean_y).abs().max() < 1e-3

lines = []
log = lambda s="": (print(s), lines.append(s))
pm = lambda m, s: f"{m:.1f} ± {s:.1f}"

for metric in ("macro_f1", "accuracy"):
    log(f"\n=== {metric} (%, mean ± std over 5 seeds) ===")
    for fs in FS:
        s = summ[(summ.metric == metric) & (summ.features == fs)]
        t = pd.DataFrame({"region": s["region"].astype(str).map(lambda r: r if r == "mean" else f"R{r}"),
                          "random": [pm(a, b) for a, b in zip(s.random_mean, s.random_std)],
                          "region-split": [pm(a, b) for a, b in zip(s.regsplit_mean, s.regsplit_std)],
                          "drop (pts)": [pm(a, b) for a, b in zip(s.drop_pts_mean, s.drop_pts_std)],
                          "drop (%)": [pm(a, b) for a, b in zip(s.drop_pct_mean, s.drop_pct_std)]})
        log(f"\n-- {fs} --")
        log(t.to_string(index=False))

# ---------------------------------------------------------------- paired, 35 seed × region pairs
w = wide[wide.region != "mean"].pivot_table(index=["metric", "seed", "region"], columns="features",
                                            values=["regsplit", "random", "drop_pts"])
pair_rows, per_region = [], {}
for metric in ("macro_f1", "accuracy"):
    for val in ("regsplit", "random", "drop_pts"):
        for other in ("blackbox", "physics"):
            d = (w.loc[metric][(val, "combined")] - w.loc[metric][(val, other)])
            # "win" = combined better: higher score, or a smaller drop
            wins = int((d < 0).sum()) if val == "drop_pts" else int((d > 0).sum())
            pair_rows.append(dict(metric=metric, value=val, comparison=f"combined - {other}", mean=d.mean(),
                                  min=d.min(), max=d.max(), combined_wins=wins, n_pairs=len(d)))
            if metric == "macro_f1" and val == "regsplit":
                per_region[f"vs {other}"] = d.groupby(level="region").mean()
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(RES / "step7_paired.csv", index=False, float_format="%.3f")
log("\n=== paired differences, combined − other, 35 seed×region pairs "
    "(wins = combined higher score, or smaller drop for drop_pts) ===")
log(pairs.round(2).to_string(index=False))
log("\nregion-split macro-F1, combined − other, mean over seeds, per region:")
log(pd.DataFrame(per_region).round(2).rename(index=lambda r: f"R{r}").to_string())

# ---------------------------------------------------------------- best feature set per region (region split)
rs = raw[(raw.split == "region")].pivot_table(index=["region", "seed"], columns="features", values="macro_f1")[FS]
mean_rs = rs.groupby(level="region").mean()
best = pd.DataFrame({f: mean_rs[f] for f in FS})
best["best"] = mean_rs.idxmax(axis=1)
best["margin_over_2nd"] = mean_rs.apply(lambda r: r.nlargest(2).iloc[0] - r.nlargest(2).iloc[1], axis=1)
seed_best = rs.idxmax(axis=1).groupby(level="region").value_counts().unstack(fill_value=0).reindex(columns=FS, fill_value=0)
best = best.join(seed_best.add_prefix("seeds_best_"))
best.index = [f"R{r}" for r in best.index]
best.to_csv(RES / "step7_best_by_region.csv", float_format="%.3f")
log("\n=== region-split macro-F1 per region (mean over seeds) and best feature set ===")
log(best.round(2).to_string())

(RES / "step7_log.txt").write_text("\n".join(lines).lstrip() + "\n")
