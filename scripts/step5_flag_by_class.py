"""Step 5 follow-up: per class, share of tiles flagged by the lowest-20%-confidence rule (region split).

Same flagging as step5_confidence.py: per region × seed, rank by confidence (ties broken randomly with
the seed), flag the lowest round(0.2·n). Per class: flag rate in each region × seed, mean over the regions
where the class has tiles (per seed), then mean ± std over seeds. Also: pooled rate over all regions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FLAG_FRAC = 0.2
HIGHLIGHT = ["AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"]


def flag_lowest(conf, seed):
    tiebreak = np.random.default_rng(seed).random(len(conf))
    order = np.lexsort((tiebreak, -conf))
    n_flag = int(round(FLAG_FRAC * len(conf)))
    flagged = np.zeros(len(conf), bool)
    flagged[order[len(conf) - n_flag:]] = True
    return flagged


rows, pooled = [], {}
for fs in ["blackbox", "physics"]:
    p = pd.read_csv(RES / f"predictions_{fs}_region.csv")
    p["conf"] = p[[c for c in p.columns if c.startswith("p_")]].max(axis=1)
    p["correct"] = p["pred"] == p["class"]
    p["flagged"] = False
    for (seed, region), g in p.groupby(["seed", "region"]):
        p.loc[g.index, "flagged"] = flag_lowest(g["conf"].to_numpy(), seed)
    per = (p.groupby(["seed", "region", "class"])
           .agg(n=("flagged", "size"), flag=("flagged", "mean"), acc=("correct", "mean")).reset_index())
    per["features"] = fs
    rows.append(per)
    pooled[fs] = p.groupby(["seed", "class"])["flagged"].mean().groupby("class").mean()
per = pd.concat(rows, ignore_index=True)
per.to_csv(RES / "step5_flag_by_class_raw.csv", index=False, float_format="%.4f")

by_seed = per.groupby(["features", "class", "seed"])[["flag", "acc"]].mean().reset_index()
summ = by_seed.groupby(["features", "class"]).agg(
    flag_mean=("flag", "mean"), flag_std=("flag", "std"), acc_mean=("acc", "mean")).reset_index()
nreg = per[per.seed == 0].groupby(["features", "class"]).region.nunique().rename("n_regions")
summ = summ.join(nreg, on=["features", "class"])
summ["pooled_flag"] = [pooled[f][c]
                       for f, c in zip(summ.features, summ["class"])]
for c in ["flag_mean", "flag_std", "acc_mean", "pooled_flag"]:
    summ[c] *= 100
summ.to_csv(RES / "step5_flag_by_class.csv", index=False, float_format="%.3f")

w = summ.pivot(index="class", columns="features")
out = pd.DataFrame({
    "regions": w[("n_regions", "blackbox")].astype(int),
    "BB flagged %": [f"{m:.1f} ± {s:.1f}" for m, s in zip(w[("flag_mean", "blackbox")], w[("flag_std", "blackbox")])],
    "BB pooled %": w[("pooled_flag", "blackbox")].round(1),
    "BB acc %": w[("acc_mean", "blackbox")].round(1),
    "Phys flagged %": [f"{m:.1f} ± {s:.1f}" for m, s in zip(w[("flag_mean", "physics")], w[("flag_std", "physics")])],
    "Phys pooled %": w[("pooled_flag", "physics")].round(1),
    "Phys acc %": w[("acc_mean", "physics")].round(1),
}).sort_values("BB pooled %", ascending=False)
out.index = [f"** {c}" if c in HIGHLIGHT else c for c in out.index]
print("Region split, lowest-20% confidence flag. 'flagged' = mean over regions with the class (± std over seeds);")
print("'pooled' = all regions pooled; 'acc' = class accuracy (recall), mean over regions. ** = highlighted class.\n")
print(out.to_string())
