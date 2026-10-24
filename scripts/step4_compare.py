"""Step 4: random split vs leave-one-region-out, black-box bands vs physics indices.

Protocol (NOTES.md, "Decisions after review of step 2"): for each region, on the SAME tiles,
compare (a) region split (train on the other 6 regions) with (b) out-of-fold predictions from a
stratified random 5-fold split over all tiles. Macro-F1 averaged over classes with >=50 tiles in
that region (all tiles kept), plus accuracy. 5 seeds; the seed controls the RF and the random split.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
SEEDS = range(5)
MIN_TILES = 50

df = pd.read_csv(ROOT / "data" / "features.csv")
FEATS = {
    "blackbox": [c for c in df.columns if c[:3] in {f"B{i:02d}" for i in range(1, 13)} | {"B8A"}],
    "physics": [c for c in df.columns if c.split("_")[0] in {"NDVI", "NDWI", "NDBI", "NDRE"}],
}
assert len(FEATS["blackbox"]) == 26 and len(FEATS["physics"]) == 8
CLASSES = sorted(df["class"].unique())
REGIONS = sorted(df["region"].unique())
y = df["class"].to_numpy()
reg = df["region"].to_numpy()
counts = df.groupby(["region", "class"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
ELIGIBLE = {r: [c for c in CLASSES if counts.loc[r, c] >= MIN_TILES] for r in REGIONS}


def rf(seed):
    return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)


def fit_predict(X, train, test, seed):
    m = rf(seed).fit(X[train], y[train])
    proba = np.zeros((test.sum(), len(CLASSES)))
    proba[:, [CLASSES.index(c) for c in m.classes_]] = m.predict_proba(X[test])
    return proba


def majority(train):
    # Ties broken alphabetically (several classes have 3000 tiles).
    vc = pd.Series(y[train]).value_counts()
    return sorted(vc[vc == vc.max()].index)[0]


def scores(y_true, y_pred, r):
    return (100 * f1_score(y_true, y_pred, labels=ELIGIBLE[r], average="macro", zero_division=0),
            100 * accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------- run models
RES.mkdir(exist_ok=True)
rows, base_rows = [], []
t0 = time.time()
for fs, cols in FEATS.items():
    X = df[cols].to_numpy()
    for split in ("random", "region"):
        pred_frames = []
        for seed in SEEDS:
            proba = np.full((len(df), len(CLASSES)), np.nan)
            base = np.empty(len(df), dtype=object)
            if split == "region":
                folds = [(reg != r, reg == r) for r in REGIONS]
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                folds = []
                for tr, te in skf.split(X, y):
                    a, b = np.zeros(len(df), bool), np.zeros(len(df), bool)
                    a[tr], b[te] = True, True
                    folds.append((a, b))
            for train, test in folds:
                proba[test] = fit_predict(X, train, test, seed)
                base[test] = majority(train)
            assert not np.isnan(proba).any()
            pred = np.array(CLASSES)[proba.argmax(1)]
            out = pd.DataFrame({"path": df["path"], "class": y, "region": reg, "seed": seed,
                                "split": split, "features": fs, "pred": pred})
            out[[f"p_{c}" for c in CLASSES]] = proba
            pred_frames.append(out)
            for r in REGIONS:
                m = reg == r
                f1, acc = scores(y[m], pred[m], r)
                rows.append(dict(features=fs, split=split, seed=seed, region=r, macro_f1=f1, accuracy=acc))
                if fs == "blackbox":  # baseline does not depend on the feature set
                    bf1, bacc = scores(y[m], base[m], r)
                    base_rows.append(dict(split=split, seed=seed, region=r, majority_class=base[m][0],
                                          macro_f1=bf1, accuracy=bacc))
            print(f"{fs:8s} {split:6s} seed {seed} done  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(pred_frames).to_csv(RES / f"predictions_{fs}_{split}.csv", index=False)

raw = pd.DataFrame(rows)
raw.to_csv(RES / "step4_per_seed_region.csv", index=False)
basedf = pd.DataFrame(base_rows)
basedf.to_csv(RES / "step4_baseline.csv", index=False)

# ---------------------------------------------------------------- summary table
# Per seed: region rows plus the unweighted mean over regions ("mean"); then mean ± std over seeds.
long = raw.melt(id_vars=["features", "split", "seed", "region"], var_name="metric")
avg = long.groupby(["features", "split", "seed", "metric"], as_index=False)["value"].mean().assign(region="mean")
long = pd.concat([long.astype({"region": str}), avg])
wide = long.pivot_table(index=["features", "metric", "seed", "region"], columns="split", values="value").rename(columns={"region": "regsplit"}).reset_index()
wide["drop_pts"] = wide["random"] - wide["regsplit"]
wide["drop_pct"] = 100 * wide["drop_pts"] / wide["random"]
# Note: the "mean" row's drop_pct is computed from the region-averaged scores of that seed.

order = [str(r) for r in REGIONS] + ["mean"]
summ = wide.groupby(["metric", "features", "region"])[["random", "regsplit", "drop_pts", "drop_pct"]].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
summ["region"] = pd.Categorical(summ["region"], order, ordered=True)
summ = summ.sort_values(["metric", "features", "region"])
bsum = (basedf.groupby(["split", "region"])["macro_f1"].mean().unstack("split")
        .rename(columns=lambda s: f"baseline_f1_{s}"))
bsum.loc["mean"] = bsum.mean()
bsum.index = bsum.index.astype(str)
summ = summ.merge(bsum, left_on=summ["region"].astype(str), right_index=True, how="left").drop(columns="key_0")
summ.to_csv(RES / "step4_table.csv", index=False, float_format="%.3f")


def pm(m, s, d=1):
    return f"{m:.{d}f} ± {s:.{d}f}"


for metric in ("macro_f1", "accuracy"):
    print(f"\n=== {metric} (points, mean ± std over 5 seeds) ===")
    for fs in FEATS:
        s = summ[(summ.metric == metric) & (summ.features == fs)]
        t = pd.DataFrame({"region": s["region"].astype(str).map(lambda r: r if r == "mean" else f"R{r}"),
                          "random": [pm(a, b) for a, b in zip(s.random_mean, s.random_std)],
                          "region-split": [pm(a, b) for a, b in zip(s.regsplit_mean, s.regsplit_std)],
                          "drop (pts)": [pm(a, b) for a, b in zip(s.drop_pts_mean, s.drop_pts_std)],
                          "drop (%)": [pm(a, b) for a, b in zip(s.drop_pct_mean, s.drop_pct_std)]})
        print(f"\n-- {fs} --")
        print(t.to_string(index=False))

print("\n=== no-skill baseline (training-set majority class), macro-F1 ===")
bt = basedf.groupby(["region", "split"]).agg(cls=("majority_class", lambda v: "/".join(sorted(set(v)))),
                                             f1=("macro_f1", "mean")).unstack("split")
print(bt.round(2).to_string())

# ---------------------------------------------------------------- paired comparison
pair_rows = []
for metric in ("macro_f1", "accuracy"):
    w = wide[(wide.metric == metric) & (wide.region != "mean")]
    p = w.pivot_table(index=["seed", "region"], columns="features", values=["drop_pts", "drop_pct"])
    for unit in ("drop_pts", "drop_pct"):
        d = p[(unit, "physics")] - p[(unit, "blackbox")]
        pair_rows.append(dict(metric=metric, unit=unit, mean=d.mean(), min=d.min(), max=d.max(),
                              n_physics_drops_less=int((d < 0).sum()), n_pairs=len(d)))
        if metric == "macro_f1" and unit == "drop_pts":
            per_region = d.groupby(level="region").agg(["mean", "min", "max"])
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(RES / "step4_paired.csv", index=False, float_format="%.3f")
print("\n=== paired: (physics drop) − (black-box drop), 35 seed×region pairs; <0 means physics drops less ===")
print(pairs.round(2).to_string(index=False))
print("\nmacro-F1, points, per region:")
print(per_region.round(2).to_string())

print(f"\ntotal {time.time() - t0:.0f}s  (figure: scripts/step4_figure.py)")
