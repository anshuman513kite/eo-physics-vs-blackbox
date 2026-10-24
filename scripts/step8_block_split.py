"""Step 8: spatial-block split, the fair "seen it before" reference.

Same RF and metrics as step 4 (scripts/step4_compare.py) for all three feature sets (black box, physics, combined).
Split: GroupKFold(5, shuffle=True, random_state=seed) over the 20 km blocks of data/blocks.csv, over all 27,000 tiles,
seeds 0-4 (the seed sets the fold assignment and the RF). Out-of-fold predictions for every tile.
Each fold's probabilities go to results/step8_folds/ as soon as the fold finishes; finished folds are skipped on restart.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FOLDS_DIR = RES / "step8_folds"
SEEDS = range(5)
MIN_TILES = 50

df = pd.read_csv(ROOT / "data" / "features.csv")
blocks = pd.read_csv(ROOT / "data" / "blocks.csv")
assert (blocks.path == df.path).all()
groups = blocks.block.to_numpy()
BB = [c for c in df.columns if c[:3] in {f"B{i:02d}" for i in range(1, 13)} | {"B8A"}]
PHYS = [c for c in df.columns if c.split("_")[0] in {"NDVI", "NDWI", "NDBI", "NDRE"}]
assert len(BB) == 26 and len(PHYS) == 8
FEATURE_SETS = {"blackbox": BB, "physics": PHYS, "combined": BB + PHYS}
CLASSES = sorted(df["class"].unique())
REGIONS = sorted(df["region"].unique())
y = df["class"].to_numpy()
reg = df["region"].to_numpy()
counts = df.groupby(["region", "class"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
ELIGIBLE = {r: [c for c in CLASSES if counts.loc[r, c] >= MIN_TILES] for r in REGIONS}


def folds_for(seed):
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in gkf.split(df, y, groups)]


def fit_predict(X, tr, te, seed):
    m = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed).fit(X[tr], y[tr])
    proba = np.zeros((len(te), len(CLASSES)))
    proba[:, [CLASSES.index(c) for c in m.classes_]] = m.predict_proba(X[te])
    return proba, len(m.classes_)


FOLDS_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rows = []
for fs, cols in FEATURE_SETS.items():
    X = df[cols].to_numpy()
    pred_frames = []
    for seed in SEEDS:
        proba = np.full((len(df), len(CLASSES)), np.nan)
        for k, (tr, te) in enumerate(folds_for(seed)):
            assert not set(groups[tr]) & set(groups[te])
            f = FOLDS_DIR / f"{fs}_seed{seed}_fold{k}.npz"
            if f.exists():
                z = np.load(f)
                assert (z["test"] == te).all()
                proba[te] = z["proba"]
                print(f"{fs:8s} seed {seed} fold {k} loaded", flush=True)
                continue
            t1 = time.time()
            p, n_cls = fit_predict(X, tr, te, seed)
            tmp = FOLDS_DIR / f"{fs}_seed{seed}_fold{k}.tmp.npz"
            np.savez(tmp, test=te, proba=p)
            tmp.rename(f)  # atomic: no half-written folds
            proba[te] = p
            print(f"{fs:8s} seed {seed} fold {k} done  n_test {len(te)}  train classes {n_cls}  "
                  f"fold {time.time() - t1:.0f}s  total {time.time() - t0:.0f}s", flush=True)
        assert not np.isnan(proba).any()
        pred = np.array(CLASSES)[proba.argmax(1)]
        out = pd.DataFrame({"path": df["path"], "class": y, "region": reg, "seed": seed,
                            "split": "block", "features": fs, "pred": pred})
        out[[f"p_{c}" for c in CLASSES]] = proba
        pred_frames.append(out)
        for r in REGIONS:
            m = reg == r
            rows.append(dict(features=fs, split="block", seed=seed, region=r,
                             macro_f1=100 * f1_score(y[m], pred[m], labels=ELIGIBLE[r], average="macro", zero_division=0),
                             accuracy=100 * accuracy_score(y[m], pred[m])))
        pd.DataFrame(rows).to_csv(RES / "step8_per_seed_region.csv", index=False)  # incremental
    pd.concat(pred_frames).to_csv(RES / f"predictions_{fs}_block.csv", index=False)

print(f"total {time.time() - t0:.0f}s  (tables: scripts/step8_analyze.py)")
