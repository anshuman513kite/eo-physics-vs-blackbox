"""Step 7: the step-4 comparison for one new feature set, combined = 26 band + 8 index features (34).

Identical protocol to scripts/step4_compare.py (same RF, seeds, folds, metrics). Only the combined set is
run; black box and physics are reused from the saved step-4 results (scripts/step7_analyze.py).
Each fold's probabilities are written to results/step7_folds/ as soon as it finishes, and finished folds
are skipped on a restart, so a crash loses at most one fold.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FOLDS_DIR = RES / "step7_folds"
SEEDS = range(5)
MIN_TILES = 50

df = pd.read_csv(ROOT / "data" / "features.csv")
BB = [c for c in df.columns if c[:3] in {f"B{i:02d}" for i in range(1, 13)} | {"B8A"}]
PHYS = [c for c in df.columns if c.split("_")[0] in {"NDVI", "NDWI", "NDBI", "NDRE"}]
COLS = BB + PHYS
assert len(BB) == 26 and len(PHYS) == 8 and len(set(COLS)) == 34
CLASSES = sorted(df["class"].unique())
REGIONS = sorted(df["region"].unique())
y = df["class"].to_numpy()
reg = df["region"].to_numpy()
counts = df.groupby(["region", "class"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
ELIGIBLE = {r: [c for c in CLASSES if counts.loc[r, c] >= MIN_TILES] for r in REGIONS}
X = df[COLS].to_numpy()


def rf(seed):
    return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)


def fit_predict(X, train, test, seed):
    m = rf(seed).fit(X[train], y[train])
    proba = np.zeros((test.sum(), len(CLASSES)))
    proba[:, [CLASSES.index(c) for c in m.classes_]] = m.predict_proba(X[test])
    return proba


def folds_for(split, seed):
    if split == "region":
        return [(reg != r, reg == r) for r in REGIONS]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = []
    for tr, te in skf.split(X, y):
        a, b = np.zeros(len(df), bool), np.zeros(len(df), bool)
        a[tr], b[te] = True, True
        out.append((a, b))
    return out


FOLDS_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rows = []
for split in ("random", "region"):
    pred_frames = []
    for seed in SEEDS:
        proba = np.full((len(df), len(CLASSES)), np.nan)
        for k, (train, test) in enumerate(folds_for(split, seed)):
            f = FOLDS_DIR / f"{split}_seed{seed}_fold{k}.npz"
            if f.exists():
                z = np.load(f)
                assert (z["test"] == np.flatnonzero(test)).all()
                proba[test] = z["proba"]
                print(f"{split:6s} seed {seed} fold {k} loaded", flush=True)
                continue
            p = fit_predict(X, train, test, seed)
            np.savez(FOLDS_DIR / f"{split}_seed{seed}_fold{k}.tmp.npz", test=np.flatnonzero(test), proba=p)
            (FOLDS_DIR / f"{split}_seed{seed}_fold{k}.tmp.npz").rename(f)  # atomic: no half-written folds
            proba[test] = p
            print(f"{split:6s} seed {seed} fold {k} done  ({time.time() - t0:.0f}s)", flush=True)
        assert not np.isnan(proba).any()
        pred = np.array(CLASSES)[proba.argmax(1)]
        out = pd.DataFrame({"path": df["path"], "class": y, "region": reg, "seed": seed,
                            "split": split, "features": "combined", "pred": pred})
        out[[f"p_{c}" for c in CLASSES]] = proba
        pred_frames.append(out)
        for r in REGIONS:
            m = reg == r
            rows.append(dict(features="combined", split=split, seed=seed, region=r,
                             macro_f1=100 * f1_score(y[m], pred[m], labels=ELIGIBLE[r], average="macro", zero_division=0),
                             accuracy=100 * accuracy_score(y[m], pred[m])))
    pd.concat(pred_frames).to_csv(RES / f"predictions_combined_{split}.csv", index=False)

pd.DataFrame(rows).to_csv(RES / "step7_per_seed_region.csv", index=False)
print(f"total {time.time() - t0:.0f}s  (tables: scripts/step7_analyze.py)")
