"""Step 6: relearning cost — how many labels from a new region are needed to recover?

Protocol (NOTES.md, step 6): for each held-out region, feature set and seed, split the region's tiles
50/50 (stratified by class, seed-controlled) into a label pool and a fixed evaluation set. Train on the
other 6 regions plus k pool tiles (k = 0, 10, 50, 100, 500), evaluate on the evaluation set only.
k tiles chosen (a) at random from the pool, or (b) by uncertainty: the k pool tiles with the lowest max
class probability under the k=0 model (one shot). Both selections are nested in k. Macro-F1 over the
step-4 class list for the region (classes with >=50 tiles in the whole region), plus accuracy.
RF with 200 trees (step 4 used 500) for runtime.

Usage: step6_relearn.py [--time] [--seeds N]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5
SEEDS = range(N_SEEDS)
KS = [0, 10, 50, 100, 500]
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
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)


def scores(y_true, y_pred, r):
    return (100 * f1_score(y_true, y_pred, labels=ELIGIBLE[r], average="macro", zero_division=0),
            100 * accuracy_score(y_true, y_pred))


def time_one():
    """Time one k=500 fit + predict per feature set on the largest training set (R0 held out)."""
    r = 0
    train = np.flatnonzero(reg != r)
    pool = np.flatnonzero(reg == r)
    for fs, cols in FEATS.items():
        X = df[cols].to_numpy()
        tr = np.concatenate([train, pool[:500]])
        t0 = time.time()
        m = rf(0).fit(X[tr], y[tr])
        m.predict_proba(X[pool])
        print(f"{fs}: one fit+predict {time.time() - t0:.1f} s (n_train={len(tr)})")


def main():
    t_start = time.time()
    rows = []
    for fs, cols in FEATS.items():
        X = df[cols].to_numpy()
        for seed in SEEDS:
            for r in REGIONS:
                idx = np.flatnonzero(reg == r)
                pool, ev = train_test_split(idx, test_size=0.5, stratify=y[idx], random_state=seed)
                base = np.flatnonzero(reg != r)
                m0 = rf(seed).fit(X[base], y[base])
                f1, acc = scores(y[ev], m0.predict(X[ev]), r)
                for method in ("random", "uncertainty"):
                    rows.append(dict(features=fs, method=method, seed=seed, region=r, k=0,
                                     n_pool=len(pool), n_eval=len(ev), macro_f1=f1, accuracy=acc))
                rng = np.random.default_rng(seed)
                order = {"random": rng.permutation(pool)}
                conf = m0.predict_proba(X[pool]).max(axis=1)
                tiebreak = rng.random(len(pool))
                order["uncertainty"] = pool[np.lexsort((tiebreak, conf))]  # least confident first
                for method, o in order.items():
                    for k in KS[1:]:
                        tr = np.concatenate([base, o[:k]])
                        m = rf(seed).fit(X[tr], y[tr])
                        f1, acc = scores(y[ev], m.predict(X[ev]), r)
                        rows.append(dict(features=fs, method=method, seed=seed, region=r, k=k,
                                         n_pool=len(pool), n_eval=len(ev), macro_f1=f1, accuracy=acc))
                print(f"{fs} seed {seed} R{r} done, {time.time() - t_start:.0f} s", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RES / "step6_per_seed_region.csv", index=False, float_format="%.4f")
    print(f"total {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    time_one() if "--time" in sys.argv else main()
