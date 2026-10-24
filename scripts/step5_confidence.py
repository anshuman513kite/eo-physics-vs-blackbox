"""Step 5: confidence and silent errors, from the saved step-4 predictions (no retraining).

Confidence = highest class probability per tile. All metrics are computed per
(features, split, region, seed); "mean" = unweighted mean over the 7 regions per seed;
then mean ± std over the 5 seeds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FEATURES = ["blackbox", "physics"]
SPLITS = ["random", "region"]
HIGH = 0.8           # "confident" threshold for silent errors
FLAG_FRAC = 0.2      # flag the lowest-confidence 20%
COVERAGES = [100, 90, 80, 70, 60, 50]


def ranked_by_confidence(conf, seed):
    """Indices from most to least confident; ties broken randomly (seeded)."""
    tiebreak = np.random.default_rng(seed).random(len(conf))
    return np.lexsort((tiebreak, -conf))


def group_metrics(g, seed):
    conf = g["conf"].to_numpy()
    correct = g["correct"].to_numpy()
    n = len(g)
    wrong = ~correct
    order = ranked_by_confidence(conf, seed)
    n_flag = int(round(FLAG_FRAC * n))
    flagged, unflagged = order[n - n_flag:], order[: n - n_flag]
    m = {
        "n_tiles": n,
        "mean_conf": 100 * conf.mean(),
        "accuracy": 100 * correct.mean(),
        "overconfidence": 100 * (conf.mean() - correct.mean()),
        "mean_conf_correct": 100 * conf[correct].mean(),
        "mean_conf_wrong": 100 * conf[wrong].mean(),
        "n_wrong": int(wrong.sum()),
        "silent_err_share": 100 * (wrong & (conf >= HIGH)).sum() / wrong.sum(),
        "silent_err_rate_all": 100 * (wrong & (conf >= HIGH)).mean(),
        "share_conf_ge_0.8": 100 * (conf >= HIGH).mean(),
        "flag_acc_flagged": 100 * correct[flagged].mean(),
        "flag_acc_unflagged": 100 * correct[unflagged].mean(),
        "flag_share_of_errors_caught": 100 * wrong[flagged].sum() / wrong.sum(),
        "flag_conf_cutoff": 100 * conf[flagged].max(),
    }
    for x in COVERAGES:
        k = int(round(x / 100 * n))
        m[f"cov{x}_acc"] = 100 * correct[order[:k]].mean()
    return m


def main():
    rows = []
    for fs in FEATURES:
        for sp in SPLITS:
            p = pd.read_csv(RES / f"predictions_{fs}_{sp}.csv")
            pcols = [c for c in p.columns if c.startswith("p_")]
            probs = p[pcols].to_numpy()
            p["conf"] = probs.max(axis=1)
            p["correct"] = p["pred"] == p["class"]
            # pred must be the argmax class (RF predict); count rows where the max is tied
            argmax_cls = np.array([c[2:] for c in pcols])[probs.argmax(axis=1)]
            n_top = (probs == p["conf"].to_numpy()[:, None]).sum(axis=1)
            mismatch = int(((argmax_cls != p["pred"]) & (n_top == 1)).sum())
            print(f"{fs}/{sp}: {len(p)} rows, tied top-probability rows {int((n_top > 1).sum())}, "
                  f"pred != argmax (untied rows) {mismatch}")
            for (seed, region), g in p.groupby(["seed", "region"]):
                rows.append({"features": fs, "split": sp, "seed": seed, "region": str(region),
                             **group_metrics(g, seed)})
    raw = pd.DataFrame(rows)
    metrics = [c for c in raw.columns if c not in ("features", "split", "seed", "region")]
    mean_reg = raw.groupby(["features", "split", "seed"])[metrics].mean().reset_index()
    mean_reg["region"] = "mean"
    raw = pd.concat([raw, mean_reg], ignore_index=True)
    raw.to_csv(RES / "step5_per_seed_region.csv", index=False, float_format="%.4f")

    long = raw.melt(id_vars=["features", "split", "seed", "region"], var_name="metric")
    table = (long.groupby(["features", "split", "region", "metric"])["value"]
             .agg(mean="mean", std="std").reset_index())
    table.to_csv(RES / "step5_table.csv", index=False, float_format="%.3f")

    # Paired region − random differences per seed × region (same tiles)
    piv = raw.set_index(["features", "split", "seed", "region"])
    diff = (piv.xs("region", level="split") - piv.xs("random", level="split")).reset_index()
    diff.to_csv(RES / "step5_region_minus_random.csv", index=False, float_format="%.4f")

    regions = [r for r in sorted(raw.region.unique()) if r != "mean"] + ["mean"]

    def fmt(fs, sp, metric):
        s = table[(table.features == fs) & (table.split == sp) & (table.metric == metric)].set_index("region")
        return {r: f"{s.loc[r, 'mean']:.1f} ± {s.loc[r, 'std']:.1f}" for r in regions}

    def show(title, cols):
        df = pd.DataFrame({name: fmt(*key) for name, key in cols.items()})
        df.index = [f"R{r}" if r != "mean" else "Mean" for r in df.index]
        print(f"\n## {title}\n")
        print(df.to_string())

    lab = {"blackbox": "BB", "physics": "Phys"}
    show("Mean confidence (%) vs accuracy (%)",
         {f"{lab[fs]} {sp} {m}": (fs, sp, mt) for fs in FEATURES for sp in SPLITS
          for m, mt in (("conf", "mean_conf"), ("acc", "accuracy"))})
    show("Overconfidence = mean confidence − accuracy (pts)",
         {f"{lab[fs]} {sp}": (fs, sp, "overconfidence") for fs in FEATURES for sp in SPLITS})
    show("Silent errors: % of wrong predictions with confidence >= 0.8",
         {f"{lab[fs]} {sp}": (fs, sp, "silent_err_share") for fs in FEATURES for sp in SPLITS})
    show("Silent errors as % of ALL tiles (wrong and conf >= 0.8)",
         {f"{lab[fs]} {sp}": (fs, sp, "silent_err_rate_all") for fs in FEATURES for sp in SPLITS})
    show("Flag lowest 20% by confidence, region split: accuracy (%) and share of errors caught (%)",
         {f"{lab[fs]} {m}": (fs, "region", mt) for fs in FEATURES
          for m, mt in (("flagged", "flag_acc_flagged"), ("unflagged", "flag_acc_unflagged"),
                        ("errs caught", "flag_share_of_errors_caught"))})
    for fs in FEATURES:
        show(f"Coverage curve, region split, {lab[fs]}: accuracy (%) on the top X% most confident",
             {f"{x}%": (fs, "region", f"cov{x}_acc") for x in COVERAGES})

    print("\n## Paired region − random, 35 seed × region pairs (mean [min, max], n pairs with region < random)\n")
    d = diff[diff.region != "mean"]
    for fs in FEATURES:
        for mt in ("mean_conf", "accuracy", "overconfidence", "silent_err_share"):
            v = d[d.features == fs][mt]
            print(f"{lab[fs]:4s} {mt:18s} {v.mean():+6.2f} [{v.min():+6.2f}, {v.max():+6.2f}]  "
                  f"region<random in {(v < 0).sum()}/{len(v)}")


if __name__ == "__main__":
    main()
