"""Step 3: per-tile features. Black-box = mean/std of 13 raw bands; physics = mean/std of 4 spectral indices."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT = Path(__file__).resolve().parent.parent

# Band order of the EuroSAT_MS GeoTIFFs (files carry no band descriptions). B8A is LAST, not after B08.
# Source: TorchGeo EuroSAT.all_band_names; confirmed empirically in step 3 (see NOTES.md):
# native-resolution smoothness tiers (10 m: 2,3,4,8; 20 m: 5,6,7,11,12,13; 60 m: 1,9,10) and value levels
# (band 10 ~10 DN = cirrus B10; band 9 near zero over water = water-vapour B09).
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
IDX = {b: i for i, b in enumerate(BANDS)}

# Normalized differences (a - b) / (a + b).
INDICES = {
    "NDVI": ("B08", "B04"),  # vegetation
    "NDWI": ("B03", "B08"),  # open water
    "NDBI": ("B11", "B08"),  # built-up
    "NDRE": ("B8A", "B05"),  # red-edge, crop vigour
}


def norm_diff(a, b):
    """(a-b)/(a+b); pixels where a+b == 0 (only possible if both are 0, data is uint16) are set to 0."""
    s = a + b
    out = np.zeros_like(s)
    np.divide(a - b, s, out=out, where=s != 0)
    return out, int((s == 0).sum())


meta = pd.read_csv(PROJECT / "data" / "metadata.csv")
rows = []
vmin = np.full(13, np.inf)
vmax = np.full(13, -np.inf)
n_zero_denom = dict.fromkeys(INDICES, 0)
for i, p in enumerate(meta["path"]):
    with rasterio.open(PROJECT / p) as src:
        assert src.count == 13, p
        a = src.read().astype(np.float64)
    flat = a.reshape(13, -1)
    vmin = np.minimum(vmin, flat.min(1))
    vmax = np.maximum(vmax, flat.max(1))
    f = {}
    for b, j in IDX.items():
        f[f"{b}_mean"] = flat[j].mean()
        f[f"{b}_std"] = flat[j].std()
    for name, (x, y) in INDICES.items():
        v, nz = norm_diff(flat[IDX[x]], flat[IDX[y]])
        n_zero_denom[name] += nz
        f[f"{name}_mean"] = v.mean()
        f[f"{name}_std"] = v.std()
    rows.append(f)
    if (i + 1) % 5000 == 0:
        print(f"  {i + 1}/{len(meta)} tiles")

feat = pd.DataFrame(rows)
out = pd.concat([meta[["path", "class", "region"]], feat], axis=1)
assert not out.isna().any().any()
out.to_csv(PROJECT / "data" / "features.csv", index=False)
print(f"\nWrote data/features.csv: {out.shape[0]} rows, "
      f"{sum(c.startswith('B') for c in feat.columns)} black-box + "
      f"{sum(not c.startswith('B') for c in feat.columns)} physics feature columns")

print("\nBand order used:", ", ".join(f"{i + 1}={b}" for i, b in enumerate(BANDS)))
print("\nRaw value range per band (uint16 DN, all 27,000 tiles):")
print(pd.DataFrame({"min": vmin.astype(int), "max": vmax.astype(int)}, index=BANDS).T.to_string())
print(f"Overall: min={int(vmin.min())}, max={int(vmax.max())}")
print("\nPixels with zero denominator (set to 0):", n_zero_denom)

print("\nSanity check: tile-mean NDVI / NDWI averaged per class")
pd.set_option("display.width", 200)
cls = out.groupby("class")[["NDVI_mean", "NDWI_mean"]].mean()
print(cls.sort_values("NDVI_mean", ascending=False).round(3).to_string())
print()
print(cls.sort_values("NDWI_mean", ascending=False).round(3).to_string())
