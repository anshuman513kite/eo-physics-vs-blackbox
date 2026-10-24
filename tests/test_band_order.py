"""(a) The band order used for the features matches the order confirmed from the data in step 3."""
import numpy as np

from conftest import CONFIRMED_BANDS, script_constant

# Native Sentinel-2 resolution of each band; after resampling to 10 m, coarser bands are smoother.
RES_M = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20,
         "B08": 10, "B09": 60, "B10": 60, "B11": 20, "B12": 20, "B8A": 20}


def test_script_band_order_is_confirmed_order():
    assert script_constant("make_features.py", "BANDS") == CONFIRMED_BANDS


def test_index_formulas_use_named_bands():
    assert script_constant("make_features.py", "INDICES") == {
        "NDVI": ("B08", "B04"), "NDWI": ("B03", "B08"), "NDBI": ("B11", "B08"), "NDRE": ("B8A", "B05")}


def test_feature_columns_follow_band_order(features):
    band_cols = [c for c in features.columns if c.split("_")[0] in CONFIRMED_BANDS]
    assert band_cols == [f"{b}_{s}" for b in CONFIRMED_BANDS for s in ("mean", "std")]


def test_smoothness_tiers_match_native_resolution(raw_sample):
    """Roughness = mean |horizontal neighbour diff| / std, per band position (the step-3 check)."""
    rough = np.zeros(13)
    for a in raw_sample:
        img = a.reshape(13, 64, 64)
        d = np.abs(np.diff(img, axis=2)).mean(axis=(1, 2))
        rough += d / np.maximum(img.std(axis=(1, 2)), 1e-9)
    rough /= len(raw_sample)
    tiers = {res: [rough[i] for i, b in enumerate(CONFIRMED_BANDS) if RES_M[b] == res] for res in (10, 20, 60)}
    # every 10 m band is rougher than every 20 m band, which is rougher than every 60 m band
    assert min(tiers[10]) > max(tiers[20]) > 0
    assert min(tiers[20]) > max(tiers[60])
