"""(b) Every NDVI/NDWI/NDBI/NDRE value is within [-1, 1]."""
import numpy as np
import pytest

from conftest import CONFIRMED_BANDS, INDEX_NAMES, script_constant, script_function


@pytest.fixture(scope="module")
def norm_diff():
    return script_function("make_features.py", "norm_diff", {"np": np})


def test_norm_diff_bounds_on_uint16_range(norm_diff):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**16, 100_000).astype(np.float64)
    b = rng.integers(0, 2**16, 100_000).astype(np.float64)
    edges = np.array([0, 0, 65535, 65535, 0, 1], dtype=np.float64)
    a, b = np.concatenate([a, edges]), np.concatenate([b, edges[[0, 2, 0, 2, 1, 0]]])
    v, n_zero = norm_diff(a, b)
    assert np.isfinite(v).all()
    assert v.min() >= -1 and v.max() <= 1
    zero = (a + b) == 0
    assert n_zero == zero.sum() >= 2 and (v[zero] == 0).all()  # 0/0 -> 0


def test_pixel_indices_on_raw_tiles(norm_diff, raw_sample):
    idx = {b: i for i, b in enumerate(CONFIRMED_BANDS)}
    for name, (x, y) in script_constant("make_features.py", "INDICES").items():
        for a in raw_sample:
            v, _ = norm_diff(a[idx[x]], a[idx[y]])
            assert np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1, name


def test_feature_table_indices(features):
    for name in INDEX_NAMES:
        m, s = features[f"{name}_mean"], features[f"{name}_std"]
        assert m.notna().all() and m.between(-1, 1).all(), name
        assert s.notna().all() and s.between(0, 1).all(), name  # std of values in [-1, 1] is at most 1
