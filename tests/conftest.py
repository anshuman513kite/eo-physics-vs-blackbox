"""Shared helpers. The scripts run their whole pipeline at import time, so the tests read constants and
functions from the script source with `ast` instead of importing them. Tests that need data/ or results/
(not in the repository) are skipped when those files are absent."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RAW = DATA / "raw" / "EuroSAT_MS"

# Confirmed in step 3 from the data (NOTES.md): B8A is the LAST band.
CONFIRMED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "NDRE"]


def script_node(script, name):
    """Return the top-level assignment value or function definition called `name` in scripts/<script>."""
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(f"{name} not found in scripts/{script}")


def script_constant(script, name):
    return ast.literal_eval(script_node(script, name))


def script_function(script, name, namespace):
    module = ast.Module(body=[script_node(script, name)], type_ignores=[])
    exec(compile(module, f"scripts/{script}", "exec"), namespace)
    return namespace[name]


def need(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present (see README: Data / How to reproduce)")
    return path


@pytest.fixture(scope="session")
def features():
    return pd.read_csv(need(DATA / "features.csv"))


@pytest.fixture(scope="session")
def metadata():
    """Committed table with the same rows, in the same order, as data/features.csv."""
    return pd.read_csv(need(DATA / "metadata.csv"))


@pytest.fixture(scope="session")
def blocks():
    return pd.read_csv(need(DATA / "blocks.csv"))


@pytest.fixture(scope="session")
def raw_sample(features):
    """A few tiles per class, read as float64 (13, 64*64)."""
    rasterio = pytest.importorskip("rasterio")
    need(RAW)
    paths = features.groupby("class").head(5)["path"]
    out = []
    for p in paths:
        with rasterio.open(ROOT / p) as src:
            out.append(src.read().astype(np.float64).reshape(13, -1))
    return out
