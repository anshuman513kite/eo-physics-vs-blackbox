"""Step 8a: assign every tile to a 20 km x 20 km spatial block.

Tile centre (lat, lon from data/metadata.csv, WGS84) -> EPSG:3035 (ETRS89 / LAEA Europe, equal-area, metres),
then floor(x / 20 km), floor(y / 20 km). Output data/blocks.csv: path, class, region, x, y, bx, by, block.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parent.parent
SIZE = 20_000  # metres

meta = pd.read_csv(ROOT / "data" / "metadata.csv")
x, y = transform("EPSG:4326", "EPSG:3035", meta.lon.tolist(), meta.lat.tolist())
out = meta[["path", "class", "region"]].copy()
out["x"], out["y"] = np.asarray(x), np.asarray(y)
assert np.isfinite(out[["x", "y"]]).all().all()
out["bx"] = np.floor(out.x / SIZE).astype(int)
out["by"] = np.floor(out.y / SIZE).astype(int)
out["block"] = out.bx.astype(str) + "_" + out.by.astype(str)
out.to_csv(ROOT / "data" / "blocks.csv", index=False)

feat = pd.read_csv(ROOT / "data" / "features.csv", usecols=["path"])
assert (feat.path == out.path).all(), "row order must match features.csv"

per_block = out.groupby("block")
n_reg = per_block.region.nunique()
print(f"tiles {len(out)}, blocks {out.block.nunique()}, tiles/block median {per_block.size().median():.0f}, "
      f"max {per_block.size().max()}, blocks with 1 tile {(per_block.size() == 1).sum()}")
print(f"blocks spanning >1 region: {(n_reg > 1).sum()} ({out.block.isin(n_reg[n_reg > 1].index).sum()} tiles)")
rows = []
for r, g in out.groupby("region"):
    s = g.groupby("block").size()
    rows.append(dict(region=f"R{r}", tiles=len(g), blocks=len(s), tiles_per_block_median=s.median(),
                     tiles_per_block_max=s.max(), largest_block_share_pct=100 * s.max() / len(g)))
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results" / "step8_blocks_per_region.csv", index=False, float_format="%.1f")
print(t.to_string(index=False, float_format="%.1f"))
