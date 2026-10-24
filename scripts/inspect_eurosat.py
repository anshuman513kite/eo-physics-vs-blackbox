"""Step 1 sanity check: inspect random EuroSAT_MS GeoTIFFs and count images per class."""
import random
import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform

root = Path(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

tifs = sorted(root.rglob("*.tif"))
print(f"Found {len(tifs)} .tif files under {root}\n")

random.seed(seed)
for p in random.sample(tifs, 5):
    with rasterio.open(p) as src:
        cx = (src.bounds.left + src.bounds.right) / 2
        cy = (src.bounds.bottom + src.bounds.top) / 2
        lon, lat = transform(src.crs, "EPSG:4326", [cx], [cy])
        print(f"{p.name}")
        print(f"  class folder : {p.parent.name}")
        print(f"  shape (b,h,w): ({src.count}, {src.height}, {src.width})")
        print(f"  bands        : {src.count}  dtype={src.dtypes[0]}")
        print(f"  CRS          : {src.crs}")
        print(f"  centre WGS84 : lat={lat[0]:.5f}, lon={lon[0]:.5f}\n")

counts = {}
for p in tifs:
    counts[p.parent.name] = counts.get(p.parent.name, 0) + 1
print("Images per class:")
for k in sorted(counts):
    print(f"  {k:22s} {counts[k]:6d}")
print(f"  {'TOTAL':22s} {sum(counts.values()):6d}")
