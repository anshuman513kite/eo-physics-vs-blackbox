"""Step 2: per-tile centre coordinates, k-means geographic regions, class-by-region table, map."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from sklearn.cluster import KMeans

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "raw" / "EuroSAT_MS"
N_REGIONS = 8
SEED = 0
MIN_PER_CLASS = 50
# Review decision (step 3): R6 (Finland/Baltics, 774 tiles, 6 classes <50) is merged into R4.
# Original IDs are kept, so there is no R6 and the 7 regions are 0,1,2,3,4,5,7.
MERGE = {6: 4}

# Validated categorical palette (dataviz skill reference, light mode), fixed slot order.
REGION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]

# 1. Centre coordinates. Gather native-CRS centres, then reproject once per CRS.
rows = []
by_crs = defaultdict(list)
for p in sorted(DATA.rglob("*.tif")):
    with rasterio.open(p) as src:
        b = src.bounds
        epsg = src.crs.to_epsg()
        by_crs[epsg].append(len(rows))
        rows.append({"path": str(p.relative_to(PROJECT)), "class": p.parent.name,
                     "x": (b.left + b.right) / 2, "y": (b.bottom + b.top) / 2, "epsg": epsg})

df = pd.DataFrame(rows)
df["lat"] = np.nan
df["lon"] = np.nan
for epsg, idx in by_crs.items():
    lon, lat = transform(f"EPSG:{epsg}", "EPSG:4326", df.loc[idx, "x"].tolist(), df.loc[idx, "y"].tolist())
    df.loc[idx, "lon"] = lon
    df.loc[idx, "lat"] = lat
df = df.drop(columns=["x", "y"])[["path", "class", "lat", "lon", "epsg"]]

# 2. K-means on (lat, lon); relabel regions west->east by centroid longitude.
km = KMeans(n_clusters=N_REGIONS, random_state=SEED, n_init=10).fit(df[["lat", "lon"]].to_numpy())
order = np.argsort(km.cluster_centers_[:, 1])
remap = {old: new for new, old in enumerate(order)}
df["region"] = [MERGE.get(remap[l], remap[l]) for l in km.labels_]
REGIONS = sorted(df["region"].unique())
# Merged centroid = mean of member tiles (identical to the k-means centroid for unmerged regions).
centres = df.groupby("region")[["lat", "lon"]].mean().loc[REGIONS].to_numpy()

(PROJECT / "data").mkdir(exist_ok=True)
df.to_csv(PROJECT / "data" / "metadata.csv", index=False)
print(f"Wrote data/metadata.csv ({len(df)} rows, {df[['lat', 'lon']].isna().any(axis=1).sum()} missing coords)\n")

# 3. Region x class table.
tab = pd.crosstab(df["region"], df["class"])
tab["TOTAL"] = tab.sum(axis=1)
tab["centre(lat,lon)"] = [f"{la:.1f},{lo:.1f}" for la, lo in centres]
classes = sorted(df["class"].unique())
tab["FLAG(<50)"] = ["" if (tab.loc[r, classes] >= MIN_PER_CLASS).all()
                    else ",".join(c for c in classes if tab.loc[r, c] < MIN_PER_CLASS)
                    for r in tab.index]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(tab.to_string())
tab.to_csv(PROJECT / "data" / "region_class_counts.csv")

# 4. Map.
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
for r in REGIONS:
    s = df[df["region"] == r]
    ax.scatter(s["lon"], s["lat"], s=3, color=REGION_COLORS[r], alpha=0.6, linewidths=0,
               label=f"Region {r} (n={len(s)})")
for r, (la, lo) in zip(REGIONS, centres):
    ax.annotate(f"R{r}", (lo, la), ha="center", va="center", fontsize=11, fontweight="bold",
                color="#1a1a19", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=REGION_COLORS[r], lw=2))
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"EuroSAT_MS tiles by region (k-means k={N_REGIONS}, seed={SEED}, R6 merged into R4; n={len(df)})", loc="left")
ax.set_aspect(1 / np.cos(np.deg2rad(df["lat"].mean())))
ax.grid(color="#e4e3dd", lw=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(markerscale=4, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
(PROJECT / "figures").mkdir(exist_ok=True)
fig.savefig(PROJECT / "figures" / "regions_map.png")
print("\nWrote figures/regions_map.png")
