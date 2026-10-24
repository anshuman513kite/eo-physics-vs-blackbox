# eo-physics-vs-blackbox — project notes

Goal: a small, honest benchmark on public satellite data testing whether land-use
models generalize to **unseen geographic regions**.

Environment: `.venv/` (Python 3.14.7) — numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1,
rasterio 1.5.2, matplotlib 3.11.2. Run scripts with `.venv/bin/python -I scripts/<name>.py`.

---

## Step 1 — Environment + data download (2026-10-22)

**What:** Created the venv; downloaded EuroSAT **multispectral** (`EuroSAT_MS.zip`, 2,065,402,329 bytes)
from the official Zenodo record 7711810 (https://zenodo.org/records/7711810).
MD5 verified: `091174add3c8e680a49244acf185b9f0` (matches Zenodo). Extracted to
`data/raw/EuroSAT_MS/<Class>/<Class>_<n>.tif`. Zip kept in `data/raw/` as the checksummed original.

**Why:** MS (13 Sentinel-2 bands) rather than RGB, so that physically-motivated features
(spectral indices etc.) are possible later; the GeoTIFFs keep georeferencing, which the
RGB JPEGs do not — needed for region-based splits.

**Key facts:** script `scripts/inspect_eurosat.py`.
- 27,000 tiles, each 13 bands × 64 × 64 px, uint16.
- Each tile is in its local UTM zone CRS (EPSG:326xx), so every tile is georeferenced.
- Per class: AnnualCrop 3000, Forest 3000, HerbaceousVegetation 3000, Residential 3000,
  SeaLake 3000, Highway 2500, Industrial 2500, PermanentCrop 2500, River 2500, Pasture 2000.

## Step 2 — Geographic regions (2026-10-22)

**What:** script `scripts/make_regions.py`.
1. Computed the centre of each tile in its native UTM CRS and reprojected to WGS84.
   Output `data/metadata.csv`: `path, class, lat, lon, epsg, region` (27,000 rows, 0 missing coords).
2. K-means (k=8, `random_state=0`, `n_init=10`) on raw (lat, lon) in degrees. Cluster IDs
   were renumbered **west→east by centroid longitude** so that region IDs are deterministic and readable.
3. Region × class counts saved to `data/region_class_counts.csv`.
4. Map: `figures/regions_map.png`.

**Why:** Regions are the unit of held-out evaluation. A model that is tested on a region it
never saw in training measures geographic generalization, not interpolation between
neighbouring (spatially autocorrelated) tiles.

**Key numbers as originally produced (k=8; superseded — R6 was later merged into R4, see Decisions):**

| Region | Centre (lat, lon) | Rough area | Tiles | Classes < 50 |
|---|---|---|---|---|
| R0 | 39.3, −3.6 | Iberia (+ Canaries) | 2495 | Pasture (13) |
| R1 | 53.4, −3.1 | UK/Ireland (+ Iceland) | 3493 | AnnualCrop (42), Forest (0), PermanentCrop (0) |
| R2 | 46.4, 3.5 | France | 4898 | — |
| R3 | 50.3, 8.1 | Benelux/W. Germany/Switzerland | 6035 | — |
| R4 | 55.8, 14.3 | Denmark/S. Sweden/N. Poland | 2458 | HerbVeg (21), Pasture (49), PermanentCrop (0) |
| R5 | 46.8, 16.7 | Central Europe/Italy | 4073 | HerbVeg (26) |
| R6 | 59.3, 25.2 | Finland/Baltics | 774 | 6 classes (AnnualCrop 9, Forest 31, HerbVeg 0, Pasture 40, PermanentCrop 0, Residential 30) |
| R7 | 40.4, 26.3 | Balkans/Greece (+ Cyprus) | 2774 | Pasture (42) |

**Caveats / observations:**
- Only R2 and R3 have ≥50 tiles of every class. The class mix differs strongly by region
  (e.g. R1 has 0 Forest and 0 PermanentCrop; R1 is 38% Pasture). So any held-out-region
  result mixes **label/prior shift** with **appearance shift**. Later metrics need to be
  per-class or class-balanced, and must handle classes that are absent from a test region.
- K-means on raw degrees is not equal-area (1° of longitude shrinks toward the north). This
  is acceptable for coarse grouping, but it is not a principled distance measure.
- Outlying islands are absorbed into their nearest cluster (Iceland→R1, Canaries→R0, Cyprus→R7).

---

## Decisions after review of step 2 (2026-10-23)

1. **Merge R6 into R4 → 7 regions.** R6 (Finland/Baltics) had only 774 tiles and 6 classes <50.
   Implemented in `scripts/make_regions.py` (`MERGE = {6: 4}`, applied after the west→east relabel);
   `data/metadata.csv`, `data/region_class_counts.csv` and `figures/regions_map.png` regenerated.
   **Original IDs kept: regions are 0,1,2,3,4,5,7 (there is no R6).** Verified: exactly the 774
   former-R6 tiles changed label (all 6→4), and no other tile changed.

   | Region | Centre (lat, lon) | Rough area | Tiles | Classes < 50 |
   |---|---|---|---|---|
   | R0 | 39.3, −3.6 | Iberia (+ Canaries) | 2495 | Pasture (13) |
   | R1 | 53.4, −3.1 | UK/Ireland (+ Iceland) | 3493 | AnnualCrop (42), Forest (0), PermanentCrop (0) |
   | R2 | 46.4, 3.5 | France | 4898 | — |
   | R3 | 50.3, 8.1 | Benelux/W. Germany/Switzerland | 6035 | — |
   | R4 | 56.6, 16.9 | Denmark/S. Sweden/N. Poland + Finland/Baltics | 3232 | HerbVeg (21), PermanentCrop (0) |
   | R5 | 46.8, 16.7 | Central Europe/Italy | 4073 | HerbVeg (26) |
   | R7 | 40.4, 26.3 | Balkans/Greece (+ Cyprus) | 2774 | Pasture (42) |

   (R4 Pasture rose from 49 to 89, so it is no longer flagged.)

2. **Evaluation protocol (for step 4).** For each region, compare on the SAME tiles:
   - (a) *region split*: train on the other 6 regions, predict this region;
   - (b) *random split*: out-of-fold predictions for this region's tiles from a stratified (by class)
     random 5-fold split over all data.
   - Metric: **macro-F1 averaged only over classes with ≥50 tiles in that region**. All test tiles are
     kept (predictions on rare/absent classes still count as errors/false positives); only the
     averaging is restricted. Plain accuracy is reported as a side column.
   - 5 seeds.

## Step 3 — Per-tile features (2026-10-23)

**What:** script `scripts/make_features.py` → `data/features.csv` (27,000 rows):
`path, class, region`, then 26 black-box columns `<band>_mean, <band>_std` for the 13 raw bands, then
8 physics columns `<index>_mean, <index>_std` for NDVI=(B08−B04)/(B08+B04), NDWI=(B03−B08)/(B03+B08),
NDBI=(B11−B08)/(B11+B08), NDRE=(B8A−B05)/(B8A+B05). Std is per-tile population std (ddof=0) over the
64×64 pixels. Division by zero: a pixel whose denominator is 0 (only when both bands are 0) gets index 0.
This happened for 3 NDVI pixels in the whole dataset and never for the other indices. There are no NaNs.

**Band order — confirmed, not assumed: `B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A`
(B8A is the LAST band, not band 9).**
- The GeoTIFFs have no band descriptions or tags. Zenodo record 7711810 and the EuroSAT GitHub README
  only say "13 Sentinel-2 bands in the original value range", without giving the order.
- Documented order: TorchGeo `EuroSAT.all_band_names` = (B01…B12, B8A).
- Independent empirical checks on a stratified sample:
  - *Native resolution:* all bands are resampled to 10 m, so the bands acquired at coarser resolution
    are spatially smoother. Roughness = mean |neighbour diff| / std falls into three tiers that match
    Sentinel-2 exactly: ≈0.30 for bands 2,3,4,8 (10 m: B02,B03,B04,B08); ≈0.15 for bands 5,6,7,11,12,13
    (20 m: B05,B06,B07,B11,B12,B8A); ≈0.08–0.10 for bands 1,9,10 (60 m: B01,B09,B10).
  - *Value levels:* band 10 ≈ 10 DN everywhere (cirrus B10, 1375 nm, absorbed by the atmosphere).
    Band 9 is well below NIR and near zero over water (water-vapour B09). Band 13 is at NIR level and has
    r=0.996 with band 7 (B07) and r=0.978 with band 8 (B08).
- The common assumption (B8A in position 9) is therefore wrong for this file set. It would have put
  B09 into NDRE and the 20 m B12 in place of B8A.

**Raw value range:** uint16 DN, overall 0 … 28,003. Typical surface values are ~100–4,000 (e.g. class-mean
B08 ≈ 2,000–3,800 on land, ≈ 380 on SeaLake). The presence of B10 indicates Level-1C TOA reflectance ×10,000.
Per-band max ≈ 28,000 on most bands, which looks like a saturation/clip value. B10 max is 183. A few pixels are 0 in
B02/B03/B04/B08 (B01 minimum is 808). Zeros are rare (only 3 pixels have B08=B04=0). Saturated pixels were
not counted. All values are kept as-is.

**Sanity check (class mean of tile-mean NDVI / NDWI) — all expectations met:**

| Class | NDVI | NDWI |
|---|---|---|
| Forest | 0.713 | −0.569 |
| Pasture | 0.654 | −0.544 |
| Highway | 0.446 | −0.392 |
| HerbaceousVegetation | 0.385 | −0.370 |
| AnnualCrop | 0.376 | −0.396 |
| PermanentCrop | 0.372 | −0.383 |
| Residential | 0.340 | −0.311 |
| River | 0.327 | −0.209 |
| Industrial | 0.223 | −0.205 |
| SeaLake | −0.253 | **0.456** (highest; the only positive class) |

- Forest and Pasture are the top two classes for NDVI, Industrial is far below Forest, and SeaLake is highest for NDWI.
- Highway ranks 3rd in NDVI, above the crop classes. The highway tiles are mostly the vegetated land around a road.
  Crop tiles include bare or harvested fields depending on acquisition date.
- River and Industrial have nearly the same NDWI (−0.21). NDWI only separates open water when water dominates the tile.

**Why:** Both feature sets are deliberately simple per-tile summaries with the same structure
(mean + std), so a later comparison tests *which* inputs generalize across regions, not model capacity.
No model has been trained yet.

## Step 4 — Main comparison: random split vs held-out region (2026-10-23)

**What:** script `scripts/step4_compare.py` (models + tables, 1023 s on 8 cores), `scripts/step4_figure.py` (figure).
Follows the step-4 protocol above exactly. `RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)`,
defaults otherwise, the same for both feature sets. Black box = 26 band features; physics = 8 index features (`data/features.csv`).
Seeds 0–4; the seed sets the RF and the `StratifiedKFold(5, shuffle=True)` split. Region split = leave-one-region-out over the
7 regions (training always contains all 10 classes). Random split = out-of-fold predictions for all 27,000 tiles. Per region, on
the same tiles: macro-F1 via `f1_score(labels=<classes with ≥50 tiles in region>, average="macro")` (all tiles kept), plus accuracy.
"Mean" = unweighted mean over the 7 regions, computed per seed; then mean ± std over the 5 seeds. Drop % = drop / random-split score,
per seed.

**Outputs:**
- `results/predictions_{blackbox,physics}_{random,region}.csv`: per tile × seed, `pred` + `p_<class>` probabilities (135,000 rows each).
- `results/step4_table.csv` (summary; `regsplit_*` = region-split columns; `baseline_f1_*` = no-skill macro-F1),
  `results/step4_per_seed_region.csv` (raw), `results/step4_paired.csv`, `results/step4_baseline.csv`, `results/step4_log.txt` (printed tables).
- `figures/random_vs_region.png`.

**Macro-F1 (%), mean ± std over 5 seeds**

| Region | BB random | BB region | BB drop pts (%) | Phys random | Phys region | Phys drop pts (%) |
|---|---|---|---|---|---|---|
| R0 | 82.9 ± 0.1 | 62.7 ± 0.4 | 20.2 ± 0.4 (24.3%) | 80.0 ± 0.5 | 68.5 ± 0.2 | 11.6 ± 0.4 (14.5%) |
| R1 | 87.4 ± 0.5 | 74.9 ± 0.4 | 12.5 ± 0.4 (14.3%) | 84.0 ± 0.2 | 73.6 ± 0.2 | 10.4 ± 0.1 (12.4%) |
| R2 | 88.5 ± 0.1 | 76.5 ± 0.2 | 12.0 ± 0.2 (13.6%) | 83.1 ± 0.1 | 73.5 ± 0.1 | 9.6 ± 0.2 (11.6%) |
| R3 | 88.7 ± 0.1 | 74.4 ± 0.2 | 14.3 ± 0.2 (16.1%) | 82.3 ± 0.3 | 69.9 ± 0.2 | 12.3 ± 0.4 (15.0%) |
| R4 | 86.7 ± 0.3 | 82.5 ± 0.3 | 4.2 ± 0.5 (4.9%) | 79.3 ± 0.4 | 77.9 ± 0.2 | 1.4 ± 0.4 (1.8%) |
| R5 | 86.4 ± 0.2 | 77.5 ± 0.1 | 8.9 ± 0.3 (10.3%) | 81.1 ± 0.3 | 75.2 ± 0.1 | 5.9 ± 0.2 (7.3%) |
| R7 | 83.0 ± 0.5 | 71.0 ± 0.1 | 11.9 ± 0.4 (14.4%) | 81.3 ± 0.5 | 72.7 ± 0.2 | 8.6 ± 0.6 (10.6%) |
| **Mean** | **86.2 ± 0.2** | **74.2 ± 0.1** | **12.0 ± 0.2 (13.9%)** | **81.6 ± 0.1** | **73.0 ± 0.0** | **8.6 ± 0.1 (10.5%)** |

**Accuracy (%), mean over regions:** BB 89.0 → 74.7 (drop 14.3 pts, 16.1%); physics 84.7 → 74.4 (drop 10.3 pts, 12.1%).
Per-region accuracy is in `results/step4_log.txt`.

**No-skill baseline** (predict the training-set majority class; ties broken alphabetically). Macro-F1 is 0–4.3 in every region,
under both splits. Random split always predicts AnnualCrop. Region split predicts SeaLake / Forest / AnnualCrop / HerbVeg depending
on which region is held out.

**Paired, (physics drop) − (black-box drop), 35 seed × region pairs:**
- Macro-F1, points: mean −3.44, range [−9.39, −1.41]. **Physics drops less in 35/35.**
- Macro-F1, % of random-split score: mean −3.53, range [−10.87, −0.52], 35/35.
- Accuracy, points: mean −4.06, range [−9.46, −1.08], 35/35.
- Per region (macro-F1 pts, mean): R0 −8.6, R7 −3.3, R5 −3.0, R4 −2.8, R2 −2.4, R1 −2.1, R3 −2.0.

**Interpretation (honest reading):**
- Physics features are **more stable** across regions: they drop less in every seed × region pair, including when the drop is
  measured relative to the starting score. The largest gap is in R0 (Iberia).
- They are **not more accurate on unseen regions overall**. Region-split mean macro-F1: black box 74.2 vs physics 73.0.
  Black box is higher in R1, R2, R3, R4 and R5. Physics is higher only in R0 (+5.8) and R7 (+1.7). Region-split accuracy is a near tie (74.7 vs 74.4).
- Most of the smaller drop comes from physics starting lower: it is 4.6 pts lower under the random split. The extra
  random-split skill of the 26 band features mostly does not transfer to new regions. It disappears, apart from about 1 pt on average.
- The random split overstates held-out-region macro-F1 by 12.0 pts (black box) and 8.6 pts (physics). It also changes the ranking
  of the two feature sets. Under the random split black box leads by 4.6 pts; on held-out regions the gap shrinks to 1.2 pts and is
  reversed in 2 of 7 regions.
- Seed std is small (≤0.6 pts) compared with the between-region differences. The 5 seeds measure RF and fold randomness, not
  uncertainty about the regions themselves (there are only 7 of them).

## Step 5 — Confidence and silent errors (2026-10-23)

**What:** script `scripts/step5_confidence.py` (tables), `scripts/step5_figure.py` (figure). Uses **only the saved step-4
predictions** (`results/predictions_*.csv`). Nothing was retrained. Confidence = max class probability (`p_*`) per tile. Every metric is
computed per (features, split, region, seed). "Mean" = unweighted mean over the 7 regions, computed per seed. Then mean ± std over 5 seeds.
Checks: `pred` equals the argmax class in every row with a single top probability. 97–186 rows per file (≤0.14%) have a tied top probability.
- Silent-error share = (wrong and conf ≥ 0.8) / wrong. Also reported: the same count / all tiles.
- Flagging: in each region × seed, tiles are ranked by confidence and the lowest round(0.2·n) are flagged. Ties are broken randomly with
  the seed. Coverage at X% = accuracy on the round(X/100·n) most confident tiles, using the same ranking.
- All metrics here use **plain accuracy**, not the step-4 macro-F1.

**Outputs:** `results/step5_table.csv` (long format: features, split, region, metric, mean, std), `results/step5_per_seed_region.csv` (raw),
`results/step5_region_minus_random.csv` (paired differences), `results/step5_log.txt` (printed tables), `figures/coverage_curve.png`.

**Mean over regions (%), mean ± std over seeds ≤ 0.3 everywhere**

| | BB random | BB region | Phys random | Phys region |
|---|---|---|---|---|
| Mean confidence | 79.6 | 69.5 | 78.2 | 72.0 |
| Accuracy | 89.0 | 74.7 | 84.7 | 74.4 |
| Confidence − accuracy (pts) | −9.4 | −5.2 | −6.5 | −2.4 |
| Silent errors, % of wrong | 6.7 | 4.8 | 9.7 | 8.4 |
| Silent errors, % of all tiles | 0.7 | 1.2 | 1.4 | 2.0 |

Per region, region split (BB / Phys): confidence R0 57.5/63.5, R1 63.6/65.6, R2 65.3/67.7, R3 74.9/76.5, R4 83.6/85.0, R5 74.9/75.9,
R7 66.8/69.7. See `results/step5_log.txt` for all per-region values.

**Paired region − random (35 seed × region pairs):** confidence falls in 35/35 pairs for both feature sets (BB −10.1 pts, range −18.5…−2.7;
Phys −6.2, range −12.9…−0.9). Accuracy falls more (BB −14.3; Phys −10.3). Confidence − accuracy rises by about 4 pts (BB 35/35; Phys 33/35).

**Flag the lowest-confidence 20%, region split (mean over regions):** BB flagged 40.5% accurate vs unflagged 83.2%, and the flag catches
50.7% of all errors. Phys flagged 41.3% vs unflagged 82.7%, catching 48.9% of errors. Flagging at random would catch 20%. Weakest in R0
(BB catches 34%, Phys 37%). Strongest in R4 (68–69%).

**Coverage curve, region split, accuracy (%) (mean over regions):**

| Coverage | 100 | 90 | 80 | 70 | 60 | 50 |
|---|---|---|---|---|---|---|
| Black box | 74.7 | 79.0 | 83.2 | 86.9 | 90.0 | 92.8 |
| Physics | 74.4 | 79.0 | 82.7 | 86.1 | 89.1 | 91.7 |

Spread over regions at 50% coverage: BB 78.3 (R0) … 99.9 (R4); Phys 82.6 (R0) … 99.4 (R4).

**Interpretation (honest reading):**
- **The random forest is underconfident everywhere.** Mean confidence is below accuracy in all 140 feature set × split × region × seed
  groups. In R0 on the region split it is only just below (by 0.4–0.6 pts). RF "probabilities" are tree-vote fractions, not calibrated probabilities. Every threshold-based
  number here (≥ 0.8) depends on this.
- **Does the model know it is in a new region? Partly.** Confidence falls in every region × seed pair, and it falls most where accuracy
  falls most (R0 largest, R4 smallest). But it falls only about 60–70% as much as accuracy does (BB 10.1 vs 14.3 pts; Phys 6.2 vs 10.3).
  The model "looks better calibrated" on new regions (gap −9.4 → −5.2 for BB) only because it started underconfident and the shift used
  up that margin. In R0 confidence ≈ accuracy for both feature sets. A model that started out calibrated would be overconfident there.
- **Silent errors:** the *share* of errors made with confidence ≥ 0.8 goes **down** on new regions (BB 6.7 → 4.8%, Phys 9.7 → 8.4%), because
  most new errors are low-confidence. But the *absolute rate* goes **up** (BB 0.7 → 1.2% of tiles, Phys 1.4 → 2.0%). So shift creates more
  confident mistakes in absolute terms, while most of the extra errors are visible as low confidence.
- **Physics features make more confident mistakes than the black box** in both splits (8.4 vs 4.8% of errors on new regions; 2.0 vs 1.2%
  of all tiles). Physics confidence drops less under shift (it was also more stable in accuracy, step 4), but it is not more trustworthy at
  high confidence.
- **Flagging works but is far from perfect.** The lowest-20% flag is about 2× more accurate on unflagged than flagged tiles and catches about
  half of the errors. It catches the fewest in the region where it is needed most (R0).
- **Coverage:** accepting the 50% most confident tiles raises held-out-region accuracy from about 74.5% to 92–93% for both feature sets. The
  two curves are equal at 90–100% coverage. Black box is slightly ahead at lower coverage (by up to 1.1 pts at 50%). The region spread is wide:
  in R0, even 50% coverage gives only 78% (BB) / 83% (Phys), which is below the *full-coverage* accuracy of R3–R5. Physics' R0 advantage from
  step 4 persists at every coverage level.
- Caveats: accuracy is not class-balanced, and region class mixes differ (step 2). Abstaining by confidence may drop some classes much more
  than others. That was not checked here. The seed std measures RF/fold randomness, not uncertainty over regions (n = 7).

### Step 5 follow-up — which classes get flagged? (2026-10-23)

**What:** `scripts/step5_flag_by_class.py` uses the saved predictions only, with the same lowest-20% flag as above (region split). It computes the
per-class flag rate in each region × seed. This is averaged over the regions where the class has tiles (PermanentCrop 5 regions, Forest 6,
the rest 7), then mean ± std over seeds. The rates include small classes in some regions (e.g. R0 Pasture, 13 tiles), so "pooled" (all
regions together) is given as a check. Outputs: `results/step5_flag_by_class.csv`, `results/step5_flag_by_class_raw.csv`,
`results/step5_flag_by_class_log.txt`.

| Class | BB flagged % | Phys flagged % | BB recall % | Phys recall % |
|---|---|---|---|---|
| Highway | 38.9 | 43.3 | 51.0 | 49.2 |
| **PermanentCrop** | 36.2 | 31.9 | 56.2 | 54.6 |
| **AnnualCrop** | 34.6 | 27.6 | 67.6 | 73.3 |
| **HerbaceousVegetation** | 33.0 | 25.2 | 66.9 | 66.4 |
| **Pasture** | 21.8 | 27.4 | 72.9 | 61.6 |
| Residential | 17.2 | 27.1 | 82.2 | 74.1 |
| River | 16.3 | 12.5 | 81.1 | 86.3 |
| Industrial | 9.8 | 12.8 | 91.8 | 89.2 |
| Forest | 4.0 | 5.7 | 88.4 | 92.5 |
| SeaLake | 1.4 | 1.3 | 97.9 | 99.1 |

**Reading:** the flag is very uneven across classes. Overall 20% of tiles are flagged, but the vegetation/crop classes and Highway get
about 22–43% flagged, while SeaLake and Forest get 1–6%. So "auto-accept the confident 80%" mostly keeps easy classes (water, forest,
industrial). About a third of the crop/herbaceous tiles, and up to ~40% of Highway, go to review. Class-balanced accuracy on accepted tiles is probably
lower than the 83% plain accuracy in the step-5 table (not computed). Physics flags fewer AnnualCrop/HerbVeg/PermanentCrop tiles than the black box,
but more Pasture and Residential. Flag rate roughly follows each class's error rate (lowest recall → most flagged).

## Step 6 — Relearning cost: how many new-region labels are needed to recover? (2026-10-24)

**What:** `scripts/step6_relearn.py` (models), `scripts/step6_analyze.py` (tables, half-gap, figure). For each held-out region × feature set
× seed 0–4: the region's tiles are split 50/50, stratified by class (`train_test_split(random_state=seed)`), into a label pool and a fixed
evaluation set. Train on the other 6 regions + k pool tiles, k ∈ {0, 10, 50, 100, 500}, and evaluate on the evaluation half only.
Two ways to choose the k tiles:
- (a) *random*: a seeded permutation of the pool;
- (b) *uncertainty*: the k pool tiles with the lowest max class probability under the k=0 model. One shot; ties broken randomly with the seed.

Both selections are nested in k, so k=10 ⊂ k=50 ⊂ … . The k=0 model is shared, so k=0 is identical for both methods (verified).
Metric: macro-F1 over the step-4 class list (classes with ≥50 tiles in the whole region; all evaluation tiles kept), plus accuracy.
Reference = step-4 random-split macro-F1 for that region.
- **Model: `RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)`. Step 6 uses 200 trees (step 4 used 500) for runtime, so
  k=0 differs slightly from step 4.** It is also evaluated on half of the region's tiles. Region-split mean macro-F1: BB 74.1 here vs 74.2 in
  step 4; Phys 73.2 vs 73.0.
- The reference is not perfectly like-for-like: 500 trees, scored on all of the region's tiles, not just the evaluation half.
- Pool sizes differ by region: R0 1247, R7 1387, R4 1616, R1 1746, R5 2036, R2 2449, R3 3017. So k=500 is 17–40% of the pool.
- **Runtime:** the estimate from one timed fit was ~36 min, so all 5 seeds were used (the limit was 45 min). The actual run took **65 min**
  (3882 s); black-box fits ran ~2.3× slower than the timed fit. By the time this showed up, stopping would have discarded the finished
  work (results are written only at the end), and a 3-seed restart would not have been faster. So the run was finished with 5 seeds.

**Outputs:** `results/step6_per_seed_region.csv` (raw, 700 rows), `results/step6_table.csv` (per-region and mean rows, with the reference),
`results/step6_half_gap.csv`, `results/step6_log.txt` (printed tables), `results/step6_run_log.txt`, `figures/relearning_curve.png`.

**Macro-F1 (%), mean over 7 regions, mean ± std over 5 seeds**

| k | BB random | BB uncertainty | Phys random | Phys uncertainty |
|---|---|---|---|---|
| 0 | 74.1 ± 0.4 | 74.1 ± 0.4 | 73.2 ± 0.2 | 73.2 ± 0.2 |
| 10 | 74.5 ± 0.3 | 74.8 ± 0.3 | 73.3 ± 0.3 | 73.5 ± 0.2 |
| 50 | 75.5 ± 0.4 | 76.2 ± 0.3 | 73.7 ± 0.2 | 74.4 ± 0.1 |
| 100 | 76.4 ± 0.5 | 77.4 ± 0.4 | 74.1 ± 0.0 | 75.1 ± 0.2 |
| 500 | 80.1 ± 0.5 | 82.2 ± 0.3 | 76.6 ± 0.2 | 78.2 ± 0.2 |
| reference | 86.2 | 86.2 | 81.6 | 81.6 |

Accuracy at k = 0 → 500: BB random 74.6 → 82.9, BB uncertainty → 84.8; Phys random 74.4 → 79.0, Phys uncertainty → 80.5.

**Labels needed to close half the gap from k=0 to the reference** (mean curve; linear interpolation in k):

| | k=0 | Reference | Target | F1 at k=500 | Gap closed at 500 | k for half gap |
|---|---|---|---|---|---|---|
| BB random | 74.1 | 86.2 | 80.2 | 80.1 | 49.5% | **not reached** (just short) |
| BB uncertainty | 74.1 | 86.2 | 80.2 | 82.2 | 66.8% | **≈ 331** |
| Phys random | 73.2 | 81.6 | 77.4 | 76.6 | 40.1% | **not reached** |
| Phys uncertainty | 73.2 | 81.6 | 77.4 | 78.2 | 59.6% | **≈ 396** |

Per region (k for half gap, BB rand / BB unc / Phys rand / Phys unc): R0 454/366/—/341, R1 197/189/481/369, R2 —/375/—/474,
R3 —/238/—/487, R4 —/353/—/409, R5 —/447/—/—, R7 —/333/407/189 (— = not reached by k=500).

**Uncertainty vs random, paired (35 seed × region pairs), macro-F1 pts:** BB +0.3 / +0.7 / +1.0 / +2.1 at k = 10 / 50 / 100 / 500
(uncertainty wins 22 / 22 / 26 / 33 of 35); Phys +0.2 / +0.6 / +1.0 / +1.6 (wins 20 / 26 / 29 / 35).

**Interpretation (honest reading):**
- **Relearning is slow at small k.** 10 labels change almost nothing (≤ +0.7 pts). 100 labels recover only 2–3 pts of a 12-pt (BB) or
  8-pt (Phys) gap. Even 500 labels from the new region close only 40–67% of the gap. Halfway needs ≈ 330–400 uncertainty-chosen labels
  per region; with random labels it is not reached within 500.
- **Uncertainty selection helps, and more as k grows.** It is clearly better at k=500 (33/35 and 35/35 pairs), but at k=10–50 it is
  inconsistent (wins 20–26 of 35). With one-shot selection, the gain at small k is within region-to-region noise.
- **Physics features recover more slowly than the black box**, in absolute points and as a fraction of their (smaller) gap. The black-box
  lead on new regions grows from 0.9 pts at k=0 to 3.5 (random) / 4.0 (uncertainty) at k=500. BB with *random* labels at k=500 (80.1)
  beats physics with *uncertainty* labels (78.2). The physics advantage from step 4 (a smaller drop) does not turn into cheaper adaptation.
  Once some local labels exist, the richer band features use them better.
- The linear interpolation between k=100 and k=500 is coarse (only one point in between). The half-gap k values are rough (± ~100).
  Regions differ a lot (R1 and R7 reach the target much sooner than R5).
- As before, seed std (≤ 0.5) reflects RF/split randomness, not uncertainty over regions (n = 7).

## Step 7 — Combined features: 26 band + 8 index features (2026-10-24)

**What:** `scripts/step7_combined.py` (models), `scripts/step7_analyze.py` (tables), `scripts/step7_figure.py` (figure).
This repeats the step-4 protocol exactly: same RF (500 trees, defaults), seeds 0–4, the same folds (the seed sets the RF and the
`StratifiedKFold`), the same macro-F1 (classes with ≥50 tiles in the region, all tiles kept) and accuracy, and the same "mean"/drop
definitions. There is one new feature set: **combined = the 26 black-box columns + the 8 physics columns (34)**. Only combined was
trained. Black box and physics are the saved step-4 results (`results/step4_per_seed_region.csv`). The analysis script asserts
that they reproduce `results/step4_table.csv`. RF `max_features` = int(√34) = 5, the same as black box (int(√26) = 5); physics uses 2.
- Each fold's probabilities are saved to `results/step7_folds/<split>_seed<s>_fold<k>.npz` as soon as the fold finishes. A restart
  skips finished folds.
- **Runtime:** the estimate from step 4's black-box time (691 s) was ~12–15 min. The first folds ran ~15× slower than in step 4. A
  50-tree benchmark showed this was not caused by the features (combined 18.7 s vs black box 20.8 s on the same fold). The whole
  machine was slow: a numpy matmul test was also several times slower than expected. Pausing the job with SIGSTOP for the benchmark
  ended it, but 1 fold was saved. The resumed run sped up partway through and finished in 1433 s (~24 min; 1 fold reused).

**Outputs:** `results/predictions_combined_{random,region}.csv` (same format as step 4), `results/step7_per_seed_region.csv` (raw,
combined only), `results/step7_table.csv` (summary, all three sets), `results/step7_paired.csv`, `results/step7_best_by_region.csv`,
`results/step7_log.txt` (printed tables), `results/step7_run_log.txt`.
`figures/random_vs_region.png` now shows all three sets (made by `scripts/step7_figure.py`). **Running `scripts/step4_figure.py` would
overwrite it with the two-set version.**

**Macro-F1 (%), mean ± std over 5 seeds** (BB and Phys blocks are unchanged from step 4)

| Region | BB random | BB region | BB drop pts (%) | Phys random | Phys region | Phys drop pts (%) | Comb random | Comb region | Comb drop pts (%) |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 82.9 ± 0.1 | 62.7 ± 0.4 | 20.2 (24.3%) | 80.0 ± 0.5 | 68.5 ± 0.2 | 11.6 (14.5%) | 86.0 ± 0.2 | 68.3 ± 0.4 | 17.7 ± 0.4 (20.6%) |
| R1 | 87.4 ± 0.5 | 74.9 ± 0.4 | 12.5 (14.3%) | 84.0 ± 0.2 | 73.6 ± 0.2 | 10.4 (12.4%) | 89.7 ± 0.3 | 78.3 ± 0.2 | 11.4 ± 0.5 (12.7%) |
| R2 | 88.5 ± 0.1 | 76.5 ± 0.2 | 12.0 (13.6%) | 83.1 ± 0.1 | 73.5 ± 0.1 | 9.6 (11.6%) | 89.6 ± 0.2 | 79.0 ± 0.2 | 10.6 ± 0.3 (11.9%) |
| R3 | 88.7 ± 0.1 | 74.4 ± 0.2 | 14.3 (16.1%) | 82.3 ± 0.3 | 69.9 ± 0.2 | 12.3 (15.0%) | 90.1 ± 0.1 | 76.2 ± 0.2 | 13.9 ± 0.2 (15.4%) |
| R4 | 86.7 ± 0.3 | 82.5 ± 0.3 | 4.2 (4.9%) | 79.3 ± 0.4 | 77.9 ± 0.2 | 1.4 (1.8%) | 87.2 ± 0.3 | 84.5 ± 0.1 | 2.7 ± 0.3 (3.1%) |
| R5 | 86.4 ± 0.2 | 77.5 ± 0.1 | 8.9 (10.3%) | 81.1 ± 0.3 | 75.2 ± 0.1 | 5.9 (7.3%) | 87.0 ± 0.2 | 78.6 ± 0.2 | 8.3 ± 0.3 (9.6%) |
| R7 | 83.0 ± 0.5 | 71.0 ± 0.1 | 11.9 (14.4%) | 81.3 ± 0.5 | 72.7 ± 0.2 | 8.6 (10.6%) | 86.6 ± 0.3 | 78.0 ± 0.2 | 8.7 ± 0.2 (10.0%) |
| **Mean** | **86.2** | **74.2** | **12.0 (13.9%)** | **81.6** | **73.0** | **8.6 (10.5%)** | **88.0 ± 0.1** | **77.6 ± 0.0** | **10.5 ± 0.0 (11.9%)** |

**Accuracy (%), mean over regions:** combined 90.5 → 78.0 (drop 12.5 pts, 13.8%), vs BB 89.0 → 74.7 and Phys 84.7 → 74.4.

**Paired, region-split macro-F1, combined − other, 35 seed × region pairs:**
- vs black box: mean **+3.34** pts, range [+0.87, +7.28], **combined higher in 35/35**.
- vs physics: mean **+4.53** pts, range [−0.70, +7.01], **combined higher in 32/35**. All 3 losses are in R0.
- Per region (mean over seeds), vs BB / vs Phys: R0 +5.5 / −0.2, R1 +3.4 / +4.7, R2 +2.5 / +5.5, R3 +1.9 / +6.3, R4 +2.0 / +6.7,
  R5 +1.1 / +3.4, R7 +6.9 / +5.2.
- Drop (macro-F1 pts): combined drops less than BB in 35/35 pairs (mean −1.5). It drops more than physics in 33/35 (mean +1.9).
- Region-split accuracy: combined higher than BB in 35/35 (+3.4), and higher than Phys in 30/35 (+3.6).

**Best feature set per region, region split (macro-F1, mean over seeds):** combined in R1, R2, R3, R4, R5, R7 (5/5 seeds each; margin over
2nd 1.1–5.2 pts). **R0: physics 68.45 vs combined 68.29, a tie** (0.16 pts; physics best in 3 of 5 seeds, combined in 2). Black box is never best.

**Interpretation (honest reading):**
- **Combined is the best of the three on held-out regions:** 77.6 vs 74.2 (BB) and 73.0 (Phys) mean macro-F1. It beats black box in every
  seed × region pair, and physics in all but 3 R0 pairs. It is also best under the random split (88.0).
- The index features add information that RF does not build from the raw band means/stds by itself. This may be because ratios are
  hard to approximate with axis-aligned splits, and because the index stds summarize within-tile *pixel-wise* ratios, which the per-band
  stds cannot give. This was not tested further. The gain over black box is larger on held-out regions (+3.3) than under the random
  split (+1.8), so the index features help transfer more than they help interpolation.
- **Combined does not inherit physics' stability.** Its drop (10.5 pts) is between BB (12.0) and Phys (8.6), and it drops more than physics
  in 33/35 pairs. Its higher held-out score comes from a much higher starting point, not from a smaller drop.
- **R0 (Iberia) is the exception:** combined only ties physics there. The band features seem to hurt in R0 about as much as the indices help,
  i.e. combined still drops 17.7 pts there. In R7 (also southern) combined gains the most over both (+6.9 vs BB, +5.2 vs Phys).
- The random split still overstates held-out macro-F1 by 10.5 pts for combined. The ranking does not flip this time (combined is first under
  both splits), but the gap between combined and physics narrows from 6.5 (random) to 4.5 pts (region).
- Caveats as before: seed std (≤ 0.5) measures RF/fold randomness, not uncertainty over regions (n = 7). The "35/35" counts are 7 regions ×
  5 correlated seeds, not 35 independent tests. Combined has more features than either other set, so this compares feature *sets* with a
  fixed RF, not equal-capacity models.

## Step 8 — Spatial-block split: a fair "seen it before" reference (2026-10-24)

**What:** `scripts/step8_blocks.py` (block assignment), `scripts/step8_block_split.py` (models), `scripts/step8_analyze.py` (tables,
decomposition, step-6 half-gap), `scripts/step8_figure.py` (figure).
1. **Blocks:** tile centre (lat, lon from `data/metadata.csv`) → EPSG:3035 (LAEA Europe, equal-area, m) via `rasterio.warp.transform`
   (no pyproj in the venv), then floor(x/20 km), floor(y/20 km). Output `data/blocks.csv` (`path, class, region, x, y, bx, by, block`,
   same row order as `features.csv`). **2,331 blocks**; tiles/block median 5, max 249; 418 single-tile blocks. 15 blocks (166 tiles) span
   two regions (K-means borders); this does not matter for the block split, which ignores regions.

   | Region | Tiles | Blocks | Tiles/block median | Largest block, % of region |
   |---|---|---|---|---|
   | R0 | 2495 | 199 | 6 | 2.8 |
   | R1 | 3493 | 253 | 6 | 3.8 |
   | R2 | 4898 | 362 | 6 | 3.5 |
   | R3 | 6035 | 526 | 7 | 3.3 |
   | R4 | 3232 | 412 | 4 | 5.0 |
   | R5 | 4073 | 393 | 5 | 3.5 |
   | R7 | 2774 | 201 | 5 | 9.0 |

   (Per-region block counts sum to 2,346 because of the 15 straddling blocks.)
   Near-twin check (centre-to-centre, tiles are 640 m wide): nearest-neighbour distance median 0.79 km (p90 3.4 km); the nearest neighbour
   has the same class for 80% of tiles and lies in the same block for 90.7%. **3.9% of tiles have a neighbour < 2 km away in another block**,
   so some edge leakage remains in the block split. The leakage estimate below is therefore a lower bound.
2. **Split:** `GroupKFold(n_splits=5, shuffle=True, random_state=seed)` over blocks, all 27,000 tiles, seeds 0–4 (the seed sets the folds
   and the RF). Out-of-fold predictions for every tile. With shuffle, GroupKFold does not balance fold sizes: test folds have 4.7k–6.7k
   tiles. It is not class-stratified, but every training fold contained all 10 classes (checked per fold).
3. **Model/metrics:** identical to step 4. RF 500 trees, defaults, `random_state=seed`; all three feature sets; macro-F1 over the step-4
   classes (≥50 tiles in the region, all tiles kept) + accuracy; "Mean" = unweighted over 7 regions per seed, then mean ± std over seeds.
   Random- and region-split values are the saved step-4/step-7 results.
4. **Checkpointing:** each fold → `results/step8_folds/<features>_seed<s>_fold<k>.npz` (atomic rename), skipped on restart;
   `results/step8_per_seed_region.csv` is rewritten after every seed.
- **Runtime:** estimate ~12 min (from step-4 timings); actual 707 s (~12 min), background run, ~10 s per BB/combined fold, ~6 s physics.

**Outputs:** `data/blocks.csv`, `results/step8_blocks_per_region.csv`, `results/predictions_{blackbox,physics,combined}_block.csv`,
`results/step8_per_seed_region.csv` (raw), `results/step8_table.csv` (random/block/region/leakage/shift per features × region × metric),
`results/step8_half_gap.csv`, `results/step8_log.txt` (all printed tables), `results/step8_run_log.txt`.
`figures/random_vs_region.png` now has random (open circle), block (half diamond), region (filled) for all three sets (made by
`scripts/step8_figure.py`; **`step4_figure.py` / `step7_figure.py` would overwrite it**).

**Macro-F1 (%), mean ± std over 5 seeds. Leakage = random − block; shift = block − region; leakage + shift = step-4 drop.**

| Region | BB random | BB block | BB region | BB leak | BB shift | Phys random | Phys block | Phys region | Phys leak | Phys shift | Comb random | Comb block | Comb region | Comb leak | Comb shift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 82.9 | 78.0 ± 0.3 | 62.7 | 4.9 | 15.2 | 80.0 | 77.2 ± 0.4 | 68.5 | 2.9 | 8.7 | 86.0 | 82.3 ± 0.4 | 68.3 | 3.7 | 14.0 |
| R1 | 87.4 | 86.0 ± 0.6 | 74.9 | 1.4 | 11.1 | 84.0 | 82.9 ± 0.3 | 73.6 | 1.1 | 9.3 | 89.7 | 88.2 ± 0.5 | 78.3 | 1.5 | 10.0 |
| R2 | 88.5 | 87.4 ± 0.2 | 76.5 | 1.1 | 10.9 | 83.1 | 81.4 ± 0.2 | 73.5 | 1.7 | 7.9 | 89.6 | 88.6 ± 0.2 | 79.0 | 1.0 | 9.6 |
| R3 | 88.7 | 86.9 ± 0.3 | 74.4 | 1.8 | 12.5 | 82.3 | 78.9 ± 0.9 | 69.9 | 3.3 | 9.0 | 90.1 | 88.2 ± 0.8 | 76.2 | 1.9 | 12.0 |
| R4 | 86.7 | 85.7 ± 0.2 | 82.5 | 1.0 | 3.2 | 79.3 | 78.7 ± 0.6 | 77.9 | 0.6 | 0.8 | 87.2 | 86.6 ± 0.2 | 84.5 | 0.7 | 2.1 |
| R5 | 86.4 | 84.7 ± 0.2 | 77.5 | 1.7 | 7.1 | 81.1 | 79.4 ± 0.3 | 75.2 | 1.7 | 4.2 | 87.0 | 85.3 ± 0.4 | 78.6 | 1.7 | 6.6 |
| R7 | 83.0 | 76.9 ± 1.0 | 71.0 | 6.1 | 5.9 | 81.3 | 77.1 ± 1.4 | 72.7 | 4.2 | 4.4 | 86.6 | 83.1 ± 0.5 | 78.0 | 3.5 | 5.1 |
| **Mean** | **86.2** | **83.7 ± 0.2** | **74.2** | **2.6 ± 0.2** | **9.4 ± 0.3** | **81.6** | **79.4 ± 0.2** | **73.0** | **2.2 ± 0.2** | **6.3 ± 0.2** | **88.0** | **86.0 ± 0.2** | **77.6** | **2.0 ± 0.1** | **8.5 ± 0.1** |

Share of the step-4 drop that is leakage (mean): BB 21%, Phys 26%, Comb 19%. Per region it ranges from 9% (R2) to 41–51% (R7).
**Accuracy (%), mean:** BB 89.0 / 86.3 / 74.7 (leak 2.7, shift 11.7); Phys 84.7 / 82.4 / 74.4 (2.3, 8.0); Comb 90.5 / 88.5 / 78.0 (2.0, 10.5).
Per-region accuracy and leakage std are in `results/step8_log.txt`.

**Paired (35 seed × region pairs, macro-F1):** block < random in 35/35 (BB, Comb) and 34/35 (Phys; one R4 pair −0.3); region < block in
35/35 for all three. Block-split ranking: Comb 86.0 > BB 83.7 > Phys 79.4 (Comb > BB 35/35, Comb > Phys 35/35, BB > Phys 32/35), the same
ranking as the random split.

**Step 6 recomputed with the block-split reference** (no re-run; step-6 curves from `results/step6_table.csv`, same linear interpolation):

| | k=0 | F1 at k=500 | Ref random → block | Gap closed at 500: random ref → block ref | k for half gap: random ref → block ref |
|---|---|---|---|---|---|
| BB random | 74.1 | 80.1 | 86.2 → 83.7 | 49.5% → **62.8%** | not reached → **≈ 367** |
| BB uncertainty | 74.1 | 82.2 | 86.2 → 83.7 | 66.8% → **84.7%** | ≈ 331 → **≈ 225** |
| Phys random | 73.2 | 76.6 | 81.6 → 79.4 | 40.1% → **54.5%** | not reached → **≈ 454** |
| Phys uncertainty | 73.2 | 78.2 | 81.6 → 79.4 | 59.6% → **80.9%** | ≈ 396 → **≈ 253** |

Per region (block ref) see `results/step8_log.txt`. With the block reference, R7 is passed at k=500 (gap closed 120–149% for 3 of 4 curves),
and the R4 physics gap is only 0.8 pts, so its half-gap k (99 for uncertainty) is noise-dominated.

**Interpretation (honest reading):**
- **Most of the step-4 drop is real region shift, not near-twin leakage.** On average, near-twin leakage accounts for 2.0–2.6 pts of macro-F1
  (about 20–26% of the drop); 6.3–9.4 pts remain when the model has never seen anything within the same 20 km block. Every one of the 35
  seed × region pairs still drops from block to region, for every feature set.
- **Leakage is very uneven by region.** It is about 1–2 pts in R1–R5 (exception: Phys R3, 3.3) but 3–6 pts in R0 (Iberia) and R7 (Balkans). In R7 it is about half the
  drop (BB 6.1 leak vs 5.9 shift). Why these two regions leak more was not checked. R7 has the largest single block
  (9% of the region), and block-split std is highest there (1.0–1.4), so its fold assignment matters more.
- **Physics vs black box:** physics also leaks less (2.2 vs 2.6) and shifts less (6.3 vs 9.4). Its stability advantage from step 4 therefore
  holds against the fair reference: most of the 3.4-pt difference in drop is in the shift part (3.1 pts), not leakage. Under the block split
  black box still leads physics by 4.3 pts, so the ranking change seen in step 4 (BB lead 4.6 → 1.2) is about region shift, not leakage.
- **Combined** leaks the least (2.0) but its shift (8.5) is between physics and BB, as in step 7.
- **Relearning (step 6) looks less hopeless against the fair reference.** The block reference is 2.2–2.5 pts lower, so the gaps are
  smaller: 500 uncertainty-chosen labels close 81–85% of the gap (was 60–67%), and half the gap needs ≈ 225 (BB) / ≈ 253 (Phys) labels
  (was 331 / 396). Random selection now reaches half the gap within 500 (≈ 367 / ≈ 454). The relative findings of step 6 (uncertainty > random,
  BB recovers faster than physics) do not depend on the reference.
- **Caveats:** the block split still leaks at block edges (3.9% of tiles have a < 2 km neighbour across the border), so 2–2.6 pts is a lower
  bound on leakage, and "shift" is an upper bound on region shift at this scale. 20 km is one chosen scale, and larger blocks would move some
  of the "shift" into "leakage"; not tested. The step-6 recomputation mixes models (step 6: 200 trees, evaluated on half the region; reference:
  500 trees, all tiles), as before. Seed std measures RF/fold randomness, not uncertainty over regions (n = 7), and the 35 pairs are not independent.

## Shipping — README, tests, repo files (2026-10-24)

- `README.md` written from the numbers in this file and `results/` (approved after one review round).
- `tests/` (pytest 9.1.1): (a) `BANDS` in `scripts/make_features.py` equals the confirmed order, feature columns follow it, and
  the smoothness tiers on 5 raw tiles per class match the native resolutions; (b) `norm_diff` stays in [−1, 1] on the full uint16
  range (0/0 → 0), on raw-tile pixels, and the `*_mean`/`*_std` index columns of `data/features.csv` are within bounds;
  (c) leave-one-region-out folds share no tile and test each tile once, and the saved `results/step7_folds/region_*` checkpoints
  match; (d) 20 km `GroupKFold` folds share no block for seeds 0–4, and so do all saved `results/step8_folds/` checkpoints.
  The scripts run at import time, so tests read constants/functions from the source with `ast`. Data-dependent tests skip
  when `data/` or `results/` files are absent. **Result: 18 passed** (a fresh clone without raw data: 11 passed, 7 skipped).
  The split tests read the committed `data/metadata.csv` (same rows/order as `features.csv`, checked by a test).
- `.gitignore` (venv, raw data, predictions, fold checkpoints, `data/features.csv` > 5 MB, `results/*_run_log.txt`), `requirements.txt` (`pip freeze`
  of the venv, Python 3.14.7), MIT `LICENSE`.
- Committed: 59 files, 6.9 MB, including `data/metadata.csv`, `data/blocks.csv` (derived tables, no imagery). `results/step4_log.txt`: home path replaced by `~/`; no committed file contains the home path.
