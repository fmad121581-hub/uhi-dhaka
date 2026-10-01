# v2 changelog (pipeline corrections and deeper analysis)

v1 outputs are kept in `data/output_v1/` and the original scripts in `scripts/_v1_original/`.
v2 writes to `data/processed_v2/` and `data/output_v2/`.

## Bugs fixed
| # | Problem in v1 | Effect | Fix |
|---|---|---|---|
| 1 | Fill pixels (DN 0) not masked in ST_B10 | 149 K = -124 C values in min, mean, std, correlations | Mask DN 0 / nodata, plausibility range 0-72 C (`uhi_common.py`, `02_lst_derivation.py`) |
| 2 | NDVI emissivity correction applied to ST_B10 | ST_B10 is already surface temperature; corrected twice | Removed; LST = 0.00341802*DN + 149 - 273.15 |
| 3 | NDVI from raw DN | The -0.2 reflectance offset does not cancel in the ratio | Convert to reflectance (2.75e-5*DN - 0.2) first |
| 4 | Bilinear resampling of LULC class codes and DN bands | Invented classes at edges, fill blended into real pixels | Nearest neighbour for categorical and DN layers |
| 5 | Water included in NDVI-LST regression | Water is cool and NDVI <= 0, flips the sign (r = -0.07) | Report all-pixel and land-only results side by side |
| 7 | WorldCover tile N21E090 stops at 24 N | 59 km2 (4%) of the study area, mostly north Dhamrai/Savar, had no land cover | Mosaic tile N24E090 (`01_preprocess.py`); all results re-run |
| 6 | No cloud/shadow mask | Scene cloud cover 5.4% | QA_PIXEL downloaded and applied (5.4% scene cloud, 2,158 pixels removed in scene 1) |

## Headline numbers, v1 -> v2
- Built-up mean LST: 31.0 C (std 5.7) -> 29.3 C (std 1.8)
- Hottest ward mean LST: 34.3 C -> 32.1 C (Ward 90, Shyampur)
- NDVI-LST r: -0.068 -> -0.112 (all pixels) and **-0.584 (land only)**
- Built-up indicator r: 0.378 -> 0.484 (land only)
- Vegetated area (NDVI > 0.2): 45.7% -> 87.9%

## New in v2 (`09_deep_analysis.py`)
- SUHI intensity: urban core vs rural reference = 2.87 C (95% CI 2.64-3.10, spatial block bootstrap)
- Urban-rural gradient: 30.4 C within 6 km of the centre falling to 26.9 C beyond 40 km
- Ward spatial autocorrelation: Moran's I = 0.84 (p = 0.001, 196 wards)
- Gi* hotspots: 45 hot, 45 cold, 106 not significant (p < 0.05); hot cluster sits in the southern core (Old Dhaka, Shyampur, Kadamtali) and a corridor north of it
- Random Forest driver model (30k pixels): spatial-block CV R2 = 0.65 vs random CV R2 = 0.75
  (random CV overstates skill by ~0.10 because neighbouring pixels leak across folds)
- Permutation importance: distance from centre 0.43, built fraction 330 m 0.18, NDVI 0.12, population density 0.07
- Exposure: 48% of the population lives in pixels >= mean+1SD (30.0 C); 25% in the hottest decile (>= 30.7 C)

## Still open
1. Add `QA_PIXEL` for scene 01 and re-run 01-03 (cloud mask).
2. Wet-season scene (Jun-Sep) plus one more dry-season year: needed for the multi-temporal claim.
   Download from USGS EarthExplorer (needs login), not reachable from the analysis environment.
3. Ground truth: compare LST against Dhaka weather-station air temperature (BMD) on scene dates.
4. Interpretation caveat: Feb 24, 10:25 local time, dry season. SUHI here is a daytime surface
   measure, not an air-temperature UHI, and one date cannot give seasonal mean intensity.

## Multi-date analysis (`10_multitemporal.py`)
Scenes under `data/raw/landsat/scene_*/` are auto-discovered, cloud-masked with QA_PIXEL (cloud, shadow, cirrus, snow flags grown by 150 m), put on one 30 m grid and compared. Dates with < 60% clear pixels inside the boundary are dropped.

| Date | Season | Sensor | Clear | Mean land LST | SUHI (urban - rural) | NDVI-LST r (land) |
|---|---|---|---|---|---|---|
| 2022-02-24 | dry / winter | L9 | 99% | 28.3 C | 2.9 C | -0.59 |
| 2026-02-19 | dry / winter | L9 | 95% | 30.1 C | 4.5 C | -0.65 |
| 2025-05-07 | pre-monsoon | L9 | 87% | 35.6 C | 4.7 C | -0.62 |
| 2026-06-03 | monsoon onset | L8 | 73% | 39.3 C | 7.1 C | -0.63 |
| 2025-10-22 | post-monsoon | L8 | 96% | 35.4 C | 4.2 C | -0.38 |

Dropped: 2025-07-10 (0% clear), 2026-06-27 (51%), 2024-09-09 (53%). Mid-monsoon (Jul-Aug) is effectively unobservable over Dhaka, so say "monsoon onset" in the write-up, not "monsoon".

- SUHI ranges 2.9 to 7.1 C across the year. Report the range and the season, never one number.
- Ward LST rank agreement across all 5 dates: Spearman 0.74 to 0.89. Three wards are in the top decile on every date (`persistent_hotspots.csv`).
- NDVI-LST r is -0.59 to -0.65 on four of five dates; post-monsoon (-0.38) is weaker, plausibly because crops/wet soil decouple NDVI from surface temperature.
- Before the cloud buffer, 2024-09-09 gave SUHI = 11.1 C. That was thin-cloud contamination, not a real signal, which is why the buffer and the 60% rule exist.

## Robustness and validation (`12_robustness.py`, `13_validation.py`)
- Exposure and Random Forest repeated on all five dates. Spatial-CV R2 0.63 to 0.78. Population above mean + 1 SD: 40 to 56%; in the hottest decile: 23 to 39%.
- Correction to the earlier single-date reading: distance from the centre ranked first only in Feb 2022 (0.43). On the other dates local built-up share and NDVI led (distance 0.04 to 0.21). The README no longer quotes one ranking.
- Validation: LST within 5 km of Dhaka airport vs ERA5-Land air temperature at overpass, r = 0.97 (n = 5), LST 3.4 to 10.0 C above air. NOAA station data existed only for 2025-05-07 (air 31.8 C, LST 38.2 C).
