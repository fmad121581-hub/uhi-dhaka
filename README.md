# Dhaka Urban Heat Island Analysis

**Fahim Ahmed** | BUET Urban & Regional Planning (2nd Year)  
Landsat 9 · ESA WorldCover · SRTM · WorldPop · NASA POWER  
Python · geopandas · rasterio · scikit-learn · matplotlib

---

## Overview

This project quantifies the Urban Heat Island (UHI) effect across the Dhaka Metropolitan Area using freely available satellite data and a fully reproducible Python pipeline. It derives Land Surface Temperature (LST) from Landsat 9 thermal imagery, correlates it with vegetation cover, land use, and population density, and maps the spatial pattern of urban heat at ward level.

**Key findings:**
- Urban core (Shyampur, Hazaribagh) is **~7°C hotter** than the rural fringe (Dohar, Dhamrai)
- Built-up area shows the strongest correlation with LST (Pearson r = 0.38)
- 45.7% of the study area retains vegetation cover despite rapid urbanisation
- Rainfall trend shows +55.5 mm/yr increase over 2004–2023, contextualising seasonal LST variation

---

## Study Area

Dhaka Metropolitan Area, Bangladesh  
Boundary: WGS84 (EPSG:4326) | Analysis CRS: UTM Zone 45N (EPSG:32645)  
Landsat scene: Path 137, Row 044 | Acquisition date: 24 February 2022

---

## Data Sources

| Layer | Source | Resolution |
|---|---|---|
| Landsat 9 OLI/TIRS C2 L2 (Bands 4, 5, 10) | USGS EarthExplorer | 30m |
| ESA WorldCover 2021 v2.0 | ESA / Vito | 10m |
| SRTM DEM | USGS EarthExplorer | 30m |
| WorldPop Population Density 2020 | WorldPop Hub | 100m |
| GADM Admin Boundaries Level 4 | GADM v4.1 | Vector |
| Monthly Rainfall 2004–2023 | NASA POWER MERRA-2 | Point |

All data is free and publicly available. Download instructions are in `data/raw/`.

---

## Project Structure

```
uhi_dhaka/
├── data/
│   ├── raw/
│   │   ├── landsat/scene_01_dhaka/   ← Landsat 9 bands + MTL.txt
│   │   ├── lulc/                     ← ESA WorldCover TIF
│   │   ├── dem/                      ← SRTM 30m
│   │   ├── population/               ← WorldPop TIF
│   │   ├── admin/                    ← GADM Level-4 shapefile
│   │   └── rainfall/                 ← NASA POWER monthly CSV
│   ├── processed/                    ← clipped + reprojected rasters
│   └── output/                       ← final maps and CSVs
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_lst_derivation.py
│   ├── 03_ndvi.py
│   ├── 04_lulc_reclassify.py
│   ├── 05_zonal_stats.py
│   ├── 06_correlation.py
│   ├── 07_maps.py
│   └── 08_rainfall_analysis.py
├── dhaka_boundary.shp                ← study area boundary (+ companion files)
└── README.md
```

---

## Methodology

### Step 1 — Preprocessing (`01_preprocess.py`)
All rasters are clipped to the Dhaka boundary and reprojected to UTM Zone 45N (EPSG:32645). The boundary CRS is matched to each source raster before clipping to handle Landsat bands stored in UTM projection.

### Step 2 — LST Derivation (`02_lst_derivation.py`)
Land Surface Temperature is derived from Landsat 9 Band 10 (TIRS) using the Landsat Collection 2 Level-2 scale factors from the MTL metadata file. An emissivity correction is applied using the NDVI Threshold Method (Sobrino et al. 2004):
- NDVI < 0.2 → ε = 0.97 (built-up/bare)
- NDVI > 0.5 → ε = 0.99 (dense vegetation)
- 0.2 ≤ NDVI ≤ 0.5 → ε estimated from fractional vegetation cover

LST range: 24°C – 52°C across the study area.

### Step 3 — NDVI (`03_ndvi.py`)
NDVI = (Band 5 − Band 4) / (Band 5 + Band 4)  
Mean NDVI: 0.178 | Vegetation cover (NDVI > 0.2): 45.7%

### Step 4 — LULC Reclassification (`04_lulc_reclassify.py`)
ESA WorldCover 11-class map reclassified into 4 classes:

| Class | Area (km²) | Share |
|---|---|---|
| Built-up | 226.5 | 18.3% |
| Vegetation | 937.0 | 75.7% |
| Water | 83.5 | 6.7% |
| Bare soil | 10.7 | 0.9% |

### Step 5 — Zonal Statistics (`05_zonal_stats.py`)
Mean LST per LULC class and per GADM Level-4 admin unit (203 units within boundary). LULC resampled from 10m to 30m to align with LST grid before comparison.

| LULC Class | Mean LST |
|---|---|
| Built-up | 31.0°C |
| Bare soil | 30.1°C |
| Vegetation | 29.0°C |
| Water | 27.2°C |

Hottest ward: **Ward No-90, Shyampur (34.3°C)**  
Coolest ward: **Muksudpur, Dohar (26.7°C)**  
UHI intensity: **~7.6°C**

### Step 6 — Correlation Analysis (`06_correlation.py`)
Sampled at a 250m regular grid (23,743 valid points). OLS regression and Pearson correlation between LST and three predictors:

| Predictor | Pearson r | R² |
|---|---|---|
| Built-up indicator | +0.378 | 0.143 |
| Population density | +0.252 | 0.063 |
| NDVI | −0.068 | 0.005 |

### Step 7 — Maps (`07_maps.py`)
Four publication-quality PNG maps at 150 dpi: LST, NDVI, LULC, and UHI intensity summary panel.

### Step 8 — Rainfall Analysis (`08_rainfall_analysis.py`)
20-year NASA POWER rainfall trend (2004–2023): mean 2,263 mm/yr, trend +55.5 mm/yr. Monsoon season (Jun–Sep) accounts for 63% of annual total. The February acquisition date falls in the dry season (mean 52 mm), when LST is highest and UHI effect is most pronounced.

---

## Results

![UHI Summary Panel](data/output/uhi_summary_panel.png)

![Rainfall Analysis](data/output/rainfall_analysis.png)

---

## How to Reproduce

### Requirements
```
pip install geopandas rasterio numpy pandas matplotlib scikit-learn
```

### Run in order
```bash
python scripts/01_preprocess.py
python scripts/02_lst_derivation.py
python scripts/03_ndvi.py
python scripts/04_lulc_reclassify.py
python scripts/05_zonal_stats.py
python scripts/06_correlation.py
python scripts/07_maps.py
python scripts/08_rainfall_analysis.py
```

All paths are relative. Clone the repo, place the raw data files in the correct folders (see Data Sources above), and run the scripts in order.

---

## References

- Avdan, U. & Jovanovska, G. (2016). Algorithm for automated mapping of land surface temperature using Landsat 8 satellite data. *Journal of Sensors*.
- Sobrino, J.A. et al. (2004). Land surface temperature retrieval from LANDSAT TM 5. *Remote Sensing of Environment*.
- Weng, Q. et al. (2004). Estimation of land surface temperature with Landsat ETM+ data. *Remote Sensing of Environment*.
- ESA WorldCover 2021 v2.0 — https://esa-worldcover.org
- USGS Landsat Collection 2 Level-2 Science Product Guide

---

*Part of a GIS + data science portfolio. BUET Urban & Regional Planning, 2026.*
