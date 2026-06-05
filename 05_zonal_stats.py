"""
05_zonal_stats.py
-----------------
Step 5 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Calculates mean, min, max, and std LST for each LULC class
  - Calculates mean LST for each GADM Level-4 admin unit
  - Saves both result tables as CSVs to data/output/

Fix: LST (30m) and LULC (10m) have different pixel dimensions.
     We resample LULC to match LST resolution before comparing.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling as RIOResampling
from rasterio.features import rasterize

# Path constants
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT    = os.path.join(ROOT, "data", "output")

LST_PATH    = os.path.join(PROCESSED, "lst_celsius.tif")
LULC_PATH   = os.path.join(PROCESSED, "lulc_4class.tif")
ADMIN_PATH  = os.path.join(ROOT, "data", "raw", "admin", "gadm41_bgd_4.shp")
BOUNDARY    = os.path.join(PROCESSED, "dhaka_boundary_utm.shp")

LULC_CSV  = os.path.join(OUTPUT, "lst_by_lulc_class.csv")
ADMIN_CSV = os.path.join(OUTPUT, "lst_by_admin_unit.csv")

LULC_LABELS = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Bare soil"}


def load_and_align_rasters(lst_path, lulc_path):
    """
    Load LST and LULC rasters, resampling LULC to match LST dimensions exactly.
    LST is 30m resolution; LULC is 10m. We resample LULC to 30m to align them.
    Returns: (lst_array, lulc_array, lst_profile)
    """
    # Load LST as the reference grid
    with rasterio.open(lst_path) as lst_src:
        lst_data = lst_src.read(1).astype(np.float32)
        if lst_src.nodata is not None:
            lst_data[lst_data == lst_src.nodata] = np.nan
        lst_profile = lst_src.profile
        lst_height  = lst_src.height
        lst_width   = lst_src.width
        lst_transform = lst_src.transform
        lst_crs     = lst_src.crs

    # Load LULC and resample it to exactly match the LST grid
    with rasterio.open(lulc_path) as lulc_src:
        # read() with out_shape resamples on the fly to the target dimensions
        # nearest neighbour is correct for categorical data (class codes)
        lulc_data = lulc_src.read(
            1,
            out_shape   = (lst_height, lst_width),
            resampling  = RIOResampling.nearest
        ).astype(np.uint8)

    print(f"  LST shape:  {lst_data.shape}")
    print(f"  LULC shape: {lulc_data.shape} (resampled to match LST)")

    return lst_data, lulc_data, lst_profile


def zonal_stats_by_lulc(lst_array, lulc_array):
    """
    Compute mean, min, max, std of LST for each LULC class.
    Both arrays must have identical shapes.
    """
    rows = []
    for code, label in LULC_LABELS.items():
        # Boolean mask: pixels in this LULC class with valid LST
        mask = (lulc_array == code) & (~np.isnan(lst_array))
        lst_zone = lst_array[mask]

        if lst_zone.size == 0:
            rows.append({"class_code": code, "class_label": label,
                          "pixel_count": 0, "mean_lst": np.nan,
                          "min_lst": np.nan, "max_lst": np.nan, "std_lst": np.nan})
            continue

        rows.append({
            "class_code"  : code,
            "class_label" : label,
            "pixel_count" : int(lst_zone.size),
            "mean_lst"    : round(float(np.mean(lst_zone)), 2),
            "min_lst"     : round(float(np.min(lst_zone)),  2),
            "max_lst"     : round(float(np.max(lst_zone)),  2),
            "std_lst"     : round(float(np.std(lst_zone)),  2),
        })

    return pd.DataFrame(rows)


def zonal_stats_by_admin(lst_src, admin_gdf):
    """
    Compute mean LST for each GADM Level-4 admin unit by rasterising each polygon.
    """
    lst_array = lst_src.read(1).astype(np.float32)
    if lst_src.nodata is not None:
        lst_array[lst_array == lst_src.nodata] = np.nan

    out_shape = (lst_src.height, lst_src.width)
    transform = lst_src.transform

    # Clip admin polygons to Dhaka boundary
    dhaka = gpd.read_file(BOUNDARY)
    admin_clipped = gpd.overlay(admin_gdf, dhaka, how="intersection")
    print(f"  Admin units within Dhaka boundary: {len(admin_clipped)}")

    rows = []
    for idx, row in admin_clipped.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Rasterise the polygon to a binary mask
        mask_raster = rasterize(
            shapes    = [(geom.__geo_interface__, 1)],
            out_shape = out_shape,
            transform = transform,
            fill      = 0,
            dtype     = np.uint8
        )

        lst_in_zone = lst_array[mask_raster == 1]
        lst_in_zone = lst_in_zone[~np.isnan(lst_in_zone)]

        if lst_in_zone.size == 0:
            continue

        rows.append({
            "admin_name"  : row.get("NAME_4", row.get("NAME_3", f"Unit_{idx}")),
            "district"    : row.get("NAME_3", ""),
            "division"    : row.get("NAME_2", ""),
            "pixel_count" : int(lst_in_zone.size),
            "mean_lst"    : round(float(np.mean(lst_in_zone)), 2),
            "min_lst"     : round(float(np.min(lst_in_zone)),  2),
            "max_lst"     : round(float(np.max(lst_in_zone)),  2),
            "std_lst"     : round(float(np.std(lst_in_zone)),  2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_lst", ascending=False).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("Step 5 - Zonal Statistics: LST by LULC and Admin Unit")
    print("=" * 60)

    os.makedirs(OUTPUT, exist_ok=True)

    # Part A: LST by LULC class
    print("\nA) LST statistics by LULC class...")

    lst_data, lulc_data, _ = load_and_align_rasters(LST_PATH, LULC_PATH)

    lulc_df = zonal_stats_by_lulc(lst_data, lulc_data)

    print(f"\n  {'Class':<15} {'Pixels':>10} {'Mean LST':>10} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("  " + "-" * 60)
    for _, r in lulc_df.iterrows():
        print(f"  {r['class_label']:<15} {r['pixel_count']:>10,} "
              f"{r['mean_lst']:>9.2f}C  {r['min_lst']:>7.2f}C  "
              f"{r['max_lst']:>7.2f}C  {r['std_lst']:>7.2f}C")

    lulc_df.to_csv(LULC_CSV, index=False)
    print(f"\n  Saved: {os.path.relpath(LULC_CSV, ROOT)}")

    # Part B: LST by admin unit
    print("\nB) LST statistics by GADM Level-4 admin unit...")

    if not os.path.exists(ADMIN_PATH):
        print(f"  [SKIP] Admin shapefile not found: {os.path.relpath(ADMIN_PATH, ROOT)}")
    else:
        admin_gdf = gpd.read_file(ADMIN_PATH).to_crs("EPSG:32645")
        print(f"  Admin features loaded: {len(admin_gdf)}")

        with rasterio.open(LST_PATH) as lst_src:
            admin_df = zonal_stats_by_admin(lst_src, admin_gdf)

        print(f"\n  Top 5 hottest admin units:")
        print(admin_df[["admin_name", "district", "mean_lst"]].head(5).to_string(index=False))
        print(f"\n  Top 5 coolest admin units:")
        print(admin_df[["admin_name", "district", "mean_lst"]].tail(5).to_string(index=False))

        admin_df.to_csv(ADMIN_CSV, index=False)
        print(f"\n  Saved: {os.path.relpath(ADMIN_CSV, ROOT)}")

    print("\n" + "=" * 60)
    print("Zonal statistics complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
