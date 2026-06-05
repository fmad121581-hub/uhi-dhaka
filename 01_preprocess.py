"""
01_preprocess.py
----------------
Step 1 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Reads the Dhaka boundary shapefile (WGS84)
  - Reprojects it to UTM Zone 45N (EPSG:32645) for metric-unit analysis
  - Clips every raw raster to the Dhaka boundary extent
  - Reprojects every clipped raster to UTM 45N
  - Saves results to data/processed/

Run this before any other script.
"""

import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# Path constants - all relative to project root (uhi_dhaka/)
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY  = os.path.join(ROOT, "dhaka_boundary.shp")
RAW_DIR   = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")

# Coordinate reference systems
WGS84  = "EPSG:4326"   # source CRS of all input data
UTM_45N = "EPSG:32645" # target CRS for all analysis (metres)

# Raw raster catalogue - maps short names to file paths
RASTERS = {
    "lulc"       : os.path.join(RAW_DIR, "lulc",
                       "ESA_WorldCover_10m_2021_v200_N21E090_Map.tif"),
    "dem"        : os.path.join(RAW_DIR, "dem",
                       "srtm_dhaka_30m.tif"),
    "population" : os.path.join(RAW_DIR, "population",
                       "BGD_ppp_2020_adj_v2.tif"),
    "band4"      : os.path.join(RAW_DIR, "landsat", "scene_01_dhaka",
                       "LC09_L2SP_137044_20220224_20230426_02_T1_SR_B4.TIF"),
    "band5"      : os.path.join(RAW_DIR, "landsat", "scene_01_dhaka",
                       "LC09_L2SP_137044_20220224_20230426_02_T1_SR_B5.TIF"),
    "band10"     : os.path.join(RAW_DIR, "landsat", "scene_01_dhaka",
                       "LC09_L2SP_137044_20220224_20230426_02_T1_ST_B10.TIF"),
}


def load_boundary_utm():
    """Load the Dhaka boundary shapefile and reproject to UTM 45N."""
    boundary = gpd.read_file(BOUNDARY)
    boundary_utm = boundary.to_crs(UTM_45N)
    print(f"  Boundary loaded: {len(boundary_utm)} feature(s), CRS = {boundary_utm.crs}")
    return boundary_utm


def clip_and_reproject(src_path, dst_path, boundary_wgs84, dst_crs=UTM_45N):
    """
    Clip a raster to the Dhaka boundary, then reproject to dst_crs.
    The boundary is reprojected to match the source raster CRS before clipping,
    which handles Landsat bands that are stored in UTM rather than WGS84.
    """
    with rasterio.open(src_path) as src:

        # Reproject boundary to match the source raster CRS exactly.
        # This is the key fix: Landsat bands are in a UTM projection,
        # so we cannot clip them using WGS84 coordinates directly.
        boundary_matched = boundary_wgs84.to_crs(src.crs)
        geoms = [geom.__geo_interface__ for geom in boundary_matched.geometry]

        # mask() clips the raster to the boundary polygon
        nodata_val = src.nodata if src.nodata is not None else -9999
        clipped_data, clipped_transform = mask(
            src, geoms, crop=True, nodata=nodata_val
        )

        # Update metadata to reflect new clipped dimensions
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height"    : clipped_data.shape[1],
            "width"     : clipped_data.shape[2],
            "transform" : clipped_transform,
            "nodata"    : nodata_val
        })

        # calculate_default_transform finds the best transform for reprojection
        new_transform, new_width, new_height = calculate_default_transform(
            src.crs,
            dst_crs,
            clipped_meta["width"],
            clipped_meta["height"],
            *rasterio.transform.array_bounds(
                clipped_meta["height"],
                clipped_meta["width"],
                clipped_transform
            )
        )

        # Empty array to receive reprojected pixels
        reprojected = np.empty(
            (clipped_data.shape[0], new_height, new_width),
            dtype=clipped_data.dtype
        )

        # Reproject band by band
        for band_idx in range(clipped_data.shape[0]):
            reproject(
                source         = clipped_data[band_idx],
                destination    = reprojected[band_idx],
                src_transform  = clipped_transform,
                src_crs        = src.crs,
                dst_transform  = new_transform,
                dst_crs        = dst_crs,
                resampling     = Resampling.bilinear
            )

        # Write the final reprojected raster
        final_meta = clipped_meta.copy()
        final_meta.update({
            "crs"       : dst_crs,
            "transform" : new_transform,
            "width"     : new_width,
            "height"    : new_height,
            "driver"    : "GTiff"
        })

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with rasterio.open(dst_path, "w", **final_meta) as dst:
            dst.write(reprojected)

    print(f"  Saved: {os.path.relpath(dst_path, ROOT)}")


def main():
    print("=" * 60)
    print("Step 1 - Preprocessing: clip + reproject all rasters")
    print("=" * 60)

    # Load boundary in WGS84 for clipping (reprojected per-raster inside the function)
    boundary_wgs84 = gpd.read_file(BOUNDARY)
    print(f"  Boundary CRS: {boundary_wgs84.crs}")
    print(f"  Boundary bounds: {boundary_wgs84.total_bounds}")

    # Also save the UTM boundary shapefile for downstream scripts
    boundary_utm = boundary_wgs84.to_crs(UTM_45N)
    boundary_out = os.path.join(PROCESSED, "dhaka_boundary_utm.shp")
    os.makedirs(PROCESSED, exist_ok=True)
    boundary_utm.to_file(boundary_out)
    print(f"  Boundary (UTM) saved: data/processed/dhaka_boundary_utm.shp")

    # Process each raster
    for name, src_path in RASTERS.items():
        if not os.path.exists(src_path):
            print(f"  [SKIP] {name}: file not found at {os.path.relpath(src_path, ROOT)}")
            continue

        print(f"\nProcessing: {name}")

        # Print source raster CRS for debugging
        with rasterio.open(src_path) as src:
            print(f"  Source CRS: {src.crs}")
            print(f"  Source bounds: {src.bounds}")

        dst_path = os.path.join(PROCESSED, f"{name}_clipped_utm.tif")
        clip_and_reproject(
            src_path       = src_path,
            dst_path       = dst_path,
            boundary_wgs84 = boundary_wgs84,
            dst_crs        = UTM_45N
        )

    print("\n" + "=" * 60)
    print("Preprocessing complete. Outputs in data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
