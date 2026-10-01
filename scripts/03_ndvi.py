"""
03_ndvi.py
----------
Step 3 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Reads preprocessed (clipped + UTM) Landsat Band 4 (Red) and Band 5 (NIR)
  - Computes NDVI: (NIR - Red) / (NIR + Red)
  - Saves the NDVI raster to data/processed/ndvi.tif

NDVI (Normalised Difference Vegetation Index) background:
  Values range from -1 to +1:
    < 0.0  — water, bare rock, built structures (little to no vegetation)
    0.0–0.2 — bare soil, urban surfaces
    0.2–0.5 — sparse to moderate vegetation
    > 0.5  — dense, healthy vegetation

  In UHI studies NDVI is a key predictor: higher vegetation cover correlates
  with lower surface temperatures (urban cooling effect).

Note: Band 4 and Band 5 are Collection 2 Level-2 Surface Reflectance (uint16 DN).
Reflectance = 2.75e-05 * DN - 0.2. The -0.2 offset does NOT cancel in the
NDVI ratio, so DNs must be converted to reflectance first (v1 skipped this
and biased NDVI). Fill pixels (DN 0) and out-of-range DNs are masked, and the
QA_PIXEL cloud mask is applied if available.
"""

import os
import numpy as np          # array arithmetic on raster data
import rasterio             # read/write GeoTIFF rasters
from uhi_common import read_mtl, qa_clear_mask, SR_DN_MIN, SR_DN_MAX

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed_v2")

BAND4_IN  = os.path.join(PROCESSED, "band4_clipped_utm.tif")   # Red band
BAND5_IN  = os.path.join(PROCESSED, "band5_clipped_utm.tif")   # NIR band
NDVI_OUT  = os.path.join(PROCESSED, "ndvi.tif")                # output NDVI
QA_IN     = os.path.join(PROCESSED, "qa_pixel_clipped_utm.tif")
MTL_FILE  = os.path.join(ROOT, "data", "raw", "landsat", "scene_01_dhaka",
                         "LC09_L2SP_137044_20220224_20230426_02_T1_MTL.txt")


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_ndvi(band4_path, band5_path, output_path):
    """
    Calculate NDVI from Band 4 (Red) and Band 5 (NIR) and write to disk.

    Parameters
    ----------
    band4_path  : str – path to clipped/UTM Band 4 GeoTIFF
    band5_path  : str – path to clipped/UTM Band 5 GeoTIFF
    output_path : str – where to save ndvi.tif
    """
    # Open Band 4 (Red) and read pixel values
    with rasterio.open(band4_path) as b4_src:
        # Cast to float32 immediately so all arithmetic stays in floating-point
        red  = b4_src.read(1).astype(np.float32)
        nodata_b4 = b4_src.nodata    # nodata fill value (often 0 or 65535)
        profile = b4_src.profile     # save metadata for output

    # Open Band 5 (NIR) and read pixel values
    with rasterio.open(band5_path) as b5_src:
        nir  = b5_src.read(1).astype(np.float32)
        nodata_b5 = b5_src.nodata

    mtl = read_mtl(MTL_FILE)
    mult = float(mtl["REFLECTANCE_MULT_BAND_4"])
    add  = float(mtl["REFLECTANCE_ADD_BAND_4"])
    assert mult == float(mtl["REFLECTANCE_MULT_BAND_5"]) and add == float(mtl["REFLECTANCE_ADD_BAND_5"])
    print(f"  SR scale: reflectance = {mult} * DN + ({add})")

    # Mask fill, nodata and out-of-range DNs
    invalid = ~((red >= SR_DN_MIN) & (red <= SR_DN_MAX) & (nir >= SR_DN_MIN) & (nir <= SR_DN_MAX))
    clear = qa_clear_mask(QA_IN, red.shape)
    if clear is None:
        print("  [WARN] QA_PIXEL not found - no cloud/shadow mask applied")
    else:
        invalid |= ~clear
    print(f"  Masked pixels: {invalid.sum():,} of {invalid.size:,}")

    red = red * mult + add
    nir = nir * mult + add
    red[invalid] = np.nan
    nir[invalid] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))

    # Clamp to theoretical NDVI bounds to remove any sensor noise artefacts
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # ── Print summary statistics ───────────────────────────────────────────────
    valid_pixels = ndvi[~np.isnan(ndvi)]   # exclude NaN before computing stats
    print(f"  NDVI min  : {valid_pixels.min():.4f}")
    print(f"  NDVI max  : {valid_pixels.max():.4f}")
    print(f"  NDVI mean : {valid_pixels.mean():.4f}")
    print(f"  NDVI std  : {valid_pixels.std():.4f}")

    # Fraction of study area covered by vegetation (NDVI > 0.2 threshold)
    veg_fraction = np.sum(valid_pixels > 0.2) / len(valid_pixels) * 100
    print(f"  Vegetation cover (NDVI > 0.2): {veg_fraction:.1f}% of study area")

    # ── Write NDVI raster to disk ──────────────────────────────────────────────
    out_profile = profile.copy()
    out_profile.update({
        "dtype"  : "float32",    # NDVI values are decimals, need float
        "nodata" : np.nan,       # use NaN as nodata in the output
        "count"  : 1             # single-band raster
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)   # write NDVI to band 1

    print(f"  NDVI saved: {os.path.relpath(output_path, ROOT)}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 3 — NDVI Derivation from Landsat Bands 4 & 5")
    print("=" * 60)

    # Check that the required preprocessed bands exist
    for label, path in [("Band 4 (Red)", BAND4_IN),
                         ("Band 5 (NIR)", BAND5_IN)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found: {os.path.relpath(path, ROOT)}\n"
                "Run 01_preprocess.py first."
            )

    print("\nCalculating NDVI...")
    compute_ndvi(BAND4_IN, BAND5_IN, NDVI_OUT)

    print("\n" + "=" * 60)
    print("NDVI derivation complete. Output: data/processed/ndvi.tif")
    print("=" * 60)


if __name__ == "__main__":
    main()
