"""
02_lst_derivation.py
--------------------
Step 2: Land Surface Temperature from Landsat 9 Collection 2 Level-2 ST_B10.

Why there is no extra emissivity step any more:
  ST_B10 is already a surface temperature product (Kelvin). USGS derives it
  with ASTER-GED emissivity and an atmospheric correction. Applying an
  NDVI-based emissivity correction on top of it (as v1 did) corrects the
  same effect twice.

  LST[K] = 0.00341802 * DN + 149.0   (scale factors read from the MTL file)

Masking:
  - DN == 0 (fill) and the raster nodata value are removed. In v1 these became
    149 K = -124 C and dominated the min, mean and standard deviation.
  - Values outside 0-72 C are treated as retrieval artefacts.
  - QA_PIXEL cloud/shadow mask is applied if the file is present.
"""
import os
import numpy as np
import rasterio
from uhi_common import read_mtl, qa_clear_mask, LST_MIN_K, LST_MAX_K

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed_v2")
SCENE_DIR = os.path.join(ROOT, "data", "raw", "landsat", "scene_01_dhaka")
MTL_FILE  = os.path.join(SCENE_DIR, "LC09_L2SP_137044_20220224_20230426_02_T1_MTL.txt")
BAND10_IN = os.path.join(PROCESSED, "band10_clipped_utm.tif")
QA_IN     = os.path.join(PROCESSED, "qa_pixel_clipped_utm.tif")
LST_OUT   = os.path.join(PROCESSED, "lst_celsius.tif")


def derive_lst():
    mtl = read_mtl(MTL_FILE)
    ml = float(mtl["TEMPERATURE_MULT_BAND_ST_B10"])
    al = float(mtl["TEMPERATURE_ADD_BAND_ST_B10"])
    print(f"  Scale factors: ML={ml}, AL={al}")

    with rasterio.open(BAND10_IN) as src:
        dn = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile

    n_total = dn.size
    invalid = (dn == 0)
    if nodata is not None:
        invalid |= (dn == nodata)
    print(f"  Fill/nodata pixels removed: {invalid.sum():,} of {n_total:,}")

    kelvin = ml * dn + al
    kelvin[invalid] = np.nan

    bad_range = (kelvin < LST_MIN_K) | (kelvin > LST_MAX_K)
    print(f"  Out-of-range pixels removed: {np.nansum(bad_range):,}")
    kelvin[bad_range] = np.nan

    clear = qa_clear_mask(QA_IN, kelvin.shape)
    if clear is None:
        print("  [WARN] QA_PIXEL not found - no cloud/shadow mask applied")
    else:
        print(f"  QA mask removes {(~clear & ~np.isnan(kelvin)).sum():,} pixels")
        kelvin[~clear] = np.nan

    lst_c = (kelvin - 273.15).astype(np.float32)
    v = lst_c[~np.isnan(lst_c)]
    print(f"  LST min/mean/max: {v.min():.2f} / {v.mean():.2f} / {v.max():.2f} C  (std {v.std():.2f})")

    profile.update(dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(LST_OUT, "w", **profile) as dst:
        dst.write(lst_c, 1)
    print(f"  Saved: {os.path.relpath(LST_OUT, ROOT)}")


def main():
    print("=" * 60)
    print("Step 2 - LST from Landsat 9 ST_B10 (Collection 2 Level-2)")
    print("=" * 60)
    for label, path in [("Band 10", BAND10_IN), ("MTL", MTL_FILE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}\nRun 01_preprocess.py first.")
    derive_lst()


if __name__ == "__main__":
    main()
