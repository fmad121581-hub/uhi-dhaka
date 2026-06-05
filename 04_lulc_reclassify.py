"""
04_lulc_reclassify.py
---------------------
Step 4 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Reads the preprocessed ESA WorldCover 10m LULC raster (clipped + UTM)
  - Reclassifies the original 11 ESA classes into 4 simplified classes:
      1 → Built-up
      2 → Vegetation
      3 → Water
      4 → Bare soil
  - Saves the reclassified raster to data/processed/lulc_4class.tif
  - Prints the pixel count and area (km²) for each class

ESA WorldCover 2021 class codes (v2.0.0):
  10 — Tree cover
  20 — Shrubland
  30 — Grassland
  40 — Cropland
  50 — Built-up
  60 — Bare / sparse vegetation
  70 — Snow and ice
  80 — Permanent water bodies
  90 — Herbaceous wetland
  95 — Mangroves
  100 — Moss and lichen

Reclassification logic for Dhaka (tropical monsoon urban context):
  Built-up   → ESA 50
  Vegetation → ESA 10, 20, 30, 40, 90, 95, 100  (all vegetated types)
  Water      → ESA 80
  Bare soil  → ESA 60, 70  (bare ground; snow/ice absent in Bangladesh)
"""

import os
import numpy as np          # array maths and boolean masking
import rasterio             # read/write GeoTIFF rasters

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")

LULC_IN   = os.path.join(PROCESSED, "lulc_clipped_utm.tif")    # ESA original classes
LULC_OUT  = os.path.join(PROCESSED, "lulc_4class.tif")         # reclassified output

# ── Reclassification mapping ───────────────────────────────────────────────────
# Keys are ESA WorldCover class codes; values are our 4-class output codes.
# Any ESA code not listed here will be mapped to 0 (unclassified/nodata).
RECLASSIFY_MAP = {
    10  : 2,   # Tree cover       → Vegetation
    20  : 2,   # Shrubland        → Vegetation
    30  : 2,   # Grassland        → Vegetation
    40  : 2,   # Cropland         → Vegetation (agricultural land)
    50  : 1,   # Built-up         → Built-up
    60  : 4,   # Bare/sparse veg  → Bare soil
    70  : 4,   # Snow/ice         → Bare soil (not expected in Dhaka but handled)
    80  : 3,   # Water bodies     → Water
    90  : 2,   # Herbaceous wetland → Vegetation
    95  : 2,   # Mangroves        → Vegetation
    100 : 2,   # Moss/lichen      → Vegetation
}

# Human-readable labels for summary reporting
CLASS_LABELS = {
    1: "Built-up",
    2: "Vegetation",
    3: "Water",
    4: "Bare soil",
    0: "Unclassified"
}


# ── Helper functions ──────────────────────────────────────────────────────────

def reclassify(lulc_path, output_path, remap):
    """
    Read the ESA LULC raster, apply the reclassification mapping, and save.

    Parameters
    ----------
    lulc_path   : str  – path to the preprocessed ESA WorldCover GeoTIFF
    output_path : str  – where to save the 4-class reclassified GeoTIFF
    remap       : dict – {original_code: new_code} mapping
    """
    with rasterio.open(lulc_path) as src:
        # Read the LULC pixel values (uint8: ESA class codes 10–100)
        lulc = src.read(1).astype(np.uint8)
        profile = src.profile   # copy metadata to reuse for output
        nodata  = src.nodata

        # Get the pixel resolution in metres (both dimensions should be ~10m)
        pixel_area_m2 = abs(src.res[0] * src.res[1])   # res is (x_size, y_size)
        pixel_area_km2 = pixel_area_m2 / 1e6            # convert m² to km²

    # ── Reclassify ────────────────────────────────────────────────────────────
    # Start with an output array filled with 0 (unclassified)
    reclassified = np.zeros_like(lulc, dtype=np.uint8)

    # Apply each mapping: set all pixels with code `esa_code` to `new_code`
    for esa_code, new_code in remap.items():
        # Boolean mask: True where pixels equal the current ESA code
        reclassified[lulc == esa_code] = new_code

    # Preserve nodata pixels as 0 (already 0, but explicit for clarity)
    if nodata is not None:
        reclassified[lulc == int(nodata)] = 0

    # ── Print class statistics ─────────────────────────────────────────────────
    print(f"\n  Pixel resolution: {abs(src.res[0]):.1f} m × {abs(src.res[1]):.1f} m")
    print(f"  Pixel area: {pixel_area_m2:.1f} m²  ({pixel_area_km2:.6f} km²)")
    print("\n  Class summary:")
    print(f"  {'Class':<5} {'Label':<15} {'Pixels':>10} {'Area (km²)':>12}")
    print("  " + "-" * 45)

    for class_code, label in CLASS_LABELS.items():
        if class_code == 0:
            continue   # skip unclassified in the summary
        # Count pixels belonging to this class
        pixel_count = int(np.sum(reclassified == class_code))
        area_km2    = pixel_count * pixel_area_km2
        print(f"  {class_code:<5} {label:<15} {pixel_count:>10,} {area_km2:>12.2f}")

    # ── Write reclassified raster to disk ──────────────────────────────────────
    out_profile = profile.copy()
    out_profile.update({
        "dtype"  : "uint8",    # 4 classes fit in uint8 (0–255)
        "nodata" : 0,          # 0 = unclassified / nodata
        "count"  : 1
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(reclassified, 1)

    print(f"\n  Reclassified LULC saved: {os.path.relpath(output_path, ROOT)}")
    return reclassified, pixel_area_km2


# ── Main routine ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 4 — LULC Reclassification (ESA → 4 classes)")
    print("=" * 60)

    # Verify the preprocessed LULC raster exists
    if not os.path.exists(LULC_IN):
        raise FileNotFoundError(
            f"Preprocessed LULC not found: {os.path.relpath(LULC_IN, ROOT)}\n"
            "Run 01_preprocess.py first."
        )

    print(f"\nInput : {os.path.relpath(LULC_IN, ROOT)}")
    print(f"Output: {os.path.relpath(LULC_OUT, ROOT)}")

    reclassify(LULC_IN, LULC_OUT, RECLASSIFY_MAP)

    print("\n" + "=" * 60)
    print("LULC reclassification complete.")
    print("Class codes: 1=Built-up  2=Vegetation  3=Water  4=Bare soil")
    print("Output: data/processed/lulc_4class.tif")
    print("=" * 60)


if __name__ == "__main__":
    main()
