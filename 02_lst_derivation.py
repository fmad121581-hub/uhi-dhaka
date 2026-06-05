"""
02_lst_derivation.py
--------------------
Step 2 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Reads the preprocessed (clipped + UTM) Landsat Band 10 thermal raster
  - Reads the MTL.txt metadata file to get the official scale factors
  - Converts raw digital numbers (DN) → Top-of-Atmosphere radiance (W/m²/sr/µm)
  - Converts TOA radiance → Brightness Temperature in Kelvin
  - Applies an NDVI-based emissivity correction (NDVI Threshold Method)
  - Outputs Land Surface Temperature (LST) in Celsius to data/processed/

Physical background (brief):
  Landsat Collection 2 Level-2 Band 10 stores thermal data as scaled integers.
  The MTL file provides:
    - TEMPERATURE_MULT_BAND_ST_B10 (multiplicative rescale factor, ML)
    - TEMPERATURE_ADD_BAND_ST_B10  (additive rescale factor, AL)
  LST (Kelvin) = ML × DN + AL
  Then convert: LST (Celsius) = LST (Kelvin) − 273.15

  A further emissivity correction refines this using NDVI:
    emissivity (ε) is estimated from NDVI via the NDVI Threshold Method
    LST_corrected = LST / (1 + (λ × LST / ρ) × ln(ε))
    where λ = 10.895 µm (Band 10 centre wavelength), ρ = h·c/σ = 1.438×10⁻² m·K
"""

import os
import re                        # regular expressions for parsing the MTL text file
import numpy as np               # array maths on raster data
import rasterio                  # reading/writing GeoTIFF rasters
from rasterio.transform import from_bounds   # not used here but useful for reference

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
MTL_FILE  = os.path.join(ROOT, "data", "raw", "landsat", "scene_01_dhaka",
                          "LC09_L2SP_137044_20220224_20230426_02_T1_MTL.txt")

# Preprocessed Band 10 (thermal) from Step 1
BAND10_IN = os.path.join(PROCESSED, "band10_clipped_utm.tif")
# Preprocessed NDVI bands from Step 1 (needed for emissivity)
BAND4_IN  = os.path.join(PROCESSED, "band4_clipped_utm.tif")
BAND5_IN  = os.path.join(PROCESSED, "band5_clipped_utm.tif")
# Final LST output
LST_OUT   = os.path.join(PROCESSED, "lst_celsius.tif")

# ── Physical constants for emissivity correction ───────────────────────────────
LAMBDA = 10.895e-6       # Band 10 centre wavelength in metres (10.895 µm)
RHO    = 1.438e-2        # h·c/σ (Planck × speed of light / Boltzmann) in metre·Kelvin


# ── Helper functions ──────────────────────────────────────────────────────────

def parse_mtl(mtl_path):
    """
    Parse the Landsat MTL.txt file and return a dict of all key-value pairs.
    We specifically need the Band 10 thermal rescale factors.
    """
    params = {}
    # Read the plain-text MTL file line by line
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            # Each useful line looks like:  KEY = VALUE
            # re.match captures everything before and after ' = '
            match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if match:
                key   = match.group(1).strip()
                value = match.group(2).strip().strip('"')   # remove any quotes
                params[key] = value
    return params


def get_thermal_scale_factors(mtl_path):
    """
    Extract the multiplicative (ML) and additive (AL) scale factors
    for Band ST (Surface Temperature) from the MTL file.
    These factors convert DN → Kelvin directly for Landsat Collection 2 Level-2.
    """
    params = parse_mtl(mtl_path)

    # Landsat Collection 2 Level-2 key names for ST scale factors
    # Try the standard key first; fall back to alternate naming in some MTL versions
    ml_key = "TEMPERATURE_MULT_BAND_ST_B10"
    al_key = "TEMPERATURE_ADD_BAND_ST_B10"

    if ml_key not in params:
        # Alternate key names used in some Collection 2 MTL versions
        ml_key = "ST_SCALE"
        al_key = "ST_OFFSET"

    if ml_key not in params or al_key not in params:
        # Print all keys so we can diagnose which key name to use
        temp_keys = [k for k in params if "TEMP" in k or "ST_" in k]
        raise KeyError(
            f"Could not find thermal scale factors in {mtl_path}.\n"
            f"Temperature-related keys found: {temp_keys}"
        )

    ml = float(params[ml_key])   # multiplicative rescale factor
    al = float(params[al_key])   # additive rescale factor
    print(f"  ML (mult) = {ml},  AL (add) = {al}")
    return ml, al


def compute_ndvi(band4_path, band5_path):
    """
    Compute NDVI from Band 4 (Red) and Band 5 (NIR).
    NDVI = (NIR - Red) / (NIR + Red)
    Returns: ndvi array, rasterio profile (metadata) of Band 5
    """
    with rasterio.open(band4_path) as b4_src:
        # Read the pixel values as float32 to avoid integer division later
        red  = b4_src.read(1).astype(np.float32)
        nodata_b4 = b4_src.nodata

    with rasterio.open(band5_path) as b5_src:
        nir  = b5_src.read(1).astype(np.float32)
        nodata_b5 = b5_src.nodata
        profile = b5_src.profile  # reuse this for output metadata

    # Mask out nodata pixels so they don't skew calculations
    if nodata_b4 is not None:
        red[red == nodata_b4] = np.nan
    if nodata_b5 is not None:
        nir[nir == nodata_b5] = np.nan

    # Suppress divide-by-zero warnings; result is nan where denominator = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(
            (nir + red) == 0,    # avoid division by zero
            np.nan,
            (nir - red) / (nir + red)
        )

    # Clamp NDVI to valid range [-1, 1] to remove any sensor artefacts
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi, profile


def estimate_emissivity(ndvi):
    """
    Estimate land surface emissivity (ε) using the NDVI Threshold Method
    (Sobrino et al. 2004 — widely used in urban heat island studies).

    Rules:
      - NDVI < 0.2  → bare soil/built-up: ε = 0.97
      - NDVI > 0.5  → dense vegetation:   ε = 0.99
      - 0.2 ≤ NDVI ≤ 0.5 → mixed pixels: ε estimated from fractional vegetation cover (Pv)
        Pv = ((NDVI - 0.2) / (0.5 - 0.2)) ²
        ε  = 0.004 × Pv + 0.986
    """
    # Fractional vegetation cover for mixed pixels
    pv = ((ndvi - 0.2) / (0.5 - 0.2)) ** 2
    pv = np.clip(pv, 0, 1)   # ensure Pv stays in [0, 1]

    # Start with the mixed-pixel emissivity everywhere
    emissivity = 0.004 * pv + 0.986

    # Override with bare-soil/built-up value where NDVI is very low
    emissivity = np.where(ndvi < 0.2, 0.97, emissivity)

    # Override with full-vegetation value where NDVI is high
    emissivity = np.where(ndvi > 0.5, 0.99, emissivity)

    return emissivity


def derive_lst(band10_path, mtl_path, ndvi, output_path):
    """
    Full LST derivation pipeline:
      DN → Kelvin (using MTL scale factors) → emissivity-corrected LST → Celsius

    Parameters
    ----------
    band10_path : str       – preprocessed thermal band raster
    mtl_path    : str       – Landsat MTL metadata text file
    ndvi        : np.ndarray – NDVI array aligned to Band 10 extent
    output_path : str       – where to write lst_celsius.tif
    """
    # Get the scale factors from the metadata file
    ml, al = get_thermal_scale_factors(mtl_path)

    with rasterio.open(band10_path) as src:
        # Read raw Digital Numbers (DN) as float to allow floating-point maths
        dn = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile   # we'll reuse this metadata for the output

        # Mask nodata pixels (usually 0 or a large fill value)
        if nodata is not None:
            dn[dn == nodata] = np.nan

        # ── Step A: DN → Brightness Temperature (Kelvin) ───────────────────
        # Formula from the Landsat Collection 2 Level-2 Science Product Guide:
        # ST (Kelvin) = ML × DN + AL
        bt_kelvin = ml * dn + al
        print(f"  BT range: {np.nanmin(bt_kelvin):.1f} K – {np.nanmax(bt_kelvin):.1f} K")

        # ── Step B: Emissivity correction ──────────────────────────────────
        # Resize ndvi to match Band 10 dimensions if they differ due to resampling
        if ndvi.shape != bt_kelvin.shape:
            # Use simple nearest-neighbour resize via numpy (avoids scipy dependency)
            zoom_y = bt_kelvin.shape[0] / ndvi.shape[0]
            zoom_x = bt_kelvin.shape[1] / ndvi.shape[1]
            # Build row/col index arrays that map Band 10 pixels back to NDVI pixels
            row_idx = (np.arange(bt_kelvin.shape[0]) / zoom_y).astype(int)
            col_idx = (np.arange(bt_kelvin.shape[1]) / zoom_x).astype(int)
            # Clamp indices to avoid out-of-bounds on edge pixels
            row_idx = np.clip(row_idx, 0, ndvi.shape[0] - 1)
            col_idx = np.clip(col_idx, 0, ndvi.shape[1] - 1)
            # Fancy indexing to create the resized array
            ndvi_resized = ndvi[np.ix_(row_idx, col_idx)]
        else:
            ndvi_resized = ndvi

        # Estimate emissivity from the aligned NDVI
        emissivity = estimate_emissivity(ndvi_resized)

        # ── Step C: Emissivity-corrected LST (still Kelvin) ────────────────
        # Formula (Weng et al. 2004 / Avdan & Jovanovska 2016):
        # LST = BT / (1 + (λ × BT / ρ) × ln(ε))
        lst_kelvin = bt_kelvin / (
            1 + (LAMBDA * bt_kelvin / RHO) * np.log(emissivity)
        )

        # ── Step D: Convert Kelvin → Celsius ───────────────────────────────
        lst_celsius = lst_kelvin - 273.15
        print(f"  LST range: {np.nanmin(lst_celsius):.1f} °C – {np.nanmax(lst_celsius):.1f} °C")

        # ── Step E: Write output GeoTIFF ────────────────────────────────────
        out_profile = profile.copy()
        out_profile.update({
            "dtype"  : "float32",   # float needed for decimal Celsius values
            "nodata" : np.nan,
            "count"  : 1            # single-band output
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(lst_celsius.astype(np.float32), 1)

    print(f"  LST saved: {os.path.relpath(output_path, ROOT)}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 2 — LST Derivation from Landsat Band 10")
    print("=" * 60)

    # Check required inputs exist before proceeding
    for label, path in [("Band 10", BAND10_IN),
                         ("Band 4",  BAND4_IN),
                         ("Band 5",  BAND5_IN),
                         ("MTL",     MTL_FILE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found: {os.path.relpath(path, ROOT)}\n"
                "Run 01_preprocess.py first and check file names match."
            )

    # Step A: compute NDVI (needed for emissivity)
    print("\nComputing NDVI for emissivity estimation...")
    ndvi, _ = compute_ndvi(BAND4_IN, BAND5_IN)
    print(f"  NDVI range: {np.nanmin(ndvi):.3f} – {np.nanmax(ndvi):.3f}")

    # Step B: derive LST with emissivity correction
    print("\nDeriving Land Surface Temperature...")
    derive_lst(BAND10_IN, MTL_FILE, ndvi, LST_OUT)

    print("\n" + "=" * 60)
    print("LST derivation complete. Output: data/processed/lst_celsius.tif")
    print("=" * 60)


if __name__ == "__main__":
    main()
