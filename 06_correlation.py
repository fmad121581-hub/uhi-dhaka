"""
06_correlation.py
-----------------
Step 6 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Samples LST, NDVI, population density, and built-up percentage
    at a regular grid of points across the study area
  - Computes Pearson correlation coefficients between LST and each predictor
  - Runs Ordinary Least Squares (OLS) regression for each pair
  - Saves a summary table to data/output/correlation_summary.csv
  - Saves scatter plots with regression lines to data/output/

Why correlation analysis:
  This answers the core research questions:
    1. Does more vegetation (higher NDVI) mean cooler surfaces?
    2. Do denser populations experience hotter surfaces?
    3. Does higher proportion of built-up land drive higher LST?
  These relationships are the statistical backbone of UHI research.

Sampling strategy:
  We cannot directly correlate rasters of different resolutions
  (10m LULC vs 30m Landsat vs 100m WorldPop) pixel-by-pixel.
  Instead we sample all rasters at the same set of grid points,
  which aligns the values correctly regardless of resolution.
"""

import os
import numpy as np                          # array maths
import pandas as pd                         # tabular data
import matplotlib.pyplot as plt             # plotting
import matplotlib.gridspec as gridspec      # flexible subplot layout
import geopandas as gpd                     # reading boundary shapefile
import rasterio                             # reading rasters
from rasterio.sample import sample_gen      # samples raster at (x, y) coordinates
from sklearn.linear_model import LinearRegression   # OLS regression
from sklearn.metrics import r2_score                 # R² goodness-of-fit metric

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT    = os.path.join(ROOT, "data", "output")

LST_PATH   = os.path.join(PROCESSED, "lst_celsius.tif")
NDVI_PATH  = os.path.join(PROCESSED, "ndvi.tif")
POP_PATH   = os.path.join(PROCESSED, "population_clipped_utm.tif")
LULC_PATH  = os.path.join(PROCESSED, "lulc_4class.tif")
BOUNDARY   = os.path.join(PROCESSED, "dhaka_boundary_utm.shp")

CORR_CSV   = os.path.join(OUTPUT, "correlation_summary.csv")
SCATTER_OUT = os.path.join(OUTPUT, "correlation_scatter_plots.png")

# ── Sampling parameters ────────────────────────────────────────────────────────
# Sample every N metres across the study area (a coarser grid runs faster)
GRID_SPACING_M = 250   # 250m grid → roughly balanced sample size for Dhaka (~700 km²)


# ── Helper functions ──────────────────────────────────────────────────────────

def build_sample_grid(boundary_shp, spacing_m):
    """
    Create a regular grid of (x, y) sample points within the study boundary.

    Parameters
    ----------
    boundary_shp : str   – path to the UTM boundary shapefile
    spacing_m    : float – grid spacing in metres

    Returns: numpy array of shape (N, 2) with (easting, northing) coordinates
    """
    # Read boundary and get its bounding box in UTM metres
    gdf = gpd.read_file(boundary_shp)
    minx, miny, maxx, maxy = gdf.total_bounds   # (min_easting, min_northing, ...)

    # Build evenly-spaced 1-D arrays along each axis
    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)

    # Create all combinations of x and y → 2-D grid
    xx, yy = np.meshgrid(xs, ys)

    # Flatten to a list of (x, y) coordinate pairs
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    print(f"  Grid: {len(xs)} cols × {len(ys)} rows = {len(coords):,} sample points")
    return coords


def sample_raster(raster_path, coords):
    """
    Sample a raster at a list of (x, y) coordinates.
    Returns a 1-D numpy array of values at those points.
    Nodata → NaN.
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        # sample_gen() is a rasterio generator that reads pixel values at given coords
        # It yields one tuple per point; we take index [0] for band 1
        values = np.array([val[0] for val in sample_gen(src, coords)],
                          dtype=np.float32)
    # Replace nodata fill values with NaN
    if nodata is not None:
        values[values == nodata] = np.nan
    return values


def compute_buildup_fraction(lulc_path, coords, radius_m=250):
    """
    For each sample point, estimate the fraction of Built-up (class 1) pixels
    within a radius_m neighbourhood.

    This is done by sampling the LULC raster at the point, then computing
    the local built-up fraction from a window of pixels around each point.

    Simplified approach: for speed we sample the LULC class at the exact
    point and use it as a proxy (1 = built-up, 0 = not built-up).
    A full moving-window approach would require more complex windowing code.
    """
    with rasterio.open(lulc_path) as src:
        nodata = src.nodata
        # Sample LULC class code at each grid point
        lulc_at_points = np.array([val[0] for val in sample_gen(src, coords)],
                                   dtype=np.float32)
    if nodata is not None:
        lulc_at_points[lulc_at_points == nodata] = np.nan

    # Binary: 1.0 where Built-up (class 1), 0.0 everywhere else
    # This acts as a built-up indicator at the point level
    buildup = np.where(lulc_at_points == 1, 1.0, 0.0)
    buildup[np.isnan(lulc_at_points)] = np.nan
    return buildup


def pearson_r(x, y):
    """
    Compute Pearson correlation coefficient between two 1-D arrays,
    ignoring rows where either value is NaN.
    Returns (r, n) where n is the number of valid pairs used.
    """
    # Build a combined mask: True where BOTH x and y are valid (not NaN)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_v, y_v = x[valid], y[valid]
    n = len(x_v)
    if n < 3:
        return np.nan, n   # not enough points for a meaningful correlation
    # np.corrcoef returns a 2×2 correlation matrix; [0, 1] is the off-diagonal r
    r = np.corrcoef(x_v, y_v)[0, 1]
    return round(float(r), 4), n


def ols_regression(x, y):
    """
    Fit a simple OLS (Ordinary Least Squares) linear regression: y = a + b*x
    Returns: (slope, intercept, R²)
    """
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_v, y_v = x[valid].reshape(-1, 1), y[valid]

    model = LinearRegression()
    model.fit(x_v, y_v)          # fit the line

    y_pred = model.predict(x_v)  # predicted y values
    r2 = r2_score(y_v, y_pred)   # coefficient of determination

    return round(float(model.coef_[0]), 4), round(float(model.intercept_), 4), round(r2, 4)


# ── Plotting function ──────────────────────────────────────────────────────────

def make_scatter_plots(df, output_path):
    """
    Create a 1×3 figure with scatter plots for each LST predictor.
    Each subplot shows the sampled data points and the OLS regression line.
    """
    predictors = [
        ("ndvi",     "NDVI",                      "NDVI",                  "tab:green"),
        ("pop_dens", "Population Density (pp/km²)", "Population Density",  "tab:orange"),
        ("buildup",  "Built-up Indicator (0/1)",   "Built-up Fraction",    "tab:red"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("LST Correlation Analysis — Dhaka Urban Heat Island",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (col, xlabel, title, color) in zip(axes, predictors):
        x = df[col].values
        y = df["lst"].values

        # Remove NaN pairs for this subplot
        valid = ~np.isnan(x) & ~np.isnan(y)
        x_v, y_v = x[valid], y[valid]

        # Scatter plot (alpha controls transparency for dense point clouds)
        ax.scatter(x_v, y_v, alpha=0.08, s=5, color=color, label="Sample points")

        # Draw the OLS regression line
        slope, intercept, r2 = ols_regression(x, y)
        x_line = np.linspace(np.nanmin(x_v), np.nanmax(x_v), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="black", linewidth=1.5,
                label=f"OLS: slope={slope:.3f}\nR²={r2:.3f}")

        # Labels and formatting
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("LST (°C)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plots saved: {os.path.relpath(output_path, ROOT)}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 6 — Correlation Analysis: LST vs Predictors")
    print("=" * 60)

    os.makedirs(OUTPUT, exist_ok=True)

    # ── Step A: Build sampling grid ────────────────────────────────────────────
    print(f"\nBuilding {GRID_SPACING_M}m sample grid within Dhaka boundary...")
    if not os.path.exists(BOUNDARY):
        raise FileNotFoundError(
            f"Boundary not found: {os.path.relpath(BOUNDARY, ROOT)}\n"
            "Run 01_preprocess.py first."
        )
    coords = build_sample_grid(BOUNDARY, GRID_SPACING_M)

    # ── Step B: Sample all rasters at grid points ──────────────────────────────
    print("\nSampling rasters at grid points...")

    rasters_needed = {
        "LST"        : LST_PATH,
        "NDVI"       : NDVI_PATH,
        "Population" : POP_PATH,
        "LULC"       : LULC_PATH,
    }
    for label, path in rasters_needed.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} raster not found: {os.path.relpath(path, ROOT)}\n"
                "Run earlier pipeline scripts first."
            )

    lst_vals  = sample_raster(LST_PATH, coords)
    ndvi_vals = sample_raster(NDVI_PATH, coords)
    pop_vals  = sample_raster(POP_PATH, coords)    # WorldPop: persons/km² (100m res)
    bu_vals   = compute_buildup_fraction(LULC_PATH, coords)

    print(f"  LST sampled:   {np.sum(~np.isnan(lst_vals)):,} valid points")
    print(f"  NDVI sampled:  {np.sum(~np.isnan(ndvi_vals)):,} valid points")
    print(f"  Pop sampled:   {np.sum(~np.isnan(pop_vals)):,} valid points")
    print(f"  LULC sampled:  {np.sum(~np.isnan(bu_vals)):,} valid points")

    # ── Step C: Build a DataFrame of aligned samples ──────────────────────────
    df = pd.DataFrame({
        "lst"      : lst_vals,
        "ndvi"     : ndvi_vals,
        "pop_dens" : pop_vals,
        "buildup"  : bu_vals,
    })

    # ── Step D: Compute Pearson r and OLS for each predictor ─────────────────
    print("\nCorrelation results:")
    print(f"  {'Predictor':<25} {'Pearson r':>10} {'Slope':>10} {'Intercept':>10} {'R²':>8} {'n':>8}")
    print("  " + "-" * 75)

    summary_rows = []
    predictor_info = [
        ("ndvi",     "NDVI"),
        ("pop_dens", "Population Density"),
        ("buildup",  "Built-up Indicator"),
    ]

    for col, label in predictor_info:
        r, n       = pearson_r(df[col].values, df["lst"].values)
        slope, intercept, r2 = ols_regression(df[col].values, df["lst"].values)

        print(f"  {label:<25} {r:>10.4f} {slope:>10.4f} {intercept:>10.4f} {r2:>8.4f} {n:>8,}")

        summary_rows.append({
            "predictor"    : label,
            "pearson_r"    : r,
            "ols_slope"    : slope,
            "ols_intercept": intercept,
            "r_squared"    : r2,
            "n_valid"      : n,
        })

    # ── Step E: Save correlation summary CSV ──────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CORR_CSV, index=False)
    print(f"\n  Summary saved: {os.path.relpath(CORR_CSV, ROOT)}")

    # ── Step F: Produce scatter plots ─────────────────────────────────────────
    print("\nGenerating scatter plots...")
    make_scatter_plots(df, SCATTER_OUT)

    print("\n" + "=" * 60)
    print("Correlation analysis complete.")
    print("Outputs:")
    print("  data/output/correlation_summary.csv")
    print("  data/output/correlation_scatter_plots.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
