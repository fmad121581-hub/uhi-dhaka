"""
07_maps.py
----------
Step 7 of the Dhaka Urban Heat Island Analysis pipeline.

What this script does:
  - Produces 4 publication-quality maps saved to data/output/:
      1. lst_map.png          — Land Surface Temperature (UHI intensity)
      2. ndvi_map.png         — NDVI (vegetation cover)
      3. lulc_map.png         — LULC (4-class reclassified)
      4. uhi_summary_panel.png — Combined 2×2 summary panel for portfolio

All maps include:
  - A study area boundary outline
  - North arrow
  - Scale bar
  - Colorbar / legend
  - Title and data source credit

Output standards: PNG, 150 dpi minimum (portfolio requirement).
"""

import os
import numpy as np                      # array operations on raster data
import matplotlib.pyplot as plt         # plotting engine
import matplotlib.patches as mpatches   # for custom legend patches (LULC)
import matplotlib.colors as mcolors     # for discrete colour maps
import matplotlib.ticker as ticker      # for scale bar formatting
import geopandas as gpd                 # reading boundary shapefile
import rasterio                         # reading rasters
from rasterio.plot import plotting_extent  # converts raster metadata to matplotlib extent

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT    = os.path.join(ROOT, "data", "output")

LST_PATH  = os.path.join(PROCESSED, "lst_celsius.tif")
NDVI_PATH = os.path.join(PROCESSED, "ndvi.tif")
LULC_PATH = os.path.join(PROCESSED, "lulc_4class.tif")
BOUNDARY  = os.path.join(PROCESSED, "dhaka_boundary_utm.shp")

DPI = 150   # minimum DPI for portfolio output maps


# ── Helper utilities ──────────────────────────────────────────────────────────

def load_raster_for_plot(path):
    """
    Open a raster and return (array, extent, crs_string).
    extent is the (left, right, bottom, top) tuple matplotlib expects.
    Nodata values become NaN so they render as transparent.
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        # plotting_extent() converts the raster transform + dimensions
        # to the (xmin, xmax, ymin, ymax) format that imshow() needs
        extent = plotting_extent(src)
    return data, extent


def add_north_arrow(ax, x=0.95, y=0.95):
    """
    Add a simple north arrow annotation to the axes at normalised position (x, y).
    Uses ax.annotate with an arrowhead pointing upward.
    """
    ax.annotate(
        "N",
        xy      = (x, y),          # tip of the arrow (axes fraction)
        xytext  = (x, y - 0.07),   # tail of the arrow
        xycoords= "axes fraction",
        textcoords="axes fraction",
        ha      = "center",
        va      = "center",
        fontsize= 11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        color   = "black",
    )


def add_scale_bar(ax, length_km=10):
    """
    Add a simple horizontal scale bar to the lower-left of the axes.
    The bar is drawn in axes-fraction units, so it works for any map extent.

    Parameters
    ----------
    ax        : matplotlib Axes
    length_km : approximate bar length in kilometres
    """
    # Position the bar in the lower-left corner (axes fraction coordinates)
    x_start = 0.05
    y_pos   = 0.05
    bar_len = 0.15   # fraction of axes width (approximate visual length)

    # Draw the bar as a thick black line
    ax.plot([x_start, x_start + bar_len], [y_pos, y_pos],
            transform=ax.transAxes,
            color="black", linewidth=4, solid_capstyle="butt")

    # Label the bar
    ax.text(x_start + bar_len / 2, y_pos + 0.015,
            f"≈ {length_km} km",
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="black")


def overlay_boundary(ax, boundary_shp):
    """
    Overlay the Dhaka boundary as a black outline on the given axes.
    GeoDataFrame.plot() with ax= overlays on an existing axes object.
    """
    if os.path.exists(boundary_shp):
        gdf = gpd.read_file(boundary_shp)
        # facecolor='none' makes it a hollow outline (no fill)
        gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2)


# ── Individual map functions ──────────────────────────────────────────────────

def map_lst(lst_path, boundary, output_path):
    """
    Map 1: Land Surface Temperature in Celsius.
    Hot colours = urban heat island; cool colours = vegetation/water.
    """
    data, extent = load_raster_for_plot(lst_path)

    fig, ax = plt.subplots(figsize=(8, 9))

    # imshow() renders the 2-D array as an image with geographic extent
    im = ax.imshow(
        data,
        extent = extent,
        cmap   = "RdYlBu_r",    # reversed: red=hot, blue=cool (intuitive for temperature)
        vmin   = np.nanpercentile(data, 2),   # clip extreme outliers at 2nd percentile
        vmax   = np.nanpercentile(data, 98),  # clip at 98th percentile
        interpolation="nearest"
    )

    # Overlay boundary, north arrow, scale bar
    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    add_scale_bar(ax, length_km=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Land Surface Temperature (°C)", fontsize=11)

    ax.set_title("Land Surface Temperature\nDhaka Metropolitan Area",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m, UTM 45N)", fontsize=9)
    ax.set_ylabel("Northing (m, UTM 45N)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Credit line at the bottom
    fig.text(0.5, 0.01, "Source: Landsat Collection 2 L2 | Analysis: Fahim Ahmed, BUET",
             ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  LST map saved: {os.path.relpath(output_path, ROOT)}")


def map_ndvi(ndvi_path, boundary, output_path):
    """
    Map 2: NDVI vegetation index.
    Dark green = dense vegetation; brown/grey = built-up/bare.
    """
    data, extent = load_raster_for_plot(ndvi_path)

    fig, ax = plt.subplots(figsize=(8, 9))

    im = ax.imshow(
        data,
        extent = extent,
        cmap   = "RdYlGn",    # red (low NDVI) → yellow → green (high NDVI)
        vmin   = -0.1,        # slightly below 0 to show water clearly
        vmax   = 0.7,         # cap at 0.7; rarely higher in urban Dhaka
        interpolation="nearest"
    )

    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    add_scale_bar(ax, length_km=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("NDVI", fontsize=11)

    ax.set_title("Normalised Difference Vegetation Index (NDVI)\nDhaka Metropolitan Area",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m, UTM 45N)", fontsize=9)
    ax.set_ylabel("Northing (m, UTM 45N)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.text(0.5, 0.01, "Source: Landsat Collection 2 L2 | Analysis: Fahim Ahmed, BUET",
             ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  NDVI map saved: {os.path.relpath(output_path, ROOT)}")


def map_lulc(lulc_path, boundary, output_path):
    """
    Map 3: LULC 4-class reclassified map.
    Uses a discrete colour map with a categorical legend.
    """
    data, extent = load_raster_for_plot(lulc_path)

    # Define colours for each class (must match 04_lulc_reclassify.py codes)
    class_colours = {
        1: "#E74C3C",   # Built-up   → red
        2: "#27AE60",   # Vegetation → green
        3: "#2980B9",   # Water      → blue
        4: "#F39C12",   # Bare soil  → orange
    }
    class_labels = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Bare soil"}

    # Build a ListedColormap so each integer code maps to one colour
    # We need 5 colours for values 0–4 (0 = nodata/transparent)
    colour_list = ["none",                 # 0 → transparent (nodata)
                   class_colours[1],       # 1 → built-up
                   class_colours[2],       # 2 → vegetation
                   class_colours[3],       # 3 → water
                   class_colours[4]]       # 4 → bare soil
    cmap = mcolors.ListedColormap(colour_list)

    fig, ax = plt.subplots(figsize=(8, 9))

    # vmin=0, vmax=4 ensures each integer code maps to the correct colour index
    ax.imshow(data, extent=extent, cmap=cmap, vmin=0, vmax=4,
              interpolation="nearest")

    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    add_scale_bar(ax, length_km=10)

    # Build a manual legend using coloured Rectangle patches
    legend_patches = [
        mpatches.Patch(facecolor=colour, label=label, edgecolor="grey", linewidth=0.5)
        for code, (colour, label) in zip(class_colours.keys(),
                                          zip(class_colours.values(), class_labels.values()))
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10,
              title="Land Use / Land Cover", title_fontsize=10,
              framealpha=0.9)

    ax.set_title("Land Use / Land Cover (4-class)\nDhaka Metropolitan Area",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m, UTM 45N)", fontsize=9)
    ax.set_ylabel("Northing (m, UTM 45N)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.text(0.5, 0.01, "Source: ESA WorldCover 2021 v2.0 | Analysis: Fahim Ahmed, BUET",
             ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  LULC map saved: {os.path.relpath(output_path, ROOT)}")


def map_summary_panel(lst_path, ndvi_path, lulc_path, boundary, output_path):
    """
    Map 4: 2×2 summary panel combining LST, NDVI, LULC, and a UHI intensity
    difference map (LST deviation from mean — highlights hot spots).

    This is the portfolio showcase figure.
    """
    lst_data,  lst_ext  = load_raster_for_plot(lst_path)
    ndvi_data, ndvi_ext = load_raster_for_plot(ndvi_path)
    lulc_data, lulc_ext = load_raster_for_plot(lulc_path)

    # UHI intensity = LST deviation from the study-area mean
    # Positive = hotter than average; negative = cooler than average
    lst_mean = np.nanmean(lst_data)
    uhi_intensity = lst_data - lst_mean   # subtract the spatial mean

    # LULC colour setup (same as map_lulc above)
    lulc_colours = ["none", "#E74C3C", "#27AE60", "#2980B9", "#F39C12"]
    lulc_cmap    = mcolors.ListedColormap(lulc_colours)
    lulc_patches = [
        mpatches.Patch(facecolor=c, label=l, edgecolor="grey", linewidth=0.5)
        for c, l in zip(lulc_colours[1:], ["Built-up", "Vegetation", "Water", "Bare soil"])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(
        "Dhaka Urban Heat Island Analysis — Summary Panel\n"
        "Fahim Ahmed | BUET Urban & Regional Planning",
        fontsize=15, fontweight="bold", y=1.01
    )

    # ── Panel A: LST ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    im_a = ax.imshow(lst_data, extent=lst_ext, cmap="RdYlBu_r",
                     vmin=np.nanpercentile(lst_data, 2),
                     vmax=np.nanpercentile(lst_data, 98),
                     interpolation="nearest")
    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    cbar_a = plt.colorbar(im_a, ax=ax, fraction=0.046, pad=0.02)
    cbar_a.set_label("°C", fontsize=9)
    ax.set_title("A) Land Surface Temperature", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    # ── Panel B: NDVI ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    im_b = ax.imshow(ndvi_data, extent=ndvi_ext, cmap="RdYlGn",
                     vmin=-0.1, vmax=0.7, interpolation="nearest")
    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    cbar_b = plt.colorbar(im_b, ax=ax, fraction=0.046, pad=0.02)
    cbar_b.set_label("NDVI", fontsize=9)
    ax.set_title("B) NDVI — Vegetation Cover", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    # ── Panel C: LULC ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.imshow(lulc_data, extent=lulc_ext, cmap=lulc_cmap,
              vmin=0, vmax=4, interpolation="nearest")
    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    ax.legend(handles=lulc_patches, loc="lower right", fontsize=8,
              title="LULC", title_fontsize=8, framealpha=0.85)
    ax.set_title("C) Land Use / Land Cover (4-class)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    # ── Panel D: UHI Intensity (LST deviation from mean) ─────────────────────
    ax = axes[1, 1]
    # Diverging colourmap centred at 0 (mean) — red = hotter, blue = cooler
    abs_max = np.nanpercentile(np.abs(uhi_intensity), 97)   # symmetric colour scale
    im_d = ax.imshow(uhi_intensity, extent=lst_ext, cmap="bwr",
                     vmin=-abs_max, vmax=abs_max, interpolation="nearest")
    overlay_boundary(ax, boundary)
    add_north_arrow(ax)
    cbar_d = plt.colorbar(im_d, ax=ax, fraction=0.046, pad=0.02)
    cbar_d.set_label("ΔT from mean (°C)", fontsize=9)
    ax.set_title("D) UHI Intensity (deviation from mean LST)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    # Common axis labels
    for row in axes:
        for ax_item in row:
            ax_item.set_xlabel("Easting (m, UTM 45N)", fontsize=8)
            ax_item.set_ylabel("Northing (m, UTM 45N)", fontsize=8)

    fig.text(0.5, -0.01,
             "Sources: Landsat C2 L2 (USGS) · ESA WorldCover 2021 · SRTM 30m · WorldPop 2020",
             ha="center", fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Summary panel saved: {os.path.relpath(output_path, ROOT)}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 7 — Final Map Production")
    print("=" * 60)

    os.makedirs(OUTPUT, exist_ok=True)

    # Check all required inputs
    required = {
        "LST"      : LST_PATH,
        "NDVI"     : NDVI_PATH,
        "LULC"     : LULC_PATH,
        "Boundary" : BOUNDARY,
    }
    missing = [k for k, v in required.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Missing required files: {missing}\n"
            "Run all earlier pipeline scripts first."
        )

    # ── Map 1: LST ────────────────────────────────────────────────────────────
    print("\nGenerating Map 1: Land Surface Temperature...")
    map_lst(LST_PATH, BOUNDARY,
            os.path.join(OUTPUT, "lst_map.png"))

    # ── Map 2: NDVI ───────────────────────────────────────────────────────────
    print("Generating Map 2: NDVI...")
    map_ndvi(NDVI_PATH, BOUNDARY,
             os.path.join(OUTPUT, "ndvi_map.png"))

    # ── Map 3: LULC ───────────────────────────────────────────────────────────
    print("Generating Map 3: LULC 4-class...")
    map_lulc(LULC_PATH, BOUNDARY,
             os.path.join(OUTPUT, "lulc_map.png"))

    # ── Map 4: Summary panel ──────────────────────────────────────────────────
    print("Generating Map 4: Summary panel (portfolio figure)...")
    map_summary_panel(
        LST_PATH, NDVI_PATH, LULC_PATH, BOUNDARY,
        os.path.join(OUTPUT, "uhi_summary_panel.png")
    )

    print("\n" + "=" * 60)
    print("All maps complete. Outputs:")
    print("  data/output/lst_map.png")
    print("  data/output/ndvi_map.png")
    print("  data/output/lulc_map.png")
    print("  data/output/uhi_summary_panel.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
