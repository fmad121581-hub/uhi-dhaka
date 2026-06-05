"""
08_rainfall_analysis.py
-----------------------
Bonus analysis for the Dhaka Urban Heat Island portfolio.

What this script does:
  - Reads the 20-year NASA POWER monthly rainfall CSV (2004–2023)
  - Cleans the raw NASA POWER header format into a usable DataFrame
  - Computes annual totals and seasonal breakdowns (pre-monsoon, monsoon, dry)
  - Plots:
      A) Annual rainfall bar chart with trend line (2004–2023)
      B) Monthly climatology (average rainfall by calendar month)
      C) Heatmap of monthly rainfall across all years
  - Saves all outputs to data/output/

Why rainfall matters for UHI analysis:
  Rainfall affects surface moisture, which modulates LST.
  Monsoon months suppress LST; dry months allow urban surfaces to overheat.
  Understanding the rainfall regime contextualises the satellite overpass timing
  and helps explain seasonal UHI intensity.

Seasons used (Bangladesh meteorological convention):
  Pre-monsoon : March – May
  Monsoon     : June – September
  Post-monsoon: October – November
  Dry         : December – February
"""

import os
import numpy as np          # array operations and statistics
import pandas as pd         # data manipulation
import matplotlib.pyplot as plt          # plotting
import matplotlib.colors as mcolors      # for heatmap colour scaling

# ── Path constants ─────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(ROOT, "data", "raw", "rainfall",
                           "rainfall_dhaka_monthly_2004_2023.csv")
OUTPUT     = os.path.join(ROOT, "data", "output")
RAIN_PNG   = os.path.join(OUTPUT, "rainfall_analysis.png")
RAIN_CSV   = os.path.join(OUTPUT, "rainfall_annual_seasonal.csv")

# ── Month and season definitions ───────────────────────────────────────────────
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

SEASONS = {
    "Pre-monsoon"  : ["MAR", "APR", "MAY"],
    "Monsoon"      : ["JUN", "JUL", "AUG", "SEP"],
    "Post-monsoon" : ["OCT", "NOV"],
    "Dry"          : ["DEC", "JAN", "FEB"],
}


# ── Data loading and cleaning ─────────────────────────────────────────────────

def load_rainfall_csv(path):
    """
    Parse the NASA POWER format CSV, skipping the text header block.
    NASA POWER CSVs have a multi-line header ending with '-END HEADER-';
    the actual data starts after that line.

    Returns a clean pandas DataFrame with columns:
      YEAR, JAN, FEB, ..., DEC, ANN
    """
    # Find which line the data header row starts on
    skip_rows = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            # The data header row begins with 'PARAMETER'
            if line.strip().startswith("PARAMETER"):
                skip_rows = i   # this line is the column header
                break

    # Read the CSV starting from the column header row
    df = pd.read_csv(path, skiprows=skip_rows)

    # Drop the 'PARAMETER' column (it just says 'PRECTOTCORR_SUM' in every row)
    df = df.drop(columns=["PARAMETER"], errors="ignore")

    # Convert all columns to numeric (handles any stray spaces or strings)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace the NASA missing-data sentinel (-999) with NaN
    df = df.replace(-999, np.nan)

    # Sort by year just in case the file is not in order
    df = df.sort_values("YEAR").reset_index(drop=True)

    print(f"  Rows loaded: {len(df)}  (years {int(df['YEAR'].min())}–{int(df['YEAR'].max())})")
    return df


def add_seasonal_totals(df):
    """
    Add seasonal rainfall columns (sum of monthly values per season per year).
    """
    for season, months in SEASONS.items():
        # Sum only the columns for the months in this season
        df[season] = df[months].sum(axis=1, skipna=True)
    return df


# ── Trend line ────────────────────────────────────────────────────────────────

def linear_trend(x, y):
    """
    Fit a simple linear trend to (x, y) using numpy polyfit.
    Returns: (slope, intercept, y_trend_array)
    """
    # Remove NaN years before fitting
    valid = ~np.isnan(y)
    coeffs = np.polyfit(x[valid], y[valid], deg=1)   # degree 1 = linear
    slope, intercept = coeffs
    y_trend = np.polyval(coeffs, x)   # evaluate the line at all x values
    return slope, intercept, y_trend


# ── Plotting functions ─────────────────────────────────────────────────────────

def plot_rainfall_analysis(df, output_path):
    """
    Produce a 3-panel figure:
      A) Annual total rainfall bar chart + trend line
      B) Monthly climatology (average ± std)
      C) Year × Month rainfall heatmap
    """
    years   = df["YEAR"].values.astype(int)
    annual  = df["ANN"].values.astype(float)

    # Monthly climatology: mean and std across all years for each month
    monthly_mean = df[MONTHS].mean(axis=0).values
    monthly_std  = df[MONTHS].std(axis=0).values

    # Build the month × year heatmap matrix (12 rows = months, N cols = years)
    heatmap_data = df[MONTHS].T.values   # shape: (12, 20)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Rainfall Analysis — Dhaka (2004–2023)\n"
        "Source: NASA POWER MERRA-2 | Fahim Ahmed, BUET",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Use a 2-row, 2-column grid; panel C spans the full bottom row
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, :])   # top row, full width
    ax_b = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_c = fig.add_subplot(gs[1, 1])   # bottom-right

    # ── Panel A: Annual bar chart ─────────────────────────────────────────────
    bar_colors = ["#2980B9" if v < np.nanmean(annual) else "#E74C3C"
                  for v in annual]   # blue = below average, red = above average

    ax_a.bar(years, annual, color=bar_colors, alpha=0.75,
             edgecolor="white", linewidth=0.5, label="Annual total")

    # Linear trend line
    slope, intercept, y_trend = linear_trend(years.astype(float), annual)
    ax_a.plot(years, y_trend, color="black", linewidth=2, linestyle="--",
              label=f"Trend: {slope:+.1f} mm/yr")

    # Long-term mean horizontal line
    mean_val = np.nanmean(annual)
    ax_a.axhline(mean_val, color="grey", linewidth=1, linestyle=":",
                 label=f"Mean: {mean_val:.0f} mm")

    ax_a.set_title("A) Annual Rainfall Total (2004–2023)", fontweight="bold")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Rainfall (mm)")
    ax_a.set_xticks(years)
    ax_a.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax_a.legend(fontsize=10)
    ax_a.grid(axis="y", alpha=0.3)

    # Annotate the wettest and driest years
    wettest = years[np.nanargmax(annual)]
    driest  = years[np.nanargmin(annual)]
    ax_a.annotate(f"Wettest: {wettest}",
                  xy=(wettest, np.nanmax(annual)),
                  xytext=(wettest + 0.5, np.nanmax(annual) * 0.98),
                  fontsize=8, color="red",
                  arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax_a.annotate(f"Driest: {driest}",
                  xy=(driest, np.nanmin(annual)),
                  xytext=(driest + 0.5, np.nanmin(annual) * 1.1),
                  fontsize=8, color="navy",
                  arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))

    # ── Panel B: Monthly climatology bar ──────────────────────────────────────
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Colour each bar by season
    season_colours = {
        "Dry"          : "#3498DB",
        "Pre-monsoon"  : "#F39C12",
        "Monsoon"      : "#27AE60",
        "Post-monsoon" : "#8E44AD",
    }
    # Build a colour list aligned to the 12 months
    month_season_map = {}
    for season, months in SEASONS.items():
        for m in months:
            month_season_map[m] = season
    bar_month_colours = [
        season_colours[month_season_map.get(m, "Dry")] for m in MONTHS
    ]

    ax_b.bar(month_labels, monthly_mean, color=bar_month_colours, alpha=0.8,
             edgecolor="white")
    # Error bars show inter-annual variability (± 1 std)
    ax_b.errorbar(month_labels, monthly_mean, yerr=monthly_std,
                  fmt="none", color="black", capsize=4, linewidth=1.2)

    ax_b.set_title("B) Monthly Climatology (mean ± 1 std)", fontweight="bold")
    ax_b.set_xlabel("Month")
    ax_b.set_ylabel("Rainfall (mm)")
    ax_b.grid(axis="y", alpha=0.3)

    # Add a legend for the seasons
    season_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.8, label=s)
        for s, c in season_colours.items()
    ]
    ax_b.legend(handles=season_patches, fontsize=8, loc="upper left")

    # ── Panel C: Heatmap (Year × Month) ───────────────────────────────────────
    # imshow() with the months on Y axis and years on X axis
    im = ax_c.imshow(
        heatmap_data,
        aspect="auto",     # stretch to fill the axes
        cmap="YlGnBu",     # light = dry, dark blue = heavy rain
        interpolation="nearest"
    )

    # Set tick labels
    ax_c.set_xticks(range(len(years)))
    ax_c.set_xticklabels(years, rotation=90, fontsize=7)
    ax_c.set_yticks(range(12))
    ax_c.set_yticklabels(month_labels, fontsize=9)
    ax_c.set_title("C) Monthly Rainfall Heatmap (mm)", fontweight="bold")
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Month")

    cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Rainfall (mm)", fontsize=9)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Rainfall figure saved: {os.path.relpath(output_path, ROOT)}")


# ── Main routine ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Bonus — Rainfall Trend Analysis (2004–2023)")
    print("=" * 60)

    os.makedirs(OUTPUT, exist_ok=True)

    # Load and clean the CSV
    print(f"\nReading: {os.path.relpath(RAW_CSV, ROOT)}")
    df = load_rainfall_csv(RAW_CSV)

    # Add seasonal breakdowns
    df = add_seasonal_totals(df)

    # Print summary statistics
    print("\n  Annual rainfall statistics:")
    print(f"  Mean : {df['ANN'].mean():.1f} mm")
    print(f"  Std  : {df['ANN'].std():.1f} mm")
    print(f"  Min  : {df['ANN'].min():.1f} mm  ({int(df.loc[df['ANN'].idxmin(), 'YEAR'])})")
    print(f"  Max  : {df['ANN'].max():.1f} mm  ({int(df.loc[df['ANN'].idxmax(), 'YEAR'])})")

    years  = df["YEAR"].values.astype(float)
    annual = df["ANN"].values.astype(float)
    slope, _, _ = linear_trend(years, annual)
    print(f"  Trend: {slope:+.1f} mm/yr over the period")

    print("\n  Seasonal averages (mm):")
    for season in SEASONS:
        print(f"    {season:<15}: {df[season].mean():.1f} mm")

    # Save the enriched table
    out_cols = ["YEAR"] + MONTHS + ["ANN"] + list(SEASONS.keys())
    df[out_cols].to_csv(RAIN_CSV, index=False, float_format="%.2f")
    print(f"\n  Processed CSV saved: {os.path.relpath(RAIN_CSV, ROOT)}")

    # Produce the 3-panel figure
    print("\nGenerating rainfall analysis figure...")
    plot_rainfall_analysis(df, RAIN_PNG)

    print("\n" + "=" * 60)
    print("Rainfall analysis complete.")
    print("Outputs:")
    print("  data/output/rainfall_analysis.png")
    print("  data/output/rainfall_annual_seasonal.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
