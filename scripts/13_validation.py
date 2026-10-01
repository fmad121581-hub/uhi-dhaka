"""
13_validation.py
----------------
Sanity check of Landsat LST against independent air temperature at Dhaka (Hazrat
Shahjalal airport, WMO 41923; 23.8433 N, 90.3978 E) at the satellite overpass
(about 04:25 UTC, 10:25 local).

Air temperature sources (be explicit about them in any write-up):
  * ERA5-Land hourly 2 m temperature, linearly interpolated to 04:25 UTC (all 5 dates).
    This is a 9 km reanalysis, not a station.
  * NOAA ISD synoptic station report, interpolated between 03 and 06 UTC. Only the
    2025-05-07 report was available for these dates.

LST is not air temperature (daytime LST is usually higher), so the check is whether
LST tracks air temperature across seasons, and how large the offset is.

Outputs (data/output_v2/): validation_vs_air_temp.csv, fig_validation.png
"""
import os
import numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "data", "processed_v2", "scenes")
OUT = os.path.join(ROOT, "data", "output_v2")
LON, LAT = 90.3978, 23.8433
FRAC = 1.42 / 3.0  # not used for ERA5; kept for the ISD interpolation below

# ERA5-Land temperature_2m at 04:00 and 05:00 UTC (Open-Meteo archive, degC)
era5 = {"2022-02-24": (25.2, 26.7), "2025-05-07": (29.2, 30.5), "2025-10-22": (30.7, 31.5),
        "2026-02-19": (27.1, 29.0), "2026-06-03": (32.3, 33.0)}
# NOAA ISD 41923099999, 03:00 and 06:00 UTC reports (degC)
isd = {"2025-05-07": (30.2, 33.5)}
T = 25 / 60.0  # 04:25 UTC as a fraction of the 04->05 UTC hour

rows = []
for d, (a, b) in era5.items():
    with rasterio.open(os.path.join(SC, f"{d}_lst.tif")) as s:
        lst = s.read(1)
        x, y = Transformer.from_crs("EPSG:4326", s.crs, always_xy=True).transform(LON, LAT)
        r, c = s.index(x, y)
        rr, cc = np.indices(lst.shape)
        dist = np.hypot((cc - c) * s.res[0], (rr - r) * s.res[1])
    ok = np.isfinite(lst)
    w1, w5 = ok & (dist <= 1000), ok & (dist <= 5000)
    era = a + (b - a) * T
    st = np.nan
    if d in isd:
        s3, s6 = isd[d]; st = s3 + (s6 - s3) * (85 / 180.0)   # 04:25 is 85 min after 03:00
    rows.append({"date": d, "lst_1km_C": lst[w1].mean() if w1.sum() > 20 else np.nan,
                 "lst_5km_C": lst[w5].mean() if w5.sum() > 100 else np.nan, "lst_citywide_C": np.nanmean(lst),
                 "era5land_air_C": era, "station_air_C": st,
                 "n_px_1km": int(w1.sum()), "n_px_5km": int(w5.sum())})
df = pd.DataFrame(rows)
df["lst_minus_air_C"] = df.lst_5km_C - df.era5land_air_C
df.round(2).to_csv(os.path.join(OUT, "validation_vs_air_temp.csv"), index=False)
print(df.round(2).to_string(index=False))
m = df.dropna(subset=["lst_5km_C"])
r_p = pearsonr(m.era5land_air_C, m.lst_5km_C)[0]; r_s = spearmanr(m.era5land_air_C, m.lst_5km_C)[0]
slope = np.polyfit(m.era5land_air_C, m.lst_5km_C, 1)
print(f"n={len(m)}  Pearson r={r_p:.2f}  Spearman={r_s:.2f}  slope={slope[0]:.2f}  mean(LST-air)={m.lst_minus_air_C.mean():.2f} C")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.scatter(m.era5land_air_C, m.lst_5km_C, s=60, color="#c0392b", zorder=3)
for _, q in m.iterrows(): ax.annotate(q.date[2:], (q.era5land_air_C, q.lst_5km_C), textcoords="offset points", xytext=(6, -10), fontsize=8)
lo, hi = m.era5land_air_C.min() - 1, m.era5land_air_C.max() + 1
ax.plot([lo, hi], [lo, hi], color="#888", ls="--", label="LST = air temperature")
ax.plot([lo, hi], np.polyval(slope, [lo, hi]), color="#34495e", label=f"fit (r = {r_p:.2f}, n = {len(m)})")
st = m.dropna(subset=["station_air_C"])
if len(st): ax.scatter(st.station_air_C, st.lst_5km_C, marker="^", s=70, color="#2e7d32", zorder=4, label="station air temp (7 May 2025)")
ax.set_xlabel("Air temperature at overpass, ERA5-Land (°C)"); ax.set_ylabel("Landsat LST within 5 km of airport (°C)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_validation.png"), dpi=200)
