"""
11_persistent_hotspots.py
-------------------------
Profiles the wards that are in the top decile of LST on EVERY usable date
(output of 10_multitemporal.py): who they are, what land cover they have,
how many people live there, and where they sit. Also draws a map of the
multi-date mean LST z-score with the persistent hotspots outlined.

Outputs (data/output_v2/): persistent_hotspot_profile.csv, fig_persistent_hotspots.png
"""
import os, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed_v2")
OUT = os.path.join(ROOT, "data", "output_v2")
ADMIN = os.path.join(ROOT, "data", "raw", "admin", "gadm41_BGD_4.shp")

with rasterio.open(os.path.join(PROC, "lst_celsius.tif")) as s:
    shape, tf, crs = (s.height, s.width), s.transform, s.crs

def to_grid(path, dtype, resampling, fill=0):
    dst = np.full(shape, fill, dtype=dtype)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs, src_nodata=s.nodata,
                  dst_transform=tf, dst_crs=crs, dst_nodata=fill, resampling=resampling)
    return dst

lulc = to_grid(os.path.join(PROC, "lulc_4class.tif"), "uint8", Resampling.nearest)
pop = to_grid(os.path.join(PROC, "population_clipped_utm.tif"), "float32", Resampling.average, fill=0)
pop[pop < 0] = 0
bnd = gpd.read_file(os.path.join(PROC, "dhaka_boundary_utm.shp"))
adm = gpd.read_file(ADMIN, bbox=tuple(bnd.to_crs(4326).total_bounds)).to_crs(crs)
wards = gpd.overlay(adm, bnd[["geometry"]], how="intersection", keep_geom_type=True).reset_index(drop=True)
wards["ward_id"] = np.arange(1, len(wards) + 1)
wid = rasterize(((g, i) for g, i in zip(wards.geometry, wards.ward_id)), out_shape=shape,
                transform=tf, fill=0, dtype="int32")
nW = len(wards) + 1

area_km2 = np.bincount(wid[lulc > 0], minlength=nW) * 0.0009
comp = {}
for code, nm in {1: "built_up", 2: "vegetation", 3: "water", 4: "bare"}.items():
    comp[nm] = np.bincount(wid[(lulc == code)], minlength=nW) / np.maximum(np.bincount(wid[lulc > 0], minlength=nW), 1)
# pop was averaged to 30 m: persons per 100 m cell -> multiply by 0.09 ha / 1 ha cell area ratio (0.09)
pop_tot = np.bincount(wid.ravel(), weights=(pop * 0.09).ravel(), minlength=nW)

ward = pd.read_csv(os.path.join(OUT, "multidate_ward_lst.csv"))
prof = wards[["ward_id", "NAME_4", "NAME_3"]].rename(columns={"NAME_4": "ward", "NAME_3": "thana"})
prof["area_km2"] = area_km2[1:]
for k, v in comp.items():
    prof[k + "_pct"] = (v[1:] * 100).round(1)
prof["population"] = pop_tot[1:].round(0)
prof["pop_density_km2"] = (prof.population / prof.area_km2.replace(0, np.nan)).round(0)
prof = prof.merge(ward.drop(columns=["name", "thana"]), on="ward_id")
dates = [c for c in prof.columns if c[:2] == "20"]
prof = prof.sort_values("mean_z", ascending=False)
pers = prof[prof.persistent_hotspot]
pers.to_csv(os.path.join(OUT, "persistent_hotspot_profile.csv"), index=False)
print(pers[["ward", "thana", "area_km2", "built_up_pct", "vegetation_pct", "population", "pop_density_km2",
            "mean_z"] + dates].round(2).to_string(index=False))
big = prof[(prof.area_km2 > 0.5)]
print(f"\nPersistent hotspots: {len(pers)} wards, {pers.population.sum():,.0f} people "
      f"({pers.population.sum() / big.population.sum() * 100:.1f}% of the study-area population)")
print("Persistent vs all other wards, mean built-up %:", round(pers.built_up_pct.mean(), 1), "vs",
      round(prof[~prof.persistent_hotspot].built_up_pct.mean(), 1))

g = wards.merge(prof[["ward_id", "mean_z", "persistent_hotspot"]], on="ward_id")
fig, ax = plt.subplots(figsize=(7.5, 7))
g.plot(column="mean_z", cmap="YlOrRd", ax=ax, edgecolor="white", linewidth=0.2,
       legend=True, legend_kwds={"label": f"Mean LST z-score across {len(dates)} dates", "shrink": 0.6})
g[g.persistent_hotspot == True].boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.4)
ax.set_title("Persistent heat hotspots (outlined: top decile on every date)")
ax.set_axis_off()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_persistent_hotspots.png"), dpi=200)
print("Done.")
