"""
10_multitemporal.py
-------------------
Step 10: multi-date (seasonal) analysis. Every Landsat scene found under
data/raw/landsat/scene_*/ is clipped/reprojected onto the same 30 m UTM grid as
the v2 LST raster, cloud-masked with QA_PIXEL, converted to LST (ST_B10) and NDVI,
and compared across dates.

Per date:   clear-sky fraction, mean land LST, SUHI (urban core vs rural reference),
            NDVI-LST correlation and slope (land only), ward mean LST.
Across dates: ward LST rank stability (Spearman), persistent hotspot wards,
            seasonal change of SUHI.

Outputs (data/output_v2/): multidate_summary.csv, multidate_ward_lst.csv,
persistent_hotspots.csv, fig_multidate.png, and per-date rasters in
data/processed_v2/scenes/.
"""
import os, re, glob, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed_v2")
OUT = os.path.join(ROOT, "data", "output_v2")
SCN = os.path.join(PROC, "scenes")
ADMIN = os.path.join(ROOT, "data", "raw", "admin", "gadm41_BGD_4.shp")
os.makedirs(SCN, exist_ok=True)
MIN_CLEAR = 0.6          # drop dates with < 50% clear pixels inside the boundary

# Standard Collection 2 Level-2 constants (used only if the MTL file is missing)
DEFAULTS = dict(ST_ML=0.00341802, ST_AL=149.0, SR_ML=2.75e-05, SR_AL=-0.2)


def find(folder, suffix):
    hits = [p for p in glob.glob(os.path.join(folder, "*")) if re.search(suffix, os.path.basename(p), re.I)
            and not p.lower().endswith(".crdownload")]
    return sorted(hits)[0] if hits else None


def read_mtl(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path, errors="ignore"):
            if "=" in line:
                k, v = [x.strip().strip('"') for x in line.split("=", 1)]
                out.setdefault(k, v)
    return out


def to_grid(path, shape, tf, crs, resampling=Resampling.nearest, dtype="float32", fill=0):
    dst = np.full(shape, fill, dtype=dtype)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs,
                  src_nodata=s.nodata, dst_transform=tf, dst_crs=crs, dst_nodata=fill,
                  resampling=resampling)
    return dst


def local_mean(arr, valid, size):
    num = ndi.uniform_filter(np.where(valid, arr, 0).astype("float32"), size=size, mode="constant")
    den = ndi.uniform_filter(valid.astype("float32"), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ───────────────────── static layers (same grid as v2 LST) ──────────────────
with rasterio.open(os.path.join(PROC, "lst_celsius.tif")) as s:
    shape, tf, crs = (s.height, s.width), s.transform, s.crs
lulc = to_grid(os.path.join(PROC, "lulc_4class.tif"), shape, tf, crs, dtype="uint8")
inside = lulc > 0
water = lulc == 3
built = lulc == 1
b990 = local_mean(built.astype("float32"), inside, 33)
b1k = local_mean(built.astype("float32"), inside, 67)
urban = built & (b990 >= 0.5) & ~water
rural = (lulc == 2) & (b1k <= 0.03)

bnd = gpd.read_file(os.path.join(PROC, "dhaka_boundary_utm.shp"))
adm = gpd.read_file(ADMIN, bbox=tuple(bnd.to_crs(4326).total_bounds)).to_crs(crs)
wards = gpd.overlay(adm, bnd[["geometry"]], how="intersection", keep_geom_type=True).reset_index(drop=True)
wards["ward_id"] = np.arange(1, len(wards) + 1)
wid = rasterize(((g, i) for g, i in zip(wards.geometry, wards.ward_id)), out_shape=shape,
                transform=tf, fill=0, dtype="int32")
nW = len(wards) + 1

# ───────────────────── per-scene processing ────────────────────────────────
rows, ward_cols = [], {}
for folder in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "landsat", "scene_*"))):
    p_st = find(folder, r"_ST_B10\.tif$")
    p_r = find(folder, r"_SR_B4\.tif$")
    p_n = find(folder, r"_SR_B5\.tif$")
    p_qa = find(folder, r"_QA_PIXEL\.tif+$")
    p_mtl = find(folder, r"_MTL\.txt$")
    if not (p_st and p_r and p_n):
        print(f"[skip] {os.path.basename(folder)}: missing bands"); continue
    parts = os.path.basename(p_st).split("_")
    date = f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:]}"
    sensor = parts[0]
    print(f"\n== {date} ({sensor}) {os.path.basename(folder)}")
    mtl = read_mtl(p_mtl)
    st_ml = float(mtl.get("TEMPERATURE_MULT_BAND_ST_B10", DEFAULTS["ST_ML"]))
    st_al = float(mtl.get("TEMPERATURE_ADD_BAND_ST_B10", DEFAULTS["ST_AL"]))
    sr_ml = float(mtl.get("REFLECTANCE_MULT_BAND_4", DEFAULTS["SR_ML"]))
    sr_al = float(mtl.get("REFLECTANCE_ADD_BAND_4", DEFAULTS["SR_AL"]))

    st = to_grid(p_st, shape, tf, crs, dtype="float32")
    red = to_grid(p_r, shape, tf, crs, dtype="float32")
    nir = to_grid(p_n, shape, tf, crs, dtype="float32")
    bad = (st == 0) | (red < 7273) | (red > 43636) | (nir < 7273) | (nir > 43636)
    if p_qa:
        qa = to_grid(p_qa, shape, tf, crs, dtype="uint16", fill=1)
        flags = np.zeros(shape, dtype=bool)
        for bit in range(6):
            flags |= ((qa >> bit) & 1).astype(bool)
        # QA misses thin cloud edges and haze: grow cloud/shadow/cirrus flags by 5 px (150 m)
        cloudy = (((qa >> 1) | (qa >> 2) | (qa >> 3) | (qa >> 4)) & 1).astype(bool)
        flags |= ndi.binary_dilation(cloudy, iterations=5)
        bad |= flags
    else:
        print("   [WARN] no QA_PIXEL - cloud mask skipped")
    clear_frac = float((~bad & inside).sum() / inside.sum())
    print(f"   clear-sky fraction inside boundary: {clear_frac:.2f}")
    if clear_frac < MIN_CLEAR:
        print("   [skip] too cloudy"); continue

    lst = st * st_ml + st_al - 273.15
    lst[bad | (lst < 0) | (lst > 72)] = np.nan
    r = red * sr_ml + sr_al; n = nir * sr_ml + sr_al
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (n - r) / (n + r)
    ndvi[bad] = np.nan; ndvi = np.clip(ndvi, -1, 1)

    # save rasters
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32",
                crs=crs, transform=tf, nodata=np.nan, compress="deflate")
    for nm, arr in (("lst", lst), ("ndvi", ndvi)):
        with rasterio.open(os.path.join(SCN, f"{date}_{nm}.tif"), "w", **prof) as d:
            d.write(arr.astype("float32"), 1)

    land = np.isfinite(lst) & np.isfinite(ndvi) & ~water & inside
    u = urban & land; ru = rural & land
    suhi = lst[u].mean() - lst[ru].mean() if u.sum() > 1000 and ru.sum() > 1000 else np.nan
    suhi_med = np.median(lst[u]) - np.median(lst[ru]) if u.sum() > 1000 and ru.sum() > 1000 else np.nan
    sub = np.flatnonzero(land.ravel())[::5]
    x, y = ndvi.ravel()[sub], lst.ravel()[sub]
    rr = np.corrcoef(x, y)[0, 1]
    slope = np.polyfit(x, y, 1)[0]
    s_w = np.bincount(wid[land], weights=lst[land], minlength=nW)
    n_w = np.bincount(wid[land], minlength=nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        w_mean = s_w / n_w
    w_mean[n_w < 100] = np.nan
    ward_cols[date] = w_mean[1:]
    rows.append({"date": date, "sensor": sensor, "scene": os.path.basename(folder),
                 "clear_fraction": round(clear_frac, 3),
                 "mean_land_lst_C": lst[land].mean(), "lst_p95_C": np.percentile(lst[land], 95),
                 "urban_mean_C": lst[u].mean(), "rural_mean_C": lst[ru].mean(), "suhi_C": suhi, "suhi_median_C": suhi_med, "n_urban_px": int(u.sum()), "n_rural_px": int(ru.sum()),
                 "ndvi_lst_r_land": rr, "ndvi_lst_slope_C_per_NDVI": slope, "mean_ndvi_land": ndvi[land].mean()})
    print(f"   LST {lst[land].mean():.2f} C | SUHI {suhi:.2f} C | NDVI-LST r {rr:.2f}")

if not rows:
    raise SystemExit("No usable scenes found.")
summ = pd.DataFrame(rows).sort_values("date")
summ.to_csv(os.path.join(OUT, "multidate_summary.csv"), index=False)
print("\n", summ[["date", "clear_fraction", "mean_land_lst_C", "suhi_C", "ndvi_lst_r_land"]].round(2).to_string(index=False))

# ───────────────────── across dates ────────────────────────────────────────
wl = pd.DataFrame(ward_cols)
wl.insert(0, "ward_id", wards.ward_id.values)
wl.insert(1, "name", wards.get("NAME_4", wards.ward_id.astype(str)).values)
wl.insert(2, "thana", wards.get("NAME_3", "").values)
dates = list(ward_cols.keys())
if len(dates) >= 2:
    z = wl[dates].apply(lambda c: (c - c.mean()) / c.std())
    wl["mean_z"] = z.mean(axis=1)
    top = wl[dates].apply(lambda c: c >= c.quantile(0.9))
    wl["n_dates_top_decile"] = top.sum(axis=1)
    wl["persistent_hotspot"] = wl["n_dates_top_decile"] == len(dates)
    wl.sort_values("mean_z", ascending=False).to_csv(os.path.join(OUT, "multidate_ward_lst.csv"), index=False)
    wl[wl.persistent_hotspot].sort_values("mean_z", ascending=False).to_csv(
        os.path.join(OUT, "persistent_hotspots.csv"), index=False)
    rho = pd.DataFrame({a: {b: spearmanr(wl[a], wl[b], nan_policy="omit")[0] for b in dates} for a in dates})
    rho.to_csv(os.path.join(OUT, "multidate_ward_rank_corr.csv"))
    print("\nWard rank correlation (Spearman):\n", rho.round(2).to_string())
    print(f"Persistent hotspot wards (top decile on all {len(dates)} dates): {int(wl.persistent_hotspot.sum())}")
else:
    wl.to_csv(os.path.join(OUT, "multidate_ward_lst.csv"), index=False)

# ───────────────────── figure ──────────────────────────────────────────────
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
lab = summ.date.values
ax[0].bar(lab, summ.suhi_C, color="#c0392b"); ax[0].set_title("SUHI intensity by date"); ax[0].set_ylabel("°C")
ax[1].bar(lab, summ.ndvi_lst_r_land, color="#2e7d32"); ax[1].set_title("NDVI-LST r (land only)")
ax[2].bar(lab, summ.mean_land_lst_C, color="#34495e"); ax[2].set_title("Mean land LST"); ax[2].set_ylabel("°C")
for a in ax: a.tick_params(axis="x", rotation=35)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_multidate.png"), dpi=200); plt.close(fig)
print("Done.")
