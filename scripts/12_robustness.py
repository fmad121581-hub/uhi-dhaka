"""
12_robustness.py
----------------
Repeats the population-exposure and Random Forest driver analysis (from 09) on every
usable date, to check whether the February 2022 results hold up.

Usage:  python 12_robustness.py 2022-02-24     # one date, appends to robustness_by_date.csv
        python 12_robustness.py --plot         # figure from the CSV
Outputs (data/output_v2/): robustness_by_date.csv, fig_robustness.png
"""
import os, sys, warnings
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed_v2")
OUT = os.path.join(ROOT, "data", "output_v2")
CSV = os.path.join(OUT, "robustness_by_date.csv")
PX, BLOCK_PX = 30.0, 100


def plot():
    d = pd.read_csv(CSV).sort_values("date")
    feats = [c for c in d.columns if c.startswith("imp_")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].bar(d.date, d.exposed_sd_pct, color="#c0392b"); ax[0].set_title("Population in pixels ≥ mean + 1 SD (%)")
    ax[1].bar(d.date, d.rf_spatial_r2, color="#34495e"); ax[1].set_title("Random Forest R² (spatial block CV)")
    ax[1].set_ylim(0, 1)
    top = ["imp_dist_center_km", "imp_built_frac_330m", "imp_ndvi", "imp_pop_density_km2"]
    w = 0.2
    for i, f in enumerate(top):
        ax[2].bar(np.arange(len(d)) + i * w, d[f], w, label=f[4:])
    ax[2].set_xticks(np.arange(len(d)) + 0.3); ax[2].set_xticklabels(d.date)
    ax[2].set_title("Permutation importance (R² drop)"); ax[2].set_ylim(0, 0.68); ax[2].legend(fontsize=8, ncol=2, loc="upper right")
    for a in ax: a.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_robustness.png"), dpi=200)
    print("figure saved"); sys.exit(0)


if sys.argv[1] == "--plot":
    plot()
date = sys.argv[1]

with rasterio.open(os.path.join(PROC, "lst_celsius.tif")) as s:
    shape, tf, crs = (s.height, s.width), s.transform, s.crs


def rd(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32"); a[~np.isfinite(a)] = np.nan; return a


def to_grid(path, dtype, rs, fill=0):
    dst = np.full(shape, fill, dtype=dtype)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs, src_nodata=s.nodata,
                  dst_transform=tf, dst_crs=crs, dst_nodata=fill, resampling=rs)
    return dst


def local_mean(arr, valid, size):
    num = ndi.uniform_filter(np.where(valid, arr, 0).astype("float32"), size=size, mode="constant")
    den = ndi.uniform_filter(valid.astype("float32"), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


lst = rd(os.path.join(PROC, "scenes", f"{date}_lst.tif"))
ndvi = rd(os.path.join(PROC, "scenes", f"{date}_ndvi.tif"))
lulc = to_grid(os.path.join(PROC, "lulc_4class.tif"), "uint8", Resampling.nearest)
dem = to_grid(os.path.join(PROC, "dem_clipped_utm.tif"), "float32", Resampling.bilinear, fill=np.nan)
pop30 = to_grid(os.path.join(PROC, "population_clipped_utm.tif"), "float32", Resampling.bilinear, fill=np.nan)
pop30[pop30 < 0] = np.nan
inside, water, built = lulc > 0, lulc == 3, lulc == 1
land = np.isfinite(lst) & np.isfinite(ndvi) & ~water & inside

rows, cols = np.indices(shape)
X = tf.c + (cols + 0.5) * tf.a; Y = tf.f + (rows + 0.5) * tf.e
cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(90.4125, 23.7306)
feat = {
    "ndvi": ndvi,
    "built_frac_330m": local_mean(built.astype("float32"), inside, 11),
    "built_frac_990m": local_mean(built.astype("float32"), inside, 33),
    "ndvi_mean_990m": local_mean(np.nan_to_num(ndvi), inside & np.isfinite(ndvi), 33),
    "water_frac_990m": local_mean(water.astype("float32"), inside, 33),
    "dist_water_m": ndi.distance_transform_edt(~water) * PX,
    "elevation_m": dem,
    "pop_density_km2": pop30 / 0.01,
    "dist_center_km": np.hypot(X - cx, Y - cy) / 1000.0,
}
names = list(feat)

# exposure
with rasterio.open(os.path.join(PROC, "population_clipped_utm.tif")) as ps:
    pop = ps.read(1).astype("float32")
    if ps.nodata is not None: pop[pop == ps.nodata] = 0
    pop[~np.isfinite(pop)] = 0; pop[pop < 0] = 0
    ptf, pshape, pcrs = ps.transform, pop.shape, ps.crs
lv = lst[land]
res = {"date": date, "n_land_px": int(land.sum()), "mean_land_lst": float(lv.mean())}
for lab, thr in (("sd", lv.mean() + lv.std()), ("p90", np.percentile(lv, 90))):
    hot = (land & (lst >= thr)).astype("float32")
    frac = np.zeros(pshape, dtype="float32")
    reproject(hot, frac, src_transform=tf, src_crs=crs, dst_transform=ptf, dst_crs=pcrs, resampling=Resampling.average)
    res[f"exposed_{lab}_pct"] = float((pop * frac).sum() / pop.sum() * 100)
    res[f"thr_{lab}_C"] = float(thr)

# Random Forest, spatial block CV
rng = np.random.default_rng(42)
stack = np.stack([feat[k] for k in names], axis=-1)
ok = land & np.all(np.isfinite(stack), axis=-1)
take = rng.choice(np.flatnonzero(ok.ravel()), size=min(20000, int(ok.sum())), replace=False)
Xs = stack.reshape(-1, len(names))[take]; ys = lst.ravel()[take]
grp = ((rows // BLOCK_PX) * 10000 + (cols // BLOCK_PX)).ravel()[take]
r2s, imps = [], []
for tr, te in GroupKFold(n_splits=5).split(Xs, ys, groups=grp):
    m = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=42).fit(Xs[tr], ys[tr])
    r2s.append(r2_score(ys[te], m.predict(Xs[te])))
    sub = rng.choice(te, size=min(3000, len(te)), replace=False)
    imps.append(permutation_importance(m, Xs[sub], ys[sub], n_repeats=2, random_state=42, n_jobs=-1).importances_mean)
res["rf_spatial_r2"] = float(np.mean(r2s)); res["rf_spatial_r2_sd"] = float(np.std(r2s))
for k, v in zip(names, np.mean(imps, axis=0)): res["imp_" + k] = float(v)

df = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
if len(df): df = df[df.date != date]
pd.concat([df, pd.DataFrame([res])]).sort_values("date").to_csv(CSV, index=False)
print(date, {k: round(v, 3) for k, v in res.items() if k in ("exposed_sd_pct", "exposed_p90_pct", "rf_spatial_r2", "imp_dist_center_km", "imp_built_frac_330m", "imp_ndvi")})
