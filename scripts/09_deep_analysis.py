"""
09_deep_analysis.py
-------------------
Step 9: Beyond correlations. Uses the corrected v2 rasters (land only, water masked
out of every statistic) and produces:

  A) Surface UHI (SUHI) intensity: urban core vs rural reference, with a
     spatial block-bootstrap confidence interval, plus an urban-rural gradient
     by distance from the city centre.
  B) Ward-level spatial statistics: global Moran's I and Getis-Ord Gi* hotspots
     (203 wards), with SUHI per ward.
  C) Driver model: Random Forest on 30 m pixels with SPATIAL block cross-validation
     (random CV is reported next to it to show how much it flatters the model),
     permutation importance on held-out blocks.
  D) Heat exposure: WorldPop population living in the hottest pixels, by ward.

Outputs go to data/output_v2/.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy import ndimage as ndi
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from libpysal.weights import Queen, KNN
from esda.moran import Moran
from esda.getisord import G_Local

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed_v2")
OUT = os.path.join(ROOT, "data", "output_v2")
ADMIN = os.path.join(ROOT, "data", "raw", "admin", "gadm41_BGD_4.shp")
os.makedirs(OUT, exist_ok=True)

CENTER_WGS84 = (90.4125, 23.7306)   # Gulistan / Motijheel core
PX = 30.0                           # Landsat pixel (m)
BLOCK_PX = 100                      # 3 km spatial blocks


# ───────────────────────────── grid helpers ──────────────────────────────
def load_ref(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        a[~np.isfinite(a)] = np.nan
        return a, s.transform, s.crs


def to_grid(path, dst_shape, dst_tf, dst_crs, resampling, fill=np.nan):
    out = np.full(dst_shape, fill, dtype="float32")
    with rasterio.open(path) as s:
        src_nodata = s.nodata
        reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs,
                  src_nodata=src_nodata, dst_transform=dst_tf, dst_crs=dst_crs,
                  dst_nodata=fill, resampling=resampling)
    return out


def local_mean(arr, valid, size):
    """Moving-window mean that ignores invalid cells (no edge bias from nodata)."""
    num = ndi.uniform_filter(np.where(valid, arr, 0).astype("float32"), size=size, mode="constant")
    den = ndi.uniform_filter(valid.astype("float32"), size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ───────────────────────────── load layers ───────────────────────────────
print("Loading layers...")
lst, tf, crs = load_ref(os.path.join(PROC, "lst_celsius.tif"))
shape = lst.shape
ndvi, _, _ = load_ref(os.path.join(PROC, "ndvi.tif"))
lulc = to_grid(os.path.join(PROC, "lulc_4class.tif"), shape, tf, crs, Resampling.nearest, fill=0)
dem = to_grid(os.path.join(PROC, "dem_clipped_utm.tif"), shape, tf, crs, Resampling.bilinear)
pop30 = to_grid(os.path.join(PROC, "population_clipped_utm.tif"), shape, tf, crs, Resampling.bilinear)
pop30[pop30 < 0] = np.nan

water = lulc == 3
built = (lulc == 1)
data_ok = np.isfinite(lst) & np.isfinite(ndvi)
land = data_ok & ~water & (lulc > 0)
inside = lulc > 0                      # inside Dhaka boundary (LULC fill is 0)

rows, cols = np.indices(shape)
X = tf.c + (cols + 0.5) * tf.a
Y = tf.f + (rows + 0.5) * tf.e
cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*CENTER_WGS84)
dist_center = np.hypot(X - cx, Y - cy) / 1000.0     # km

# ───────────────────────────── features ──────────────────────────────────
print("Building neighbourhood features...")
valid_nb = inside
feat = {
    "ndvi": ndvi,
    "built_frac_330m": local_mean(built.astype("float32"), valid_nb, 11),
    "built_frac_990m": local_mean(built.astype("float32"), valid_nb, 33),
    "ndvi_mean_990m": local_mean(np.nan_to_num(ndvi), valid_nb & np.isfinite(ndvi), 33),
    "water_frac_990m": local_mean(water.astype("float32"), valid_nb, 33),
    "dist_water_m": ndi.distance_transform_edt(~water) * PX,
    "elevation_m": dem,
    "pop_density_km2": pop30 / 0.01,       # WorldPop persons per 100 m cell -> per km2
    "dist_center_km": dist_center,
}
built_1km = local_mean(built.astype("float32"), valid_nb, 67)

# ───────────────────────────── A) SUHI ───────────────────────────────────
print("A) SUHI intensity and urban-rural gradient...")
urban = land & built & (feat["built_frac_990m"] >= 0.5)
rural = land & (lulc == 2) & (built_1km <= 0.03)
print(f"   urban ref pixels: {urban.sum():,}   rural ref pixels: {rural.sum():,}")

blk = (rows // BLOCK_PX) * 10_000 + (cols // BLOCK_PX)
ub, inv = np.unique(blk[land], return_inverse=True)
def per_block(mask):
    s = np.bincount(inv, weights=np.where(mask[land], lst[land], 0), minlength=len(ub))
    n = np.bincount(inv, weights=mask[land].astype(float), minlength=len(ub))
    return s, n
us, un = per_block(urban); rs, rn = per_block(rural)
suhi = us.sum() / un.sum() - rs.sum() / rn.sum()
boots = []
for _ in range(2000):
    k = RNG.integers(0, len(ub), len(ub))
    if un[k].sum() > 0 and rn[k].sum() > 0:
        boots.append(us[k].sum() / un[k].sum() - rs[k].sum() / rn[k].sum())
lo, hi = np.percentile(boots, [2.5, 97.5])
suhi_df = pd.DataFrame([{
    "urban_mean_lst": us.sum() / un.sum(), "rural_mean_lst": rs.sum() / rn.sum(),
    "suhi_intensity_C": suhi, "ci95_low": lo, "ci95_high": hi,
    "n_urban_px": int(un.sum()), "n_rural_px": int(rn.sum()),
    "ci_method": "spatial block bootstrap, 3 km blocks, 2000 resamples"}])
suhi_df.to_csv(os.path.join(OUT, "suhi_summary.csv"), index=False)
print(f"   SUHI = {suhi:.2f} C  (95% CI {lo:.2f} to {hi:.2f})")

ring_edges = np.arange(0, np.nanmax(dist_center[inside]) + 2, 2)
ring_id = np.digitize(dist_center, ring_edges) - 1
ring_rows = []
for r in range(len(ring_edges) - 1):
    m = land & (ring_id == r)
    if m.sum() > 500:
        ring_rows.append({"ring_km_from": ring_edges[r], "ring_km_to": ring_edges[r + 1],
                          "mean_lst": lst[m].mean(), "mean_ndvi": ndvi[m].mean(),
                          "mean_built_frac": built[m].mean(), "n_px": int(m.sum())})
ring_df = pd.DataFrame(ring_rows)
ring_df.to_csv(os.path.join(OUT, "gradient_by_distance.csv"), index=False)

# ───────────────────────────── B) wards ──────────────────────────────────
print("B) Ward-level hotspots (Moran's I, Gi*)...")
bnd = gpd.read_file(os.path.join(PROC, "dhaka_boundary_utm.shp"))
adm = gpd.read_file(ADMIN, bbox=tuple(bnd.to_crs(4326).total_bounds)).to_crs(crs)
wards = gpd.overlay(adm, bnd[["geometry"]], how="intersection", keep_geom_type=True).reset_index(drop=True)
wards["ward_id"] = np.arange(1, len(wards) + 1)
wards["name"] = wards.get("NAME_4", wards["ward_id"].astype(str))
wards["thana"] = wards.get("NAME_3", "")
wid = rasterize(((g, i) for g, i in zip(wards.geometry, wards.ward_id)), out_shape=shape,
                transform=tf, fill=0, dtype="int32")
nW = len(wards) + 1
def zmean(vals, mask):
    s = np.bincount(wid[mask], weights=vals[mask], minlength=nW)
    n = np.bincount(wid[mask], minlength=nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, n
w_lst, w_n = zmean(lst, land)
w_ndvi, _ = zmean(ndvi, land)
w_built, _ = zmean(built.astype("float32"), land)
rural_ref = lst[rural].mean()
wards["mean_lst"] = w_lst[1:]; wards["land_px"] = w_n[1:]
wards["mean_ndvi"] = w_ndvi[1:]; wards["built_frac"] = w_built[1:]
wards["suhi_C"] = wards["mean_lst"] - rural_ref
ok = wards["land_px"] >= 100                    # drop sliver wards (too few pixels)
wd = wards[ok].reset_index(drop=True)
try:
    w = Queen.from_dataframe(wd, use_index=False)
    if w.islands:
        w = KNN.from_dataframe(wd, k=5)
except Exception:
    w = KNN.from_dataframe(wd, k=5)
w.transform = "r"
mi = Moran(wd["mean_lst"].values, w, permutations=999)
gi = G_Local(wd["mean_lst"].values, w, transform="r", star=True, permutations=999)
wd["gi_z"] = gi.Zs; wd["gi_p"] = gi.p_sim
wd["hotspot"] = np.where((wd.gi_z > 0) & (wd.gi_p < 0.05), "hot",
                  np.where((wd.gi_z < 0) & (wd.gi_p < 0.05), "cold", "ns"))
print(f"   wards used: {len(wd)}   Moran's I = {mi.I:.3f} (p = {mi.p_sim:.3f})")
print("   hotspot counts:", wd.hotspot.value_counts().to_dict())

# ───────────────────────────── D) exposure ───────────────────────────────
print("D) Heat exposure of population...")
land_lst = lst[land]
thr_sd = land_lst.mean() + land_lst.std()
thr_p90 = np.percentile(land_lst, 90)
exp_rows = []
with rasterio.open(os.path.join(PROC, "population_clipped_utm.tif")) as ps:
    pop = ps.read(1).astype("float32")
    if ps.nodata is not None: pop[pop == ps.nodata] = 0
    pop[~np.isfinite(pop)] = 0; pop[pop < 0] = 0
    ptf, pshape, pcrs = ps.transform, pop.shape, ps.crs
    wid_p = rasterize(((g, i) for g, i in zip(wards.geometry, wards.ward_id)), out_shape=pshape,
                      transform=ptf, fill=0, dtype="int32")
    wards["population"] = np.bincount(wid_p.ravel(), weights=pop.ravel(), minlength=nW)[1:]
    for label, thr in [("LST >= mean+1SD", thr_sd), ("LST >= 90th percentile", thr_p90)]:
        hot = (land & (lst >= thr)).astype("float32")
        frac = np.zeros(pshape, dtype="float32")
        reproject(hot, frac, src_transform=tf, src_crs=crs, dst_transform=ptf, dst_crs=pcrs,
                  resampling=Resampling.average)
        exposed = pop * frac
        col = "exposed_" + ("sd" if "SD" in label else "p90")
        wards[col] = np.bincount(wid_p.ravel(), weights=exposed.ravel(), minlength=nW)[1:]
        exp_rows.append({"hot_definition": label, "threshold_C": round(float(thr), 2),
                         "population_total": float(pop.sum()), "population_exposed": float(exposed.sum()),
                         "share_exposed_pct": float(exposed.sum() / pop.sum() * 100)})
exp_df = pd.DataFrame(exp_rows)
exp_df.to_csv(os.path.join(OUT, "exposure_summary.csv"), index=False)
print(exp_df.to_string(index=False))

wd = wd.merge(wards[["ward_id", "population", "exposed_sd", "exposed_p90"]], on="ward_id")
wd.drop(columns="geometry").to_csv(os.path.join(OUT, "ward_hotspots.csv"), index=False)

# ───────────────────────────── C) Random Forest ──────────────────────────
print("C) Random Forest with spatial block CV...")
names = list(feat.keys())
stack = np.stack([feat[k] for k in names], axis=-1)
cand = land & np.all(np.isfinite(stack), axis=-1)
idx = np.flatnonzero(cand.ravel())
take = RNG.choice(idx, size=min(30_000, len(idx)), replace=False)
Xs = stack.reshape(-1, len(names))[take]
ys = lst.ravel()[take]
groups = blk.ravel()[take]
def make_rf():
    return RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=-1, random_state=42)
def cv(splitter, **kw):
    r2s, rmses, imps = [], [], []
    for tr, te in splitter.split(Xs, ys, **kw):
        m = make_rf().fit(Xs[tr], ys[tr])
        p = m.predict(Xs[te])
        r2s.append(r2_score(ys[te], p)); rmses.append(mean_squared_error(ys[te], p) ** 0.5)
        sub = RNG.choice(te, size=min(4000, len(te)), replace=False)
        pi = permutation_importance(m, Xs[sub], ys[sub], n_repeats=3, random_state=42, n_jobs=-1)
        imps.append(pi.importances_mean)
    return np.array(r2s), np.array(rmses), np.mean(imps, axis=0)
r2_sp, rmse_sp, imp_sp = cv(GroupKFold(n_splits=5), groups=groups)
r2_rd, rmse_rd, _ = cv(KFold(n_splits=5, shuffle=True, random_state=42))
cv_df = pd.DataFrame([
    {"cv": "spatial block (3 km)", "r2_mean": r2_sp.mean(), "r2_sd": r2_sp.std(), "rmse_C": rmse_sp.mean()},
    {"cv": "random k-fold", "r2_mean": r2_rd.mean(), "r2_sd": r2_rd.std(), "rmse_C": rmse_rd.mean()}])
cv_df.to_csv(os.path.join(OUT, "rf_cv_summary.csv"), index=False)
imp_df = pd.DataFrame({"feature": names, "perm_importance_R2_drop": imp_sp}).sort_values(
    "perm_importance_R2_drop", ascending=False)
imp_df.to_csv(os.path.join(OUT, "rf_feature_importance.csv"), index=False)
print(cv_df.to_string(index=False)); print(imp_df.to_string(index=False))

# ───────────────────────────── figures ───────────────────────────────────
print("Figures...")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
mid = (ring_df.ring_km_from + ring_df.ring_km_to) / 2
ax[0].plot(mid, ring_df.mean_lst, marker="o", color="#c0392b")
ax[0].set_xlabel("Distance from city centre (km)"); ax[0].set_ylabel("Mean LST, land only (°C)")
ax[0].set_title("Urban-rural LST gradient")
ax[1].barh(["Rural reference", "Urban core"], [suhi_df.rural_mean_lst[0], suhi_df.urban_mean_lst[0]],
           color=["#2e7d32", "#c0392b"])
ax[1].set_xlim(min(suhi_df.rural_mean_lst[0], suhi_df.urban_mean_lst[0]) - 2, None)
ax[1].set_title(f"SUHI = {suhi:.1f} °C (95% CI {lo:.1f} to {hi:.1f})")
ax[1].set_xlabel("Mean LST (°C)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_suhi_gradient.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
d = imp_df.iloc[::-1]
ax[0].barh(d.feature, d.perm_importance_R2_drop, color="#34495e")
ax[0].set_xlabel("Permutation importance (drop in R² on held-out blocks)")
ax[0].set_title("What drives LST (Random Forest)")
ax[1].bar(cv_df.cv, cv_df.r2_mean, yerr=cv_df.r2_sd, color=["#34495e", "#b0b7bf"], capsize=4)
ax[1].set_ylabel("R²"); ax[1].set_title("Spatial vs random CV")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rf_drivers.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
wd_g = gpd.GeoDataFrame(wd, geometry=wards.set_index("ward_id").loc[wd.ward_id, "geometry"].values, crs=crs)
wd_g.plot(column="mean_lst", cmap="YlOrRd", legend=True, ax=ax[0], edgecolor="white", linewidth=0.2,
          legend_kwds={"label": "Ward mean LST (°C)", "shrink": 0.7})
ax[0].set_title(f"Ward mean LST (Moran's I = {mi.I:.2f}, p = {mi.p_sim:.3f})")
colors = {"hot": "#b2182b", "cold": "#2166ac", "ns": "#e0e0e0"}
wd_g.plot(color=wd_g.hotspot.map(colors), ax=ax[1], edgecolor="white", linewidth=0.2)
ax[1].set_title("Getis-Ord Gi* hotspots (p < 0.05)")
for a in ax: a.set_axis_off()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ward_hotspots.png"), dpi=200); plt.close(fig)

summary = {"suhi_C": suhi, "suhi_ci95": [lo, hi], "morans_I": mi.I, "morans_p": mi.p_sim,
           "rf_spatial_r2": r2_sp.mean(), "rf_random_r2": r2_rd.mean(),
           "rural_ref_mean_lst": rural_ref, "n_wards": len(wd)}
json.dump(summary, open(os.path.join(OUT, "deep_summary.json"), "w"), indent=2, default=float)
print("Done.", summary)
