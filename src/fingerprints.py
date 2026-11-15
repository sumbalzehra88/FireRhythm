import pandas as pd, numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.signal import get_window
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

OUT = "data/processed"
h = pd.read_parquet(f"{OUT}/harmonized.parquet")
X = h.pivot(index="month", columns="cell_id", values="h_count").fillna(0).sort_index()
meta = h.drop_duplicates("cell_id").set_index("cell_id")[["lat", "lon"]]
keep = X.columns[X.sum() >= 50]
N = len(X); f = np.fft.rfftfreq(N, d=1); w = get_window("hann", N)
i12, i6 = np.argmin(abs(f - 1/12)), np.argmin(abs(f - 1/6))

# ---- 1. STL + FFT fingerprint per cell ----
rows, stl_parts = [], []
for c in keep:
    y = np.log1p(X[c])
    r = STL(y, period=12, robust=True).fit()
    stl_parts.append(pd.DataFrame({"cell_id": c, "month": X.index, "trend": r.trend.values,
                                   "seasonal": r.seasonal.values, "resid": r.resid.values}))
    z = (y - r.trend).values * w                      # detrend + Hann window
    P = np.abs(np.fft.rfft(z)) ** 2; P[0] = 0
    amp = 2 * np.sqrt(P) / w.sum()
    p = P[1:] / P[1:].sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))   # spectral entropy 0-1
    prof = X[c].groupby(X.index.month).mean().reindex(range(1, 13), fill_value=0)
    prof = prof / prof.sum()
    rows.append(dict(cell_id=c, A12=amp[i12], A6=amp[i6], dom_period=1 / f[np.argmax(P)],
                     entropy=ent, peak_month=int(prof.idxmax()),
                     spring_share=prof[[4, 5, 6]].sum(), autumn_share=prof[[9, 10, 11, 12]].sum(),
                     mean_annual=X[c].sum() / (N / 12), **{f"p{m}": prof[m] for m in range(1, 13)}))
fp = pd.DataFrame(rows).set_index("cell_id").join(meta)
pd.concat(stl_parts).to_parquet(f"{OUT}/stl.parquet")

# ---- 2. Season timing per cell-year from daily data ----
hs = pd.read_parquet(f"{OUT}/hotspots_clean.parquet")
sw = pd.Timestamp("2012-02-01")
hs = hs[((hs.sensor == "MODIS") & (hs.acq_date < sw)) | ((hs.sensor == "VIIRS") & (hs.acq_date >= sw))].copy()
mon = hs.acq_date.dt.month
hs["season"] = np.select([mon.isin([4, 5, 6]), mon.isin([9, 10, 11, 12])], ["wheat", "rice"], "none")
hs = hs[hs.season != "none"]
hs["year"], hs["doy"] = hs.acq_date.dt.year, hs.acq_date.dt.day_of_year

def timing(g):
    d = np.sort(g.doy.values); n = len(d)
    return pd.Series({"n": n, "onset_doy": d[int(0.1*(n-1))], "mid_doy": d[int(0.5*(n-1))],
                      "end_doy": d[int(0.9*(n-1))]})
ons = hs.groupby(["cell_id", "year", "season"]).apply(timing).reset_index()
ons = ons[ons.n >= 10]
ons.to_parquet(f"{OUT}/onsets.parquet")
med = ons.groupby(["cell_id", "season"])[["onset_doy", "mid_doy"]].median().unstack()
med.columns = [f"med_{a}_{b}" for a, b in med.columns]
fp = fp.join(med)

# ---- 3. Regime clustering ----
F = pd.DataFrame({"logA12": np.log(fp.A12 + 1e-6), "ratio6": fp.A6 / (fp.A12 + 1e-6),
                  "pk_sin": np.sin(2*np.pi*fp.peak_month/12), "pk_cos": np.cos(2*np.pi*fp.peak_month/12),
                  "entropy": fp.entropy, "logmean": np.log1p(fp.mean_annual),
                  "spring": fp.spring_share, "autumn": fp.autumn_share})
Z = StandardScaler().fit_transform(F)
best = max(range(3, 7), key=lambda k: silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)))
fp["cluster"] = KMeans(best, n_init=10, random_state=0).fit_predict(Z)
cm = fp.groupby("cluster")[["spring_share", "autumn_share", "entropy", "mean_annual"]].mean()

def name(r):
    if r.spring_share >= 0.15 and r.autumn_share >= 0.3: return "Crop-residue (2-season)"
    if max(r.spring_share, r.autumn_share) >= 0.6: return "Single-season"
    return "Irregular / other"
cm["regime"] = cm.apply(name, axis=1)
cm["rank"] = cm.groupby("regime").mean_annual.rank(ascending=False).astype(int)
cm["regime"] = np.where(cm.groupby("regime").regime.transform("size") > 1,
                        cm.regime + np.where(cm["rank"] == 1, " - high", " - low"), cm.regime)
fp["regime"] = fp.cluster.map(cm.regime)
fp["regime"] = fp.regime.replace({
    "Crop-residue (2-season) - high": "Rice–wheat residue (Oct–Nov + May)",
    "Crop-residue (2-season) - low": "Mixed winter–spring burning",
    "Single-season": "Spring single-season (wheat / foothill fires)",
    "Irregular / other": "Irregular / low activity"})
fp.reset_index().to_parquet(f"{OUT}/fingerprints.parquet")
print("k =", best); print(cm.round(2)); print(fp.regime.value_counts())

# ---- 4. Figures ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for i, (rg, g) in enumerate(fp.groupby("regime")):
    col = plt.cm.viridis(i / max(1, fp.regime.nunique() - 1))
    ax[0].scatter(g.lon, g.lat, marker="s", s=60, color=col, label=rg)
    ax[1].plot(range(1, 13), g[[f"p{m}" for m in range(1, 13)]].mean(), "-o", color=col, label=rg)
ax[0].set_title("Fire regimes (from signal fingerprints)"); ax[0].legend(fontsize=8)
ax[1].set_title("Mean seasonal profile"); ax[1].set_xticks(range(1, 13)); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("docs/figures/regimes.png", dpi=150); plt.show()