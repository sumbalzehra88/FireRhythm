import pandas as pd, numpy as np, json, lightgbm as lgb
import matplotlib.pyplot as plt

g = pd.read_parquet("data/processed/grid_monthly.parquet")
m = g[g.sensor == "MODIS"].set_index(["cell_id", "month"])
v = g[g.sensor == "VIIRS"].set_index(["cell_id", "month"])["count"].rename("viirs")
d = (m[["lat", "lon", "count", "sum_frp", "mean_frp", "day_fraction", "n_sat"]]
     .rename(columns={"count": "modis"}).join(v, how="left").reset_index()
     .sort_values(["cell_id", "month"]))

d["log_modis"] = np.log1p(d.modis)
d["log_modis_lag1"] = d.groupby("cell_id").log_modis.shift(1)
d["log_sum_frp"] = np.log1p(d.sum_frp)
mo = d.month.dt.month
d["month_sin"], d["month_cos"] = np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12)
d[["mean_frp", "n_sat"]] = d[["mean_frp", "n_sat"]].fillna(0)
d["day_fraction"] = d.day_fraction.fillna(-1)
FEATS = ["log_modis", "log_modis_lag1", "log_sum_frp", "mean_frp", "day_fraction",
         "n_sat", "month_sin", "month_cos", "lat", "lon"]

ov = d[d.viirs.notna()]
tr = ov[ov.month < "2021-01-01"]
va = ov[(ov.month >= "2021-01-01") & (ov.month < "2023-01-01")]
te = ov[ov.month >= "2023-01-01"].copy()

model = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=31,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1)
model.fit(tr[FEATS], np.log1p(tr.viirs), eval_set=[(va[FEATS], np.log1p(va.viirs))],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
model.booster_.save_model("data/processed/harmonizer.txt")

# compare with ratio baseline on the same test years
ratio = tr.groupby(tr.month.dt.month).apply(lambda x: x.viirs.sum() / max(x.modis.sum(), 1))
te["base"] = te.modis * te.month.dt.month.map(ratio)
te["lgbm"] = np.clip(np.expm1(model.predict(te[FEATS])), 0, None)

def metrics(y, p):
    return dict(rmse=float(np.sqrt(((y-p)**2).mean())), mae=float((y-p).abs().mean()),
                r2=float(1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum()))
reg = te.groupby("month")[["viirs", "base", "lgbm"]].sum()
res = {lvl: {k: metrics(df_.viirs, df_[k]) for k in ["base", "lgbm"]}
       for lvl, df_ in [("cell", te), ("regional", reg)]}
json.dump(res, open("data/processed/metrics_harmonize.json", "w"), indent=2)
print(pd.DataFrame({(l, k): v for l, kv in res.items() for k, v in kv.items()}).round(3))

# harmonized series: modeled before VIIRS, observed after
d["pred"] = np.clip(np.expm1(model.predict(d[FEATS])), 0, None)
early = d.month < "2012-02-01"
d["h_count"] = np.where(early, d.pred, d.viirs)
d["source"] = np.where(early, "modeled", "observed")
d[["cell_id", "month", "lat", "lon", "h_count", "source", "modis", "viirs"]] \
    .to_parquet("data/processed/harmonized.parquet")

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
tot = d.groupby("month")[["modis", "viirs", "h_count"]].sum().replace(0, np.nan)
tot.plot(ax=ax[0], logy=True, title="Regional monthly: raw vs harmonized")
ax[1].scatter(np.log1p(te.viirs), np.log1p(te.lgbm), s=2, alpha=.3)
ax[1].plot([0, 6], [0, 6], "r--"); ax[1].set_title("Test 2023+: log1p actual vs predicted")
plt.tight_layout(); plt.savefig("docs/figures/harmonization.png", dpi=150); plt.show()