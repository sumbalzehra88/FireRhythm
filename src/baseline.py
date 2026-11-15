import pandas as pd, numpy as np, json

g = pd.read_parquet("data/processed/grid_monthly.parquet")
w = g.pivot_table(index=["cell_id", "month"], columns="sensor", values="count").dropna().reset_index()
w["m"] = w.month.dt.month
train, test = w[w.month < "2021-01-01"], w[w.month >= "2021-01-01"].copy()
ratio = train.groupby("m").apply(lambda d: d.VIIRS.sum() / max(d.MODIS.sum(), 1))
test["pred"] = test.MODIS * test.m.map(ratio)

def metrics(y, p):
    return dict(rmse=float(np.sqrt(((y - p) ** 2).mean())), mae=float((y - p).abs().mean()),
                r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))

reg = test.groupby("month")[["VIIRS", "pred"]].sum()
out = {"cell": metrics(test.VIIRS, test.pred), "regional": metrics(reg.VIIRS, reg.pred),
       "ratio_by_month": ratio.round(2).to_dict()}
json.dump(out, open("data/processed/metrics_baseline.json", "w"), indent=2)
print(json.dumps(out, indent=2))