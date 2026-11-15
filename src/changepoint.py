import pandas as pd, numpy as np, ruptures as rpt, matplotlib.pyplot as plt
from scipy import stats

h = pd.read_parquet("data/processed/harmonized.parquet")
reg = h.groupby("month")[["modis", "viirs", "h_count"]].sum()
SW = pd.Timestamp("2012-02-01")
reg["naive"] = np.where(reg.index < SW, reg.modis, reg.viirs)
PEN = 10   # tune 5-20 so the naive series shows 1-3 breaks

def deseason(s):
    x = np.log1p(s)
    return x - x.groupby(x.index.month).transform("mean")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, col, name in zip(axes, ["naive", "h_count"], ["Naive splice", "FireRhythm harmonized"]):
    x = deseason(reg[col])
    bk = rpt.Pelt(model="rbf", min_size=12).fit(x.values.reshape(-1, 1)).predict(pen=PEN)[:-1]
    pre = x[(x.index < SW) & (x.index >= SW - pd.DateOffset(months=24))]
    post = x[(x.index >= SW) & (x.index < SW + pd.DateOffset(months=24))]
    t, p = stats.ttest_ind(pre, post, equal_var=False)
    dates = [str(x.index[b].date()) for b in bk]
    print(f"{name}: breaks={dates} | jump={post.mean()-pre.mean():.2f} | p={p:.3g}")
    ax.plot(x.index, x.values)
    for b in bk: ax.axvline(x.index[b], color="red", ls="--")
    ax.axvline(SW, color="black", lw=3, alpha=.3)
    ax.set_title(f"{name} | breaks: {dates} | 2012 jump p={p:.3g}")
plt.tight_layout(); plt.savefig("docs/figures/changepoint.png", dpi=150); plt.show()