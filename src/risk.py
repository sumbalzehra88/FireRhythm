import pandas as pd, numpy as np, lightgbm as lgb, json

OUT = "data/processed"
h = pd.read_parquet(f"{OUT}/harmonized.parquet")
fp = pd.read_parquet(f"{OUT}/fingerprints.parquet").set_index("cell_id")
hs = pd.read_parquet(f"{OUT}/hotspots_clean.parquet")
CROP = fp.index[fp.regime.str.startswith("Rice") & (fp.mean_annual >= 50)]

# ---------- monthly panel with expanding (no-leak) median/MAD ----------
P = (h[h.cell_id.isin(fp.index)][["cell_id", "month", "h_count"]]
     .sort_values(["cell_id", "month"]).reset_index(drop=True))
P["y"], P["cm"] = np.log1p(P.h_count), P.month.dt.month
med, mad, v = np.full(len(P), np.nan), np.full(len(P), np.nan), P.h_count.values
for idx in P.groupby(["cell_id", "cm"]).indices.values():
    for i in range(1, len(idx)):
        past = v[idx[:i]]; m = np.median(past)
        med[idx[i]], mad[idx[i]] = m, np.median(np.abs(past - m))
P["med"], P["mad"] = med, mad
P["z"] = (P.h_count - P.med) / (1.4826 * np.maximum(P.mad, 1))
g = P.groupby("cell_id")
for k in [1, 2, 11]: P[f"lag{k}"] = g.y.shift(k)          # lag11 = same month last year
P["target"], P["tmonth"] = g.y.shift(-1), g.month.shift(-1)
P["med_next"] = g.med.shift(-1); P["med_next_log"] = np.log1p(P.med_next)
nm = P.cm % 12 + 1
P["msin"], P["mcos"] = np.sin(2*np.pi*nm/12), np.cos(2*np.pi*nm/12)
P = P.join(fp[["lat", "lon", "A12", "A6", "cluster"]], on="cell_id")
FEATS = ["y", "lag1", "lag2", "lag11", "z", "msin", "mcos", "lat", "lon",
         "A12", "A6", "cluster", "med_next_log"]

# ---------- rice-season timing (VIIRS daily, days since Sep 1) ----------
SEP = lambda y: pd.Timestamp(year=int(y), month=9, day=1)
r = hs[(hs.sensor == "VIIRS") & hs.acq_date.dt.month.isin([9, 10, 11, 12])].copy()
r["year"] = r.acq_date.dt.year
r["sday"] = (r.acq_date - pd.to_datetime(r.year.astype(str) + "-09-01")).dt.days
daily = r.groupby(["cell_id", "year", "sday"]).size()

def timing(as_of):
    yr = as_of.year if as_of.month >= 9 else as_of.year - 1
    cut, rows = (as_of - SEP(yr)).days, []
    for c, dc in daily.groupby(level=0):
        dc = dc.droplevel(0); tot = dc.groupby(level=0).sum(); past = tot[tot.index < yr]
        if len(past) < 3: continue
        T = max(3, 0.1 * past.median())                      # 10% of a typical season
        def reach(y, limit=None):
            if y not in tot.index: return np.nan
            s = dc.loc[y].sort_index()
            if limit is not None: s = s[s.index < limit]
            hit = s.cumsum()[lambda x: x >= T]
            return hit.index[0] if len(hit) else np.nan
        tp = pd.Series({y: reach(y) for y in past.index}).dropna()
        if len(tp) < 3: continue
        tmed = tp.median(); tcur = reach(yr, cut) if cut > 0 else np.nan
        drift = tcur - tmed if not np.isnan(tcur) else (cut - tmed if cut > tmed else 0.0)
        rows.append(dict(cell_id=c, t_med=tmed, t_cur=tcur, drift=drift))
    return pd.DataFrame(rows).set_index("cell_id"), yr

# ---------- full scoring as of a date ----------
def score(as_of):
    last = (as_of.to_period("M") - 1).to_timestamp()
    D = P[P.month <= last]
    L = D.dropna(subset=["target", "lag11"]); L = L[L.tmonth <= last]
    split = last - pd.DateOffset(months=24)
    tr, va = L[L.tmonth <= split], L[L.tmonth > split]
    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                              min_child_samples=50, subsample=.8, subsample_freq=1,
                              colsample_bytree=.8, verbose=-1)
    model.fit(tr[FEATS], tr.target, eval_set=[(va[FEATS], va.target)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    mae = lambda a, b: float(np.mean(np.abs(np.expm1(a) - np.expm1(b))))
    fm = {"lgbm_mae": mae(va.target, model.predict(va[FEATS])),
          "seasonal_naive_mae": mae(va.target, va.lag11)}

    cur = D[D.month == last].set_index("cell_id").copy()
    cur["forecast"] = np.expm1(model.predict(cur[FEATS]))
    contrib = model.predict(cur[FEATS], pred_contrib=True)[:, :-1]     # SHAP values
    cur["top_feature"] = np.array(FEATS)[np.abs(contrib).argmax(1)]
    tm, yr = timing(as_of)
    R = cur[["lat", "lon", "h_count", "med", "z", "forecast", "med_next", "top_feature"]] \
        .join(tm).join(fp[["regime"]])
    R["drift"] = R.drift.fillna(0)
    R["c_anom"] = R.z.clip(0, 5).fillna(0)
    R["c_early"] = (-R.drift).clip(0, 30)
    R["c_exceed"] = (R.forecast / np.maximum(R.med_next.fillna(0), 1)).clip(0, 5)
    for c in ["c_anom", "c_early", "c_exceed"]:
        rng = R[c].max() - R[c].min()
        R[c + "_n"] = (R[c] - R[c].min()) / rng if rng > 0 else 0.0
    R["score"] = .4*R.c_anom_n + .3*R.c_early_n + .3*R.c_exceed_n
    R["risk"] = np.select([R.score >= .6, R.score >= .3], ["High", "Medium"], "Low")
    def explain(x):
        k = int(np.argmax([.4*x.c_anom_n, .3*x.c_early_n, .3*x.c_exceed_n]))
        return [f"{x.h_count:.0f} hotspots last month (z={x.z:.1f} vs usual)",
                f"Rice-burning onset {abs(x.drift):.0f} days early",
                f"Next month forecast {x.forecast:.0f} = {x.c_exceed:.1f}x usual"][k]
    R["explanation"] = R.apply(explain, axis=1)
    R["pred_onset"] = SEP(yr) + pd.to_timedelta((R.t_med + R.drift).round(), unit="D")
    Dp = R[R.index.isin(CROP)].copy()
    Dp["priority"] = Dp.forecast * (1 + Dp.c_early / 30)
    Dp = Dp.sort_values("priority", ascending=False); Dp["rank"] = range(1, len(Dp) + 1)
    return R, Dp, fm

# ---------- run for today ----------
R, Dp, fm = score(pd.Timestamp.today().normalize())
R.reset_index().to_parquet(f"{OUT}/risk.parquet")
cols = ["rank", "cell_id", "lat", "lon", "regime", "pred_onset", "forecast", "risk", "explanation"]
Dp.reset_index()[cols].to_parquet(f"{OUT}/deployment_priority.parquet")
Dp.reset_index()[cols].to_csv(f"{OUT}/deployment_priority.csv", index=False)
print("Forecast (validation, last 24 mo):", fm)
print(R.risk.value_counts()); print(Dp.reset_index()[cols].head(10).to_string())

# ---------- back-test: pretend it is 2025-10-01 ----------
AS = pd.Timestamp("2025-10-01")
Rb, Db, fmb = score(AS)
actual, _ = timing(pd.Timestamp("2026-01-01"))                 # full 2025 season
cmp = Db[["t_med", "drift"]].join(actual[["t_cur"]]).dropna()
on = {"cells": len(cmp),
      "onset_mae_days_model": float((cmp.t_med + cmp.drift - cmp.t_cur).abs().mean()),
      "onset_mae_days_climatology": float((cmp.t_med - cmp.t_cur).abs().mean())}
p90 = P[(P.cm == 10) & (P.month < AS)].groupby("cell_id").h_count.quantile(.9)
act = P[P.month == AS].set_index("cell_id").h_count
ext = (act > p90).reindex(Rb.index).fillna(False)
def pr(mask):
    tp = (mask & ext).sum()
    return {"precision": float(tp / max(mask.sum(), 1)), "recall": float(tp / max(ext.sum(), 1))}
bt = {"as_of": str(AS.date()), "forecast": fmb, "onset": on,
      "high": pr(Rb.risk == "High"), "high_or_medium": pr(Rb.risk != "Low")}
open("docs/backtest.md", "w").write("# Back-test (as of 2025-10-01)\n\n```json\n"
                                    + json.dumps(bt, indent=2, default=str) + "\n```\n")
print(json.dumps(bt, indent=2, default=str))