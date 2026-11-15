import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt

RAW, OUT, RES = "data/raw/api", "data/processed", 0.25
os.makedirs(OUT, exist_ok=True); os.makedirs("docs/figures", exist_ok=True)

def load(prefix, name):
    read = lambda p: pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()
    sp, nrt = read(f"{RAW}/{prefix}_SP.csv"), read(f"{RAW}/{prefix}_NRT.csv")
    for d in (sp, nrt):
        if len(d): d["acq_date"] = pd.to_datetime(d["acq_date"])
    if len(sp) and len(nrt):
        nrt = nrt[nrt.acq_date > sp.acq_date.max()]   # NRT only after SP ends
    df = pd.concat([sp, nrt], ignore_index=True); df["sensor"] = name
    return df

def clean(df):
    if "type" in df: df = df[df["type"].isna() | (df["type"] == 0)]
    num = pd.to_numeric(df["confidence"], errors="coerce")
    letter = df["confidence"].astype(str).str.lower().str[0].isin(["n", "h"])
    df = df[np.where(num.notna(), num >= 30, letter)]
    df = df[df.longitude.between(69, 78) & df.latitude.between(27, 33)]
    return df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time", "sensor"])

df = pd.concat([clean(load("MODIS", "MODIS")), clean(load("VIIRS_SNPP", "VIIRS"))], ignore_index=True)
df["row"] = np.floor(df.latitude / RES).astype(int)
df["col"] = np.floor(df.longitude / RES).astype(int)
df["cell_id"] = df.row.astype(str) + "_" + df.col.astype(str)
df["month"] = df.acq_date.dt.to_period("M").dt.to_timestamp()
df["is_day"] = (df.daynight == "D").astype(int)
for col in df.select_dtypes("object").columns: df[col] = df[col].astype(str)
df.to_parquet(f"{OUT}/hotspots_clean.parquet")

agg = df.groupby(["cell_id", "month", "sensor"]).agg(
    count=("frp", "size"), sum_frp=("frp", "sum"), mean_frp=("frp", "mean"),
    day_fraction=("is_day", "mean"), n_sat=("satellite", "nunique")).reset_index()

# full grid: every cell x every month, zeros filled
last = (pd.Timestamp.today().to_period("M") - 1).to_timestamp()
cells = df.drop_duplicates("cell_id")[["cell_id", "row", "col"]].copy()
cells["lat"], cells["lon"] = (cells.row + 0.5) * RES, (cells.col + 0.5) * RES
frames = []
for s, start in [("MODIS", "2000-11-01"), ("VIIRS", "2012-02-01")]:
    idx = pd.MultiIndex.from_product([cells.cell_id, pd.date_range(start, last, freq="MS")],
                                     names=["cell_id", "month"]).to_frame(index=False)
    idx["sensor"] = s; frames.append(idx)
full = pd.concat(frames).merge(agg, how="left", on=["cell_id", "month", "sensor"])
full[["count", "sum_frp"]] = full[["count", "sum_frp"]].fillna(0)
full = full.merge(cells[["cell_id", "lat", "lon"]], on="cell_id")
full.to_parquet(f"{OUT}/grid_monthly.parquet")

print("hotspots:", df.groupby("sensor").size().to_dict())
print("cells:", len(cells), "| grid rows:", len(full))
tot = full.groupby(["month", "sensor"])["count"].sum().unstack()
ax = tot.plot(figsize=(14, 4), logy=True, title="Monthly hotspots, Punjab region")
plt.tight_layout(); plt.savefig("docs/figures/raw_monthly.png", dpi=150); plt.show()