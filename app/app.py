import os, io, time
import numpy as np, pandas as pd, requests
import streamlit as st, folium, pydeck as pdk
import plotly.graph_objects as go
from folium.plugins import Draw
from streamlit_folium import st_folium
from matplotlib.path import Path
from matplotlib import colormaps

T0 = time.perf_counter()
st.set_page_config(page_title="FireRhythm", page_icon="🔥", layout="wide")
D, RES = "data/processed", 0.25
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
INF = colormaps["inferno"]
INF6 = [[int(c * 255) for c in INF(x)[:3]] for x in np.linspace(.15, 1, 6)]
RISK_HEX = {"High": "#fca50a", "Medium": "#dd513a", "Low": "#57106e"}
RISK_RGB = {k: [int(v[i:i+2], 16) for i in (1, 3, 5)] for k, v in RISK_HEX.items()}
REG_PAL = ["#fca50a", "#dd513a", "#932667", "#57106e", "#fcffa4"]
ESRI_DARK = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"

# ---------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame {font-family:'Inter',sans-serif;}
footer, #MainMenu {visibility:hidden;}
.block-container {padding-top:1.2rem; max-width:1500px;}
.hero {padding:22px 26px; border-radius:18px; margin-bottom:14px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(252,165,10,.18), transparent 60%),
              radial-gradient(900px 260px at 100% 0%, rgba(147,38,103,.25), transparent 60%), #10131a;
  border:1px solid #232836;}
.hero h1 {font-family:'Space Grotesk',sans-serif; font-size:2.7rem; margin:0; line-height:1.1;
  background:linear-gradient(90deg,#fcffa4,#fca50a 30%,#dd513a 65%,#932667);
  -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero p {color:#a3a9b6; margin:.4rem 0 .7rem; max-width:900px;}
.pill {display:inline-block; padding:3px 11px; border:1px solid #2a2f3a; border-radius:999px;
  font-size:.74rem; color:#c9ced8; margin:0 6px 4px 0; background:#141822;}
.kgrid {display:grid; grid-template-columns:1fr 1fr; gap:12px;}
.kpi {background:linear-gradient(180deg,#151922,#10131a); border:1px solid #232836; border-radius:14px;
  padding:14px 16px; position:relative; overflow:hidden;}
.kpi:before {content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--a);}
.kpi .l {font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; color:#8b93a3;}
.kpi .v {font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; color:#f4f1ec; margin-top:2px;}
.kpi .s {font-size:.78rem; color:#a3a9b6;}
.up {color:#fca50a;} .down {color:#7fb3d5;}
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid #232836;}
.stTabs [data-baseweb="tab"] {background:#11141b; border-radius:10px 10px 0 0; padding:8px 16px;}
.stTabs [aria-selected="true"] {background:#1a1f2b; color:#fca50a !important;}
.live {display:inline-flex; align-items:center; gap:8px; color:#fca50a; font-weight:600; font-size:.85rem;}
.live:before {content:""; width:9px; height:9px; border-radius:50%; background:#ff4d2e;
  box-shadow:0 0 0 0 rgba(255,77,46,.7); animation:pulse 1.6s infinite;}
@keyframes pulse {to {box-shadow:0 0 0 12px rgba(255,77,46,0);}}
.legend {height:10px; border-radius:6px; background:linear-gradient(90deg,#57106e,#932667,#dd513a,#fca50a,#fcffa4);}
.lgl {display:flex; justify-content:space-between; font-size:.72rem; color:#8b93a3;}
.note {color:#8b93a3; font-size:.83rem;}
</style>""", unsafe_allow_html=True)

def card(label, value, sub="", accent="#fca50a"):
    return (f'<div class="kpi" style="--a:{accent}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')

def style(fig, h=380, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, title=title, font=dict(family="Inter, sans-serif", color="#c9ced8"),
                      margin=dict(l=10, r=10, t=50 if title else 10, b=10))
    return fig

def frp_rgb(frp):
    x = np.clip(np.log10(np.asarray(frp, float) + 1) / 2.5, .15, 1)
    return (INF(x)[:, :3] * 255).astype(int)

doy2str = lambda d: "—" if pd.isna(d) else (pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(d) - 1)).strftime("%b %d")

# ---------------------------------------------------------------- data
@st.cache_data
def load():
    h = pd.read_parquet(f"{D}/harmonized.parquet")
    fp = pd.read_parquet(f"{D}/fingerprints.parquet")
    risk = pd.read_parquet(f"{D}/risk.parquet")
    dep = pd.read_parquet(f"{D}/deployment_priority.parquet")
    hs = pd.read_parquet(f"{D}/hotspots_clean.parquet",
                         columns=["latitude", "longitude", "acq_date", "acq_time", "frp",
                                  "confidence", "satellite", "daynight", "sensor", "cell_id"])
    return h, fp, risk, dep, hs

@st.cache_data(ttl=900, show_spinner="Fetching live detections from NASA FIRMS…")
def live(bb, days):
    key = os.environ.get("FIRMS_KEY")
    if not key:
        try: key = st.secrets["FIRMS_KEY"]
        except Exception: key = None
    if not key:
        return pd.DataFrame(), "Add FIRMS_KEY to .streamlit/secrets.toml to enable live mode."
    area, frames = ",".join(f"{v:.3f}" for v in bb), []
    for s in ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]:
        try:
            r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{s}/{area}/{days}", timeout=30)
            if r.ok and r.text.startswith("latitude"):
                d = pd.read_csv(io.StringIO(r.text)); d["satellite"] = s.replace("_NRT", ""); frames.append(d)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(), "No live detections returned for this area (quiet period or API unavailable)."
    return pd.concat(frames, ignore_index=True), None

h, fp, risk, dep, hs = load()
cells = h.drop_duplicates("cell_id")[["cell_id", "lat", "lon"]]
RCOL = dict(zip(sorted(fp.regime.unique()), REG_PAL))
ymax = int(h.month.dt.year.max())

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### Area & period")
mode = st.sidebar.radio("Area of interest", ["Draw on map", "Enter coordinates"])
if mode == "Enter coordinates":
    c1, c2 = st.sidebar.columns(2)
    lat0 = c1.number_input("Min lat", 27.0, 33.0, 29.5, 0.25); lat1 = c2.number_input("Max lat", 27.0, 33.0, 31.5, 0.25)
    lon0 = c1.number_input("Min lon", 69.0, 78.0, 73.0, 0.25); lon1 = c2.number_input("Max lon", 69.0, 78.0, 76.0, 0.25)
y0, y1 = st.sidebar.slider("Years", 2001, ymax, (2001, ymax))
st.sidebar.markdown('<p class="note">Data: NASA FIRMS — MODIS C6.1 (1 km) & VIIRS S-NPP (375 m). '
                    'Pre-2012 values are modeled VIIRS-equivalents.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- hero
st.markdown(f"""<div class="hero"><h1>FireRhythm</h1>
<p>One consistent fire record from two satellite generations. Harmonized MODIS + VIIRS hotspots reveal the
rhythm of burning across the Punjab rice–wheat belt, flag unusual activity, and tell agriculture teams where
crop-residue burning will peak next.</p>
<span class="pill">NASA FIRMS</span><span class="pill">2000 – {ymax}</span><span class="pill">0.25° grid</span>
<span class="pill">LightGBM transfer function</span><span class="pill">STL + FFT fingerprints</span></div>""",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- selection map
cmap, ckpi = st.columns([1.55, 1])
with cmap:
    m = folium.Map(location=[30, 73.5], zoom_start=6, tiles=ESRI_DARK,
                   attr="Esri, HERE, Garmin, OpenStreetMap contributors")
    lr = folium.FeatureGroup("Risk (latest month)")
    for r in risk.itertuples():
        folium.Rectangle([[r.lat - RES/2, r.lon - RES/2], [r.lat + RES/2, r.lon + RES/2]], weight=0, fill=True,
                         fill_color=RISK_HEX[r.risk], fill_opacity=.55 if r.risk != "Low" else .18,
                         tooltip=f"<b>{r.risk}</b> · {r.explanation}").add_to(lr)
    lr.add_to(m)
    lg = folium.FeatureGroup("Fire regimes", show=False)
    for r in fp.itertuples():
        folium.Rectangle([[r.lat - RES/2, r.lon - RES/2], [r.lat + RES/2, r.lon + RES/2]], weight=0, fill=True,
                         fill_color=RCOL[r.regime], fill_opacity=.55, tooltip=r.regime).add_to(lg)
    lg.add_to(m)
    if mode == "Enter coordinates":
        folium.Rectangle([[lat0, lon0], [lat1, lon1]], color="#fcffa4", weight=2, fill=False).add_to(m)
    Draw(draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False,
                       "polygon": {"shapeOptions": {"color": "#fcffa4"}},
                       "rectangle": {"shapeOptions": {"color": "#fcffa4"}}},
         edit_options={"edit": False}).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out = st_folium(m, height=440, use_container_width=True, returned_objects=["all_drawings"])

sel = None
if mode == "Enter coordinates":
    sel = cells[cells.lat.between(lat0, lat1) & cells.lon.between(lon0, lon1)]
else:
    drw = (out or {}).get("all_drawings") or []
    if drw:
        ring = np.array(drw[-1]["geometry"]["coordinates"][0])
        sel = cells[Path(ring).contains_points(cells[["lon", "lat"]].values)]
custom = sel is not None and not sel.empty
if not custom:
    sel = cells
ids = set(sel.cell_id)
Hall = h[h.cell_id.isin(ids)]
H = Hall[Hall.month.dt.year.between(y0, y1)]
if H.empty:
    st.warning("No data in this selection."); st.stop()

last = Hall.month.max()
Hm = Hall[Hall.month.dt.month == last.month]
by_m = Hm.groupby(Hm.month.dt.year).h_count.sum()
cur, med = by_m.get(last.year, 0), by_m[by_m.index < last.year].median()
dlt = (cur / med - 1) * 100 if med > 0 else np.nan
mshare = Hall.groupby(Hall.month.dt.month).h_count.sum()
F = fp[fp.cell_id.isin(ids)]
R = risk[risk.cell_id.isin(ids)]
rs = R.risk.value_counts()

with ckpi:
    arrow = "" if np.isnan(dlt) else (f'<span class="up">▲ {dlt:+.0f}%</span>' if dlt > 0
                                      else f'<span class="down">▼ {dlt:+.0f}%</span>')
    st.markdown('<div class="kgrid">' + "".join([
        card("Area", f"{len(ids)} cells", "Custom selection" if custom else "Whole region — draw to focus", "#fcffa4"),
        card(f"Hotspots {y0}–{y1}", f"{H.h_count.sum():,.0f}", "harmonized, VIIRS-equivalent"),
        card(f"{last:%B %Y}", f"{cur:,.0f}", f"{arrow} vs median {med:,.0f}", "#dd513a"),
        card("Peak burning month", MONTHS[int(mshare.idxmax()) - 1], f"{mshare.max() / mshare.sum():.0%} of all hotspots", "#932667"),
        card("Typical rice-burn onset", doy2str(F.med_onset_doy_rice.median()), "10% of season reached", "#fca50a"),
        card("High-risk cells", int(rs.get("High", 0)), f'{int(rs.get("Medium", 0))} medium · {int(rs.get("Low", 0))} low', "#fca50a"),
    ]) + "</div>", unsafe_allow_html=True)
    st.markdown('<p class="note" style="margin-top:10px">Draw a rectangle or polygon on the map to analyse any area. '
                'Toggle <b>Fire regimes</b> at the top right of the map.</p>', unsafe_allow_html=True)

tabs = st.tabs(["🔥 Fire map", "📅 Burn calendar", "📈 This month vs history", "🧬 Fire regimes",
                "⚠️ Risk & deployment", "⬇️ Data", "ℹ️ Method"])

# ---------------------------------------------------------------- 1. live / archive fire map
with tabs[0]:
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.2, 1])
    src = c1.radio("Source", ["Archive", "Live · NASA FIRMS"], horizontal=True)
    view = c2.radio("View", ["Points", "Heat", "3D hexbins", "3D risk"], horizontal=True)
    bb = (sel.lon.min() - RES/2, sel.lat.min() - RES/2, sel.lon.max() + RES/2, sel.lat.max() + RES/2)
    if src == "Archive":
        dmin, dmax = hs.acq_date.min().date(), hs.acq_date.max().date()
        dr = c3.date_input("Dates", (dmax - pd.Timedelta(days=30), dmax), min_value=dmin, max_value=dmax)
        a, b = (dr if isinstance(dr, tuple) and len(dr) == 2 else (dr[0], dr[0]) if isinstance(dr, tuple) else (dr, dr))
        sens = c4.multiselect("Sensor", ["VIIRS", "MODIS"], ["VIIRS"])
        pts = hs[hs.cell_id.isin(ids) & hs.sensor.isin(sens) &
                 hs.acq_date.between(pd.Timestamp(a), pd.Timestamp(b))].copy()
        st.markdown(f'<span class="note">{len(pts):,} detections · {a:%d %b %Y} – {b:%d %b %Y}</span>',
                    unsafe_allow_html=True)
    else:
        days = c3.slider("Last N days", 1, 5, 2)
        pts, err = live(tuple(round(v, 3) for v in bb), days)
        if err:
            st.warning(err)
        else:
            st.markdown(f'<span class="live">LIVE · {len(pts):,} detections in the last {days} day(s) · '
                        f'VIIRS S-NPP / NOAA-20 / NOAA-21 + MODIS · refreshes every 15 min</span>',
                        unsafe_allow_html=True)
    if len(pts) > 200_000:
        pts = pts.sample(200_000, random_state=0); st.caption("Showing a random 200,000-point sample.")

    layers, tip = [], None
    if len(pts):
        pts = pts.copy()
        pts["frp"] = pd.to_numeric(pts.frp, errors="coerce").fillna(0)
        pts["date"] = pd.to_datetime(pts.acq_date).dt.strftime("%d %b %Y")
        pts["time"] = pd.to_numeric(pts.acq_time, errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
        rgb = frp_rgb(pts.frp); pts["r"], pts["g"], pts["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        pts["rad"] = 250 + 70 * np.sqrt(pts.frp)
        pos = "[longitude, latitude]"
        if view == "Points":
            layers.append(pdk.Layer("ScatterplotLayer", pts, get_position=pos, get_fill_color="[r, g, b, 215]",
                                    get_radius="rad", radius_min_pixels=1.6, radius_max_pixels=14,
                                    stroked=False, pickable=True))
            tip = {"html": "<b>{date}</b> · {time} UTC<br/>FRP <b>{frp}</b> MW<br/>{satellite} · conf {confidence}",
                   "style": {"backgroundColor": "#151922", "color": "#e8e6e3"}}
        elif view == "Heat":
            layers.append(pdk.Layer("HeatmapLayer", pts, get_position=pos, get_weight=1,
                                    radius_pixels=35, color_range=INF6, intensity=1.2))
        elif view == "3D hexbins":
            layers.append(pdk.Layer("HexagonLayer", pts, get_position=pos, radius=2500, elevation_scale=40,
                                    elevation_range=[0, 3000], extruded=True, coverage=.9,
                                    color_range=INF6, pickable=True))
            tip = {"html": "<b>{elevationValue}</b> detections", "style": {"backgroundColor": "#151922"}}
    if view == "3D risk":
        RR = R.copy()
        RR["polygon"] = [[[lo - RES/2, la - RES/2], [lo + RES/2, la - RES/2], [lo + RES/2, la + RES/2],
                          [lo - RES/2, la + RES/2]] for la, lo in zip(RR.lat, RR.lon)]
        RR["color"] = RR.risk.map(RISK_RGB); RR["elev"] = RR.score * 90000
        layers = [pdk.Layer("PolygonLayer", RR, get_polygon="polygon", get_fill_color="color",
                            get_elevation="elev", extruded=True, pickable=True, opacity=.85)]
        tip = {"html": "<b>{risk}</b> risk · {regime}<br/>{explanation}", "style": {"backgroundColor": "#151922"}}
    outline = pd.DataFrame({"polygon": [[[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]]})
    layers.append(pdk.Layer("PolygonLayer", outline, get_polygon="polygon", filled=False, stroked=True,
                            get_line_color=[252, 255, 164, 140], line_width_min_pixels=1.5))
    pitch = 50 if view.startswith("3D") else 0
    span = max(bb[2] - bb[0], bb[3] - bb[1])
    st.pydeck_chart(pdk.Deck(layers=layers, tooltip=tip, map_provider="carto", map_style=pdk.map_styles.CARTO_DARK,
                             initial_view_state=pdk.ViewState(latitude=(bb[1] + bb[3]) / 2,
                                                              longitude=(bb[0] + bb[2]) / 2,
                                                              zoom=float(np.clip(8.6 - np.log2(span + .25), 5.3, 10)),
                                                              pitch=pitch, bearing=-12 if pitch else 0)),
                    width="stretch", height=580)
    if view in ("Points", "Heat", "3D hexbins"):
        st.markdown('<div class="legend"></div><div class="lgl"><span>low intensity / few</span>'
                    '<span>Fire radiative power / density →</span><span>high</span></div>', unsafe_allow_html=True)
    if src == "Archive" and not len(pts):
        st.info("No detections for this area and date range — try widening the dates (e.g. Oct–Nov).")

# ---------------------------------------------------------------- 2. calendar
with tabs[1]:
    cal = (H.groupby([H.month.dt.year.rename("year"), H.month.dt.month.rename("m")]).h_count.sum()
           .unstack().reindex(columns=range(1, 13)))
    modeled = set(H[H.source == "modeled"].month.dt.year)
    ylab = [f"{y} ·" if y in modeled else str(y) for y in cal.index]
    log = st.toggle("Log colour scale", True)
    z = np.log10(cal.values + 1) if log else cal.values
    cb = dict(title="Hotspots", thickness=12)
    if log:
        tv = [v for v in range(0, 6) if v <= np.nanmax(z) + .5]
        cb.update(tickvals=tv, ticktext=[f"{10**v:,.0f}" for v in tv])
    fig = go.Figure(go.Heatmap(z=z, x=MONTHS, y=ylab, customdata=cal.values, colorscale="Inferno",
                               hovertemplate="<b>%{x} %{y}</b><br>%{customdata:,.0f} hotspots<extra></extra>",
                               colorbar=cb, xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed", type="category"); fig.update_xaxes(side="top")
    st.plotly_chart(style(fig, max(380, 24 * len(cal))), width="stretch")
    yt = cal.sum(axis=1)
    st.markdown(f'<p class="note">Most active year: <b>{yt.idxmax()}</b> ({yt.max():,.0f} hotspots) · '
                f'quietest: <b>{yt.idxmin()}</b> · rows marked “·” are modeled from MODIS (VIIRS-equivalent).</p>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- 3. this month vs history
with tabs[2]:
    c1, c2 = st.columns(2)
    q1, q3 = by_m[by_m.index < last.year].quantile([.25, .75])
    fig = go.Figure(go.Bar(x=by_m.index.astype(str), y=by_m.values,
                           marker_color=["#fca50a" if y == last.year else "#3a4152" for y in by_m.index],
                           hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    fig.add_hrect(y0=q1, y1=q3, fillcolor="rgba(221,81,58,.12)", line_width=0)
    fig.add_hline(y=med, line_dash="dash", line_color="#dd513a", annotation_text="median",
                  annotation_font_color="#dd513a")
    c1.plotly_chart(style(fig, 400, f"{last:%B} hotspots by year — {last.year} highlighted"), width="stretch")
    ann = Hall.groupby(Hall.month.dt.year).h_count.sum()
    ann = ann[ann.index < last.year + (1 if last.month == 12 else 0)]
    fig = go.Figure(go.Scatter(x=ann.index, y=ann.values, mode="lines+markers", fill="tozeroy",
                               line=dict(color="#fca50a", width=3), fillcolor="rgba(252,165,10,.15)",
                               hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    fig.add_vrect(x0=ann.index.min() - .5, x1=2011.5, fillcolor="rgba(255,255,255,.04)", line_width=0,
                  annotation_text="modeled", annotation_position="top left")
    c2.plotly_chart(style(fig, 400, "Annual hotspots (complete years)"), width="stretch")
    st.markdown(f'<p class="note">Shaded band = typical range (IQR) for {last:%B}. '
                f'{last:%B %Y}: <b>{cur:,.0f}</b> vs median <b>{med:,.0f}</b>.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 4. regimes
with tabs[3]:
    c1, c2 = st.columns([1, 1.3])
    share = (mshare.reindex(range(1, 13), fill_value=0) / mshare.sum()).values
    fig = go.Figure(go.Barpolar(r=share, theta=MONTHS,
                                marker=dict(color=share, colorscale="Inferno", line=dict(color="#0b0d12", width=1)),
                                hovertemplate="%{theta}: %{r:.1%}<extra></extra>"))
    fig.update_polars(bgcolor="rgba(0,0,0,0)", angularaxis=dict(direction="clockwise", rotation=90),
                      radialaxis=dict(showticklabels=False))
    c1.plotly_chart(style(fig, 420, "Fire clock — share of hotspots by month"), width="stretch")
    fig = go.Figure()
    for rg, g in F.groupby("regime"):
        fig.add_trace(go.Scatter(x=MONTHS, y=g[[f"p{i}" for i in range(1, 13)]].mean(), mode="lines+markers",
                                 name=f"{rg} ({len(g)})", line=dict(color=RCOL[rg], width=3)))
    fig.update_layout(legend=dict(orientation="h", y=-0.18))
    c2.plotly_chart(style(fig, 420, "Seasonal signature of each fire regime"), width="stretch")
    summ = F.groupby("regime").agg(cells=("cell_id", "size"), onset=("med_onset_doy_rice", "median"),
                                   mid=("med_mid_doy_rice", "median"),
                                   hotspots_per_year=("mean_annual", "median")).reset_index()
    summ["onset"], summ["mid"] = summ.onset.map(doy2str), summ.mid.map(doy2str)
    st.dataframe(summ, hide_index=True, width="stretch", column_config={
        "regime": "Regime", "cells": "Cells", "onset": "Rice-season onset", "mid": "Rice-season midpoint",
        "hotspots_per_year": st.column_config.NumberColumn("Median hotspots / yr", format="%.0f")})
    st.markdown('<p class="note">Regimes are discovered from each cell’s signal fingerprint — FFT amplitude at 6 and '
                '12 months, peak month and spectral entropy — with no land-cover map.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 5. risk & deployment
with tabs[4]:
    st.markdown('<div class="kgrid" style="grid-template-columns:repeat(3,1fr)">' + "".join(
        card(f"{k} risk", int(rs.get(k, 0)), "cells", RISK_HEX[k]) for k in ["High", "Medium", "Low"]) +
        "</div>", unsafe_allow_html=True)
    st.markdown("#### Highest-risk cells")
    st.dataframe(R.sort_values("score", ascending=False)
                 [["cell_id", "lat", "lon", "regime", "risk", "score", "explanation"]].head(15),
                 hide_index=True, width="stretch", column_config={
                     "score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.2f"),
                     "lat": st.column_config.NumberColumn(format="%.2f"),
                     "lon": st.column_config.NumberColumn(format="%.2f")})
    st.markdown("#### Residue-management deployment priority — rice–wheat belt")
    Dp = dep[dep.cell_id.isin(ids)].copy()
    Dp["pred_onset"] = pd.to_datetime(Dp.pred_onset).dt.strftime("%d %b %Y")
    st.dataframe(Dp.head(20), hide_index=True, width="stretch", column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "forecast": st.column_config.ProgressColumn("Expected hotspots next month", min_value=0,
                                                    max_value=float(dep.forecast.max()), format="%.0f"),
        "pred_onset": "Predicted onset", "lat": st.column_config.NumberColumn(format="%.2f"),
        "lon": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("⬇️ Download deployment list (CSV)", Dp.to_csv(index=False),
                       "firerhythm_deployment_priority.csv", "text/csv")
    st.markdown('<p class="note">Risk = 0.4 × anomaly (robust z vs the same month in past years) + 0.3 × early '
                'onset + 0.3 × forecast exceedance. This is <i>anomalous burning risk</i>; weather inputs are on '
                'the roadmap.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 6. data
with tabs[5]:
    P = hs[hs.cell_id.isin(ids) & hs.acq_date.dt.year.between(y0, y1)]
    st.markdown(f"**{len(P):,}** hotspot detections in the selection ({y0}–{y1}).")
    st.dataframe(P.head(300), hide_index=True, width="stretch")
    st.download_button("⬇️ Download hotspots (CSV)", P.to_csv(index=False),
                       f"firerhythm_hotspots_{y0}_{y1}.csv", "text/csv")

# ---------------------------------------------------------------- 7. method
with tabs[6]:
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Harmonization", "−23% RMSE", "cell level, test 2023+ vs fixed ratio"), unsafe_allow_html=True)
    c2.markdown(card("Sensor-switch jump", "p = 0.07", "was p = 3×10⁻¹¹ before harmonization", "#dd513a"),
                unsafe_allow_html=True)
    c3.markdown(card("Forecast", "−16% MAE", "vs seasonal-naive", "#932667"), unsafe_allow_html=True)
    st.markdown("""
**1 · Harmonization as system identification.** MODIS (1 km) acts like a low-pass, under-sampling view of fire
compared with VIIRS (375 m). A LightGBM transfer function learned on the 2012–2022 overlap converts MODIS counts
into VIIRS-equivalents, giving one consistent record from 2000.
**2 · Validation.** Change-point detection and a t-test show the artificial 2012 jump disappears after harmonization.
**3 · Fingerprints.** STL decomposition and FFT per cell → period, phase, amplitude, spectral entropy → k-means regimes.
**4 · Risk & early warning.** Robust anomalies, season-onset drift and a LightGBM forecast → High/Medium/Low risk
and a ranked deployment list for residue-management machinery.
""")
    for f in ["changepoint", "harmonization", "regimes"]:
        p = f"docs/figures/{f}.png"
        if os.path.exists(p):
            st.image(p, width="stretch")

st.sidebar.markdown(f'<p class="note">Rendered in {time.perf_counter() - T0:.2f} s</p>', unsafe_allow_html=True)