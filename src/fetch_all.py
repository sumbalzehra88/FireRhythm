import requests, pandas as pd, io, time, os
from datetime import date, timedelta

MAP_KEY = os.environ.get("FIRMS_KEY", "PASTE_YOUR_KEY")
AREA = "69,27,78,33"
JOBS = [("MODIS_SP", date(2000,11,1)), ("VIIRS_SNPP_SP", date(2012,1,20)),
        ("MODIS_NRT", date(2025,6,1)), ("VIIRS_SNPP_NRT", date(2025,6,1))]
os.makedirs("data/raw/api", exist_ok=True)

for src, start in JOBS:
    frames, d = [], start
    while d <= date.today():
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{src}/{AREA}/5/{d}"
        try:
            r = requests.get(url, timeout=120)
            if r.ok and r.text.startswith("latitude"):
                frames.append(pd.read_csv(io.StringIO(r.text)))
            elif not r.ok:
                print(src, d, r.status_code, r.text[:100])
        except Exception as e:
            print(src, d, "error", e)
        d += timedelta(days=5)
        time.sleep(0.3)
    if frames:
        df = pd.concat(frames).drop_duplicates()
        df.to_csv(f"data/raw/api/{src}.csv", index=False)
        print(src, len(df), "rows saved")