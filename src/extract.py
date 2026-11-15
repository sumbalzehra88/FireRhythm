import zipfile; from pathlib import Path
for zp in Path("data/raw/zips").glob("*.zip"):
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            f = Path(m).name
            if f.endswith(("_Pakistan.csv", "_India.csv")):
                s = "modis" if "modis" in (m + zp.name).lower() else "viirs"
                (Path("data/raw") / s / f).write_bytes(z.read(m)); print(s, f)