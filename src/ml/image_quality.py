"""Flag blank (no-data) NAIP images (2026-11-07).

Some NAIP exports are black in the middle (no imagery). The classifier still
scores them (about 0.21), which can fake an "absent" year. This writes
outputs/naip_image_quality.csv with dark_frac = share of near-black pixels in the
centre 30% of each image; apply.py skips images with dark_frac >= 0.9.

Usage: python src/ml/image_quality.py
"""
import pandas as pd, numpy as np
from PIL import Image
from pathlib import Path
R=Path(__file__).resolve().parents[2]
by=pd.read_csv(R/"outputs/ml_presence_by_year.csv",dtype={"facility_id":str})
def stats(fid,y):
    p=R/"data/imagery"/fid/"naip"/f"{int(y)}.png"
    if not p.exists(): return np.nan,np.nan
    a=np.asarray(Image.open(p).convert("L").reduce(8),dtype=float)
    w=a.shape[0]; c=a[w*35//100:w*65//100, w*35//100:w*65//100]
    return (c<8).mean(), c.std()
s=by.apply(lambda r: stats(r.facility_id,r.year),axis=1,result_type="expand")
by["dark_frac"]=s[0]; by["std"]=s[1]
by.to_csv(R/"outputs/naip_image_quality.csv",index=False)
print(by.dark_frac.describe())
print("centre >50% black:", (by.dark_frac>0.5).sum(), "of", len(by), "images; sites:", by[by.dark_frac>0.5].facility_id.nunique())
print(by[by.dark_frac>0.5].year.value_counts().head(10))
