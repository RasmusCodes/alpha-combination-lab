"""Kenneth French data library: factor returns and industry definitions (public files; cached under
data/raw/french, downloaded on first use if missing)."""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import pandas as pd

from alphacomb import paths

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ff3_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "strev_monthly": "F-F_ST_Reversal_Factor_CSV.zip",
    "sic48": "Siccodes48.zip",
}


def _zip(name: str) -> zipfile.ZipFile:
    p = paths.RAW_FRENCH / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "alphacomb-research"})
        with urllib.request.urlopen(req, timeout=60) as r:
            p.write_bytes(r.read())
    return zipfile.ZipFile(io.BytesIO(p.read_bytes()))


def _text(name: str) -> str:
    z = _zip(name)
    return z.read(z.namelist()[0]).decode("latin-1")


def _parse(text: str, digits: int) -> pd.DataFrame:
    """First table whose rows start with a YYYYMMDD (digits=8) or YYYYMM (digits=6) key; percent -> decimal."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(r"^\s*,", ln))
    cols = [c.strip().replace(" ", "") for c in lines[start].split(",")[1:]]
    recs = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not re.fullmatch(rf"\d{{{digits}}}", parts[0]):
            if recs:
                break
            continue
        recs.append([parts[0]] + [float(x) for x in parts[1:len(cols) + 1]])
    df = pd.DataFrame(recs, columns=["key"] + cols)
    fmt = "%Y%m%d" if digits == 8 else "%Y%m"
    idx = pd.to_datetime(df["key"], format=fmt)
    if digits == 6:
        idx = idx + pd.offsets.MonthEnd(0)
    return df.drop(columns="key").set_index(pd.DatetimeIndex(idx, name="date")) / 100.0


def daily_factors() -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RMW, CMA, RF, Mom (daily, decimal); 3-factor columns from 1926, the rest from 1963."""
    ff3 = _parse(_text(FILES["ff3_daily"]), 8)
    ff5 = _parse(_text(FILES["ff5_daily"]), 8)[["RMW", "CMA"]]
    mom = _parse(_text(FILES["mom_daily"]), 8)
    mom.columns = ["Mom"]
    return ff3.join(ff5, how="left").join(mom, how="left")


def monthly_factors() -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RF, Mom, ST_Rev from French's monthly files; RMW and CMA compounded from the daily
    five-factor file (French publishes monthly RMW/CMA in a separate file; compounding daily long-short returns
    differs from it only at second order)."""
    ff3 = _parse(_text(FILES["ff3_monthly"]), 6)
    mom = _parse(_text(FILES["mom_monthly"]), 6)
    mom.columns = ["Mom"]
    rev = _parse(_text(FILES["strev_monthly"]), 6)
    rev.columns = ["ST_Rev"]
    d = _parse(_text(FILES["ff5_daily"]), 8)[["RMW", "CMA"]]
    m5 = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    m5.index = m5.index.to_timestamp(how="end").normalize()
    m5.index.name = "date"
    return ff3.join(mom, how="left").join(rev, how="left").join(m5, how="left")


def industry_map(n: int = 48) -> pd.DataFrame:
    """(ind, short name, sic_lo, sic_hi) rows from Siccodes{n}."""
    text = _text(f"Siccodes{n}.zip")
    rows, cur = [], None
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*\S)\s*$", ln)
        r = re.match(r"^\s+(\d{4})-(\d{4})", ln)
        if r and cur:
            rows.append({**cur, "sic_lo": int(r.group(1)), "sic_hi": int(r.group(2))})
        elif m and not r:
            cur = {"ind": int(m.group(1)), "name": m.group(2)}
    return pd.DataFrame(rows)


def map_sic(sic: pd.Series, n: int = 48) -> pd.Series:
    """Fama-French industry number for each SIC code; codes outside every range -> industry n ('Other')."""
    defs = industry_map(n)
    u = pd.Series(sic.dropna().unique())
    out = {}
    for s in u:
        hit = defs[(defs["sic_lo"] <= s) & (defs["sic_hi"] >= s)]
        out[s] = int(hit["ind"].iloc[0]) if len(hit) else n
    return sic.map(out)
