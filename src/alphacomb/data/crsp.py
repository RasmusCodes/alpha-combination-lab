"""CRSP / Compustat (WRDS): pull the raw tables, build the survivorship-free monthly panel.

Universe: every security ever in the S&P 500 (crsp.msp500list), legacy CRSP stock files from 1962 (they end
2024-12-31), Compustat annual (INDL/STD/D/C) linked through the CCM link table (LU/LC links, primary P then C).

Pull (needs a WRDS account; libpq reads the password from its own password file, never from this code):
    python scripts/build_data.py --pull       (writes data/raw/crsp/*.parquet + manifest.json)
Raw files can live elsewhere: set ALPHACOMB_CRSP_RAW.

Monthly build rules (point in time):
  * total return `tr` includes delisting returns (msedelist.dlret; Shumway (1997) fill of -30% NYSE/AMEX, -55%
    Nasdaq for performance delistings without a return), booked in the month the HOLDER receives it: the delisting
    month if the stock stopped trading before that month's last trading day, else the next month (a stock that
    traded through a month end is still held by a portfolio formed at that close); a row is appended if needed
  * member = inside an S&P 500 membership spell at the month's last trading day; alive = not delisted before it
  * formation universe = member & alive & market cap known
  * industries (FF48) from the point-in-time CRSP SIC code
These rules follow the author's earlier CRSP build (statarb, research note on CRSP data, 2026-11-20).
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphacomb import paths

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
PERF_DELIST = set([500]) | set(range(520, 585))
FIRST_YEAR, LAST_YEAR = 1962, 2024
MEMBERS = "(select distinct permno from crsp.msp500list)"
LINKED = (f"(select distinct gvkey from crsp.ccmxpf_lnkhist where lpermno in {MEMBERS} "
          "and linktype in ('LU','LC') and linkprim in ('P','C'))")
FUNDA_COLS = ("gvkey, datadate, fyear, fyr, curcd, sich, revt, sale, cogs, xsga, xrd, xint, oiadp, oibdp, dp, ib, ni, "
              "at, lt, seq, ceq, pstk, pstkrv, pstkl, txditc, csho, prcc_f, act, lct, che, dlc, dltt, rect, invt, "
              "xpp, drc, drlt, ap, xacc, oancf, capx")
TABLES = {
    "msp500list": ("select permno, start, ending from crsp.msp500list", ["start", "ending"]),
    "msf": (f"select permno, date, ret, retx, prc, shrout, vol, cfacpr, cfacshr from crsp.msf "
            f"where permno in {MEMBERS} and date >= '{FIRST_YEAR}-01-01'", ["date"]),
    "msenames": (f"select permno, permco, namedt, nameendt, shrcd, exchcd, siccd, ncusip, ticker, comnam "
                 f"from crsp.msenames where permno in {MEMBERS}", ["namedt", "nameendt"]),
    "msedelist": (f"select permno, dlstdt, dlret, dlstcd from crsp.msedelist where permno in {MEMBERS}", ["dlstdt"]),
    "ccmxpf_lnkhist": (f"select gvkey, lpermno, lpermco, linktype, linkprim, linkdt, linkenddt "
                       f"from crsp.ccmxpf_lnkhist where lpermno in {MEMBERS} "
                       "and linktype in ('LU','LC') and linkprim in ('P','C')", ["linkdt", "linkenddt"]),
    "funda": (f"select {FUNDA_COLS} from comp.funda where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' "
              f"and consol = 'C' and gvkey in {LINKED} and datadate >= '1959-01-01'", ["datadate"]),
}
DSF_COLS = "permno, date, prc, ret, retx, vol, shrout, openprc, askhi, bidlo, bid, ask, cfacpr, cfacshr"
TEXT_COLS = {"gvkey": str, "ncusip": str, "ticker": str, "comnam": str, "curcd": str, "linktype": str,
             "linkprim": str}
OPEN_END = pd.Timestamp("2099-12-31")


# ----------------------------------------------------------------------------- pull
def dsf_query(year: int) -> str:
    return f"select {DSF_COLS} from crsp.dsf where permno in {MEMBERS} and date between '{year}-01-01' and '{year}-12-31'"


def _engine():
    from sqlalchemy import create_engine
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        pg = Path(os.environ.get("APPDATA", Path.home())) / "postgresql" / "pgpass.conf" if os.name == "nt" \
            else Path.home() / ".pgpass"
        if pg.exists():
            for ln in pg.read_text(encoding="utf-8").splitlines():
                if ln.startswith(f"{HOST}:{PORT}:{DB}:"):
                    user = ln.split(":")[3]
    if not user:
        raise RuntimeError("No WRDS username: set WRDS_USERNAME (the password comes from libpq's password file)")
    return create_engine(f"postgresql+psycopg2://{user}@{HOST}:{PORT}/{DB}",
                         connect_args={"sslmode": "require", "connect_timeout": 90})


def _read(sql: str, dates, eng) -> pd.DataFrame:
    buf = io.BytesIO()
    raw = eng.raw_connection()
    try:
        cur = raw.cursor()
        cur.copy_expert(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER true)", buf)
        cur.close()
    finally:
        raw.close()
    buf.seek(0)
    df = pd.read_csv(buf, dtype=TEXT_COLS, keep_default_na=False, na_values=[""], low_memory=False)
    for c in dates:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d").astype("datetime64[ns]")
    return df


def pull(out_dir: Path | None = None, daily: bool = True, log=print) -> None:
    out = Path(out_dir or paths.RAW_CRSP)
    (out / "dsf").mkdir(parents=True, exist_ok=True)
    eng = _engine()
    man = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    jobs = [(n, q, d, out / f"{n}.parquet") for n, (q, d) in TABLES.items()]
    if daily:
        jobs += [(f"dsf_{y}", dsf_query(y), ["date"], out / "dsf" / f"dsf_{y}.parquet")
                 for y in range(FIRST_YEAR, LAST_YEAR + 1)]
    for name, sql, dates, f in jobs:
        if f.exists() and name in man:
            continue
        t0 = time.time()
        df = _read(sql, dates, eng)
        tmp = f.with_name(f.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
        man[name] = {"rows": len(df), "seconds": round(time.time() - t0, 1), "query": " ".join(sql.split())}
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        log(f"{name}: {len(df):,} rows")


# ----------------------------------------------------------------------------- build
def raw_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(Path(raw_dir or paths.RAW_CRSP) / f"{name}.parquet")
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype("datetime64[ns]")
    return df


def dsf_files(raw_dir: Path | None = None) -> list[Path]:
    return sorted((Path(raw_dir or paths.RAW_CRSP) / "dsf").glob("dsf_*.parquet"))


def clean_returns(r: pd.Series) -> pd.Series:
    """CRSP missing-return codes (-66, -77, -88, -99) are never returns: anything below -100% is missing."""
    return r.astype(float).where(r >= -1.0)


def delisting_events(delist: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    ev = delist.dropna(subset=["dlstdt"]).copy()
    exch = names.sort_values("nameendt").groupby("permno")["exchcd"].last()
    code = ev["dlstcd"].fillna(0).astype(int)
    fill = ev["dlret"].isna() & code.isin(PERF_DELIST)
    ev["shumway"] = fill
    ev.loc[fill, "dlret"] = np.where(ev.loc[fill, "permno"].map(exch).eq(3).to_numpy(), -0.55, -0.30)
    return ev.dropna(subset=["dlret"]).reset_index(drop=True)


def apply_delisting(msf: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    m = msf.reset_index(drop=True).copy()
    m["ret"] = clean_returns(m["ret"])
    m["ym"] = m["date"].dt.to_period("M")
    last_day = m.groupby("ym")["date"].max()
    e = ev.assign(ym=ev["dlstdt"].dt.to_period("M"))
    late = (e["dlstdt"] >= e["ym"].map(last_day)).to_numpy()
    e.loc[late, "ym"] = e.loc[late, "ym"] + 1
    e = e[["permno", "ym", "dlret", "shumway"]]
    j = m[["permno", "ym"]].reset_index().merge(e, on=["permno", "ym"], how="inner")
    pos = j["index"].to_numpy()
    tr = m["ret"].to_numpy(dtype=float).copy()
    tr[pos] = (1 + np.nan_to_num(tr[pos])) * (1 + j["dlret"].to_numpy()) - 1
    m["tr"] = tr
    m["delist_booked"] = False
    m.loc[pos, "delist_booked"] = True
    have = pd.MultiIndex.from_frame(m[["permno", "ym"]])
    miss = e[~pd.MultiIndex.from_frame(e[["permno", "ym"]]).isin(have)]
    last = m.groupby("permno")["ym"].max()
    month_end = m.groupby("ym")["date"].max()
    miss = miss[miss["permno"].isin(last.index)]
    miss = miss[(miss["ym"] > miss["permno"].map(last)) & miss["ym"].isin(month_end.index)]
    if len(miss):
        add = pd.DataFrame({"permno": miss["permno"].to_numpy(), "date": miss["ym"].map(month_end).to_numpy(),
                            "ym": miss["ym"].to_numpy(), "tr": miss["dlret"].to_numpy(), "delist_booked": True,
                            "appended": True})
        m = pd.concat([m.assign(appended=False), add], ignore_index=True)
    else:
        m["appended"] = False
    return m.sort_values(["permno", "date"]).reset_index(drop=True)


def names_asof(df: pd.DataFrame, names: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    left = df.reset_index(drop=True).assign(_row=lambda x: np.arange(len(x)))
    n = names.sort_values("namedt")[["permno", "namedt", "nameendt", *cols]]
    x = pd.merge_asof(left.sort_values("date"), n, left_on="date", right_on="namedt", by="permno",
                      direction="backward")
    inside = (x["date"] <= x["nameendt"]).to_numpy()
    for c in cols:
        x[c] = x[c].where(inside)
    return x.sort_values("_row").drop(columns=["_row", "namedt", "nameendt"]).reset_index(drop=True)


def in_spells(df: pd.DataFrame, spells: pd.DataFrame) -> np.ndarray:
    left = df[["permno", "date"]].reset_index(drop=True).assign(_row=lambda x: np.arange(len(x)))
    s = spells.sort_values("start")[["permno", "start", "ending"]]
    x = pd.merge_asof(left.sort_values("date"), s, left_on="date", right_on="start", by="permno",
                      direction="backward")
    out = np.zeros(len(left), dtype=bool)
    out[x["_row"].to_numpy()] = (x["date"] <= x["ending"]).to_numpy()
    return out


def build_monthly(raw_dir: Path | None = None) -> pd.DataFrame:
    """One row per permno-month: tr, prc, shrout, mcap ($M), volume, split factors, point-in-time names, FF48,
    member, alive. `month` is the calendar month end (the formation timestamp); `date` the last trading day."""
    from alphacomb.data import french
    msf, names, sp, dl = (raw_table(t, raw_dir) for t in ("msf", "msenames", "msp500list", "msedelist"))
    m = apply_delisting(msf, delisting_events(dl, names))
    px = m["prc"].abs()
    m["mcap"] = (px * m["shrout"] / 1000.0).where(px > 0)
    m = names_asof(m, names, ["permco", "shrcd", "exchcd", "siccd"])
    m["siccd"] = m["siccd"].where(m["siccd"] > 0)
    m["ff48"] = french.map_sic(m["siccd"], 48)
    m["member"] = in_spells(m, sp)
    gone = dl[dl["dlstcd"].fillna(0).astype(int) != 100].set_index("permno")["dlstdt"]
    m["alive"] = ~(m["permno"].map(gone) < m["date"]).to_numpy(dtype=bool)
    m["month"] = (m["date"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    return m.drop(columns=["ym"])


def link_funda(raw_dir: Path | None = None) -> pd.DataFrame:
    """Compustat annual rows with the permno of the primary (P, else C) CCM link valid at datadate."""
    f, lk = raw_table("funda", raw_dir), raw_table("ccmxpf_lnkhist", raw_dir)
    lk = lk.assign(linkenddt=lk["linkenddt"].fillna(OPEN_END))
    x = f.merge(lk[["gvkey", "lpermno", "linkprim", "linkdt", "linkenddt"]], on="gvkey")
    x = x[(x["datadate"] >= x["linkdt"]) & (x["datadate"] <= x["linkenddt"])]
    x = x.sort_values(["lpermno", "datadate", "linkprim"], ascending=[True, True, False])
    x = x.drop_duplicates(["lpermno", "datadate"]).rename(columns={"lpermno": "permno"})
    return x.drop(columns=["linkdt", "linkenddt"]).reset_index(drop=True)
