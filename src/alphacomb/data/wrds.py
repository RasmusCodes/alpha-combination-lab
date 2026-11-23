"""WRDS access shared by the CRSP, I/B/E/S and OptionMetrics pulls.

The password is never handled here: libpq reads it from its own password file (%APPDATA%\\postgresql\\pgpass.conf
on Windows, ~/.pgpass elsewhere). The username comes from WRDS_USERNAME or that file's WRDS line. Every table is
read with COPY ... TO STDOUT (fast, low memory), written atomically to parquet, and recorded in a manifest
(rows, columns, seconds, the query), so an interrupted pull resumes where it stopped.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import pandas as pd

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
MEMBERS = "(select distinct permno from crsp.msp500list)"


def username() -> str | None:
    if os.environ.get("WRDS_USERNAME"):
        return os.environ["WRDS_USERNAME"]
    pg = (Path(os.environ.get("APPDATA", Path.home())) / "postgresql" / "pgpass.conf" if os.name == "nt"
          else Path.home() / ".pgpass")
    if pg.exists():
        for ln in pg.read_text(encoding="utf-8").splitlines():
            if ln.startswith(f"{HOST}:{PORT}:{DB}:"):
                return ln.split(":")[3]
    return None


def engine():
    from sqlalchemy import create_engine
    user = username()
    if not user:
        raise RuntimeError("No WRDS username: set WRDS_USERNAME (the password comes from libpq's password file)")
    return create_engine(f"postgresql+psycopg2://{user}@{HOST}:{PORT}/{DB}",
                         connect_args={"sslmode": "require", "connect_timeout": 90})


def read(sql: str, eng, dates=(), text_cols: dict | None = None) -> pd.DataFrame:
    buf = io.BytesIO()
    raw = eng.raw_connection()
    try:
        cur = raw.cursor()
        cur.copy_expert(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER true)", buf)
        cur.close()
    finally:
        raw.close()
    buf.seek(0)
    df = pd.read_csv(buf, dtype=text_cols or {}, keep_default_na=False, na_values=[""], low_memory=False)
    for c in dates:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d").astype("datetime64[ns]")
    return df


def check(sql: str, eng) -> None:
    """Zero-row run: a wrong table or column name fails here, before anything is downloaded."""
    pd.read_sql(f"select * from ({sql}) q limit 0", eng)


def pull(jobs: list[tuple[str, str, list, dict | None]], out: Path, log=print, tries: int = 3) -> None:
    """jobs: (name, sql, date columns, text columns). Skips names already in the manifest with their file present."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    eng = engine()
    todo = [j for j in jobs if not ((out / f"{j[0]}.parquet").exists() and j[0] in man)]
    for name, sql, _, _ in todo:
        check(sql, eng)
    log(f"{len(jobs)} tables, {len(todo)} to pull (all queries checked)")
    for name, sql, dates, text_cols in todo:
        for k in range(1, tries + 1):
            t0 = time.time()
            try:
                df = read(sql, eng, dates, text_cols)
                break
            except Exception as e:  # noqa: BLE001 - dropped connection: wait and retry
                if k == tries:
                    raise
                log(f"{name}: attempt {k} failed ({type(e).__name__}); retrying in 20 s")
                time.sleep(20)
        f = out / f"{name}.parquet"
        tmp = f.with_name(f.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
        man[name] = {"rows": int(len(df)), "columns": list(map(str, df.columns)), "seconds": round(time.time() - t0, 1),
                     "pulled_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "query": " ".join(sql.split())}
        tmpm = mf.with_name("manifest.json.tmp")
        tmpm.write_text(json.dumps(man, indent=1))
        os.replace(tmpm, mf)
        log(f"{name}: {len(df):,} rows in {time.time() - t0:.0f} s")
