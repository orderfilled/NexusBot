"""Download XAUUSD M1 bars from HistData.com (monthly ASCII zips).

HistData timestamps are EST without DST (fixed UTC-5); converted to UTC here.
HistData has no spread, so a constant --spread (MT5 points) is filled in.

Usage: python tools/local/histdata_bars.py --months 2025-01:2026-10 --out data/backups/XAUUSD_M1_histdata.parquet
"""
import argparse
import io
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import timedelta
from pathlib import Path

import polars as pl

PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{y}/{m}"
GET = "https://www.histdata.com/get.php"


def fetch_month(pair, y, m, cache):
    path = cache / f"{pair}_{y}{m:02d}.csv"
    if path.exists():
        return path
    page_url = PAGE.format(pair=pair.lower(), y=y, m=m)
    html = urllib.request.urlopen(page_url, timeout=60).read().decode("utf-8", "replace")
    tk = re.search(r'id="tk" value="([^"]+)"', html)
    if not tk:
        return None
    body = urllib.parse.urlencode({
        "tk": tk.group(1), "date": str(y), "datemonth": f"{y}{m:02d}",
        "platform": "ASCII", "timeframe": "M1", "fxpair": pair.upper(),
    }).encode()
    req = urllib.request.Request(GET, data=body, headers={"Referer": page_url})
    raw = urllib.request.urlopen(req, timeout=120).read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return None
    name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    path.write_bytes(zf.read(name))
    return path


def month_range(spec):
    a, b = spec.split(":")
    y, m = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="XAUUSD")
    ap.add_argument("--months", required=True, help="YYYY-MM:YYYY-MM")
    ap.add_argument("--spread", type=int, default=30, help="constant spread in points")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/cache/histdata")
    a = ap.parse_args()

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for y, m in month_range(a.months):
        p = fetch_month(a.pair, y, m, cache)
        print(f"{y}-{m:02d}: {p if p else 'not available'}", flush=True)
        if p:
            frames.append(pl.read_csv(
                p, separator=";", has_header=False,
                new_columns=["ts", "open", "high", "low", "close", "volume"],
            ))
        time.sleep(1)
    bars = (
        pl.concat(frames)
        .with_columns(
            (pl.col("ts").str.strptime(pl.Datetime("us"), "%Y%m%d %H%M%S") + timedelta(hours=5))
            .dt.replace_time_zone("UTC").alias("time")
        )
        .select(
            "time", "open", "high", "low", "close",
            pl.lit(1).cast(pl.Int64).alias("tick_volume"),
            pl.lit(a.spread).cast(pl.Int64).alias("spread"),
            pl.lit(0).cast(pl.Int64).alias("real_volume"),
        )
        .unique("time", keep="last")
        .sort("time")
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(a.out)
    print(f"wrote {bars.height} bars {bars['time'].min()} .. {bars['time'].max()} -> {a.out}")


if __name__ == "__main__":
    main()
