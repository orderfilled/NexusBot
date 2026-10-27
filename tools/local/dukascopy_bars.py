"""Download XAUUSD ticks from Dukascopy and aggregate to MT5-style M1 bars.

Usage: python tools/local/dukascopy_bars.py --days 90 --out data/backups/XAUUSD_M1_dukascopy.parquet
"""
import argparse
import lzma
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import polars as pl

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
DIVISOR = 1000.0  # XAUUSD price scale
POINT = 0.01      # MT5 point for XAUUSD


def fetch_hour(sym, hour):
    url = URL.format(sym=sym, y=hour.year, m=hour.month - 1, d=hour.day, h=hour.hour)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            break
        except Exception as e:  # 429 / transient
            if getattr(e, "code", None) == 404:
                return []
            time.sleep(2 ** attempt)
    else:
        return []
    if not raw:
        return []
    data = lzma.decompress(raw)
    base_ms = int(hour.timestamp() * 1000)
    out = []
    for ms, ask, bid, _av, _bv in struct.iter_unpack(">3I2f", data):
        out.append((base_ms + ms, ask / DIVISOR, bid / DIVISOR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    start = end - timedelta(days=a.days)
    hours = []
    h = start
    while h < end:
        if h.weekday() != 5:  # skip Saturdays (market closed)
            hours.append(h)
        h += timedelta(hours=1)

    ticks = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda x: fetch_hour(a.symbol, x), hours)):
            ticks.extend(res)
            if i % 200 == 0:
                print(f"{i}/{len(hours)} hours, {len(ticks)} ticks", flush=True)

    df = pl.DataFrame(ticks, schema=["ms", "ask", "bid"], orient="row")
    df = df.with_columns(
        pl.from_epoch("ms", time_unit="ms").dt.replace_time_zone("UTC").alias("ts"),
        ((pl.col("ask") - pl.col("bid")) / POINT).alias("spr"),
    ).sort("ts")
    bars = (
        df.group_by_dynamic("ts", every="1m")
        .agg(
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.len().alias("tick_volume"),
            pl.col("spr").median().round(0).cast(pl.Int64).alias("spread"),
        )
        .rename({"ts": "time"})
        .with_columns(pl.lit(0).cast(pl.Int64).alias("real_volume"))
    )
    bars.write_parquet(a.out)
    print(f"wrote {bars.height} bars {bars['time'].min()} .. {bars['time'].max()} -> {a.out}")


if __name__ == "__main__":
    main()
