"""Weekly forward test: refresh price data, backtest every NEW complete week, write a report.

Forward test = weeks starting FORWARD_START or later. Those weeks came after all data
used to choose the gate (Jul-Oct), so each run is a fresh out-of-sample check.
A week is "done" when its result file exists in tools/local/weekly/results/.

Usage: python tools/local/weekly_report.py [--jobs 4]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
MODEL = REPO / "tools/local/model/model.pt"
RESULTS = REPO / "tools/local/weekly/results"
REPORT = REPO / "tools/local/weekly/REPORT.md"
CACHE_MONTH = REPO / "data/cache/histdata"
FORWARD_START = datetime(2026, 10, 24, tzinfo=UTC)
GATE = ["--gate", "prob", "--gate-threshold", "0.36", "--threshold", "0.35"]


def run(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def refresh_data():
    now = datetime.now(UTC)
    span = f"2025-01:{now:%Y-%m}"
    # The current month is partial and changes daily: never reuse a cached copy of it.
    stale = CACHE_MONTH / f"XAUUSD_{now:%Y%m}.csv"
    stale.unlink(missing_ok=True)
    r = run([PY, "tools/local/histdata_bars.py", "--months", span,
             "--out", "data/backups/XAUUSD_M1_histdata.parquet"])
    if r.returncode:
        raise SystemExit(f"data download failed:\n{r.stderr[-800:]}")
    r = run(["nexus", "data-restore", "--source", "data/backups/XAUUSD_M1_histdata.parquet", "--force"])
    if r.returncode:
        raise SystemExit(f"data restore failed:\n{r.stdout[-800:]}")
    last = pl.read_parquet(REPO / "data/raw/XAUUSD_M1.parquet", columns=["time_utc"])["time_utc"].max()
    return last.replace(tzinfo=UTC)


def new_weeks(last_bar):
    weeks = []
    start = FORWARD_START
    while start + timedelta(days=7) <= last_bar + timedelta(minutes=1):
        if not (RESULTS / f"bt_{start:%Y%m%d}_{(start + timedelta(days=7)):%Y%m%d}.json").exists():
            weeks.append(start)
        start += timedelta(days=7)
    return weeks


def backtest_week(start):
    end = start + timedelta(days=7)
    r = run([PY, "tools/local/backtest_runner.py", "--days", "7", "--end", end.isoformat(),
             "--model", str(MODEL), "--label", "forward", "--out-dir", str(RESULTS), *GATE])
    if r.returncode:
        return f"{start:%Y-%m-%d}: FAILED {r.stderr[-200:]}"
    return f"{start:%Y-%m-%d}: ok"


def load_results():
    rows = []
    for p in sorted(RESULTS.glob("bt_*.json")):
        r = json.loads(p.read_text())
        rows.append({"start": r["window"]["start"][:10], **r["overall"],
                     "by_direction": r["by_direction"], "by_exit": r["by_exit_reason"],
                     "gate": r.get("gate", {})})
    return rows


def verdict(rows):
    if len(rows) < 8:
        return f"TOO EARLY: {len(rows)} forward week(s). Need at least 8 before judging."
    total = sum(w["pnl_usd"] for w in rows)
    wins = sum(w["pnl_usd"] > 0 for w in rows)
    if total > 0 and wins / len(rows) >= 0.6:
        return "PROMISING, NOT PROVEN: profitable forward. Still confirm on the LHFX demo."
    return "NOT PROFITABLE on forward data. Do not trade this setup."


def agg(rows, key):
    out = {}
    for w in rows:
        for k, v in w[key].items():
            a = out.setdefault(k, {"trades": 0, "pnl_usd": 0.0})
            a["trades"] += v["trades"]
            a["pnl_usd"] += v["pnl_usd"]
    return {k: {"trades": v["trades"], "pnl_usd": round(v["pnl_usd"], 2)} for k, v in out.items()}


def write_report(rows, last_bar, log):
    lines = [
        "# Forward test: rules + model probability gate (>= 0.36)",
        "",
        f"Updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Price data runs to {last_bar:%Y-%m-%d %H:%M} UTC.",
        "Forward weeks start 2026-10-24: after all data used to pick the gate.",
        "",
        f"**Verdict:** {verdict(rows)}",
        "",
    ]
    if rows:
        total = sum(w["pnl_usd"] for w in rows)
        lines += [f"Weeks: {len(rows)}. Trades: {sum(w['trades'] for w in rows)}. "
                  f"P&L: ${total:,.2f}. Profitable weeks: {sum(w['pnl_usd'] > 0 for w in rows)}.", "",
                  "| Week | Trades | Win rate | P&L | Profit factor | Cumulative |",
                  "|---|---|---|---|---|---|"]
        cum = 0.0
        for w in rows:
            cum += w["pnl_usd"]
            lines.append(f"| {w['start']} | {w['trades']} | {w['win_rate']} | ${w['pnl_usd']:,.2f} | "
                         f"{w['profit_factor']} | ${cum:,.2f} |")
        lines += ["", "By direction (all forward weeks):", ""]
        lines += [f"- {k}: {v['trades']} trades, ${v['pnl_usd']:,.2f}" for k, v in agg(rows, "by_direction").items()]
        lines += ["", "By exit reason (all forward weeks):", ""]
        lines += [f"- {k}: {v['trades']} trades, ${v['pnl_usd']:,.2f}" for k, v in agg(rows, "by_exit").items()]
    else:
        lines.append("No complete forward weeks yet.")
    lines += ["", "## This run", ""] + [f"- {x}" for x in log] + [""]
    REPORT.write_text("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    last_bar = refresh_data()
    weeks = new_weeks(last_bar)
    log = [f"Data refreshed. New complete forward weeks to test: {len(weeks)}."]
    if weeks:
        with ThreadPoolExecutor(a.jobs) as ex:
            log += list(ex.map(backtest_week, weeks))
    print(write_report(load_results(), last_bar, log))


if __name__ == "__main__":
    main()
