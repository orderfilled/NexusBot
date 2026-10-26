"""Replay the serving model over real M1 bars and report what is / isn't working.

Usage: python tools/local/backtest_runner.py --days 30 [--threshold 0.20] [--out-dir artifacts/backtests]
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402

from nexus_scalp.research.event_source import BarEventSource  # noqa: E402
from nexus_scalp.research.streaming_replay import (  # noqa: E402
    ReplaySessionConfig,
    StreamingReplayEngine,
)

MODEL = "artifacts/models/scalp/XAUUSD/70d_liquidity/model.pt"
BARS = REPO / "data/raw/XAUUSD_M1.parquet"
WARMUP_BARS = 300


def load_records(start, end):
    df = pl.read_parquet(BARS).with_columns(
        pl.col("time_utc").dt.replace_time_zone("UTC").alias("time")
    ).filter(
        (pl.col("time") >= start - timedelta(minutes=WARMUP_BARS * 3)) & (pl.col("time") < end)
    ).sort("time")
    return [
        {
            "timestamp": r["time"],
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "tick_volume": int(r["tick_volume"]), "spread": float(r["spread"]),
            "symbol": "XAUUSD", "timeframe": "M1",
        }
        for r in df.iter_rows(named=True)
    ]


def session_of(hour):
    if 0 <= hour < 7:
        return "Asia (00-07 UTC)"
    if 7 <= hour < 12:
        return "London (07-12 UTC)"
    if 12 <= hour < 17:
        return "London/NY overlap (12-17 UTC)"
    return "NY late (17-24 UTC)"


def group_stats(trades):
    pnl = [t.pnl_usd for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gp, gl = sum(wins), -sum(losses)
    return {
        "trades": len(pnl),
        "win_rate": round(len(wins) / len(pnl), 3) if pnl else None,
        "pnl_usd": round(sum(pnl), 2),
        "avg_usd": round(sum(pnl) / len(pnl), 2) if pnl else None,
        "profit_factor": round(gp / gl, 2) if gl > 0 else None,
        "avg_win": round(gp / len(wins), 2) if wins else None,
        "avg_loss": round(-gl / len(losses), 2) if losses else None,
    }


def max_drawdown(trades, start_eq):
    eq = peak = start_eq
    dd = 0.0
    for t in sorted(trades, key=lambda t: t.exit_time):
        eq += t.pnl_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return round(dd, 2), round(dd / peak * 100, 2) if peak else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--end", default=None, help="ISO end date (default: last bar)")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-dir", default=str(REPO / "artifacts/backtests"))
    a = ap.parse_args()

    last = pl.read_parquet(BARS, columns=["time_utc"])["time_utc"].max().replace(tzinfo=UTC)
    end = datetime.fromisoformat(a.end).replace(tzinfo=UTC) if a.end else last + timedelta(minutes=1)
    start = end - timedelta(days=a.days)
    records = load_records(start, end)

    t0 = time.time()
    session = ReplaySessionConfig(
        model_artifact_path=a.model,
        policy_params={"confidence_threshold": a.threshold},
        decide_on="bar_close",
    )
    run = StreamingReplayEngine(session).run(BarEventSource(records), run_id="BT")
    trades = [
        SimpleNamespace(**{**d, "entry_time": datetime.fromisoformat(d["entry_time"]),
                           "exit_time": datetime.fromisoformat(d["exit_time"])})
        for d in run.trades
    ]
    trades = [t for t in trades if t.entry_time >= start]
    elapsed = time.time() - t0

    def by(keyfn):
        g = defaultdict(list)
        for t in trades:
            g[keyfn(t)].append(t)
        return {k: group_stats(v) for k, v in sorted(g.items())}

    dd_usd, dd_pct = max_drawdown(trades, session.starting_equity_usd)
    hold = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "label": a.label,
        "model": a.model,
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": a.days},
        "params": {"confidence_threshold": a.threshold},
        "bars": len(records),
        "decisions": run.decisions,
        "replay_seconds": round(elapsed, 1),
        "overall": group_stats(trades),
        "max_drawdown_usd": dd_usd,
        "max_drawdown_pct": dd_pct,
        "avg_hold_minutes": round(sum(hold) / len(hold), 1) if hold else None,
        "by_direction": by(lambda t: t.direction),
        "by_exit_reason": by(lambda t: t.exit_reason),
        "by_session": by(lambda t: session_of(t.entry_time.hour)),
        "by_weekday": by(lambda t: t.entry_time.strftime("%a")),
        "by_week": by(lambda t: t.entry_time.strftime("%G-W%V")),
    }
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"bt_{start:%Y%m%d}_{end:%Y%m%d}_th{a.threshold}{('_' + a.label) if a.label else ''}.json"
    (out / name).write_text(json.dumps(report, indent=2, default=str))
    o = report["overall"]
    print(f"{name}: trades={o['trades']} win={o['win_rate']} pnl=${o['pnl_usd']} pf={o['profit_factor']} "
          f"dd={dd_pct}% bars={len(records)} {elapsed:.0f}s")


if __name__ == "__main__":
    main()
