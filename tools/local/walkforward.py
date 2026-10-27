"""Run backtest_runner over consecutive weekly windows in parallel and merge.

Usage: python tools/local/walkforward.py --start 2026-07-01 --end 2026-10-24 --model PATH [--threshold 0.2] [--jobs 4]
Writes artifacts/backtests/<label>/summary.json + summary.md
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run_week(args, window, out_dir):
    start, end = window
    days = (end - start).total_seconds() / 86400
    cmd = [sys.executable, str(REPO / "tools/local/backtest_runner.py"), "--days", f"{days:g}",
           "--end", end.isoformat(), "--threshold", str(args.threshold),
           "--model", args.model, "--out-dir", str(out_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]
    print(line, flush=True)


def merge(out_dir):
    reports = [json.loads(p.read_text()) for p in sorted(out_dir.glob("bt_*.json"))]
    agg = {}
    for key in ("by_direction", "by_exit_reason", "by_session", "by_weekday"):
        g = {}
        for r in reports:
            for k, v in r[key].items():
                a = g.setdefault(k, {"trades": 0, "wins": 0, "pnl_usd": 0.0, "gp": 0.0, "gl": 0.0})
                a["trades"] += v["trades"]
                a["wins"] += round((v["win_rate"] or 0) * v["trades"])
                a["pnl_usd"] += v["pnl_usd"]
                if v["avg_win"]:
                    a["gp"] += v["avg_win"] * round((v["win_rate"] or 0) * v["trades"])
                if v["avg_loss"]:
                    a["gl"] += -v["avg_loss"] * (v["trades"] - round((v["win_rate"] or 0) * v["trades"]))
        agg[key] = {k: {"trades": a["trades"],
                        "win_rate": round(a["wins"] / a["trades"], 3) if a["trades"] else None,
                        "pnl_usd": round(a["pnl_usd"], 2),
                        "profit_factor": round(a["gp"] / a["gl"], 2) if a["gl"] else None}
                    for k, a in g.items()}
    weeks = [{"start": r["window"]["start"][:10], **r["overall"]} for r in reports]
    tot = sum(w["trades"] for w in weeks)
    pnl = sum(w["pnl_usd"] for w in weeks)
    eq = peak = 10_000.0
    dd = 0.0
    for w in weeks:
        eq += w["pnl_usd"]
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    summary = {
        "model": reports[0]["model"] if reports else None,
        "params": reports[0]["params"] if reports else None,
        "weeks": len(weeks),
        "trades": tot,
        "pnl_usd": round(pnl, 2),
        "winning_weeks": sum(1 for w in weeks if w["pnl_usd"] > 0),
        "max_weekly_close_drawdown_usd": round(dd, 2),
        **{k: v for k, v in agg.items()},
        "weekly": weeks,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    out_dir = REPO / "artifacts/backtests" / a.label
    out_dir.mkdir(parents=True, exist_ok=True)
    s, e = datetime.fromisoformat(a.start), datetime.fromisoformat(a.end)
    windows = []
    t = s
    while t < e:
        windows.append((t, min(t + timedelta(days=7), e)))
        t += timedelta(days=7)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda w: run_week(a, w, out_dir), windows))
    s = merge(out_dir)
    print(json.dumps({k: s[k] for k in ("weeks", "trades", "pnl_usd", "winning_weeks")}))


if __name__ == "__main__":
    main()
