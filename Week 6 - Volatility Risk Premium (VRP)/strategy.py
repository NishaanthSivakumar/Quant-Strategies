"""
Week 6 — Volatility Risk Premium (VRP)
=======================================

Tests whether implied volatility (VIX) systematically overstates future
realised volatility, and whether a simple strategy that harvests that
gap earns a risk-adjusted premium — and what happens when it doesn't.

Hypothesis
----------
Option-implied volatility is, on average, higher than the subsequent
realised volatility of the underlying. This gap — the variance or
volatility risk premium — exists because investors demand compensation
for bearing downside risk, effectively overpaying for portfolio
insurance. A strategy that *sells* this insurance should earn a
persistent premium, punctuated by occasional severe drawdowns when
realised vol spikes above implied.

Design Choices
--------------
- Implied vol proxy: VIX (CBOE 30-day implied vol on SPX options).
- Realised vol: 21-trading-day rolling standard deviation of SPY daily
  log returns, annualised (√252).
- Signal: VRP = VIX − realised vol. When VRP > 0, implied > realised →
  the premium exists → the strategy is "short vol" (long SPY as proxy).
  When VRP ≤ 0, step aside to T-bills (risk-free proxy).
- This is a *proxy* backtest. We are NOT actually selling options or
  variance swaps. We use SPY as a stand-in for the return you'd earn
  by being exposed to the equity market when the vol premium is
  positive. Real vol-selling strategies (short straddles, short VIX
  futures) have different P&L profiles, leverage, and margin
  requirements.
- Benchmark: buy-and-hold SPY over the same period.
- Transaction cost: 10 bps per switch (matching Week 4+ convention).
- Rebalance: daily signal, but only trade on signal *changes* (not
  daily re-entry).

Data
----
- ^VIX via yfinance: daily VIX close.
- SPY via yfinance: daily adjusted close prices.
- Risk-free rate: approximated at 0 for simplicity (conservative for
  the strategy since cash periods earn nothing in the backtest).

Reference
---------
Carr, P. & Wu, L. (2009). Variance Risk Premiums. Review of Financial
Studies, 22(3), 1311–1341.

Usage
-----
    python strategy.py
    python strategy.py --lookback 10          # 10-day realised vol
    python strategy.py --threshold 0          # default: enter when VRP > 0
    python strategy.py --threshold 2          # require 2-point VRP cushion
    python strategy.py --cost-bps 5           # match Weeks 1–3
    python strategy.py --start 2006-01-01 --end 2023-12-31
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_START = "2006-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_LOOKBACK = 21          # trading days for realised vol
DEFAULT_THRESHOLD = 0.0        # VRP must exceed this to go long
DEFAULT_COST_BPS = 10          # one-way cost per switch
ANNUALISE = np.sqrt(252)


def parse_args():
    p = argparse.ArgumentParser(
        description="Week 6 — Volatility Risk Premium backtest"
    )
    p.add_argument("--start", default=DEFAULT_START,
                   help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", default=DEFAULT_END,
                   help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                   help="Trading days for realised vol (default: 21)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Minimum VRP to go long (default: 0)")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                   help="One-way transaction cost in bps (default: 10)")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────

def fetch_data(start: str, end: str) -> pd.DataFrame:
    """Download VIX and SPY, merge on date, drop NaNs."""
    print(f"Fetching SPY and ^VIX from {start} to {end} ...")

    spy = yf.download("SPY", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame({
        "spy_close": spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"],
        "vix_close": vix["Close"],
    }).dropna()

    print(f"  {len(df)} trading days loaded "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── Signal ───────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame, lookback: int,
                    threshold: float) -> pd.DataFrame:
    """
    Compute realised vol, VRP, and the long/cash signal.
    """
    # Daily log returns on SPY
    df = df.copy()
    df["log_ret"] = np.log(df["spy_close"] / df["spy_close"].shift(1))

    # Realised vol: rolling std of log returns, annualised, × 100 to
    # match VIX scale (VIX is quoted in percentage points)
    df["realised_vol"] = (
        df["log_ret"].rolling(lookback).std() * ANNUALISE * 100
    )

    # VRP = implied − realised
    df["vrp"] = df["vix_close"] - df["realised_vol"]

    # Signal: 1 = long SPY (VRP above threshold), 0 = cash
    df["signal"] = (df["vrp"] > threshold).astype(int)

    df.dropna(inplace=True)
    return df


# ── Backtest ─────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """
    Run the long/cash strategy and the buy-and-hold benchmark.
    """
    df = df.copy()
    cost = cost_bps / 10_000

    # Daily simple returns on SPY
    df["spy_ret"] = df["spy_close"].pct_change()

    # Position is yesterday's signal applied to today's return
    # (signal is known at close; we trade next open, proxied as next close)
    df["position"] = df["signal"].shift(1)

    # Identify switches
    df["switch"] = (df["position"] != df["position"].shift(1)).astype(int)

    # Strategy daily return
    df["strat_ret"] = df["position"] * df["spy_ret"] - df["switch"] * cost

    # Cumulative returns (growth of $1)
    df["strat_cum"] = (1 + df["strat_ret"]).cumprod()
    df["bench_cum"] = (1 + df["spy_ret"]).cumprod()

    df.dropna(inplace=True)
    return df


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, label: str,
                    ret_col: str) -> dict:
    """Standard summary stats for a return series."""
    rets = df[ret_col].dropna()
    cum = (1 + rets).cumprod()
    total_days = len(rets)
    years = total_days / 252

    total_ret = cum.iloc[-1] - 1
    cagr = (cum.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    ann_vol = rets.std() * ANNUALISE
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (of days in the market)
    in_market = rets[rets != 0]
    win_rate = (in_market > 0).mean() if len(in_market) > 0 else 0

    return {
        "label": label,
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "days_in_market": int((df.get("position", pd.Series(1)) == 1).sum()),
        "pct_time_in_market": round(
            (df.get("position", pd.Series(1)) == 1).mean() * 100, 1
        ),
        "num_switches": int(df.get("switch", pd.Series(0)).sum()),
        "total_days": total_days,
        "years": round(years, 1),
    }


# ── VRP analysis ─────────────────────────────────────────────────────

def vrp_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics on the VRP itself."""
    vrp = df["vrp"].dropna()
    summary = {
        "mean": round(vrp.mean(), 2),
        "median": round(vrp.median(), 2),
        "std": round(vrp.std(), 2),
        "pct_positive": round((vrp > 0).mean() * 100, 1),
        "min": round(vrp.min(), 2),
        "max": round(vrp.max(), 2),
        "p5": round(vrp.quantile(0.05), 2),
        "p95": round(vrp.quantile(0.95), 2),
    }
    return pd.DataFrame([summary])


# ── Drawdown analysis ────────────────────────────────────────────────

def worst_drawdowns(df: pd.DataFrame, ret_col: str,
                    n: int = 5) -> pd.DataFrame:
    """Identify the N worst drawdown periods."""
    cum = (1 + df[ret_col]).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    # Find drawdown periods
    in_dd = dd < 0
    periods = []
    start = None
    for i, (idx, val) in enumerate(in_dd.items()):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            end = dd.loc[start:idx].idxmin()
            trough = dd.loc[end]
            periods.append({
                "start": start.strftime("%Y-%m-%d"),
                "trough": end.strftime("%Y-%m-%d"),
                "end": idx.strftime("%Y-%m-%d"),
                "depth_pct": round(trough * 100, 2),
                "length_days": (idx - start).days,
            })
            start = None

    # Handle ongoing drawdown at end of series
    if start is not None:
        end = dd.loc[start:].idxmin()
        trough = dd.loc[end]
        periods.append({
            "start": start.strftime("%Y-%m-%d"),
            "trough": end.strftime("%Y-%m-%d"),
            "end": "ongoing",
            "depth_pct": round(trough * 100, 2),
            "length_days": (df.index[-1] - start).days,
        })

    return (pd.DataFrame(periods)
            .sort_values("depth_pct")
            .head(n)
            .reset_index(drop=True))


# ── Plotting ─────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, outdir: str):
    """Generate equity curve and VRP diagnostic plots."""

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["strat_cum"], label="VRP Strategy", linewidth=1.2)
    ax.plot(df.index, df["bench_cum"], label="Buy & Hold SPY",
            linewidth=1.2, alpha=0.7)
    ax.set_title("Week 6 — Volatility Risk Premium: Growth of $1",
                 fontsize=13)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved equity_curve.png")

    # 2. VRP time series with signal overlay
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["vix_close"], label="VIX (implied)",
             linewidth=0.8, alpha=0.8)
    ax1.plot(df.index, df["realised_vol"], label="Realised vol (21d)",
             linewidth=0.8, alpha=0.8)
    ax1.set_ylabel("Volatility (%)")
    ax1.set_title("Implied vs Realised Volatility", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(df.index, df["vrp"], 0,
                     where=df["vrp"] > 0, color="green", alpha=0.3,
                     label="VRP > 0 (long)")
    ax2.fill_between(df.index, df["vrp"], 0,
                     where=df["vrp"] <= 0, color="red", alpha=0.3,
                     label="VRP ≤ 0 (cash)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("VRP (VIX − Realised)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "vrp_diagnostic.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved vrp_diagnostic.png")

    # 3. Drawdown comparison
    fig, ax = plt.subplots(figsize=(12, 4))

    strat_cum = (1 + df["strat_ret"]).cumprod()
    bench_cum = (1 + df["spy_ret"]).cumprod()

    strat_dd = (strat_cum - strat_cum.cummax()) / strat_cum.cummax() * 100
    bench_dd = (bench_cum - bench_cum.cummax()) / bench_cum.cummax() * 100

    ax.fill_between(df.index, strat_dd, 0, alpha=0.4, label="Strategy DD")
    ax.fill_between(df.index, bench_dd, 0, alpha=0.3, label="B&H DD")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Comparison", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drawdown.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved drawdown.png")

    # 4. VRP distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["vrp"], bins=80, edgecolor="none", alpha=0.7, color="steelblue")
    ax.axvline(df["vrp"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['vrp'].mean():.1f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("VRP (VIX − Realised Vol)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of the Volatility Risk Premium", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "vrp_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved vrp_distribution.png")


# ── Annual breakdown ─────────────────────────────────────────────────

def annual_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year strategy vs benchmark returns."""
    df = df.copy()
    df["year"] = df.index.year

    rows = []
    for yr, grp in df.groupby("year"):
        s_ret = (1 + grp["strat_ret"]).prod() - 1
        b_ret = (1 + grp["spy_ret"]).prod() - 1
        pct_long = grp["position"].mean() * 100
        switches = grp["switch"].sum()
        avg_vrp = grp["vrp"].mean()
        rows.append({
            "year": yr,
            "strategy_pct": round(s_ret * 100, 2),
            "benchmark_pct": round(b_ret * 100, 2),
            "excess_pct": round((s_ret - b_ret) * 100, 2),
            "pct_time_long": round(pct_long, 1),
            "switches": int(switches),
            "avg_vrp": round(avg_vrp, 2),
        })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Create output directory
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    # 1. Fetch data
    df = fetch_data(args.start, args.end)

    # 2. Compute signals
    df = compute_signals(df, args.lookback, args.threshold)
    print(f"  Signal computed: {args.lookback}-day realised vol, "
          f"threshold = {args.threshold}")

    # 3. VRP summary
    vrp_stats = vrp_summary(df)
    print(f"\n  VRP summary:")
    print(f"    Mean:  {vrp_stats['mean'].values[0]}")
    print(f"    Median: {vrp_stats['median'].values[0]}")
    print(f"    % positive: {vrp_stats['pct_positive'].values[0]}%")
    vrp_stats.to_csv(os.path.join(outdir, "vrp_summary.csv"), index=False)

    # 4. Backtest
    df = backtest(df, args.cost_bps)

    # 5. Metrics
    strat_m = compute_metrics(df, "VRP Strategy", "strat_ret")
    bench_m = compute_metrics(df, "Buy & Hold SPY", "spy_ret")

    metrics = pd.DataFrame([strat_m, bench_m])
    metrics.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for m in [strat_m, bench_m]:
        print(f"\n  {m['label']}")
        print(f"    Total return:     {m['total_return_pct']:>8.2f}%")
        print(f"    CAGR:             {m['cagr_pct']:>8.2f}%")
        print(f"    Ann. volatility:  {m['ann_volatility_pct']:>8.2f}%")
        print(f"    Sharpe:           {m['sharpe']:>8.3f}")
        print(f"    Max drawdown:     {m['max_drawdown_pct']:>8.2f}%")
        print(f"    Calmar:           {m['calmar']:>8.3f}")
        if "pct_time_in_market" in m:
            print(f"    Time in market:   {m['pct_time_in_market']:>7.1f}%")
            print(f"    Switches:         {m['num_switches']:>8d}")

    # 6. Annual breakdown
    annual = annual_returns(df)
    annual.to_csv(os.path.join(outdir, "annual_returns.csv"), index=False)
    print(f"\n  Annual returns:")
    print(annual.to_string(index=False))

    # 7. Worst drawdowns
    strat_dd = worst_drawdowns(df, "strat_ret", n=5)
    strat_dd.to_csv(os.path.join(outdir, "worst_drawdowns.csv"), index=False)
    print(f"\n  5 worst drawdowns (strategy):")
    print(strat_dd.to_string(index=False))

    # 8. Save daily data
    export_cols = ["spy_close", "vix_close", "realised_vol", "vrp",
                   "signal", "position", "spy_ret", "strat_ret",
                   "strat_cum", "bench_cum"]
    df[export_cols].to_csv(os.path.join(outdir, "backtest.csv"))
    print(f"\n  Saved backtest.csv ({len(df)} rows)")

    # 9. Plots
    print(f"\n  Generating plots ...")
    plot_results(df, outdir)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to {outdir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
