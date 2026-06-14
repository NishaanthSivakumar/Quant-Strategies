"""
Week 2 — RSI Mean Reversion
============================

Hypothesis
----------
When the 14-day RSI drops below 30, short-term selling is exhausted and the
asset is likely to bounce — enter long. Exit when RSI crosses above 70, on
the assumption that the mean-reversion move has played out.

Why this should (or shouldn't) work
-----------------------------------
Mean reversion exists because markets overreact to short-term news. The bid
falls faster than fundamentals justify, and once forced sellers are done
trading, price drifts back up. The catch: in a strong downtrend, "oversold"
just keeps getting more oversold. This strategy is regime-dependent.

Reference: Connors & Alvarez (2009), *Short-Term Trading Strategies That Work*.

Methodology (per repo standard)
-------------------------------
- Wilder's RSI (the textbook definition) with period = 14
- Standard thresholds: enter < 30, exit > 70
- No parameter tuning by default — robustness tested separately
- 5 bps per side transaction cost on every position change
- Signals are shifted by 1 day to prevent look-ahead bias
- Benchmark is buy-and-hold on the same ticker

Run
---
    python strategy.py --ticker AAPL --start 2010-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
RSI_PERIOD = 14
OVERSOLD = 30.0
OVERBOUGHT = 70.0
COST_BPS = 5.0          # one-way, per position change
ANN_FACTOR = 252        # trading days per year


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV via yfinance, auto-adjusted for splits/dividends."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    # yfinance sometimes returns a multi-index column when only one ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


# -----------------------------------------------------------------------------
# Indicator
# -----------------------------------------------------------------------------
def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI using Wilder's smoothing (an EWMA with alpha = 1/period).
    This matches the original Wilder (1978) definition used in most charting
    platforms — distinct from the simple-moving-average variant.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# -----------------------------------------------------------------------------
# Signal logic
# -----------------------------------------------------------------------------
def generate_positions(
    rsi: pd.Series,
    oversold: float = OVERSOLD,
    overbought: float = OVERBOUGHT,
) -> pd.Series:
    """
    Enter long when RSI drops below `oversold`, exit when RSI rises above
    `overbought`. State is held in between. Returned series is shifted by one
    day so today's signal trades using tomorrow's return (no look-ahead).
    """
    raw = pd.Series(0, index=rsi.index, dtype=int)
    in_trade = False
    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            continue
        if not in_trade and r < oversold:
            in_trade = True
        elif in_trade and r > overbought:
            in_trade = False
        raw.iloc[i] = 1 if in_trade else 0

    return raw.shift(1).fillna(0).astype(int)


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
def backtest(
    prices: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Apply position to daily returns, deduct cost on every position change."""
    close = prices["Close"]
    daily_ret = close.pct_change().fillna(0.0)

    strat_ret_gross = position * daily_ret

    # Charge cost_bps on every position change (entry and exit each cost the spread)
    trades = position.diff().abs().fillna(float(position.iloc[0]))
    cost = trades * (cost_bps / 10_000.0)

    strat_ret_net = strat_ret_gross - cost

    return pd.DataFrame(
        {
            "close": close,
            "position": position,
            "asset_ret": daily_ret,
            "strat_ret": strat_ret_net,
            "equity": (1 + strat_ret_net).cumprod(),
            "benchmark": (1 + daily_ret).cumprod(),
        }
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _trade_returns(result: pd.DataFrame) -> list[float]:
    """Return per-trade % returns by walking position changes."""
    out = []
    in_trade = False
    entry_eq = None
    for i in range(len(result)):
        pos = result["position"].iloc[i]
        eq = result["equity"].iloc[i]
        if pos == 1 and not in_trade:
            in_trade, entry_eq = True, eq
        elif pos == 0 and in_trade:
            in_trade = False
            out.append(eq / entry_eq - 1)
    # Open trade at end of series — close at final equity
    if in_trade and entry_eq is not None:
        out.append(result["equity"].iloc[-1] / entry_eq - 1)
    return out


def compute_metrics(result: pd.DataFrame) -> dict:
    strat = result["strat_ret"]
    bench = result["asset_ret"]

    total = result["equity"].iloc[-1] - 1
    bench_total = result["benchmark"].iloc[-1] - 1

    sharpe = (strat.mean() / strat.std() * np.sqrt(ANN_FACTOR)) if strat.std() > 0 else 0.0
    bench_sharpe = (bench.mean() / bench.std() * np.sqrt(ANN_FACTOR)) if bench.std() > 0 else 0.0

    rolling_max = result["equity"].cummax()
    drawdown = result["equity"] / rolling_max - 1
    max_dd = drawdown.min()
    calmar = (total / abs(max_dd)) if max_dd < 0 else float("nan")

    trades = _trade_returns(result)
    n_trades = len(trades)
    win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0.0
    avg_trade = float(np.mean(trades)) if trades else 0.0
    time_in_market = float(result["position"].mean())

    return {
        "total_return": total,
        "benchmark_return": bench_total,
        "excess_return": total - bench_total,
        "sharpe": sharpe,
        "benchmark_sharpe": bench_sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade,
        "time_in_market": time_in_market,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(result: pd.DataFrame, rsi: pd.Series, ticker: str, out_path: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax1.plot(result.index, result["equity"], label="RSI Mean Reversion", linewidth=1.5)
    ax1.plot(result.index, result["benchmark"], label=f"{ticker} Buy & Hold", linewidth=1.5, alpha=0.7)
    ax1.set_ylabel("Equity (start = 1.0)")
    ax1.set_title(f"RSI(14) Mean Reversion vs Buy & Hold — {ticker}")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(rsi.index, rsi, linewidth=0.8, color="#555")
    ax2.axhline(OVERSOLD, linestyle="--", color="green", alpha=0.6, label=f"Oversold ({OVERSOLD:.0f})")
    ax2.axhline(OVERBOUGHT, linestyle="--", color="red", alpha=0.6, label=f"Overbought ({OVERBOUGHT:.0f})")
    ax2.set_ylabel("RSI(14)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.fill_between(result.index, result["position"], 0, step="post", alpha=0.4)
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Date")
    ax3.set_yticks([0, 1])
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Week 2 — RSI Mean Reversion backtest")
    p.add_argument("--ticker", default=DEFAULT_TICKER)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--period", type=int, default=RSI_PERIOD)
    p.add_argument("--oversold", type=float, default=OVERSOLD)
    p.add_argument("--overbought", type=float, default=OVERBOUGHT)
    p.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = p.parse_args()

    out_dir = Path(__file__).parent
    

    print(f"Fetching {args.ticker} ({args.start} → {args.end})...")
    prices = fetch_prices(args.ticker, args.start, args.end)

    rsi = rsi_wilder(prices["Close"], period=args.period)
    position = generate_positions(rsi, args.oversold, args.overbought)
    result = backtest(prices, position, cost_bps=args.cost_bps)
    result["rsi"] = rsi

    m = compute_metrics(result)

    print(f"\n--- {args.ticker} | RSI({args.period}) {args.oversold:.0f}/{args.overbought:.0f} | {args.cost_bps} bps ---")
    print(f"Total return:       {m['total_return']:>9.2%}")
    print(f"Buy & Hold return:  {m['benchmark_return']:>9.2%}")
    print(f"Excess:             {m['excess_return']:>9.2%}")
    print(f"Sharpe (strategy):  {m['sharpe']:>9.2f}")
    print(f"Sharpe (B&H):       {m['benchmark_sharpe']:>9.2f}")
    print(f"Max drawdown:       {m['max_drawdown']:>9.2%}")
    print(f"Calmar:             {m['calmar']:>9.2f}")
    print(f"Trades:             {m['n_trades']:>9d}")
    print(f"Win rate:           {m['win_rate']:>9.2%}")
    print(f"Avg trade:          {m['avg_trade_return']:>9.2%}")
    print(f"Time in market:     {m['time_in_market']:>9.2%}")

    # Persist outputs
    result.to_csv(out_dir / "backtest.csv")
    pd.Series(m).to_csv(out_dir / "metrics.csv", header=["value"])
    plot_results(result, rsi, args.ticker, out_dir / "equity_curve.png")
    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()
