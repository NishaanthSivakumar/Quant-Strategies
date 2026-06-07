"""
Moving Average Crossover Backtest on SPY
=========================================

Tests the classic 10/50-day MA crossover strategy on the SPY ETF (2019-2024).

Hypothesis
----------
When the 10-day moving average crosses ABOVE the 50-day moving average,
the asset is in an uptrend — go long. When it crosses back below, exit to cash.

Design Choices
--------------
- Daily close-to-close returns (no intraday)
- Signal computed at t, applied at t+1 (no look-ahead bias)
- 5 basis points per trade transaction cost (~realistic for SPY retail)
- Long-only, no leverage, no shorting
- Benchmark: buy-and-hold SPY over the same window

Usage
-----
    pip install yfinance pandas numpy matplotlib
    python ma_crossover_backtest.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
TICKER = "SPY"
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
SHORT_WINDOW = 10
LONG_WINDOW = 50
INITIAL_CAPITAL = 10_000
TRANSACTION_COST_BPS = 5      # 5 bps = 0.05% per trade
TRADING_DAYS_PER_YEAR = 252


# ============================================================
# DATA
# ============================================================
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted close from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=True)
    # yfinance sometimes returns multi-index columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].rename(columns={"Close": "close"})


# ============================================================
# STRATEGY
# ============================================================
def compute_signals(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """Generate long/flat signals from a moving-average crossover."""
    out = df.copy()
    out["ma_short"] = out["close"].rolling(short).mean()
    out["ma_long"] = out["close"].rolling(long).mean()
    out["signal"] = (out["ma_short"] > out["ma_long"]).astype(int)
    # Trade flag: 1 on days the position changes (entry or exit)
    out["trade"] = out["signal"].diff().abs().fillna(0)
    return out


# ============================================================
# BACKTEST
# ============================================================
def run_backtest(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Apply signals at t+1 and subtract transaction costs on trade days."""
    out = df.copy()
    out["returns"] = out["close"].pct_change()
    # Signal generated at close of day t → traded at open of day t+1
    out["strategy_returns"] = out["signal"].shift(1) * out["returns"]
    # Subtract cost on the day a trade occurs
    cost = cost_bps / 10_000
    out["strategy_returns"] -= out["trade"].shift(1).fillna(0) * cost
    # Equity curves
    out["equity_bh"] = (1 + out["returns"]).cumprod() * INITIAL_CAPITAL
    out["equity_strategy"] = (1 + out["strategy_returns"]).cumprod() * INITIAL_CAPITAL
    return out


# ============================================================
# METRICS
# ============================================================
def compute_metrics(returns: pd.Series, label: str) -> dict:
    """Compute total return, CAGR, Sharpe, max drawdown."""
    r = returns.dropna()
    n_years = len(r) / TRADING_DAYS_PER_YEAR
    total_return = (1 + r).prod() - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (r.mean() / r.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if r.std() > 0 else 0.0
    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "Strategy": label,
        "Total Return": f"{total_return*100:6.1f}%",
        "CAGR":         f"{cagr*100:6.1f}%",
        "Volatility":   f"{vol*100:6.1f}%",
        "Sharpe":       f"{sharpe:6.2f}",
        "Max Drawdown": f"{max_dd*100:6.1f}%",
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_results(df: pd.DataFrame) -> None:
    """Two-panel chart: equity curves + price with MAs."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2, 1]})

    # --- Panel 1: equity curves ---
    axes[0].plot(df.index, df["equity_bh"], label="Buy & Hold",
                 linewidth=2.2, color="#2563eb")
    axes[0].plot(df.index, df["equity_strategy"], label="MA Crossover",
                 linewidth=2.2, color="#dc2626")
    axes[0].set_title(f"Equity Curve — ${INITIAL_CAPITAL:,} invested in {TICKER} "
                      f"({START_DATE[:4]}–{END_DATE[:4]})", fontsize=13)
    axes[0].set_ylabel("Portfolio Value (USD)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # --- Panel 2: price with MAs and shaded long periods ---
    axes[1].plot(df.index, df["close"], label="SPY Close",
                 color="black", linewidth=0.9)
    axes[1].plot(df.index, df["ma_short"], label=f"MA{SHORT_WINDOW}",
                 alpha=0.85, color="#16a34a")
    axes[1].plot(df.index, df["ma_long"], label=f"MA{LONG_WINDOW}",
                 alpha=0.85, color="#ea580c")
    ymin, ymax = df["close"].min(), df["close"].max()
    axes[1].fill_between(df.index, ymin, ymax,
                         where=df["signal"] == 1,
                         alpha=0.08, color="green", label="In market")
    axes[1].set_ylim(ymin, ymax)
    axes[1].set_title("Price & Moving Averages")
    axes[1].set_ylabel("Price (USD)")
    axes[1].legend(loc="upper left", ncol=4)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ma_crossover_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nChart saved to ma_crossover_results.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Downloading {TICKER} from {START_DATE} to {END_DATE}...")
    df = fetch_data(TICKER, START_DATE, END_DATE)
    df = compute_signals(df, SHORT_WINDOW, LONG_WINDOW)
    df = run_backtest(df, TRANSACTION_COST_BPS)

    metrics_bh = compute_metrics(df["returns"], "Buy & Hold")
    metrics_st = compute_metrics(df["strategy_returns"], "MA Crossover")

    print("\n" + "=" * 60)
    print(f"  RESULTS  ({START_DATE} → {END_DATE})")
    print("=" * 60)
    summary = pd.DataFrame([metrics_bh, metrics_st]).set_index("Strategy").T
    print(summary.to_string())

    n_trades = int(df["trade"].sum())
    pct_in_market = df["signal"].mean() * 100
    print(f"\nTotal trades : {n_trades}")
    print(f"Time in mkt  : {pct_in_market:.1f}%")
    print(f"Cost assumed : {TRANSACTION_COST_BPS} bps per trade")

    plot_results(df)


if __name__ == "__main__":
    main()
