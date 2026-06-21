"""
Day-of-Week (Calendar) Effect — Multi-Ticker Study
====================================================

Tests whether daily returns differ systematically by day of the week,
across a basket of tickers, and whether any effect found survives
transaction costs when turned into a trading rule.

Hypothesis
----------
Classic finance literature claims a "Monday effect" (lower/negative average
returns on Mondays) and sometimes a "Friday effect" (higher average returns
before the weekend). This script tests that empirically and asks: even if
the effect is statistically real, is it big enough to trade?

Design Choices
---------------
- Daily close-to-close returns (no intraday)
- Signal computed at t, applied at t+1 (no look-ahead bias)
- 5 basis points per trade transaction cost, consistent with Weeks 1-2
- One-way long/flat rule: long only on the single best day-of-week found
  in-sample, flat otherwise. No leverage, no shorting.
- Statistical test: one-way ANOVA across the 5 day-of-week return groups,
  plus pairwise t-tests for the best vs. worst day.
- Benchmark: buy-and-hold the same ticker over the same window.
- Multiple tickers tested independently — the question is whether the
  effect is a robust, recurring pattern or a single-ticker fluke.

Usage
-----
    pip install -r ../requirements.txt
    python strategy.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

# ============================================================
# CONFIG
# ============================================================
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GLD"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 10_000
TRANSACTION_COST_BPS = 5      # 5 bps = 0.05% per trade, matches Week 1-2
TRADING_DAYS_PER_YEAR = 252
SIGNIFICANCE_LEVEL = 0.05


# ============================================================
# DATA
# ============================================================
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted close from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")
    # Recent yfinance versions return a MultiIndex on columns even for a single
    # ticker (e.g. ("Close", "SPY")). Flatten to a plain Index before renaming.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"
    return df


# ============================================================
# CALENDAR EFFECT ANALYSIS
# ============================================================
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach daily returns and day-of-week label."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["dow"] = df.index.dayofweek  # Monday=0 ... Friday=4
    df["dow_name"] = df.index.day_name()
    return df.dropna(subset=["returns"])


def dow_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, and count of returns grouped by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    g = df.groupby("dow_name")["returns"]
    summary = pd.DataFrame({
        "mean_return": g.mean(),
        "std_return": g.std(),
        "n_obs": g.count(),
    }).reindex(order)
    summary["annualized_mean"] = summary["mean_return"] * TRADING_DAYS_PER_YEAR
    return summary


def calendar_significance_test(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA across the 5 day-of-week groups, plus a t-test
    comparing the best vs. worst day found in-sample.

    Returns a dict with the test statistics and the identified
    best/worst days, so the caller can decide whether the effect
    is worth trading.
    """
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    groups = [df.loc[df["dow_name"] == d, "returns"].dropna().values for d in order]

    f_stat, p_value = stats.f_oneway(*groups)

    means = {d: g.mean() for d, g in zip(order, groups)}
    best_day = max(means, key=means.get)
    worst_day = min(means, key=means.get)

    best_returns = df.loc[df["dow_name"] == best_day, "returns"]
    worst_returns = df.loc[df["dow_name"] == worst_day, "returns"]
    t_stat, t_pvalue = stats.ttest_ind(best_returns, worst_returns, equal_var=False)

    return {
        "anova_f_stat": f_stat,
        "anova_p_value": p_value,
        "best_day": best_day,
        "worst_day": worst_day,
        "best_vs_worst_t_stat": t_stat,
        "best_vs_worst_p_value": t_pvalue,
        "significant": p_value < SIGNIFICANCE_LEVEL,
    }


# ============================================================
# STRATEGY / BACKTEST
# ============================================================
def generate_positions(df: pd.DataFrame, best_day: str) -> pd.Series:
    """
    Long-only rule: hold the asset only on the single best day-of-week
    identified in-sample (e.g. only Tuesdays), flat every other day.

    This is intentionally a simple, literal test of the calendar effect —
    not an attempt to build the most profitable possible rule.
    """
    return (df["dow_name"] == best_day).astype(int)


def backtest(df: pd.DataFrame, positions: pd.Series, cost_bps: float = TRANSACTION_COST_BPS) -> pd.DataFrame:
    """
    Apply a 1-day lag to positions (signal known at close of t, traded at t+1),
    subtract transaction costs on position changes, and compute equity curves
    for both the strategy and a buy-and-hold benchmark.
    """
    result = df.copy()
    result["position"] = positions
    result["trade"] = result["position"].diff().abs().fillna(0)

    lagged_position = result["position"].shift(1).fillna(0)
    cost = result["trade"].shift(1).fillna(0) * (cost_bps / 10_000)

    result["strategy_returns"] = lagged_position * result["returns"] - cost
    result["benchmark_returns"] = result["returns"]

    result["strategy_equity"] = INITIAL_CAPITAL * (1 + result["strategy_returns"]).cumprod()
    result["benchmark_equity"] = INITIAL_CAPITAL * (1 + result["benchmark_returns"]).cumprod()

    return result


# ============================================================
# METRICS
# ============================================================
def compute_metrics(returns: pd.Series, positions: pd.Series = None) -> dict:
    """Annualized return, volatility, Sharpe, max drawdown, Calmar, time in market."""
    returns = returns.dropna()
    equity = (1 + returns).cumprod()

    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (returns.mean() * TRADING_DAYS_PER_YEAR) / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    time_in_market = (positions.shift(1).fillna(0) != 0).mean() if positions is not None else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "time_in_market": time_in_market,
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_dow_means(summary: pd.DataFrame, ticker: str, out_path: Path):
    """Bar chart of mean return by day of week, annualized for readability."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in summary["annualized_mean"]]
    ax.bar(summary.index, summary["annualized_mean"] * 100, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ticker}: Annualized Mean Return by Day of Week")
    ax.set_ylabel("Annualized mean return (%)")
    ax.set_xlabel("Day of week")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_equity_curve(result: pd.DataFrame, ticker: str, best_day: str, out_path: Path):
    """Strategy vs. buy-and-hold equity curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.index, result["benchmark_equity"], label="Buy & Hold", color="#1f77b4")
    ax.plot(result.index, result["strategy_equity"], label=f"Long-only-{best_day}", color="#ff7f0e")
    ax.set_title(f"{ticker}: Day-of-Week Strategy vs. Buy & Hold")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def run_for_ticker(ticker: str, start: str, end: str, out_dir: Path) -> dict:
    """Run the full pipeline for a single ticker and return a results row."""
    print(f"\n=== {ticker} ===")
    raw = fetch_data(ticker, start, end)
    df = add_calendar_features(raw)

    summary = dow_summary(df)
    test = calendar_significance_test(df)

    print(summary.round(5))
    print(f"ANOVA p-value: {test['anova_p_value']:.4f} "
          f"({'significant' if test['significant'] else 'not significant'} at {SIGNIFICANCE_LEVEL})")
    print(f"Best day in-sample: {test['best_day']} | Worst day: {test['worst_day']}")

    positions = generate_positions(df, test["best_day"])
    result = backtest(df, positions)

    strat_metrics = compute_metrics(result["strategy_returns"], positions)
    bench_metrics = compute_metrics(result["benchmark_returns"])

    plot_dow_means(summary, ticker, out_dir / f"{ticker}_dow_means.png")
    plot_equity_curve(result, ticker, test["best_day"], out_dir / f"{ticker}_equity_curve.png")

    result.to_csv(out_dir / f"{ticker}_backtest.csv")
    summary.to_csv(out_dir / f"{ticker}_dow_summary.csv")

    return {
        "ticker": ticker,
        "best_day": test["best_day"],
        "worst_day": test["worst_day"],
        "anova_p_value": test["anova_p_value"],
        "significant": test["significant"],
        "strategy_total_return": strat_metrics["total_return"],
        "strategy_sharpe": strat_metrics["sharpe_ratio"],
        "strategy_max_dd": strat_metrics["max_drawdown"],
        "benchmark_total_return": bench_metrics["total_return"],
        "benchmark_sharpe": bench_metrics["sharpe_ratio"],
        "benchmark_max_dd": bench_metrics["max_drawdown"],
    }


def main():
    parser = argparse.ArgumentParser(description="Day-of-week calendar effect study")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="Tickers to test")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    rows = [run_for_ticker(t, args.start, args.end, out_dir) for t in args.tickers]

    overview = pd.DataFrame(rows).set_index("ticker")
    overview.to_csv(out_dir / "overview_all_tickers.csv")

    print("\n=== Summary across tickers ===")
    print(overview.round(4))
    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()