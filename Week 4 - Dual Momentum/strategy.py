"""
Week 4 - Dual Momentum (cross-asset)
====================================

Monthly-Rebalanced Dual Momentum Strategy

This strategy is rebalanced at the end of each month, with portfolio allocations taking effect during the following month.

Strategy Logic

Portfolio allocations are determined at the end of each month and implemented during the following month.

1. Relative Momentum
    * Calculate the trailing 12-month return for each risky asset.
    * Select the asset with the highest trailing 12-month return.
2. Absolute Momentum Filter
    * Compare the selected asset’s 12-month return against the risk-free proxy (BIL, representing U.S. Treasury Bills).
    * If the selected asset outperforms BIL, allocate the portfolio to that risky asset.
    * Otherwise, move to a defensive allocation by investing in AGG (U.S. Aggregate Bond ETF).

When only one risky asset is provided, the strategy reduces to a traditional absolute momentum strategy with a bond fallback. 
When two or more risky assets are available, it implements Antonacci’s Global Equities Momentum (GEM) framework by selecting the strongest-performing equity asset while rotating into bonds whenever none of the risky assets outperform the risk-free rate. 
For example, with SPY and VEU, the strategy invests in the ETF with the stronger 12-month momentum, provided it exceeds the return of BIL; otherwise, it allocates to AGG.

Benchmark

Performance is compared against a buy-and-hold SPY portfolio.

Reference: Antonacci (2012), "Risk Premia Harvesting Through Dual Momentum".

"""

import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
 
 
# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def download_monthly_prices(tickers, start, end):
    """Download daily adjusted prices and resample to month-end.
 
    auto_adjust=True folds dividends into the price, which matters a lot
    for BIL and AGG where most of the return IS the yield.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
 
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
 
    prices = prices[tickers].dropna(how="all")
 
    try:
        monthly = prices.resample("ME").last()
    except ValueError:
        monthly = prices.resample("M").last()
 
    monthly = monthly.dropna()
    if monthly.empty:
        raise SystemExit(
            "No overlapping monthly data. Check tickers/date range "
            "(BIL only goes back to ~2007, AGG to ~2003, VEU to ~2007)."
        )
    return monthly
 
 
# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def build_benchmark(monthly_ret, risky, spec):
    """Return a monthly benchmark return series given a --benchmark spec."""
    if spec == "equal":
        # Equal-weight, monthly-rebalanced basket of the risky sleeve.
        return monthly_ret[risky].mean(axis=1)
    if spec == "first":
        return monthly_ret[risky[0]]
    # Otherwise treat spec as a single ticker.
    if spec not in monthly_ret.columns:
        raise SystemExit(f"Benchmark ticker '{spec}' not in downloaded data.")
    return monthly_ret[spec]
 
 
# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def backtest(prices, risky, bond, cash, lookback=12, cost_bps=10.0,
             benchmark="equal"):
    """Run the dual momentum backtest across one or more risky assets."""
    monthly_ret = prices.pct_change()
    momentum = prices.pct_change(lookback)
 
    holdings = []
    index = prices.index
    prev_hold = None
    cost = cost_bps / 10_000.0
 
    # Signal at t -> return realised over (t, t+1]. No lookahead.
    for i in range(lookback, len(index) - 1):
        sig_date = index[i]
        nxt_date = index[i + 1]
 
        cash_mom = momentum.loc[sig_date, cash]
 
        # Relative momentum: best risky asset this month.
        risky_mom = momentum.loc[sig_date, risky]
        best_risky = risky_mom.idxmax()
        best_risky_mom = risky_mom.max()
 
        # Absolute gate: winner must clear cash, else go to bonds.
        hold = best_risky if best_risky_mom > cash_mom else bond
 
        gross = monthly_ret.loc[nxt_date, hold]
        if prev_hold is not None and hold != prev_hold:
            gross -= cost  # one-way cost only when we switch
 
        holdings.append((nxt_date, hold, gross))
        prev_hold = hold
 
    out = pd.DataFrame(
        holdings, columns=["date", "holding", "strategy_ret"]
    ).set_index("date")
 
    bench_ret = build_benchmark(monthly_ret, risky, benchmark)
    out["benchmark_ret"] = bench_ret.reindex(out.index)
 
    out["strategy_equity"] = (1 + out["strategy_ret"]).cumprod()
    out["benchmark_equity"] = (1 + out["benchmark_ret"]).cumprod()
    return out
 
 
# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def performance_metrics(returns, periods_per_year=12, rf_annual=0.0):
    """Standard performance stats from a series of periodic returns."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
 
    total_return = (1 + r).prod() - 1
    years = n / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
 
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    rf_period = rf_annual / periods_per_year
    excess = r - rf_period
    sharpe = (
        excess.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)
        if r.std(ddof=1) > 0
        else np.nan
    )
 
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
 
    return {
        "Total return": total_return,
        "CAGR": cagr,
        "Ann. volatility": ann_vol,
        "Sharpe": sharpe,
        "Max drawdown": max_dd,
        "Months": n,
    }
 
 
def fmt_metrics(strat, bench):
    """Build a tidy comparison table."""
    df = pd.DataFrame({"strategy": strat, "benchmark": bench})
    pct_rows = ["Total return", "CAGR", "Ann. volatility", "Max drawdown"]
    disp = df.copy()
    for row in pct_rows:
        if row in disp.index:
            disp.loc[row] = disp.loc[row].map(lambda x: f"{x:.2%}")
    if "Sharpe" in disp.index:
        disp.loc["Sharpe"] = disp.loc["Sharpe"].map(lambda x: f"{x:.2f}")
    if "Months" in disp.index:
        disp.loc["Months"] = disp.loc["Months"].map(lambda x: f"{int(x)}")
    return df, disp
 
 
# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_results(result, bench_label, out_path="equity_curve.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.index, result["strategy_equity"],
            label="Dual Momentum", linewidth=1.8)
    ax.plot(result.index, result["benchmark_equity"],
            label=bench_label, linewidth=1.4, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Dual Momentum vs Benchmark (growth of $1, log scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
 
 
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Dual momentum backtest")
    p.add_argument("--risky", nargs="+", default=["SPY", "VEU"],
                   help="Risky asset(s). One = absolute momentum; "
                        "two+ = relative-momentum rotation (GEM).")
    p.add_argument("--bond", default="AGG", help="Defensive/bond ETF")
    p.add_argument("--cash", default="BIL", help="Risk-free / cash proxy")
    p.add_argument("--benchmark", default="equal",
                   help="'equal' (equal-wt basket of risky), 'first' "
                        "(first risky asset), or a specific TICKER.")
    p.add_argument("--start", default="2007-06-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=12,
                   help="Momentum lookback in months")
    p.add_argument("--cost-bps", type=float, default=10.0,
                   help="One-way transaction cost in basis points")
    args = p.parse_args()
 
    # Assemble the download set: risky + bond + cash + any custom bench.
    needed = args.risky + [args.bond, args.cash]
    if args.benchmark not in ("equal", "first"):
        needed.append(args.benchmark)
    tickers = list(dict.fromkeys(needed))
 
    print(f"Downloading: {tickers}")
    prices = download_monthly_prices(tickers, args.start, args.end)
    print(f"Monthly data: {prices.index[0]:%Y-%m} -> "
          f"{prices.index[-1]:%Y-%m} ({len(prices)} months)\n")
 
    result = backtest(
        prices,
        risky=args.risky,
        bond=args.bond,
        cash=args.cash,
        lookback=args.lookback,
        cost_bps=args.cost_bps,
        benchmark=args.benchmark,
    )
 
    strat = performance_metrics(result["strategy_ret"])
    bench = performance_metrics(result["benchmark_ret"])
    raw_df, disp_df = fmt_metrics(strat, bench)
 
    # Human-readable benchmark label.
    if args.benchmark == "equal":
        bench_label = "Equal-wt " + "/".join(args.risky)
    elif args.benchmark == "first":
        bench_label = f"Buy & Hold {args.risky[0]}"
    else:
        bench_label = f"Buy & Hold {args.benchmark}"
 
    print("Performance summary")
    print("=" * 40)
    print(f"Risky sleeve : {', '.join(args.risky)}")
    print(f"Benchmark    : {bench_label}\n")
    print(disp_df.to_string())
 
    in_bond = (result["holding"] == args.bond).mean()
    print(f"\nTime in bonds ({args.bond}): {in_bond:.1%}")
    for a in args.risky:
        share = (result["holding"] == a).mean()
        print(f"Time in {a}: {share:.1%}")
    switches = (result["holding"] != result["holding"].shift()).sum() - 1
    print(f"Number of switches: {int(switches)}")
 
    out_dir = Path(__file__).parent
    result.to_csv(out_dir / "backtest.csv")
    raw_df.to_csv(out_dir / "metrics.csv")
    plot_results(result, bench_label, out_dir / "equity_curve.png")
    print(f"\nWritten to {out_dir}/")
    print("  backtest.csv, metrics.csv, equity_curve.png")
 
 
if __name__ == "__main__":
    main()