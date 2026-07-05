"""
Week 5 — Post-Earnings Announcement Drift (PEAD)
==================================================

This project investigates whether stocks that exceed or fall short of analyst earnings expectations generate abnormal returns 
in the weeks following their earnings announcements. The phenomenon, known as Post-Earnings Announcement Drift (PEAD), 
is one of the most extensively documented market anomalies in empirical finance.

Hypothesis
----------
Financial markets do not fully incorporate earnings information immediately after an earnings announcement. As a result:

* Stocks reporting positive earnings surprises tend to continue outperforming over the subsequent weeks.
* Stocks reporting negative earnings surprises tend to continue underperforming over the same period.

This behavioural inefficiency was first documented by Ball & Brown (1968) and later strengthened by Bernard & Thomas (1989), 
with numerous subsequent studies confirming the persistence of the anomaly across different markets and time periods.

Design Choices
--------------
- Universe: a basket of large-cap, liquid names (configurable).
- Earnings surprise: (Actual EPS − Estimate EPS) / |Estimate EPS|
  (standardised unexpected earnings, SUE). We bucket into quintiles:
  Q5 = biggest positive surprise, Q1 = biggest negative.
- Signal computed AFTER the announcement date — no look-ahead bias.
  We assume the announcement is known by the next trading day's open.
- Holding period: configurable (default 20 trading days ≈ 1 month).
- Strategy: go long Q5 (top surprises), benchmark against equal-weight
  buy-and-hold of the same universe over the same windows.
- Transaction cost: 10 bps per round-trip, matching Week 4 convention.
- Abnormal return: stock return minus equal-weight universe return over
  the same holding window (simple market-adjustment).

Data
----
- yfinance: daily adjusted prices + earnings_dates (EPS actual/estimate).
  Note: yfinance earnings_dates coverage is imperfect — some tickers or
  quarters may be missing. The script logs what it finds and drops events
  with missing data rather than imputing.

Reference
---------
Ball & Brown (1968) — An Empirical Evaluation of Accounting Income Numbers
Bernard & Thomas (1989) — Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?

Usage
-----
    pip install yfinance pandas numpy matplotlib
    python strategy.py
    python strategy.py --tickers AAPL MSFT GOOGL META AMZN NVDA --hold 40
    python strategy.py --start 2018-01-01 --end 2024-12-31 --cost-bps 10

Outputs (saved next to this file)
---------------------------------
    earnings_events.csv     all detected earnings events with SUE + drift
    quintile_returns.csv    average abnormal return by SUE quintile
    drift_curve.png         cumulative abnormal return by quintile over time
    metrics.csv             long-Q5 strategy vs equal-weight benchmark
"""

import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ============================================================
# CONFIG — defaults, overridable via CLI
# ============================================================
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "NVDA", "JPM", "JNJ", "V", "PG",
    "UNH", "HD", "MA", "DIS", "NFLX",
]
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_HOLD_DAYS = 20          # trading days to hold after earnings
DEFAULT_COST_BPS = 10.0         # round-trip transaction cost
PRE_WINDOW = 5                  # trading days BEFORE earnings (for context)
POST_WINDOW_MAX = 60            # max post-event window for drift curve
TRADING_DAYS_PER_YEAR = 252


# ============================================================
# DATA — prices + earnings events
# ============================================================
def fetch_prices(tickers, start, end):
    """Download daily adjusted close for all tickers."""
    print(f"Downloading prices for {len(tickers)} tickers ...")
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

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


def fetch_earnings_events(tickers):
    """Scrape earnings dates + EPS actual/estimate from yfinance.

    Returns a DataFrame with columns:
        ticker, date, eps_actual, eps_estimate, surprise, sue
    Rows with missing actuals or estimates are dropped.
    """
    all_events = []

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            ed = tk.earnings_dates
            if ed is None or ed.empty:
                print(f"  {ticker}: no earnings dates found — skipping")
                continue

            df = ed.reset_index()
            df.columns = [c.strip() for c in df.columns]

            # Identify columns — yfinance naming varies across versions
            date_col = df.columns[0]
            actual_col = [c for c in df.columns if "actual" in c.lower() and "eps" in c.lower()]
            est_col = [c for c in df.columns if "estimate" in c.lower() and "eps" in c.lower()]
            surprise_col = [c for c in df.columns if "surprise" in c.lower()]

            if not actual_col or not est_col:
                # Try alternate column patterns
                actual_col = [c for c in df.columns if "reported" in c.lower()]
                est_col = [c for c in df.columns if "estimate" in c.lower()]

            if not actual_col or not est_col:
                print(f"  {ticker}: could not identify EPS columns — skipping")
                print(f"    Available columns: {list(df.columns)}")
                continue

            actual_col = actual_col[0]
            est_col = est_col[0]

            sub = pd.DataFrame({
                "ticker": ticker,
                "date": pd.to_datetime(df[date_col]).dt.tz_localize(None),
                "eps_actual": pd.to_numeric(df[actual_col], errors="coerce"),
                "eps_estimate": pd.to_numeric(df[est_col], errors="coerce"),
            })

            # If yfinance provides a surprise % column, grab it too
            if surprise_col:
                sub["surprise_pct_raw"] = pd.to_numeric(
                    df[surprise_col[0]], errors="coerce"
                )

            sub = sub.dropna(subset=["eps_actual", "eps_estimate"])
            n_before = len(df)
            n_after = len(sub)
            print(f"  {ticker}: {n_after}/{n_before} earnings events with EPS data")
            all_events.append(sub)

        except Exception as e:
            print(f"  {ticker}: error fetching earnings — {e}")
            continue

    if not all_events:
        raise SystemExit(
            "\nNo earnings events found for any ticker. "
            "This can happen if yfinance's earnings endpoint is down "
            "or the tickers lack coverage. Try different tickers or "
            "check your internet connection."
        )

    events = pd.concat(all_events, ignore_index=True)

    # Compute standardised unexpected earnings (SUE)
    events["surprise"] = events["eps_actual"] - events["eps_estimate"]
    events["sue"] = np.where(
        events["eps_estimate"].abs() > 0.001,
        events["surprise"] / events["eps_estimate"].abs(),
        np.where(events["surprise"] > 0, 1.0,
                 np.where(events["surprise"] < 0, -1.0, 0.0)),
    )
    events = events.sort_values(["ticker", "date"]).reset_index(drop=True)
    return events


# ============================================================
# EVENT STUDY — measure drift after each earnings event
# ============================================================
def compute_post_event_returns(events, prices, hold_days, max_window):
    """For each earnings event, compute the stock's cumulative return
    over multiple horizons after the announcement.

    We align to the next available trading day AFTER the announcement
    date to avoid look-ahead bias (earnings often drop after close or
    pre-market — using next-day open is the conservative choice, and
    using next-day close-to-close is even more conservative).
    """
    trading_dates = prices.index.sort_values()
    results = []

    for _, row in events.iterrows():
        ticker = row["ticker"]
        ann_date = row["date"]

        if ticker not in prices.columns:
            continue

        px = prices[ticker].dropna()
        if px.empty:
            continue

        # Find the first trading day AFTER the announcement
        future_dates = trading_dates[trading_dates > ann_date]
        if len(future_dates) < 2:
            continue

        entry_date = future_dates[0]  # t+1 (day after announcement)
        entry_idx = trading_dates.get_loc(entry_date)

        # Ensure we have enough forward data
        end_idx = min(entry_idx + max_window, len(trading_dates) - 1)
        if end_idx <= entry_idx:
            continue

        entry_price = px.get(entry_date)
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Cumulative return at each horizon
        horizons = {}
        for d in range(1, max_window + 1):
            if entry_idx + d > len(trading_dates) - 1:
                break
            exit_date = trading_dates[entry_idx + d]
            exit_price = px.get(exit_date)
            if pd.notna(exit_price) and exit_price > 0:
                horizons[d] = (exit_price / entry_price) - 1.0

        if not horizons:
            continue

        rec = {
            "ticker": ticker,
            "ann_date": ann_date,
            "entry_date": entry_date,
            "eps_actual": row["eps_actual"],
            "eps_estimate": row["eps_estimate"],
            "surprise": row["surprise"],
            "sue": row["sue"],
        }

        # Store return at the strategy's holding period
        rec["hold_return"] = horizons.get(hold_days, np.nan)
        # Store full drift curve
        for d, ret in horizons.items():
            rec[f"ret_d{d}"] = ret

        results.append(rec)

    return pd.DataFrame(results)


def compute_market_returns(prices, events_df, max_window):
    """Compute equal-weight universe return over the same windows as
    each event, to use as the market-adjustment benchmark.

    For each event's entry_date, we compute the equal-weight return of
    ALL tickers in the universe over the same horizon.
    """
    trading_dates = prices.index.sort_values()
    daily_rets = prices.pct_change()
    ew_daily = daily_rets.mean(axis=1)  # equal-weight daily return

    mkt_returns = {}
    for entry_date in events_df["entry_date"].unique():
        if entry_date not in trading_dates:
            continue
        entry_idx = trading_dates.get_loc(entry_date)
        cum = 0.0
        horizons = {}
        for d in range(1, max_window + 1):
            if entry_idx + d > len(trading_dates) - 1:
                break
            dt = trading_dates[entry_idx + d]
            r = ew_daily.get(dt, 0.0)
            cum = (1 + cum) * (1 + r) - 1
            horizons[d] = cum
        mkt_returns[entry_date] = horizons

    return mkt_returns


# ============================================================
# QUINTILE ANALYSIS
# ============================================================
def assign_quintiles(events_df):
    """Assign each event to a SUE quintile (1 = worst, 5 = best)."""
    events_df = events_df.copy()
    events_df["sue_quintile"] = pd.qcut(
        events_df["sue"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    )
    events_df["sue_quintile"] = events_df["sue_quintile"].astype(int)
    return events_df


def quintile_abnormal_returns(events_df, mkt_returns, max_window):
    """Compute average cumulative abnormal return (CAR) by quintile
    at each horizon day."""
    rows = []
    for _, ev in events_df.iterrows():
        entry = ev["entry_date"]
        q = ev["sue_quintile"]
        mkt = mkt_returns.get(entry, {})

        for d in range(1, max_window + 1):
            stock_ret = ev.get(f"ret_d{d}", np.nan)
            mkt_ret = mkt.get(d, np.nan)
            if pd.notna(stock_ret) and pd.notna(mkt_ret):
                rows.append({
                    "quintile": q,
                    "horizon": d,
                    "stock_ret": stock_ret,
                    "mkt_ret": mkt_ret,
                    "abnormal_ret": stock_ret - mkt_ret,
                })

    car_df = pd.DataFrame(rows)
    if car_df.empty:
        return car_df

    summary = (
        car_df
        .groupby(["quintile", "horizon"])["abnormal_ret"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["quintile", "horizon", "avg_car", "std_car", "n_events"]
    return summary


# ============================================================
# STRATEGY BACKTEST — long Q5, benchmark = equal-weight B&H
# ============================================================
def backtest_long_q5(events_df, mkt_returns, hold_days, cost_bps):
    """Simple event-driven backtest:

    After each earnings announcement, if the stock is in SUE quintile 5
    (top surprise), go long for `hold_days` trading days. Deduct
    transaction cost (round-trip). Compare against equal-weight market
    return over the same window.

    Returns per-event results and aggregate metrics.
    """
    cost = cost_bps / 10_000  # round-trip

    q5 = events_df[events_df["sue_quintile"] == 5].copy()
    q5 = q5.dropna(subset=["hold_return"])

    if q5.empty:
        print("\nNo Q5 events with valid holding-period returns.")
        return pd.DataFrame(), {}

    # Per-event P&L
    q5["gross_return"] = q5["hold_return"]
    q5["net_return"] = q5["hold_return"] - cost
    q5["mkt_return"] = q5["entry_date"].map(
        lambda d: mkt_returns.get(d, {}).get(hold_days, np.nan)
    )
    q5["abnormal_return"] = q5["net_return"] - q5["mkt_return"]

    q5 = q5.dropna(subset=["mkt_return"])

    # Aggregate metrics
    n = len(q5)
    avg_gross = q5["gross_return"].mean()
    avg_net = q5["net_return"].mean()
    avg_mkt = q5["mkt_return"].mean()
    avg_abn = q5["abnormal_return"].mean()
    hit_rate = (q5["net_return"] > 0).mean()
    hit_rate_vs_mkt = (q5["abnormal_return"] > 0).mean()

    # Annualise (approximate)
    periods_per_year = TRADING_DAYS_PER_YEAR / hold_days
    ann_strat = avg_net * periods_per_year
    ann_mkt = avg_mkt * periods_per_year
    vol_strat = q5["net_return"].std() * np.sqrt(periods_per_year)
    vol_mkt = q5["mkt_return"].std() * np.sqrt(periods_per_year)
    sharpe_strat = ann_strat / vol_strat if vol_strat > 0 else np.nan
    sharpe_mkt = ann_mkt / vol_mkt if vol_mkt > 0 else np.nan

    metrics = {
        "n_events": n,
        "avg_gross_return": avg_gross,
        "avg_net_return": avg_net,
        "avg_mkt_return": avg_mkt,
        "avg_abnormal_return": avg_abn,
        "hit_rate": hit_rate,
        "hit_rate_vs_mkt": hit_rate_vs_mkt,
        "ann_return_strat": ann_strat,
        "ann_return_mkt": ann_mkt,
        "ann_vol_strat": vol_strat,
        "ann_vol_mkt": vol_mkt,
        "sharpe_strat": sharpe_strat,
        "sharpe_mkt": sharpe_mkt,
        "hold_days": hold_days,
        "cost_bps": cost_bps,
    }

    return q5, metrics


# ============================================================
# PLOTTING
# ============================================================
def plot_drift_curve(car_summary, out_path):
    """Plot cumulative abnormal return by SUE quintile over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {1: "#d62728", 2: "#ff7f0e", 3: "#7f7f7f", 4: "#2ca02c", 5: "#1f77b4"}
    labels = {1: "Q1 (worst miss)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (best beat)"}

    for q in sorted(car_summary["quintile"].unique()):
        sub = car_summary[car_summary["quintile"] == q].sort_values("horizon")
        ax.plot(
            sub["horizon"], sub["avg_car"] * 100,
            color=colors.get(q, "gray"),
            label=labels.get(q, f"Q{q}"),
            linewidth=2,
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Trading Days After Earnings Announcement")
    ax.set_ylabel("Cumulative Abnormal Return (%)")
    ax.set_title("Post-Earnings Announcement Drift by Surprise Quintile")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved drift curve → {out_path}")


def plot_quintile_bar(car_summary, hold_days, out_path):
    """Bar chart of average CAR at the holding horizon by quintile."""
    at_hold = car_summary[car_summary["horizon"] == hold_days]
    if at_hold.empty:
        # Fall back to closest available horizon
        closest = car_summary["horizon"].unique()
        closest = closest[closest <= hold_days]
        if len(closest) == 0:
            print("  Not enough data for quintile bar chart — skipping.")
            return
        at_hold = car_summary[car_summary["horizon"] == closest.max()]

    fig, ax = plt.subplots(figsize=(8, 5))
    qs = at_hold.sort_values("quintile")
    colors = ["#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"]
    ax.bar(
        qs["quintile"].astype(str),
        qs["avg_car"] * 100,
        color=colors[: len(qs)],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SUE Quintile (1 = worst miss → 5 = best beat)")
    ax.set_ylabel(f"Avg Cumulative Abnormal Return at Day {int(at_hold['horizon'].iloc[0])} (%)")
    ax.set_title(f"PEAD: Abnormal Return by Earnings Surprise Quintile")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved quintile bar → {out_path}")


# ============================================================
# DISPLAY
# ============================================================
def print_metrics(metrics):
    """Pretty-print strategy vs market metrics."""
    print("\n" + "=" * 55)
    print("  PEAD Long-Q5 Strategy vs Equal-Weight Market")
    print("=" * 55)
    print(f"  Earnings events traded (Q5):  {metrics['n_events']}")
    print(f"  Holding period:               {metrics['hold_days']} trading days")
    print(f"  Transaction cost:             {metrics['cost_bps']:.0f} bps round-trip")
    print()
    print(f"  {'Metric':<30} {'Strategy':>10} {'Market':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Avg return per trade':<30} {metrics['avg_net_return']:>9.2%} {metrics['avg_mkt_return']:>9.2%}")
    print(f"  {'Avg abnormal return':<30} {metrics['avg_abnormal_return']:>9.2%} {'—':>10}")
    print(f"  {'Hit rate (>0)':<30} {metrics['hit_rate']:>9.1%} {'—':>10}")
    print(f"  {'Hit rate vs market':<30} {metrics['hit_rate_vs_mkt']:>9.1%} {'—':>10}")
    print(f"  {'Annualised return (approx)':<30} {metrics['ann_return_strat']:>9.2%} {metrics['ann_return_mkt']:>9.2%}")
    print(f"  {'Annualised vol (approx)':<30} {metrics['ann_vol_strat']:>9.2%} {metrics['ann_vol_mkt']:>9.2%}")
    print(f"  {'Sharpe ratio (approx)':<30} {metrics['sharpe_strat']:>10.2f} {metrics['sharpe_mkt']:>10.2f}")
    print("=" * 55)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Week 5 — Post-Earnings Announcement Drift (PEAD)"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Tickers to study (default: 15 large-caps)"
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--hold", type=int, default=DEFAULT_HOLD_DAYS,
        help="Trading days to hold after earnings (default: 20)"
    )
    parser.add_argument(
        "--cost-bps", type=float, default=DEFAULT_COST_BPS,
        help="Round-trip transaction cost in basis points"
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    tickers = args.tickers
    hold_days = args.hold

    # ----- 1. Fetch data -----
    print("\n[1/5] Fetching earnings events ...")
    events = fetch_earnings_events(tickers)
    print(f"\n  Total earnings events with EPS data: {len(events)}")

    # Filter to date range
    events = events[
        (events["date"] >= args.start) & (events["date"] <= args.end)
    ]
    print(f"  Events in {args.start} → {args.end}: {len(events)}")

    if events.empty:
        raise SystemExit("No earnings events in the specified date range.")

    print(f"\n[2/5] Fetching prices ...")
    # Pad date range to allow for post-event windows
    price_start = (
        pd.Timestamp(args.start) - timedelta(days=30)
    ).strftime("%Y-%m-%d")
    price_end = (
        pd.Timestamp(args.end) + timedelta(days=int(POST_WINDOW_MAX * 1.6))
    ).strftime("%Y-%m-%d")
    prices = fetch_prices(tickers, price_start, price_end)
    print(f"  Price data: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Tickers with data: {len(prices.columns)}")

    # ----- 2. Compute post-event returns -----
    print(f"\n[3/5] Computing post-event returns (up to {POST_WINDOW_MAX}d) ...")
    drift_df = compute_post_event_returns(
        events, prices, hold_days, POST_WINDOW_MAX
    )
    print(f"  Events with valid return data: {len(drift_df)}")

    if drift_df.empty:
        raise SystemExit("No valid post-event returns computed.")

    mkt_returns = compute_market_returns(prices, drift_df, POST_WINDOW_MAX)

    # ----- 3. Quintile analysis -----
    print(f"\n[4/5] SUE quintile analysis ...")
    drift_df = assign_quintiles(drift_df)

    print(f"\n  SUE quintile distribution:")
    for q in range(1, 6):
        n = (drift_df["sue_quintile"] == q).sum()
        avg_sue = drift_df.loc[drift_df["sue_quintile"] == q, "sue"].mean()
        print(f"    Q{q}: {n:>4} events  (avg SUE: {avg_sue:+.3f})")

    car_summary = quintile_abnormal_returns(drift_df, mkt_returns, POST_WINDOW_MAX)

    # ----- 4. Strategy backtest -----
    print(f"\n[5/5] Backtesting long-Q5 strategy ({hold_days}d hold) ...")
    q5_trades, metrics = backtest_long_q5(
        drift_df, mkt_returns, hold_days, args.cost_bps
    )

    if metrics:
        print_metrics(metrics)
    else:
        print("\n  Could not compute strategy metrics (insufficient data).")

    # ----- 5. Save outputs -----
    print("\nSaving outputs ...")

    # Earnings events (drop the per-day return columns for readability)
    export_cols = [
        "ticker", "ann_date", "entry_date", "eps_actual", "eps_estimate",
        "surprise", "sue", "sue_quintile", "hold_return",
    ]
    export_cols = [c for c in export_cols if c in drift_df.columns]
    drift_df[export_cols].to_csv(out_dir / "earnings_events.csv", index=False)
    print(f"  Saved events → earnings_events.csv")

    if not car_summary.empty:
        car_summary.to_csv(out_dir / "quintile_returns.csv", index=False)
        print(f"  Saved quintile CARs → quintile_returns.csv")
        plot_drift_curve(car_summary, out_dir / "drift_curve.png")
        plot_quintile_bar(car_summary, hold_days, out_dir / "quintile_bar.png")

    if not q5_trades.empty:
        trade_cols = [
            "ticker", "ann_date", "entry_date", "eps_actual", "eps_estimate",
            "sue", "gross_return", "net_return", "mkt_return", "abnormal_return",
        ]
        trade_cols = [c for c in trade_cols if c in q5_trades.columns]
        q5_trades[trade_cols].to_csv(out_dir / "q5_trades.csv", index=False)
        print(f"  Saved Q5 trades → q5_trades.csv")

    if metrics:
        pd.Series(metrics).to_csv(out_dir / "metrics.csv", header=["value"])
        print(f"  Saved metrics → metrics.csv")

    print(f"\nAll outputs in {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
