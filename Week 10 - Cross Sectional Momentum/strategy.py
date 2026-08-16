"""
Week 10 — Cross-Sectional Momentum (Jegadeesh–Titman 12-1)
==========================================================

Hypothesis
----------
Rank a cross-section of stocks each month by their return over the prior 12
months, skipping the most recent month. Buy the top decile, short the bottom
decile, hold for one month, repeat. Winners keep winning; losers keep losing.

This is the Jegadeesh & Titman (1993) momentum factor, the most documented
anomaly in the asset-pricing literature and the only strategy in this series
with a genuine prior in its favour.

Design (textbook defaults, fixed before seeing results)
------------------------------------------------------
  Universe        ~140 large-cap US equities, monthly rebalance
  Formation       12-month return, skipping the most recent month (12-1)
  Ranking         cross-sectional, deciles
  Portfolio       long top decile, short bottom decile, equal weight
  Exposure        100% long / 100% short (zero-cost, 200% gross)
  Holding         1 month
  Costs           10 bps one-way on every dollar traded
  Benchmarks      SPY buy-and-hold, equal-weight universe buy-and-hold
  Control         random decile selection, 50 independent seeds

Three things are measured separately, following the series convention that
selection quality and P&L are different questions:

  1. Does the ranking contain information?      -> rank IC, decile monotonicity
  2. Where does the P&L come from?              -> long leg vs short leg
  3. Is it better than picking names at random? -> 50-seed control distribution

Known limitation — survivorship bias
------------------------------------
The universe is a fixed list of stocks that are still listed today, so names
that were delisted, acquired or went to zero over the sample are absent. This
inflates a long-only backtest badly. It is less damaging to a long-short
ranking strategy, because both legs are drawn from the same survivor pool and
the spread is relative rather than absolute — but it is not neutral. Losers
that survived are, by construction, losers that recovered, which flatters the
short leg's forward returns and therefore *understates* the strategy. The
direction of the bias is stated here rather than corrected; correcting it
needs a point-in-time constituent database, which this series does not have.

Usage
-----
    python strategy.py
    python strategy.py --formation 6 --skip 1
    python strategy.py --n-portfolios 5 --cost-bps 10
    python strategy.py --refresh          # force re-download

Outputs land in ./results/ next to this file.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_START = "1999-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_FORMATION = 12      # months in the formation window
DEFAULT_SKIP = 1            # months skipped before formation (short-term reversal)
DEFAULT_N_PORTFOLIOS = 10   # deciles
DEFAULT_COST_BPS = 10.0     # one-way, per dollar traded
DEFAULT_RANDOM_SEEDS = 50
DEFAULT_BENCHMARK = "SPY"
MONTHS_PER_YEAR = 12

# Fixed universe of large-cap US equities with long trading histories.
# Any ticker that fails to download, or that lacks history at a given
# rebalance date, is dropped for that date rather than forward-filled.
DEFAULT_UNIVERSE = [
    # Technology / semis
    "AAPL", "MSFT", "IBM", "INTC", "CSCO", "ORCL", "TXN", "ADBE", "AMD", "MU",
    "HPQ", "QCOM", "AMAT", "ADI", "KLAC", "LRCX", "NVDA", "WDC", "STX", "GLW",
    "MSI", "ADSK", "INTU",
    # Communications / media
    "T", "VZ", "DIS", "CMCSA",
    # Consumer staples & discretionary
    "KO", "PEP", "PG", "CL", "KMB", "GIS", "K", "HSY", "SYY", "MO",
    "COST", "WMT", "TGT", "HD", "LOW", "MCD", "SBUX", "YUM", "NKE", "TJX", "ROST",
    # Health care
    "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "GILD", "BIIB", "MDT",
    "SYK", "BSX", "BAX", "CAH", "MCK", "CI", "HUM", "UNH",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "SCHW",
    "BLK", "BK", "STT", "TRV", "ALL", "AIG", "MET", "PRU", "AFL",
    # Industrials / transport
    "GE", "HON", "MMM", "CAT", "DE", "BA", "LMT", "NOC", "GD", "EMR",
    "ETN", "ITW", "PH", "ROK", "CMI", "FDX", "UPS", "NSC", "UNP", "CSX", "LUV",
    # Energy
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "EOG", "MRO", "VLO",
    # Materials
    "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM", "PPG",
    # Utilities
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "ED", "PEG",
    # Real estate
    "SPG", "PSA", "O", "AMT", "EQR", "AVB", "VTR",
]


# ============================================================
# DATA
# ============================================================

def fetch_prices(tickers, start, end, cache_path, refresh=False):
    """Daily adjusted close for the universe. Cached to CSV after first pull."""
    if os.path.exists(cache_path) and not refresh:
        print(f"  Loading cached prices from {os.path.basename(cache_path)}")
        px = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return px

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is required for the first run.  pip install yfinance")

    print(f"  Downloading {len(tickers)} tickers from Yahoo Finance ...")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy()
        px.columns = tickers[:1]

    px = px.dropna(axis=1, how="all").sort_index()
    px.to_csv(cache_path)
    print(f"  Cached to {os.path.basename(cache_path)}")
    return px


def to_monthly(daily_px):
    """Last observed price of each calendar month. Version-agnostic (no resample)."""
    key = daily_px.index.to_period("M")
    monthly = daily_px.groupby(key).last()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    return monthly


# ============================================================
# SIGNAL
# ============================================================

def formation_returns(monthly_px, formation, skip):
    """
    Momentum signal observable at each month-end t:
        P[t - skip] / P[t - skip - formation] - 1

    With formation=12, skip=1 this is the standard 12-1 return: the trailing
    twelve months excluding the most recent one.
    """
    lagged = monthly_px.shift(skip)
    return lagged / lagged.shift(formation) - 1.0


def eligible_mask(monthly_px, signal):
    """Names with a computable signal AND a tradable price at the rebalance date."""
    return signal.notna() & monthly_px.notna()


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def decile_selector(sig_row, n_portfolios, rng=None):
    """Top decile long, bottom decile short, by formation return."""
    k = max(1, int(len(sig_row) / n_portfolios))
    ranked = sig_row.sort_values(ascending=False)
    return list(ranked.index[:k]), list(ranked.index[-k:])


def random_selector(sig_row, n_portfolios, rng):
    """Control: same leg sizes, names drawn at random from the same eligible set."""
    k = max(1, int(len(sig_row) / n_portfolios))
    picks = rng.permutation(np.asarray(sig_row.index, dtype=object))
    return list(picks[:k]), list(picks[k:2 * k])


def run_backtest(monthly_px, signal, n_portfolios, cost_bps, selector=decile_selector,
                 rng=None, collect_cross_section=False):
    """
    Walk forward month by month.

    At each month-end t the signal uses prices through t only; the position is
    held over month t+1 and earns r[t+1]. Nothing about month t+1 enters the
    selection.
    """
    rets = monthly_px.pct_change()
    elig = eligible_mask(monthly_px, signal)
    dates = monthly_px.index

    prev_w = pd.Series(dtype=float)
    rows, cross_section = [], []

    for i in range(len(dates) - 1):
        t, t_next = dates[i], dates[i + 1]
        names = elig.columns[elig.loc[t].values]
        if len(names) < n_portfolios * 2:
            continue

        sig_row = signal.loc[t, names]
        fwd = rets.loc[t_next, names]

        longs, shorts = selector(sig_row, n_portfolios, rng)
        k_l, k_s = len(longs), len(shorts)

        # Target weights: 100% long / 100% short, equal weight within each leg.
        w = pd.Series(0.0, index=names)
        w[longs] = 1.0 / k_l
        w[shorts] = -1.0 / k_s

        long_ret = fwd[longs].mean(skipna=True)
        short_ret = fwd[shorts].mean(skipna=True)
        long_ret = 0.0 if pd.isna(long_ret) else long_ret
        short_ret = 0.0 if pd.isna(short_ret) else short_ret
        gross = long_ret - short_ret

        # Turnover against last month's drifted book, then cost on dollars traded.
        all_names = prev_w.index.union(w.index)
        w_new = w.reindex(all_names).fillna(0.0)
        w_old = prev_w.reindex(all_names).fillna(0.0)
        turnover = float(np.abs(w_new - w_old).sum())
        cost = turnover * cost_bps / 1e4
        net = gross - cost

        drift = w * (1.0 + fwd.reindex(w.index).fillna(0.0))
        prev_w = drift

        rows.append({
            "date": t_next, "signal_date": t, "n_eligible": len(names),
            "k_per_leg": k_l, "long_ret": long_ret, "short_ret": short_ret,
            "gross_ret": gross, "turnover": turnover, "cost": cost, "net_ret": net,
            "missing_fwd": int(fwd[longs + shorts].isna().sum()),
        })

        if collect_cross_section:
            valid = fwd.notna()
            if valid.sum() >= n_portfolios * 2:
                s, f = sig_row[valid], fwd[valid]
                ic = s.rank().corr(f.rank())
                bucket = pd.qcut(s.rank(method="first"), n_portfolios,
                                 labels=False, duplicates="drop")
                cross_section.append({
                    "date": t_next, "ic": ic,
                    "bucket_means": f.groupby(bucket.values).mean(),
                })

    bt = pd.DataFrame(rows).set_index("date")
    bt["equity"] = (1.0 + bt["net_ret"]).cumprod()
    bt["equity_gross"] = (1.0 + bt["gross_ret"]).cumprod()
    bt["long_equity"] = (1.0 + bt["long_ret"]).cumprod()
    bt["short_equity"] = (1.0 - bt["short_ret"]).cumprod()
    return (bt, cross_section) if collect_cross_section else bt


# ============================================================
# METRICS
# ============================================================

def perf_metrics(returns, label):
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    equity = (1.0 + r).cumprod()
    years = n / MONTHS_PER_YEAR
    total = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    sharpe = (r.mean() * MONTHS_PER_YEAR) / vol if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    downside = r[r < 0].std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    return {
        "strategy": label,
        "months": n,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": (r.mean() * MONTHS_PER_YEAR) / downside if downside > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "win_rate": (r > 0).mean(),
        "best_month": r.max(),
        "worst_month": r.min(),
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
    }


def drawdown_table(returns, top=5):
    equity = (1.0 + returns.dropna()).cumprod()
    dd = equity / equity.cummax() - 1.0
    episodes, in_dd, start, trough, trough_val = [], False, None, None, 0.0
    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd, start, trough, trough_val = True, date, date, val
        elif in_dd:
            if val < trough_val:
                trough, trough_val = date, val
            if val >= -1e-12:
                episodes.append({"start": start, "trough": trough, "end": date,
                                 "depth": trough_val,
                                 "months_to_trough": len(dd.loc[start:trough]),
                                 "months_total": len(dd.loc[start:date])})
                in_dd = False
    if in_dd:
        episodes.append({"start": start, "trough": trough, "end": pd.NaT,
                         "depth": trough_val,
                         "months_to_trough": len(dd.loc[start:trough]),
                         "months_total": len(dd.loc[start:])})
    out = pd.DataFrame(episodes)
    return out.sort_values("depth").head(top).reset_index(drop=True) if len(out) else out


def annual_table(series_map):
    frames = {}
    for label, r in series_map.items():
        frames[label] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return pd.DataFrame(frames)


# ============================================================
# CHARTS
# ============================================================

def chart_equity(bt, bench, ew, outdir):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt.index, bt["equity"], lw=1.8, color="#1f4e79",
            label="Momentum L/S (net)")
    ax.plot(bench.index, (1 + bench).cumprod(), lw=1.4, color="#c0504d",
            label="SPY buy & hold")
    ax.plot(ew.index, (1 + ew).cumprod(), lw=1.2, color="#7f7f7f",
            ls="--", label="Equal-weight universe")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 — cross-sectional momentum vs benchmarks (log scale)")
    ax.set_ylabel("Equity (log)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curves.png"), dpi=150)
    plt.close(fig)


def chart_deciles(bucket_means, n_portfolios, outdir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(bucket_means))
    colors = ["#c0504d" if i < len(x) / 2 else "#1f4e79" for i in x]
    ax.bar(x, bucket_means.values * 100, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}" for i in x])
    ax.set_xlabel("Momentum decile (D1 = lowest formation return)")
    ax.set_ylabel("Mean forward 1-month return (%)")
    ax.set_title("Is the ranking informative? Mean next-month return by decile")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "decile_returns.png"), dpi=150)
    plt.close(fig)


def chart_ic(ic_series, outdir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(ic_series.index, ic_series.values, width=20,
           color=np.where(ic_series.values > 0, "#1f4e79", "#c0504d"), alpha=0.55)
    ax.plot(ic_series.index, ic_series.rolling(12).mean(), color="black", lw=1.6,
            label="12-month rolling mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(ic_series.mean(), color="#1f4e79", ls="--", lw=1.2,
               label=f"Full-sample mean = {ic_series.mean():.3f}")
    ax.set_title("Rank information coefficient — momentum score vs next-month return")
    ax.set_ylabel("Spearman IC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ic_series.png"), dpi=150)
    plt.close(fig)


def chart_legs(bt, outdir):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt.index, bt["long_equity"], lw=1.5, color="#1f4e79",
            label="Long leg (winners), long-only")
    ax.plot(bt.index, bt["short_equity"], lw=1.5, color="#c0504d",
            label="Short leg (losers), short-only")
    ax.plot(bt.index, bt["equity"], lw=1.8, color="black",
            label="Combined L/S (net of costs)")
    ax.set_yscale("log")
    ax.set_title("Where does the P&L come from? Leg decomposition (log scale)")
    ax.set_ylabel("Equity (log)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "leg_decomposition.png"), dpi=150)
    plt.close(fig)


def chart_random(random_cagrs, strat_cagr, outdir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(np.array(random_cagrs) * 100, bins=18, color="#a6a6a6",
            edgecolor="white", label=f"Random selection ({len(random_cagrs)} seeds)")
    ax.axvline(strat_cagr * 100, color="#1f4e79", lw=2.2,
               label=f"Momentum = {strat_cagr*100:.2f}%")
    ax.axvline(np.mean(random_cagrs) * 100, color="#c0504d", ls="--", lw=1.4,
               label=f"Random mean = {np.mean(random_cagrs)*100:.2f}%")
    ax.set_xlabel("Net CAGR (%)")
    ax.set_ylabel("Seeds")
    ax.set_title("Momentum ranking vs random name selection, same leg sizes")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "random_control.png"), dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Week 10 — Cross-Sectional Momentum (Jegadeesh-Titman 12-1)")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--formation", type=int, default=DEFAULT_FORMATION,
                   help="Formation window in months (default: 12)")
    p.add_argument("--skip", type=int, default=DEFAULT_SKIP,
                   help="Months skipped before the formation window (default: 1)")
    p.add_argument("--n-portfolios", type=int, default=DEFAULT_N_PORTFOLIOS,
                   help="Number of ranked buckets; 10 = deciles (default: 10)")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                   help="One-way cost in bps per dollar traded (default: 10)")
    p.add_argument("--random-seeds", type=int, default=DEFAULT_RANDOM_SEEDS,
                   help="Random-selection control draws (default: 50)")
    p.add_argument("--refresh", action="store_true", help="Force price re-download")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, "prices_cache.csv")

    # ---- 1. Data -------------------------------------------------------
    print("\n[1/6] Loading prices ...")
    universe = list(dict.fromkeys(args.tickers))
    px = fetch_prices(universe + [args.benchmark], args.start, args.end,
                      cache, args.refresh)
    # The cache holds whatever was downloaded first; always slice to the
    # requested window so --start/--end behave the same cached or not.
    px = px.loc[str(args.start):str(args.end)]
    bench_px = px[args.benchmark]
    px = px.drop(columns=[args.benchmark], errors="ignore")
    print(f"  {px.shape[1]} tickers with data, {px.shape[0]} trading days")

    m_px = to_monthly(px)
    m_bench = to_monthly(bench_px.to_frame())[args.benchmark]
    print(f"  {len(m_px)} month-ends, {m_px.index[0].date()} to {m_px.index[-1].date()}")

    # ---- 2. Signal and backtest ---------------------------------------
    print(f"\n[2/6] Building {args.formation}-{args.skip} momentum signal ...")
    signal = formation_returns(m_px, args.formation, args.skip)

    print("[3/6] Running walk-forward backtest ...")
    bt, cs = run_backtest(m_px, signal, args.n_portfolios, args.cost_bps,
                          selector=decile_selector, collect_cross_section=True)
    print(f"  {len(bt)} months traded, "
          f"{bt['k_per_leg'].mean():.0f} names per leg on average")

    bench_ret = m_bench.pct_change().reindex(bt.index)
    ew_ret = m_px.pct_change().mean(axis=1).reindex(bt.index)

    # ---- 3. Selection quality, scored independently of P&L ------------
    print("\n[4/6] Scoring selection quality ...")
    ic = pd.Series({d["date"]: d["ic"] for d in cs}).dropna()
    ic_t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    buckets = pd.DataFrame([d["bucket_means"] for d in cs])
    bucket_means = buckets.mean()
    monotonicity = pd.Series(bucket_means.values).corr(
        pd.Series(np.arange(len(bucket_means))), method="spearman")
    print(f"  Mean IC {ic.mean():+.4f}  (t = {ic_t:+.2f}, "
          f"{(ic > 0).mean():.1%} of months positive)")
    print(f"  Decile monotonicity (Spearman) = {monotonicity:+.3f}")

    # ---- 4. Random control --------------------------------------------
    print(f"\n[5/6] Random-selection control, {args.random_seeds} seeds ...")
    rand_rows = []
    for s in range(args.random_seeds):
        rng = np.random.default_rng(s)
        rbt = run_backtest(m_px, signal, args.n_portfolios, args.cost_bps,
                           selector=random_selector, rng=rng)
        mm = perf_metrics(rbt["net_ret"], f"random_{s}")
        gg = perf_metrics(rbt["gross_ret"], f"random_gross_{s}")
        rand_rows.append({"seed": s, "cagr": mm["cagr"], "sharpe": mm["sharpe"],
                          "max_drawdown": mm["max_drawdown"],
                          "cagr_gross": gg["cagr"], "sharpe_gross": gg["sharpe"],
                          "avg_turnover": rbt["turnover"].mean()})
        if (s + 1) % 10 == 0:
            print(f"    {s + 1}/{args.random_seeds} seeds")
    rand = pd.DataFrame(rand_rows)

    strat = perf_metrics(bt["net_ret"], "momentum_ls_net")
    strat_gross = perf_metrics(bt["gross_ret"], "momentum_ls_gross")
    pct = float((rand["cagr"] < strat["cagr"]).mean() * 100)
    z = (strat["cagr"] - rand["cagr"].mean()) / rand["cagr"].std(ddof=1)
    # Random selection rebuilds the whole book every month, so a net-basis
    # comparison rewards momentum partly for its lower turnover. The gross
    # comparison isolates selection from trading cost; both are reported.
    pct_gross = float((rand["cagr_gross"] < strat_gross["cagr"]).mean() * 100)
    z_gross = ((strat_gross["cagr"] - rand["cagr_gross"].mean())
               / rand["cagr_gross"].std(ddof=1))

    # ---- 5. Metrics ----------------------------------------------------
    print("\n[6/6] Writing outputs ...")
    metrics = pd.DataFrame([
        strat,
        perf_metrics(bt["gross_ret"], "momentum_ls_gross"),
        perf_metrics(bt["long_ret"], "long_leg_only"),
        perf_metrics(-bt["short_ret"], "short_leg_only"),
        perf_metrics(bench_ret, f"{args.benchmark}_buy_hold"),
        perf_metrics(ew_ret, "equal_weight_universe"),
    ])
    metrics.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    diagnostics = pd.DataFrame([{
        "formation_months": args.formation,
        "skip_months": args.skip,
        "n_portfolios": args.n_portfolios,
        "cost_bps_one_way": args.cost_bps,
        "avg_names_per_leg": bt["k_per_leg"].mean(),
        "avg_eligible": bt["n_eligible"].mean(),
        "avg_monthly_turnover": bt["turnover"].mean(),
        "annual_cost_drag": bt["cost"].mean() * MONTHS_PER_YEAR,
        "mean_ic": ic.mean(),
        "ic_t_stat": ic_t,
        "ic_pct_positive": (ic > 0).mean(),
        "decile_monotonicity": monotonicity,
        "random_cagr_mean": rand["cagr"].mean(),
        "random_cagr_std": rand["cagr"].std(ddof=1),
        "random_avg_turnover": rand["avg_turnover"].mean(),
        "strategy_percentile_vs_random": pct,
        "z_vs_random": z,
        "random_cagr_gross_mean": rand["cagr_gross"].mean(),
        "random_cagr_gross_std": rand["cagr_gross"].std(ddof=1),
        "strategy_percentile_vs_random_gross": pct_gross,
        "z_vs_random_gross": z_gross,
    }])
    diagnostics.to_csv(os.path.join(outdir, "diagnostics.csv"), index=False)

    annual_table({
        "momentum_ls_net": bt["net_ret"],
        "long_leg": bt["long_ret"],
        "short_leg": -bt["short_ret"],
        args.benchmark: bench_ret,
        "equal_weight": ew_ret,
    }).to_csv(os.path.join(outdir, "annual_returns.csv"), index_label="year")

    drawdown_table(bt["net_ret"]).to_csv(
        os.path.join(outdir, "worst_drawdowns.csv"), index=False)
    bt.to_csv(os.path.join(outdir, "backtest.csv"))
    bucket_means.rename("mean_fwd_return").to_frame().to_csv(
        os.path.join(outdir, "decile_returns.csv"), index_label="decile")
    ic.rename("ic").to_frame().to_csv(os.path.join(outdir, "ic_by_month.csv"),
                                      index_label="date")
    rand.to_csv(os.path.join(outdir, "random_control.csv"), index=False)

    chart_equity(bt, bench_ret, ew_ret, outdir)
    chart_deciles(bucket_means, args.n_portfolios, outdir)
    chart_ic(ic, outdir)
    chart_legs(bt, outdir)
    chart_random(rand["cagr"].tolist(), strat["cagr"], outdir)

    # ---- Console summary ------------------------------------------------
    b = perf_metrics(bench_ret, "bench")
    print("\n" + "=" * 64)
    print("  Cross-Sectional Momentum (12-1)  —  long/short deciles")
    print("=" * 64)
    print(f"  {'Metric':<26}{'Momentum':>12}{'Long leg':>12}{args.benchmark:>12}")
    print(f"  {'-'*26}{'-'*12}{'-'*12}{'-'*12}")
    lg = perf_metrics(bt["long_ret"], "long")
    for key, fmt in [("cagr", "{:>11.2%}"), ("ann_vol", "{:>11.2%}"),
                     ("sharpe", "{:>11.2f}"), ("max_drawdown", "{:>11.2%}"),
                     ("win_rate", "{:>11.2%}")]:
        print(f"  {key:<26}" + fmt.format(strat[key]) + " "
              + fmt.format(lg[key]) + " " + fmt.format(b[key]))
    print()
    print(f"  Mean rank IC               {ic.mean():>11.4f}  (t = {ic_t:+.2f})")
    print(f"  Decile monotonicity        {monotonicity:>11.3f}")
    print(f"  Avg monthly turnover       {bt['turnover'].mean():>11.2f}x")
    print(f"  Annual cost drag           {bt['cost'].mean()*MONTHS_PER_YEAR:>11.2%}")
    print(f"  Random control CAGR  net   {rand['cagr'].mean():>11.2%} "
          f"(sd {rand['cagr'].std(ddof=1):.2%}, turnover "
          f"{rand['avg_turnover'].mean():.2f}x)")
    print(f"  Percentile vs random net   {pct:>11.0f}  (z = {z:+.2f})")
    print(f"  Random control CAGR  gross {rand['cagr_gross'].mean():>11.2%} "
          f"(sd {rand['cagr_gross'].std(ddof=1):.2%})")
    print(f"  Percentile vs random gross {pct_gross:>11.0f}  (z = {z_gross:+.2f})")
    print("=" * 64)
    print(f"  Outputs -> {outdir}")


if __name__ == "__main__":
    main()
