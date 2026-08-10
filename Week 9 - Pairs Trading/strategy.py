"""
Week 9 — Pairs Trading / Cointegration
======================================

Hypothesis
----------
If two assets are cointegrated, the spread between them is stationary. When
the spread moves far from its mean, it should revert. Long the cheap leg,
short the expensive leg, and collect the convergence. The return should be
close to market-neutral, because the market exposure of the two legs largely
cancels.

The interesting question is not "does mean reversion exist" — it does — but
whether *selecting pairs by cointegration* adds anything over selecting them
some other way, once costs are paid. So the strategy is run three times with
identical machinery and only the pair-selection rule changed:

    coint   — pairs with the lowest Engle-Granger p-value   [the strategy]
    corr    — pairs with the highest return correlation     [control]
    random  — pairs drawn at random from the same universe  [control]

The random control is a *sample*, not a single number. One draw can land
anywhere in a wide band, so a single seed showing random beating (or losing to)
the strategy is a sampling artifact, not a result. It is therefore run
`--random-seeds` times and reported as a distribution in `random_control.csv`,
with the strategy's position inside that distribution in
`random_control_summary.csv`. The default of 50 draws is a runtime compromise:
the percentile estimate has a standard error of about 12 points at 12 draws,
6 points at 50, and 3 points at 100, so 50 is enough to distinguish "middle of
the pack" from "clearly better or worse" and not enough to quote to the
percentage point. Quote it as a band, not a figure.

This is the same decomposition logic as Week 4 (relative vs absolute momentum)
and Week 8 (ARIMA direction vs GARCH sizing): isolate the component that is
supposed to be doing the work and check whether it actually is.

Method
------
1. Fixed ETF universe, chosen ex ante for plausible economic linkage
   (energy, precious metals, miners, country funds, US sectors).
2. Walk forward. Every `--trading` days, look back over a `--formation`
   window and:
       a. test every pair for cointegration on log prices (Engle-Granger),
       b. rank and pick the top `--top-n`,
       c. estimate the hedge ratio beta by OLS on the formation window,
       d. record the formation-window spread mean and standard deviation.
3. Trade the following `--trading` days out of sample. The z-score uses the
   *formation* mean and sd, and the *formation* beta — nothing from the
   trading window enters the signal. Parameters are frozen for the block.
4. Entry at |z| >= `--entry-z`, exit at |z| <= `--exit-z`, stop at
   |z| >= `--stop-z` (pair disabled for the rest of the block). Positions are
   liquidated at the end of each block because betas are re-estimated.
5. Signal on day t, exposure on day t+1. Costs charged on turnover at
   `--cost-bps` per unit, matching Weeks 4-8 (10 bps per full switch).

Sizing
------
Each pair is dollar-neutral in gross terms: for hedge ratio beta the legs get
weights +1/(1+|beta|) and -beta/(1+|beta|), so gross exposure per pair is 1.
Capital is split equally across the `--top-n` pairs, so portfolio gross
exposure is at most 1.0 and no leverage is used. Idle capital earns nothing
unless `--cash-yield` is set, which understates a market-neutral book's real
return — see notes in the README.

Signal quality vs P&L
---------------------
Week 8 established that forecast quality should be scored separately from
P&L. The equivalent here is *relationship* quality, and it is measured three
ways in `signal_quality.csv`, independently of whether the trades made money:

    - the fraction of selected pairs whose spread is still stationary
      out of sample (ADF on the forward window, formation beta),
    - the fraction of entries that actually converged before stopping out,
      timing out, or hitting the end of the block,
    - formation half-life vs realised out-of-sample half-life.

A caveat that matters more than it looks: the ADF test has almost no power on
a 63-day window. Simulating a genuinely stationary spread with an 11-day
half-life, the test rejects a unit root only ~10% of the time at n=63, ~61% at
n=252. So the block-level p-value is reported but must not be read as evidence
that a relationship broke; `oos_adf_pvalue_fwd` uses a `--diag-window` day
forward window instead, which has usable power. That window extends past the
trading block, so it is strictly a post-hoc diagnostic and is never available
at trade time and never touches the signal.

A pair can be beautifully cointegrated in sample and still lose money after
costs, and it can make money for reasons that have nothing to do with
cointegration. These files let those two things be told apart.

Parameters
----------
All thresholds are textbook defaults (252/63 day walk-forward, 2.0 entry,
0.5 exit, 4.0 stop, top 5 pairs, p < 0.05). Nothing has been tuned on the
reported results. They are exposed as CLI arguments so anyone can check the
sensitivity themselves, which is the point.

Usage
-----
    python strategy.py
    python strategy.py --top-n 10 --entry-z 1.5
    python strategy.py --skip-controls          # main variant only, faster
    python strategy.py --cache                  # reuse downloaded prices
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")

# Anchor all output to the script's own folder, not the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_FILE = os.path.join(RESULTS_DIR, "prices_cache.csv")

TRADING_DAYS = 252

# Fixed ex-ante universe. Chosen for plausible economic linkage, not screened
# on results. Duplicate share classes of the same index are excluded, because
# they cointegrate trivially and the spread is smaller than the spread cost.
DEFAULT_UNIVERSE = [
    "XLE", "XOP", "OIH",            # energy / E&P / oil services
    "GLD", "SLV", "GDX",            # precious metals and miners
    "EWA", "EWC", "EWG", "EWJ",     # country funds
    "XLF", "KRE",                   # financials / regional banks
    "XLK", "XLV", "XLP",            # tech / healthcare / staples
    "XLU", "XLI", "XLB", "XLY",     # utilities / industrials / materials / discretionary
    "IYR",                          # real estate
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def download_prices(tickers, start, end, use_cache=False):
    """Adjusted closes for the universe, aligned to common trading days."""
    if use_cache and os.path.exists(CACHE_FILE):
        px = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        print(f"Loaded cached prices: {CACHE_FILE}")
        return px

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("No data returned. Check tickers and date range.")

    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = pd.DataFrame(px)
    if px.shape[1] == 1 and len(tickers) == 1:
        px.columns = list(tickers)

    px = px.ffill(limit=5)
    coverage = px.notna().mean()
    dropped = sorted(coverage[coverage < 0.95].index.tolist())
    if dropped:
        print(f"Dropped for insufficient history: {', '.join(dropped)}")
    px = px[coverage[coverage >= 0.95].index]
    px = px.dropna().sort_index()

    if px.shape[1] < 4:
        raise RuntimeError(f"Only {px.shape[1]} usable tickers; need at least 4.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    px.to_csv(CACHE_FILE)
    return px


def download_benchmark(ticker, index, start, end):
    """Buy-and-hold benchmark returns aligned to the strategy calendar."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    s = pd.Series(np.asarray(close).ravel(), index=raw.index, name=ticker)
    return s.reindex(index).ffill().pct_change().fillna(0.0)


# --------------------------------------------------------------------------
# Pair estimation
# --------------------------------------------------------------------------
def ols_hedge_ratio(y, x):
    """Regress y on x with a constant. Returns (alpha, beta)."""
    design = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(alpha), float(beta)


def half_life(spread):
    """Ornstein-Uhlenbeck half-life from d(spread) ~ a + b * spread_lag."""
    s = np.asarray(spread, dtype=float)
    if len(s) < 20:
        return np.nan
    lag, delta = s[:-1], np.diff(s)
    design = np.column_stack([np.ones_like(lag), lag])
    b = np.linalg.lstsq(design, delta, rcond=None)[0][1]
    if not np.isfinite(b) or b >= 0 or b <= -1:
        return np.nan
    return float(-np.log(2.0) / np.log(1.0 + b))


def fit_pair(logp, a, b, f0, f1, maxlag, run_coint=True):
    """Estimate hedge ratio, spread moments and cointegration p-value."""
    ya = logp[f0:f1, a]
    yb = logp[f0:f1, b]
    alpha, beta = ols_hedge_ratio(ya, yb)
    resid = ya - (alpha + beta * yb)
    sigma = float(resid.std(ddof=1))

    pval = np.nan
    if run_coint:
        try:
            pval = float(coint(ya, yb, trend="c", maxlag=maxlag, autolag=None)[1])
        except Exception:
            pval = np.nan

    denom = 1.0 + abs(beta)
    return {
        "a": a,
        "b": b,
        "alpha": alpha,
        "beta": beta,
        "mu": float(resid.mean()),
        "sigma": sigma,
        "pval": pval,
        "half_life": half_life(resid),
        "ua": 1.0 / denom,
        "ub": -beta / denom,
    }


def usable(fit, cfg):
    """Reject degenerate hedge ratios and dead spreads.

    A negative beta is rejected outright, not on its absolute value. Beta < 0
    means the OLS fit wants BOTH legs long, which is a levered directional bet
    rather than a hedged pair: the market exposure of the two legs adds instead
    of cancelling, and the resulting "spread" has no economic interpretation.
    The first version of this filter tested |beta| and let them through.
    """
    if not np.isfinite(fit["beta"]) or not np.isfinite(fit["sigma"]):
        return False
    if fit["sigma"] <= 1e-8:
        return False
    if fit["beta"] <= 0:
        return False
    return cfg.beta_min <= fit["beta"] <= cfg.beta_max


def select_pairs(logp, rets, f0, f1, candidates, cfg, method, rng):
    """Choose pairs for the next trading block using formation-window data only."""
    if method == "coint":
        fits = []
        for a, b in candidates:
            fit = fit_pair(logp, a, b, f0, f1, cfg.maxlag, run_coint=True)
            if np.isfinite(fit["pval"]) and fit["pval"] <= cfg.pval and usable(fit, cfg):
                fits.append(fit)
        fits.sort(key=lambda f: f["pval"])
        return fits[: cfg.top_n]

    if method == "corr":
        block = rets[f0:f1]
        cm = np.corrcoef(block, rowvar=False)
        ranked = sorted(candidates, key=lambda p: -cm[p[0], p[1]])
    elif method == "random":
        order = rng.permutation(len(candidates))
        ranked = [candidates[i] for i in order]
    else:
        raise ValueError(f"Unknown selection method: {method}")

    out = []
    for a, b in ranked:
        fit = fit_pair(logp, a, b, f0, f1, cfg.maxlag, run_coint=False)
        if usable(fit, cfg):
            out.append(fit)
        if len(out) == cfg.top_n:
            break
    return out


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def run_backtest(px, cfg, method="coint", seed=0):
    """Walk-forward pairs backtest. Returns weights, trades, selections, quality."""
    tickers = list(px.columns)
    logp = np.log(px.to_numpy(dtype=float))
    rets = px.pct_change().fillna(0.0).to_numpy(dtype=float)
    dates = px.index
    n, k_assets = logp.shape

    candidates = list(itertools.combinations(range(k_assets), 2))
    rng = np.random.default_rng(seed)

    weights = np.zeros((n, k_assets))
    # Counted explicitly rather than inferred from non-zero weight columns:
    # two pairs sharing a leg occupy 3 columns, not 4, so the column count
    # undercounts and produces spurious half-pairs.
    pair_count = np.zeros(n)
    trades, selections, quality, z_blocks = [], [], [], {}

    for i0 in range(cfg.formation, n, cfg.trading):
        f0, f1 = i0 - cfg.formation, i0
        end = min(i0 + cfg.trading, n)
        if end - i0 < 5:
            break

        picks = select_pairs(logp, rets, f0, f1, candidates, cfg, method, rng)
        if not picks:
            continue

        # Size at 1/top_n, NOT 1/len(picks). If fewer than top_n pairs pass the
        # filter, the shortfall stays in cash. Dividing by len(picks) silently
        # concentrated the whole book into a single spread whenever only one
        # pair qualified, which is where the fat left tail came from.
        k = cfg.top_n
        block_start = dates[i0]
        for p in picks:
            selections.append(
                {
                    "block_start": block_start,
                    "block_end": dates[end - 1],
                    "asset_a": tickers[p["a"]],
                    "asset_b": tickers[p["b"]],
                    "beta": p["beta"],
                    "coint_pvalue": p["pval"],
                    "spread_sd": p["sigma"],
                    "half_life_days": p["half_life"],
                }
            )

        state = [{"pos": 0, "entry": None, "entry_z": np.nan, "off": False} for _ in picks]
        zs = np.full((end - i0, k), np.nan)

        for t in range(i0, end):
            last = t == end - 1
            for j, p in enumerate(picks):
                spread = logp[t, p["a"]] - p["alpha"] - p["beta"] * logp[t, p["b"]]
                z = (spread - p["mu"]) / p["sigma"]
                zs[t - i0, j] = z
                s = state[j]

                if s["pos"] != 0:
                    held = t - s["entry"]
                    if abs(z) >= cfg.stop_z:
                        reason = "stop"
                    elif abs(z) <= cfg.exit_z:
                        reason = "converged"
                    elif held >= cfg.max_hold:
                        reason = "time"
                    elif last:
                        reason = "block_end"
                    else:
                        reason = None

                    if reason:
                        span = slice(s["entry"] + 1, t + 1)
                        pnl = float(
                            np.sum(
                                s["pos"]
                                * (p["ua"] * rets[span, p["a"]] + p["ub"] * rets[span, p["b"]])
                            )
                        )
                        trades.append(
                            {
                                "asset_a": tickers[p["a"]],
                                "asset_b": tickers[p["b"]],
                                "direction": "long_spread" if s["pos"] == 1 else "short_spread",
                                "entry_date": dates[s["entry"]],
                                "exit_date": dates[t],
                                "days_held": held,
                                "entry_z": s["entry_z"],
                                "exit_z": z,
                                "exit_reason": reason,
                                "converged": reason == "converged",
                                "beta": p["beta"],
                                "gross_return_unit_gross": pnl,
                            }
                        )
                        s["pos"], s["entry"], s["entry_z"] = 0, None, np.nan
                        if reason == "stop":
                            s["off"] = True

                elif (not s["off"]) and (not last) and abs(z) >= cfg.entry_z:
                    s["pos"] = -1 if z > 0 else 1
                    s["entry"], s["entry_z"] = t, z

                if s["pos"] != 0:
                    weights[t, p["a"]] += s["pos"] * p["ua"] / k
                    weights[t, p["b"]] += s["pos"] * p["ub"] / k
                    pair_count[t] += 1

        # Out-of-sample relationship quality, scored regardless of P&L.
        # Two windows: the block itself (what was traded, but underpowered) and
        # a longer forward window (post-hoc diagnostic only, never a signal).
        fwd_end = min(i0 + cfg.diag_window, n)
        for j, p in enumerate(picks):
            def spread_over(lo, hi):
                return logp[lo:hi, p["a"]] - p["alpha"] - p["beta"] * logp[lo:hi, p["b"]]

            def adf_p(series):
                try:
                    return float(adfuller(series, maxlag=cfg.maxlag, autolag=None)[1])
                except Exception:
                    return np.nan

            oos = spread_over(i0, end)
            fwd = spread_over(i0, fwd_end)
            fwd_p = adf_p(fwd) if len(fwd) >= cfg.formation // 2 else np.nan

            quality.append(
                {
                    "block_start": block_start,
                    "asset_a": tickers[p["a"]],
                    "asset_b": tickers[p["b"]],
                    "formation_coint_pvalue": p["pval"],
                    "formation_half_life": p["half_life"],
                    "oos_adf_pvalue_block": adf_p(oos),
                    "oos_adf_pvalue_fwd": fwd_p,
                    "oos_days_fwd": len(fwd),
                    "oos_stationary_fwd": fwd_p <= cfg.pval if np.isfinite(fwd_p) else np.nan,
                    "oos_half_life": half_life(fwd),
                    "oos_max_abs_z": float(np.nanmax(np.abs(zs[:, j]))),
                    "oos_spread_sd_ratio": float(np.nanstd(oos, ddof=1) / p["sigma"]),
                }
            )

        z_blocks[block_start] = (dates[i0:end], picks, zs, tickers)

    weights = pd.DataFrame(weights, index=dates, columns=tickers)
    return {
        "weights": weights,
        "active_pairs": pd.Series(pair_count, index=dates),
        "returns": px.pct_change().fillna(0.0),
        "trades": pd.DataFrame(trades),
        "selections": pd.DataFrame(selections),
        "quality": pd.DataFrame(quality),
        "z_blocks": z_blocks,
    }


def portfolio_series(weights, rets, cost_bps, cash_yield=0.0, active_pairs=None):
    """Turn a weight path into daily gross/net returns. Position lags one day."""
    lagged = weights.shift(1).fillna(0.0)
    gross = (lagged * rets).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1).iloc[0])
    turnover = turnover.shift(1).fillna(0.0)
    costs = turnover * (cost_bps / 10_000.0)

    idle = (1.0 - lagged.abs().sum(axis=1)).clip(lower=0.0)
    cash = idle * (cash_yield / TRADING_DAYS)

    net = gross + cash - costs
    return pd.DataFrame(
        {
            "gross_return": gross,
            "cash_return": cash,
            "turnover": turnover,
            "cost": costs,
            "net_return": net,
            "gross_exposure": lagged.abs().sum(axis=1),
            "net_exposure": lagged.sum(axis=1),
            "long_leg_return": (lagged.clip(lower=0.0) * rets).sum(axis=1),
            "short_leg_return": (lagged.clip(upper=0.0) * rets).sum(axis=1),
            "active_pairs": (
                active_pairs.shift(1).fillna(0.0)
                if active_pairs is not None
                else (lagged.abs() > 1e-12).sum(axis=1) / 2.0
            ),
        }
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def perf_metrics(ret, bench=None):
    ret = ret.dropna()
    if len(ret) == 0:
        return {}
    equity = (1.0 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    sd = ret.std(ddof=1)
    downside = ret[ret < 0].std(ddof=1)
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())

    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    out = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "ann_volatility": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "sortino": float(ret.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "hit_rate_daily": float((ret > 0).mean()),
        "best_day": float(ret.max()),
        "worst_day": float(ret.min()),
        "skew": float(ret.skew()),
        "kurtosis": float(ret.kurtosis()),
    }
    if bench is not None:
        b = bench.reindex(ret.index).fillna(0.0)
        var_b = b.var(ddof=1)
        out["beta_to_benchmark"] = float(ret.cov(b) / var_b) if var_b > 0 else np.nan
        out["corr_to_benchmark"] = float(ret.corr(b))
    return out


def annual_returns(net, gross, bench):
    frame = pd.DataFrame({"strategy_net": net, "strategy_gross": gross, "benchmark": bench})
    yearly = frame.groupby(frame.index.year).apply(lambda g: (1.0 + g).prod() - 1.0)
    yearly.index.name = "year"
    return yearly.reset_index()


def worst_drawdowns(ret, top=5):
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    underwater = dd < 0

    episodes, start = [], None
    for i, flag in enumerate(underwater.to_numpy()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i - 1, i))
            start = None
    if start is not None:
        episodes.append((start, len(dd) - 1, None))

    rows = []
    for s, e, rec in episodes:
        window = dd.iloc[s : e + 1]
        trough = int(window.to_numpy().argmin())
        rows.append(
            {
                "peak_date": dd.index[s - 1] if s > 0 else dd.index[s],
                "trough_date": window.index[trough],
                "recovery_date": dd.index[rec] if rec is not None else pd.NaT,
                "depth": float(window.min()),
                "length_days": (e - s + 1),
                "days_to_trough": trough + 1,
                "recovered": rec is not None,
            }
        )
    frame = pd.DataFrame(rows).sort_values("depth").head(top).reset_index(drop=True)
    return frame


def trade_stats(trades):
    if trades is None or len(trades) == 0:
        return {}
    r = trades["gross_return_unit_gross"]
    return {
        "n_trades": int(len(trades)),
        "trade_win_rate": float((r > 0).mean()),
        "avg_trade_return": float(r.mean()),
        "median_days_held": float(trades["days_held"].median()),
        "convergence_rate": float(trades["converged"].mean()),
        "pct_stopped_out": float((trades["exit_reason"] == "stop").mean()),
        "pct_timed_out": float(trades["exit_reason"].isin(["time", "block_end"]).mean()),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def save_charts(daily, bench, result, variants, cfg, random_curves=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    eq_s = (1.0 + daily["net_return"]).cumprod()
    eq_b = (1.0 + bench).cumprod()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eq_s.index, eq_s, label="Pairs (net of costs)", lw=1.4)
    ax.plot(eq_b.index, eq_b, label=f"Buy & hold {cfg.benchmark}", lw=1.4, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Week 9 — Cointegration pairs vs buy & hold")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "equity_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for series, label in [(eq_s, "Pairs"), (eq_b, f"Buy & hold {cfg.benchmark}")]:
        ax.fill_between(series.index, series / series.cummax() - 1.0, 0, alpha=0.4, label=label)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "drawdown.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    if random_curves:
        for i, curve in enumerate(random_curves):
            ax.plot(
                curve.index,
                curve,
                lw=0.9,
                color="grey",
                alpha=0.45,
                label=f"random pairs ({len(random_curves)} seeds)" if i == 0 else None,
            )
    for name, frame in variants.items():
        if name.startswith("random pairs"):
            continue
        curve = (1.0 + frame["net_return"]).cumprod()
        ax.plot(curve.index, curve, lw=1.6, label=name, zorder=3)
    ax.set_title("Does cointegration selection add anything? (all net of costs)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "selection_decomposition.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(daily.index, daily["active_pairs"], lw=0.9)
    axes[0].set_ylabel("Active pairs")
    axes[0].set_title("Capital deployment and trading intensity")
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        daily.index,
        daily["cost"].rolling(TRADING_DAYS).sum(),
        lw=0.9,
        color="firebrick",
    )
    axes[1].set_ylabel("Rolling 1y cost drag")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exposure_and_costs.png"), dpi=150)
    plt.close(fig)

    # One representative spread, chosen as the pair with the most trades in the
    # first block that produced at least two round trips.
    trades = result["trades"]
    picked = None
    if len(trades) > 0:
        for block_start, (idx, picks, zs, tickers) in result["z_blocks"].items():
            block_trades = trades[trades["entry_date"] >= block_start]
            block_trades = block_trades[block_trades["entry_date"] <= idx[-1]]
            if len(block_trades) < 2:
                continue
            counts = block_trades.groupby(["asset_a", "asset_b"]).size().sort_values()
            a_name, b_name = counts.index[-1]
            for j, p in enumerate(picks):
                if tickers[p["a"]] == a_name and tickers[p["b"]] == b_name:
                    picked = (idx, zs[:, j], a_name, b_name)
                    break
            if picked:
                break

    if picked:
        idx, z, a_name, b_name = picked
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(idx, z, lw=1.1, color="steelblue")
        for level, style in [
            (cfg.entry_z, "--"),
            (-cfg.entry_z, "--"),
            (cfg.exit_z, ":"),
            (-cfg.exit_z, ":"),
            (cfg.stop_z, "-."),
            (-cfg.stop_z, "-."),
        ]:
            ax.axhline(level, ls=style, lw=0.8, color="grey")
        ax.axhline(0, lw=0.8, color="black")
        ax.set_title(f"Spread z-score, {a_name} / {b_name} (one out-of-sample block)")
        ax.set_ylabel("z-score (formation mean and sd)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "spread_example.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Week 9 — pairs trading / cointegration")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--formation", type=int, default=252, help="formation window, days")
    p.add_argument("--trading", type=int, default=63, help="out-of-sample block, days")
    p.add_argument("--top-n", type=int, default=5, help="pairs held per block")
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--stop-z", type=float, default=4.0)
    p.add_argument("--max-hold", type=int, default=None, help="defaults to --trading")
    p.add_argument("--pval", type=float, default=0.05, help="cointegration p-value cutoff")
    p.add_argument("--maxlag", type=int, default=1, help="ADF lag for coint/adfuller")
    p.add_argument(
        "--diag-window",
        type=int,
        default=252,
        help="forward window for the OOS stationarity diagnostic (not tradeable)",
    )
    p.add_argument("--beta-min", type=float, default=0.05)
    p.add_argument("--beta-max", type=float, default=20.0)
    p.add_argument("--cost-bps", type=float, default=10.0, help="bps per unit turnover")
    p.add_argument("--cash-yield", type=float, default=0.0, help="annual yield on idle capital")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument(
        "--random-seeds",
        type=int,
        default=50,
        help="draws of the random-pair control, reported as a distribution",
    )
    p.add_argument("--cache", action="store_true", help="reuse results/prices_cache.csv")
    cfg = p.parse_args(argv)
    if cfg.max_hold is None:
        cfg.max_hold = cfg.trading
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    px = download_prices(cfg.tickers, cfg.start, cfg.end, use_cache=cfg.cache)
    bench = download_benchmark(cfg.benchmark, px.index, cfg.start, cfg.end)
    print(f"{px.shape[1]} tickers, {len(px)} days, {px.index[0].date()} to {px.index[-1].date()}")

    n_pairs = px.shape[1] * (px.shape[1] - 1) // 2
    print(f"Testing {n_pairs} candidate pairs per formation window...")

    result = run_backtest(px, cfg, method="coint", seed=cfg.seed)
    rets = result["returns"]
    daily = portfolio_series(
        result["weights"], rets, cfg.cost_bps, cfg.cash_yield, result["active_pairs"]
    )

    variants = {"cointegration (strategy)": daily}
    decomposition = []

    def add_row(label, frame, trades=None):
        row = {"variant": label}
        row.update(perf_metrics(frame["net_return"], bench))
        row["ann_cost_drag"] = float(frame["cost"].mean() * TRADING_DAYS)
        row["ann_turnover"] = float(frame["turnover"].mean() * TRADING_DAYS)
        if trades is not None and len(trades) > 0:
            row.update(trade_stats(trades))
        decomposition.append(row)

    add_row("cointegration (strategy)", daily, result["trades"])
    add_row("cointegration, zero cost", portfolio_series(result["weights"], rets, 0.0, cfg.cash_yield))

    if not cfg.skip_controls:
        ctrl = run_backtest(px, cfg, method="corr", seed=cfg.seed)
        frame = portfolio_series(
            ctrl["weights"], rets, cfg.cost_bps, cfg.cash_yield, ctrl["active_pairs"]
        )
        variants["correlation-selected"] = frame
        add_row("correlation-selected", frame, ctrl["trades"])

        # The random control is a SAMPLE, not a number. A single draw can land
        # anywhere in a wide band, so reporting one seed invites reading a
        # sampling artifact as a result. Run --random-seeds draws and report
        # where the strategy falls inside that distribution.
        draws, random_curves = [], []
        for i in range(cfg.random_seeds):
            rnd = run_backtest(px, cfg, method="random", seed=cfg.seed + i)
            rframe = portfolio_series(
                rnd["weights"], rets, cfg.cost_bps, cfg.cash_yield, rnd["active_pairs"]
            )
            rm = perf_metrics(rframe["net_return"], bench)
            rm["seed"] = cfg.seed + i
            rm["ann_turnover"] = float(rframe["turnover"].mean() * TRADING_DAYS)
            rm.update(trade_stats(rnd["trades"]))
            draws.append(rm)
            random_curves.append((1.0 + rframe["net_return"]).cumprod())
            if i == 0:
                variants[f"random pairs (seed {cfg.seed})"] = rframe
                add_row(f"random pairs (seed {cfg.seed})", rframe, rnd["trades"])

        random_draws = pd.DataFrame(draws)
        random_draws.to_csv(os.path.join(RESULTS_DIR, "random_control.csv"), index=False)

        strat_cagr = perf_metrics(daily["net_return"])["cagr"]
        strat_sharpe = perf_metrics(daily["net_return"])["sharpe"]
        random_summary = {
            "n_seeds": cfg.random_seeds,
            "random_cagr_mean": float(random_draws["cagr"].mean()),
            "random_cagr_sd": float(random_draws["cagr"].std(ddof=1)),
            "random_cagr_min": float(random_draws["cagr"].min()),
            "random_cagr_max": float(random_draws["cagr"].max()),
            "random_sharpe_mean": float(random_draws["sharpe"].mean()),
            "strategy_cagr": float(strat_cagr),
            "strategy_sharpe": float(strat_sharpe),
            # Where the cointegration strategy sits inside the random band.
            # ~50% means selection on cointegration is indistinguishable from
            # drawing pairs at random.
            "strategy_cagr_percentile": float((random_draws["cagr"] < strat_cagr).mean()),
            "strategy_sharpe_percentile": float((random_draws["sharpe"] < strat_sharpe).mean()),
            "strategy_beats_random_seeds": int((random_draws["cagr"] < strat_cagr).sum()),
        }
        pd.DataFrame([random_summary]).T.rename(columns={0: "value"}).to_csv(
            os.path.join(RESULTS_DIR, "random_control_summary.csv"), index_label="metric"
        )
    else:
        random_draws, random_summary, random_curves = None, None, None

    add_row(f"buy & hold {cfg.benchmark}", pd.DataFrame({"net_return": bench, "cost": 0.0, "turnover": 0.0}))

    # ---- metrics.csv
    metrics = {"strategy_" + k: v for k, v in perf_metrics(daily["net_return"], bench).items()}
    metrics.update({"benchmark_" + k: v for k, v in perf_metrics(bench).items()})
    metrics.update(
        {
            "gross_sharpe": perf_metrics(daily["gross_return"]).get("sharpe"),
            "ann_cost_drag": float(daily["cost"].mean() * TRADING_DAYS),
            "ann_turnover": float(daily["turnover"].mean() * TRADING_DAYS),
            "total_cost_paid": float(daily["cost"].sum()),
            "avg_gross_exposure": float(daily["gross_exposure"].mean()),
            "avg_net_exposure": float(daily["net_exposure"].mean()),
            "pct_days_invested": float((daily["gross_exposure"] > 1e-12).mean()),
            "long_leg_ann_contrib": float(daily["long_leg_return"].mean() * TRADING_DAYS),
            "short_leg_ann_contrib": float(daily["short_leg_return"].mean() * TRADING_DAYS),
        }
    )
    invested = daily["gross_exposure"] > 1e-12
    metrics["hit_rate_invested_days"] = (
        float((daily.loc[invested, "net_return"] > 0).mean()) if invested.any() else np.nan
    )
    metrics.update(trade_stats(result["trades"]))
    q = result["quality"]
    if len(q) > 0:
        metrics["oos_stationarity_rate_fwd"] = float(q["oos_stationary_fwd"].mean(skipna=True))
        metrics["oos_stationarity_rate_block"] = float(
            (q["oos_adf_pvalue_block"] <= cfg.pval).mean()
        )
        metrics["median_formation_half_life"] = float(q["formation_half_life"].median())
        metrics["median_oos_half_life"] = float(q["oos_half_life"].median())
    pd.DataFrame([metrics]).T.rename(columns={0: "value"}).to_csv(
        os.path.join(RESULTS_DIR, "metrics.csv"), index_label="metric"
    )

    # ---- remaining outputs
    annual_returns(daily["net_return"], daily["gross_return"], bench).to_csv(
        os.path.join(RESULTS_DIR, "annual_returns.csv"), index=False
    )
    worst_drawdowns(daily["net_return"]).to_csv(
        os.path.join(RESULTS_DIR, "worst_drawdowns.csv"), index=False
    )

    backtest = daily.copy()
    backtest["benchmark_return"] = bench
    backtest["equity"] = (1.0 + daily["net_return"]).cumprod()
    backtest["benchmark_equity"] = (1.0 + bench).cumprod()
    backtest.to_csv(os.path.join(RESULTS_DIR, "backtest.csv"), index_label="date")

    pd.DataFrame(decomposition).to_csv(os.path.join(RESULTS_DIR, "decomposition.csv"), index=False)
    result["selections"].to_csv(os.path.join(RESULTS_DIR, "pair_selection.csv"), index=False)
    result["trades"].to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)
    result["quality"].to_csv(os.path.join(RESULTS_DIR, "signal_quality.csv"), index=False)

    save_charts(daily, bench, result, variants, cfg, random_curves)

    # ---- console summary
    sm = perf_metrics(daily["net_return"], bench)
    bm = perf_metrics(bench)
    def row(label, a, b=None, pct=True):
        fmt = (lambda v: f"{v:>13.2%}") if pct else (lambda v: f"{v:>13.3f}")
        print(f"{label:<28}" + fmt(a) + ("" if b is None else fmt(b)))

    print("\n" + "=" * 56)
    print(f"{'':<28}{'Pairs':>13}{'Buy & hold':>13}")
    print("-" * 56)
    row("Total return", sm["total_return"], bm["total_return"])
    row("CAGR", sm["cagr"], bm["cagr"])
    row("Ann. volatility", sm["ann_volatility"], bm["ann_volatility"])
    row("Sharpe", sm["sharpe"], bm["sharpe"], pct=False)
    row("Max drawdown", sm["max_drawdown"], bm["max_drawdown"])
    print("-" * 56)
    row(f"Beta to {cfg.benchmark}", sm.get("beta_to_benchmark", np.nan), pct=False)
    row("Days with a position on", metrics["pct_days_invested"])
    row("Avg gross exposure", metrics["avg_gross_exposure"])
    row("Trades", float(metrics.get("n_trades", 0)), pct=False)
    row("Convergence rate", metrics.get("convergence_rate", np.nan))
    row("OOS stationarity (fwd)", metrics.get("oos_stationarity_rate_fwd", np.nan))
    row("Ann. cost drag", metrics["ann_cost_drag"])
    if random_summary is not None:
        print("-" * 56)
        print(f"{'Random-pair control':<28}{'(' + str(random_summary['n_seeds']) + ' seeds)':>13}")
        row("  random CAGR, mean", random_summary["random_cagr_mean"])
        row("  random CAGR, sd", random_summary["random_cagr_sd"])
        row("  random CAGR, range lo", random_summary["random_cagr_min"])
        row("  random CAGR, range hi", random_summary["random_cagr_max"])
        row("  strategy percentile", random_summary["strategy_cagr_percentile"])
    print("=" * 56)
    print(f"\nOutputs written to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())