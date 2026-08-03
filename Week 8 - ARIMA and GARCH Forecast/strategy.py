"""
Week 8 — ARIMA + GARCH Forecast Strategy
=========================================

Week 7 threw a Random Forest at 15 technical features and failed to beat the
naive base rate. This week swaps the black box for the classical time-series
toolkit: ARIMA for the conditional MEAN of returns, GARCH(1,1) for the
conditional VARIANCE, refit walk-forward.

The hypothesis is deliberately split in two, because the literature is blunt
about which half works:

    MEAN  (ARIMA)   — daily equity returns are close to a martingale. An ARIMA
                      mean forecast is expected to be near-useless. We measure
                      it honestly (out-of-sample R^2, directional accuracy)
                      rather than assuming either way.
    VAR   (GARCH)   — volatility clusters, and GARCH(1,1) is a genuinely good
                      one-step-ahead variance forecaster. If this strategy has
                      an edge, it should come from SIZING, not from DIRECTION.

Approach
--------
1. Fetch daily OHLCV, compute log returns.
2. Walk forward: every `refit_every` days, refit ARIMA(p,d,q) on the trailing
   `window` days, then fit GARCH(1,1) to that ARIMA's residuals.
3. Hold parameters FIXED across the block and filter them forward to produce a
   one-step-ahead forecast of the mean (mu) and volatility (sigma) for each day
   in the block. Filtering with fixed parameters uses only past observations,
   so no future data enters any forecast.
4. Convert (mu, sigma) into a position:
       mean       — long if mu > threshold, else cash (direction only)
       voltarget  — always long, weight = target_vol / forecast_vol (size only)
       combo      — direction from ARIMA, size from GARCH  [default]
5. Charge `cost_bps` on turnover and benchmark against buy-and-hold over the
   identical out-of-sample window.

Honesty notes
-------------
* Every forecast for day t+1 is built from data up to and including day t.
  Parameters are estimated only on data up to the START of each block, then
  frozen — they are never re-estimated using the days they are used to trade.
* The backtest applies `position.shift(1) * ret`, so a decision made at the
  close of day t is paid the return of day t+1.
* The first `window` days are burned on the initial fit and are excluded from
  both the strategy and the benchmark. Results are out-of-sample only.
* Forecast quality is scored separately from P&L in forecast_diagnostics.csv.
  A strategy can make money with a bad model (and lose with a good one), so
  the two are reported apart.
* Vol targeting can imply leverage > 1. `--max-weight` defaults to 1.0, i.e.
  NO leverage, so the comparison against buy-and-hold stays honest. Raising it
  above 1.0 borrows at 0% and flatters the strategy.
* Continuous vol targeting rebalances every day, and at 10 bps a trade that
  turns into a large cost drag. `--rebal-band` only moves the book when the
  target weight has drifted past a threshold. That is a trading rule, not a
  cost assumption — costs themselves are never discounted.
* The GARCH variance forecast is clipped to a band around the training
  window's realised vol. A near-unit-root fit can otherwise drive the variance
  recursion toward zero and imply an absurd position size.

Usage
-----
    python strategy.py
    python strategy.py --ticker QQQ --mode voltarget --target-vol 0.15
    python strategy.py --p 2 --q 2 --window 750 --refit-every 42

"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_prices(ticker, start, end):
    """Download daily prices and compute simple + log returns."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                      progress=False)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    df["close"] = raw["Close"]
    df["ret"] = df["close"].pct_change()
    df["logret"] = np.log(df["close"]).diff()
    return df.dropna()


# --------------------------------------------------------------------------- #
# Walk-forward ARIMA + GARCH forecasting
# --------------------------------------------------------------------------- #
def fit_block(train_logret, p, d, q):
    """
    Fit ARIMA(p,d,q) to the mean and GARCH(1,1) to its residuals.

    Returns the two fitted result objects. Returns are scaled to PERCENT for
    the GARCH fit, which is what `arch` expects for numerical stability.
    """
    arima_res = ARIMA(train_logret, order=(p, d, q),
                      trend="c" if d == 0 else "n").fit()
    resid_pct = pd.Series(arima_res.resid, index=train_logret.index) * 100.0
    garch_res = arch_model(resid_pct.dropna(), mean="Zero", vol="GARCH",
                           p=1, q=1, dist="normal").fit(disp="off")
    return arima_res, garch_res


def garch_recursion(resid_pct, omega, alpha, beta):
    """
    Roll the GARCH(1,1) variance recursion forward with FIXED parameters.

        sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

    sigma_t depends only on information through t-1, so the returned series is
    a genuine one-step-ahead forecast at every point. Initialised at the
    unconditional variance implied by the parameters.
    """
    eps = np.asarray(resid_pct, dtype=float)
    n = len(eps)
    var = np.empty(n)
    persistence = alpha + beta
    uncond = omega / (1.0 - persistence) if persistence < 1.0 else np.var(eps)
    var[0] = uncond
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    return np.sqrt(var) / 100.0  # back to decimal units


def walk_forward(df, p, d, q, window, refit_every):
    """
    Roll through history refitting on a trailing window and forecasting ahead.

    For each block the parameters are estimated on data ending at the block
    start, then frozen and filtered forward across the block. Filtering with
    fixed parameters is what keeps this honest: the model never sees the days
    it is being used to trade.

    Adds two columns, both indexed at day t and describing day t+1:
        mu_next     — forecast mean log return for the next day
        sigma_next  — forecast volatility (daily, decimal) for the next day
    """
    logret = df["logret"]
    n = len(logret)
    if n <= window + refit_every:
        raise SystemExit(
            f"Not enough data: {n} rows for window={window}. Use an earlier "
            f"--start or a smaller --window.")

    mu_fit = pd.Series(np.nan, index=df.index)     # E[r_t | info up to t-1]
    sigma_fit = pd.Series(np.nan, index=df.index)  # sd[r_t | info up to t-1]
    fit_log = []

    starts = list(range(window, n, refit_every))
    for k, start_i in enumerate(starts):
        end_i = min(start_i + refit_every, n)
        train = logret.iloc[start_i - window:start_i]

        try:
            arima_res, garch_res = fit_block(train, p, d, q)
        except Exception as exc:                      # singular / non-converged
            fit_log.append({"block": k, "train_end": str(train.index[-1].date()),
                            "status": f"failed: {type(exc).__name__}"})
            continue

        # Filter frozen parameters over train + block. `.apply` re-runs the
        # Kalman filter without re-estimating, so fittedvalues[t] is a true
        # one-step-ahead prediction of day t from days < t.
        extended = logret.iloc[start_i - window:end_i]
        applied = arima_res.apply(extended, refit=False)

        block_idx = extended.index[window:]
        mu_fit.loc[block_idx] = applied.fittedvalues.iloc[window:].values

        resid_pct = pd.Series(applied.resid, index=extended.index) * 100.0
        gp = garch_res.params
        sig = garch_recursion(resid_pct, gp["omega"], gp["alpha[1]"], gp["beta[1]"])

        # Guard rail: a near-unit-root GARCH fit can send the variance
        # recursion to ~0 and imply absurd position sizes. Clip the forecast to
        # a sane band around the training window's own realised vol. On real
        # equity data this almost never binds; it stops pathological blocks
        # from dominating the backtest.
        train_sd = float(train.std())
        sigma_fit.loc[block_idx] = np.clip(sig[window:],
                                           0.25 * train_sd, 4.0 * train_sd)

        fit_log.append({
            "block": k,
            "train_start": str(train.index[0].date()),
            "train_end": str(train.index[-1].date()),
            "n_train": len(train),
            "arima_aic": round(float(arima_res.aic), 2),
            "garch_alpha": round(float(gp["alpha[1]"]), 4),
            "garch_beta": round(float(gp["beta[1]"]), 4),
            "garch_persistence": round(float(gp["alpha[1]"] + gp["beta[1]"]), 4),
            "status": "ok",
        })

    # Shift so that row t carries the forecast FOR day t+1, i.e. the number an
    # investor actually has in hand at the close of day t.
    df = df.copy()
    df["mu_next"] = mu_fit.shift(-1)
    df["sigma_next"] = sigma_fit.shift(-1)
    return df, pd.DataFrame(fit_log)


# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
def apply_rebal_band(target, band):
    """
    Only move the book when the target weight has drifted more than `band`.

    Without this, vol targeting rebalances every single day for a fraction of a
    percent and the cost line eats the strategy alive. The band is the cheapest
    honest fix: it is a real trading rule, not a cost assumption.
    """
    out = np.empty(len(target))
    current = 0.0
    for i, t in enumerate(np.asarray(target, dtype=float)):
        if np.isnan(t):
            t = 0.0
        if abs(t - current) > band:
            current = t
        out[i] = current
    return out


def build_positions(df, mode, threshold_bps, target_vol, max_weight, rebal_band):
    """
    Map (mu_next, sigma_next) onto a portfolio weight for the next day.

    mean       direction only  — 1.0 if mu > threshold else 0.0
    voltarget  size only       — target_vol / annualised forecast vol, capped
    combo      both            — vol-target size, gated by ARIMA direction
    """
    out = df.copy()
    thresh = threshold_bps / 10_000.0
    ann_sigma = out["sigma_next"] * np.sqrt(TRADING_DAYS)

    long_gate = (out["mu_next"] > thresh).astype(float)
    vol_weight = (target_vol / ann_sigma).clip(upper=max_weight, lower=0.0)

    if mode == "mean":
        target = long_gate
    elif mode == "voltarget":
        target = vol_weight
    elif mode == "combo":
        target = vol_weight * long_gate
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    out["ann_vol_forecast"] = ann_sigma
    out["long_gate"] = long_gate
    out["vol_weight"] = vol_weight
    out["target_weight"] = target.fillna(0.0)
    out = out.dropna(subset=["mu_next", "sigma_next"])
    out["position"] = apply_rebal_band(out["target_weight"], rebal_band)
    return out


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(df, cost_bps):
    """
    Apply the position with a one-day lag and charge costs on turnover.

    `position.shift(1) * ret` is the lag that matters: a weight chosen at the
    close of day t earns day t+1's return. Turnover is |w_t - w_{t-1}|, so a
    full 0 -> 1 switch costs the headline cost_bps and partial rebalances cost
    proportionally.
    """
    out = df.copy()
    held = out["position"].shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())

    out["held"] = held
    out["turnover"] = turnover
    out["cost"] = turnover * (cost_bps / 10_000.0)
    out["gross_ret"] = held * out["ret"]
    out["strat_ret"] = out["gross_ret"] - out["cost"]
    out["bench_ret"] = out["ret"]

    out["strat_cum"] = (1.0 + out["strat_ret"]).cumprod()
    out["bench_cum"] = (1.0 + out["bench_ret"]).cumprod()
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(df, label, ret_col):
    r = df[ret_col].dropna()
    if r.empty:
        raise SystemExit(f"No returns to score for {label}")

    cum = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = cum.iloc[-1] - 1.0
    cagr = cum.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan
    dd = cum / cum.cummax() - 1.0
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    m = {
        "label": label,
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "n_days": len(r),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_volatility_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "win_rate_pct": round((r > 0).mean() * 100, 2),
    }

    if ret_col == "strat_ret":
        held = df["held"]
        m["pct_time_in_market"] = round((held > 0).mean() * 100, 1)
        m["avg_exposure"] = round(held.mean(), 3)
        m["max_exposure"] = round(held.max(), 3)
        m["num_switches"] = int(((held > 0).astype(int).diff().abs() > 0).sum())
        m["total_turnover"] = round(df["turnover"].sum(), 1)
        m["total_cost_pct"] = round(df["cost"].sum() * 100, 2)
    return m


def annual_returns(df):
    rows = []
    for year, grp in df.groupby(df.index.year):
        s = (1.0 + grp["strat_ret"]).prod() - 1.0
        b = (1.0 + grp["bench_ret"]).prod() - 1.0
        rows.append({
            "year": int(year),
            "strategy_pct": round(s * 100, 2),
            "benchmark_pct": round(b * 100, 2),
            "excess_pct": round((s - b) * 100, 2),
            "avg_exposure": round(grp["held"].mean(), 3),
        })
    return pd.DataFrame(rows)


def worst_drawdowns(df, ret_col, n=5):
    cum = (1.0 + df[ret_col]).cumprod()
    dd = cum / cum.cummax() - 1.0
    in_dd = dd < 0

    episodes, start = [], None
    for i, flag in enumerate(in_dd):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            seg = dd.iloc[start:i]
            episodes.append((dd.index[start], seg.idxmin(), dd.index[i - 1],
                             seg.min()))
            start = None
    if start is not None:
        seg = dd.iloc[start:]
        episodes.append((dd.index[start], seg.idxmin(), dd.index[-1], seg.min()))

    episodes.sort(key=lambda e: e[3])
    return pd.DataFrame([{
        "start": str(s.date()),
        "trough": str(t.date()),
        "end": str(e.date()),
        "depth_pct": round(d * 100, 2),
        "length_days": (e - s).days,
    } for s, t, e, d in episodes[:n]])


# --------------------------------------------------------------------------- #
# Forecast quality — scored separately from P&L
# --------------------------------------------------------------------------- #
def forecast_diagnostics(df):
    """
    Score the two halves of the model on their own terms.

    Mean model  — out-of-sample R^2 against a zero-forecast, plus directional
                  accuracy against the naive always-up base rate.
    Vol model   — MSE and QLIKE against squared returns (a noisy but unbiased
                  variance proxy), benchmarked against a 21-day trailing
                  realised-vol forecast. Lower is better for both.
    """
    d = df.dropna(subset=["mu_next", "sigma_next"]).copy()
    realised = d["ret"].shift(-1)          # the day the forecast describes
    d = d.assign(realised=realised).dropna(subset=["realised"])

    err = d["realised"] - d["mu_next"]
    ss_res = float((err ** 2).sum())
    ss_zero = float((d["realised"] ** 2).sum())
    r2_vs_zero = 1.0 - ss_res / ss_zero if ss_zero > 0 else np.nan

    pred_up = d["mu_next"] > 0
    actual_up = d["realised"] > 0
    accuracy = float((pred_up == actual_up).mean())
    base_rate = float(actual_up.mean())

    var_f = d["sigma_next"] ** 2
    proxy = d["realised"] ** 2
    naive = (d["ret"].rolling(21).std().shift(1) ** 2).reindex(d.index)
    valid = naive.notna() & (naive > 0) & (var_f > 0)

    def qlike(v):
        return float(np.mean(proxy[valid] / v[valid] - np.log(proxy[valid] / v[valid]) - 1.0))

    return pd.DataFrame([{
        "n_forecasts": len(d),
        "mean_r2_vs_zero": round(r2_vs_zero, 5),
        "mean_directional_accuracy": round(accuracy, 4),
        "naive_base_rate_up": round(base_rate, 4),
        "mean_pred_pct_up": round(float(pred_up.mean()), 4),
        "vol_forecast_mse": float(np.mean((var_f[valid] - proxy[valid]) ** 2)),
        "vol_naive21_mse": float(np.mean((naive[valid] - proxy[valid]) ** 2)),
        "vol_forecast_qlike": round(qlike(var_f), 5),
        "vol_naive21_qlike": round(qlike(naive), 5),
        "vol_corr_with_abs_ret": round(float(
            d.loc[valid, "sigma_next"].corr(d.loc[valid, "realised"].abs())), 4),
        "mean_ann_vol_forecast_pct": round(float(d["ann_vol_forecast"].mean() * 100), 2),
        "realised_ann_vol_pct": round(float(d["ret"].std() * np.sqrt(TRADING_DAYS) * 100), 2),
    }])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_results(df, fits, diag, ticker, mode, outdir):
    # 1. Equity curve + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1]})
    ax1.plot(df.index, df["strat_cum"], lw=1.5, color="#1f4e79",
             label=f"ARIMA+GARCH ({mode})")
    ax1.plot(df.index, df["bench_cum"], lw=1.5, color="#999999",
             label=f"Buy & Hold {ticker}")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log scale)")
    ax1.set_title(f"Week 8 — ARIMA+GARCH vs Buy & Hold ({ticker}, out-of-sample)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    for col, lab, c in (("strat_cum", "Strategy", "#1f4e79"),
                        ("bench_cum", "Buy & Hold", "#999999")):
        dd = df[col] / df[col].cummax() - 1.0
        ax2.fill_between(df.index, dd * 100, 0, alpha=0.35, color=c, label=lab)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curve.png"), dpi=140)
    plt.close(fig)

    # 2. Volatility forecast vs realised
    fig, ax = plt.subplots(figsize=(13, 5))
    realised21 = df["ret"].rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
    ax.plot(df.index, df["ann_vol_forecast"] * 100, lw=1.1, color="#c0392b",
            label="GARCH(1,1) forecast (annualised)")
    ax.plot(df.index, realised21, lw=1.1, color="#333333", alpha=0.6,
            label="Realised vol, 21-day trailing")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("GARCH one-step-ahead volatility forecast vs realised")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "volatility_forecast.png"), dpi=140)
    plt.close(fig)

    # 3. Exposure through time
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(df.index, df["held"], 0, color="#1f4e79", alpha=0.5, lw=0)
    ax.axhline(1.0, color="#999999", ls="--", lw=1, label="Buy & Hold weight")
    ax.set_ylabel("Portfolio weight")
    ax.set_title(f"Position size through time (mode = {mode})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exposure.png"), dpi=140)
    plt.close(fig)

    # 4. Mean-forecast scatter + GARCH persistence through refits
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))
    realised = df["ret"].shift(-1)
    axa.scatter(df["mu_next"] * 100, realised * 100, s=5, alpha=0.25,
                color="#1f4e79")
    axa.axhline(0, color="#999999", lw=0.8)
    axa.axvline(0, color="#999999", lw=0.8)
    axa.set_xlabel("ARIMA forecast for next day (%)")
    axa.set_ylabel("Realised next-day return (%)")
    acc = float(diag["mean_directional_accuracy"].iloc[0])
    base = float(diag["naive_base_rate_up"].iloc[0])
    axa.set_title(f"Mean forecast vs reality\naccuracy {acc:.3f} vs base rate {base:.3f}")
    axa.grid(alpha=0.3)

    ok = fits[fits["status"] == "ok"] if "status" in fits else fits
    if len(ok):
        axb.plot(pd.to_datetime(ok["train_end"]), ok["garch_persistence"],
                 marker="o", ms=3, lw=1, color="#c0392b")
        axb.axhline(1.0, color="#999999", ls="--", lw=1, label="Unit persistence")
        axb.set_ylabel(r"GARCH $\alpha + \beta$")
        axb.set_title("GARCH persistence across walk-forward refits")
        axb.legend()
        axb.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "forecast_quality.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Week 8 — ARIMA mean + GARCH volatility forecast strategy")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--p", type=int, default=1, help="ARIMA AR order")
    p.add_argument("--d", type=int, default=0, help="ARIMA differencing order")
    p.add_argument("--q", type=int, default=1, help="ARIMA MA order")
    p.add_argument("--window", type=int, default=1000,
                   help="trailing days used to fit each block")
    p.add_argument("--refit-every", type=int, default=21,
                   help="days between parameter refits")
    p.add_argument("--mode", default="combo",
                   choices=["mean", "voltarget", "combo"])
    p.add_argument("--threshold", type=float, default=0.0,
                   help="mean-forecast entry threshold in bps")
    p.add_argument("--target-vol", type=float, default=0.15,
                   help="annualised vol target for sizing")
    p.add_argument("--max-weight", type=float, default=1.0,
                   help="exposure cap; 1.0 = no leverage")
    p.add_argument("--rebal-band", type=float, default=0.05,
                   help="only rebalance when the target weight moves this much")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir if os.path.isabs(args.outdir) \
        else os.path.join(script_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Week 8 — ARIMA({args.p},{args.d},{args.q}) + GARCH(1,1): {args.ticker}")
    print(f"{'='*60}")

    # 1. Data
    df = fetch_prices(args.ticker, args.start, args.end)
    print(f"  Fetched {len(df)} rows: {df.index[0].date()} -> {df.index[-1].date()}")

    # 2. Walk-forward forecasts
    print(f"  Walk-forward: window={args.window}, refit every {args.refit_every} days")
    print(f"  Fitting ... (this is the slow part)")
    df, fits = walk_forward(df, args.p, args.d, args.q, args.window,
                            args.refit_every)
    n_ok = int((fits["status"] == "ok").sum()) if len(fits) else 0
    print(f"  Refits: {n_ok} converged, {len(fits) - n_ok} failed")
    fits.to_csv(os.path.join(outdir, "fit_log.csv"), index=False)
    if n_ok:
        ok = fits[fits["status"] == "ok"]
        print(f"  Mean GARCH persistence (alpha+beta): "
              f"{ok['garch_persistence'].mean():.4f}")

    # 3. Positions + backtest
    sig = build_positions(df, args.mode, args.threshold, args.target_vol,
                          args.max_weight, args.rebal_band)
    bt = backtest(sig, args.cost_bps)
    print(f"  Out-of-sample window: {bt.index[0].date()} -> {bt.index[-1].date()} "
          f"({len(bt)} days)")

    # 4. Forecast quality
    diag = forecast_diagnostics(sig)
    diag.to_csv(os.path.join(outdir, "forecast_diagnostics.csv"), index=False)
    print(f"\n  FORECAST QUALITY (scored independently of P&L)")
    print(f"    Mean model  R^2 vs zero:      {diag['mean_r2_vs_zero'].iloc[0]:>9.5f}")
    print(f"    Mean model  accuracy:         {diag['mean_directional_accuracy'].iloc[0]:>9.4f}"
          f"   (base rate {diag['naive_base_rate_up'].iloc[0]:.4f})")
    print(f"    Vol model   QLIKE:            {diag['vol_forecast_qlike'].iloc[0]:>9.5f}")
    print(f"    Naive 21d   QLIKE:            {diag['vol_naive21_qlike'].iloc[0]:>9.5f}"
          f"   (lower is better)")
    print(f"    Vol vs |return| correlation:  {diag['vol_corr_with_abs_ret'].iloc[0]:>9.4f}")

    # 5. Metrics
    strat_m = compute_metrics(bt, f"ARIMA+GARCH ({args.mode})", "strat_ret")
    bench_m = compute_metrics(bt, f"Buy & Hold {args.ticker}", "bench_ret")
    pd.DataFrame([strat_m, bench_m]).to_csv(
        os.path.join(outdir, "metrics.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS (out-of-sample)")
    print(f"{'='*60}")
    for m in (strat_m, bench_m):
        print(f"\n  {m['label']}")
        print(f"    Total return:     {m['total_return_pct']:>8.2f}%")
        print(f"    CAGR:             {m['cagr_pct']:>8.2f}%")
        print(f"    Ann. volatility:  {m['ann_volatility_pct']:>8.2f}%")
        print(f"    Sharpe:           {m['sharpe']:>8.3f}")
        print(f"    Max drawdown:     {m['max_drawdown_pct']:>8.2f}%")
        print(f"    Calmar:           {m['calmar']:>8.3f}")
        if "pct_time_in_market" in m:
            print(f"    Time in market:   {m['pct_time_in_market']:>7.1f}%")
            print(f"    Avg exposure:     {m['avg_exposure']:>8.3f}")
            print(f"    Switches:         {m['num_switches']:>8d}")
            print(f"    Total cost:       {m['total_cost_pct']:>8.2f}%")

    # 6. Annual breakdown
    annual = annual_returns(bt)
    annual.to_csv(os.path.join(outdir, "annual_returns.csv"), index=False)
    print(f"\n  Annual returns:")
    print(annual.to_string(index=False))

    # 7. Worst drawdowns
    dd = worst_drawdowns(bt, "strat_ret", n=5)
    dd.to_csv(os.path.join(outdir, "worst_drawdowns.csv"), index=False)
    print(f"\n  5 worst drawdowns (strategy):")
    print(dd.to_string(index=False))

    # 8. Daily data
    export_cols = ["close", "ret", "mu_next", "sigma_next", "ann_vol_forecast",
                   "long_gate", "vol_weight", "target_weight", "position",
                   "held", "turnover",
                   "strat_ret", "bench_ret", "strat_cum", "bench_cum"]
    bt[export_cols].to_csv(os.path.join(outdir, "backtest.csv"))
    print(f"\n  Saved backtest.csv ({len(bt)} rows)")

    # 9. Plots
    print(f"  Generating plots ...")
    plot_results(bt, fits, diag, args.ticker, args.mode, outdir)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to {outdir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
