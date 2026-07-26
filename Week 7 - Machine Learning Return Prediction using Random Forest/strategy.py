"""
Week 7 — Machine-Learning Return Prediction (Random Forest)
===========================================================

Can a Random Forest, fed only backward-looking technical features derived from
OHLCV data, predict the *direction* of the next day's return well enough to beat
buy-and-hold after costs?

Approach
--------
1. Build a feature matrix from OHLCV: lagged returns, moving-average ratios,
   RSI(14), rolling realised volatility, volume ratio, daily range, momentum.
2. Label each row with the SIGN of the NEXT period's return (binary target).
   The label is used only for training — it is never a feature.
3. Split the sample CHRONOLOGICALLY (no shuffling). Fit the model on the first
   `train_frac` of history and evaluate strictly out-of-sample on the tail.
4. Trade the out-of-sample predictions only: go long when the model predicts an
   up day, hold cash otherwise (long/short optional via --long-short).
5. Benchmark against buy-and-hold over the identical out-of-sample window.

Honesty notes (why this is the "hard" week)
-------------------------------------------
* The single biggest way to lie to yourself with ML on markets is LOOK-AHEAD
  BIAS. Everything here is engineered against it: features use only past data
  (`.rolling()` is backward-looking, target uses `.shift(-1)` and is never a
  feature), the train/test split is time-ordered, the model is fit on train
  only, and the backtest uses `position.shift(1) * ret` so a decision made at
  the close of day t is paid the return of day t+1.
* Results are reported OUT-OF-SAMPLE ONLY. In-sample accuracy on a Random Forest
  is near-meaningless (it can memorise the training set).
* Directional accuracy near 50% is the honest expectation. A model that clears
  ~52-54% out-of-sample after costs is already doing something. If you see performance substantially 
  higher than that on daily equity index data, the first assumption should be data leakage—not that 
  you’ve discovered an exceptional model.

Usage
-----
    python strategy.py                         # SPY, default settings
    python strategy.py --ticker QQQ --start 2010-01-01
    python strategy.py --horizon 5 --long-short
    python strategy.py --n-estimators 500 --max-depth 4 --train-frac 0.7

Dependencies: pandas, numpy, yfinance, matplotlib, scikit-learn
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

import yfinance as yf


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def fetch_prices(ticker, start, end):
    """Download daily OHLCV from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna()
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker / dates.")
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering  (ALL features are strictly backward-looking)
# --------------------------------------------------------------------------- #
def _rsi_wilder(close, window=14):
    """Wilder's RSI — same implementation used in Week 2, kept consistent."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df, horizon=1):
    """
    Construct the feature matrix and the classification target.

    Target = 1 if the return over the NEXT `horizon` days is positive, else 0.
    The target is derived from data strictly in the future of each row and is
    used only as a label — never as a model input.
    """
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    feats = {}

    # Lagged returns (momentum / autocorrelation structure)
    for lag in (1, 2, 3, 5, 10):
        feats[f"ret_lag_{lag}"] = df["close"].pct_change(lag)

    # Price relative to moving averages (trend)
    for w in (5, 10, 20):
        feats[f"close_sma_{w}"] = df["close"] / df["close"].rolling(w).mean() - 1

    # RSI (mean-reversion / overbought-oversold)
    feats["rsi_14"] = _rsi_wilder(df["close"], 14)

    # Rolling realised volatility (regime)
    for w in (10, 21):
        feats[f"vol_{w}"] = df["ret"].rolling(w).std()

    # Volume relative to its own average (participation)
    feats["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

    # Normalised daily range (intraday pressure)
    feats["range"] = (df["high"] - df["low"]) / df["close"]

    # Momentum over medium horizons
    feats["mom_10"] = df["close"] / df["close"].shift(10) - 1
    feats["mom_20"] = df["close"] / df["close"].shift(20) - 1

    feat_df = pd.DataFrame(feats, index=df.index)
    feature_cols = list(feat_df.columns)

    out = df.join(feat_df)

    # Forward return over the prediction horizon, and its sign as the label.
    out["fwd_ret"] = out["close"].pct_change(horizon).shift(-horizon)
    out["target"] = (out["fwd_ret"] > 0).astype(int)

    # Drop rows with NaNs from feature warm-up or the un-observable final label.
    out = out.dropna(subset=feature_cols + ["target"])
    return out, feature_cols


# --------------------------------------------------------------------------- #
# 3. Train / predict  (chronological, out-of-sample only)
# --------------------------------------------------------------------------- #
def train_predict(df, feature_cols, train_frac=0.7, n_estimators=300,
                  max_depth=5, min_leaf=50, seed=42):
    """
    Fit a Random Forest on the first `train_frac` of the sample and predict on
    the remaining tail. Returns (test_df, model, split_date, class_report).

    Nothing from the test window touches the fit. This is the whole ballgame:
    the split is by time, not random, so the model never sees the future.
    """
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split]
    test = df.iloc[split:].copy()

    if len(test) < 30:
        raise ValueError("Test window too small — lower --train-frac or widen dates.")

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,   # heavy regularisation: markets are noisy
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test["pred"] = model.predict(X_test)
    test["pred_proba"] = model.predict_proba(X_test)[:, 1]

    report = {
        "n_train": len(train),
        "n_test": len(test),
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "accuracy": accuracy_score(y_test, test["pred"]),
        "precision_up": precision_score(y_test, test["pred"], zero_division=0),
        "base_rate_up": y_test.mean(),   # naive "always long" hit rate
    }
    tn, fp, fn, tp = confusion_matrix(y_test, test["pred"]).ravel()
    report.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})

    return test, model, test.index[0], report


# --------------------------------------------------------------------------- #
# 4. Backtest
# --------------------------------------------------------------------------- #
def backtest(df, cost_bps=10.0, long_short=False):
    """
    Convert predictions into positions and P&L on the out-of-sample window.

    position(t) is decided at the close of day t; it earns ret(t+1). We enforce
    that with position.shift(1) * ret. Costs are charged on turnover.
    """
    df = df.copy()

    if long_short:
        df["position"] = np.where(df["pred"] == 1, 1.0, -1.0)
    else:
        df["position"] = np.where(df["pred"] == 1, 1.0, 0.0)  # long / cash

    # Return earned today comes from yesterday's position (no look-ahead).
    df["strat_ret_gross"] = df["position"].shift(1) * df["ret"]

    # Transaction cost on every change of position.
    turnover = df["position"].diff().abs().fillna(0.0)
    df["cost"] = turnover * (cost_bps / 1e4)
    df["strat_ret"] = df["strat_ret_gross"] - df["cost"]

    df["spy_ret"] = df["ret"]  # benchmark: buy & hold the same instrument
    df = df.dropna(subset=["strat_ret"])

    df["strat_cum"] = (1 + df["strat_ret"]).cumprod()
    df["bench_cum"] = (1 + df["spy_ret"]).cumprod()

    df["num_switches"] = (turnover.loc[df.index] > 0).sum()
    return df


# --------------------------------------------------------------------------- #
# 5. Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(df, label, ret_col):
    r = df[ret_col].dropna()
    n = len(r)
    total_return = (1 + r).prod() - 1
    years = n / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan

    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan

    m = {
        "label": label,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "ann_volatility_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "calmar": calmar,
    }
    if ret_col == "strat_ret" and "position" in df.columns:
        active = df["position"].abs() > 0
        m["pct_time_in_market"] = active.mean() * 100
        m["num_switches"] = int((df["position"].diff().abs() > 0).sum())
    return m


def annual_returns(df):
    rows = []
    for year, grp in df.groupby(df.index.year):
        rows.append({
            "year": year,
            "strategy_pct": ((1 + grp["strat_ret"]).prod() - 1) * 100,
            "buy_hold_pct": ((1 + grp["spy_ret"]).prod() - 1) * 100,
        })
    return pd.DataFrame(rows)


def worst_drawdowns(df, ret_col, n=5):
    cum = (1 + df[ret_col]).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    spells, in_dd, start = [], False, None
    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd, start = True, date
        elif val >= 0 and in_dd:
            in_dd = False
            spell = dd.loc[start:date]
            spells.append((start, spell.idxmin(), date, spell.min() * 100))
    if in_dd:
        spell = dd.loc[start:]
        spells.append((start, spell.idxmin(), dd.index[-1], spell.min() * 100))

    out = pd.DataFrame(spells, columns=["start", "trough", "recovered", "depth_pct"])
    out = out.sort_values("depth_pct").head(n).reset_index(drop=True)
    out["start"] = out["start"].dt.date
    out["trough"] = out["trough"].dt.date
    out["recovered"] = out["recovered"].dt.date
    return out


# --------------------------------------------------------------------------- #
# 6. Plots
# --------------------------------------------------------------------------- #
def plot_results(df, model, feature_cols, report, outdir):
    # 1. Equity curve (out-of-sample)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["strat_cum"], label="RF Strategy", linewidth=1.2)
    ax.plot(df.index, df["bench_cum"], label="Buy & Hold",
            linewidth=1.2, alpha=0.7)
    ax.set_title("Week 7 — Random Forest Direction Model: Growth of $1 (out-of-sample)",
                 fontsize=13)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity_curve.png"), dpi=150)
    plt.close(fig)
    print("  Saved equity_curve.png")

    # 2. Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp.index, imp.values, alpha=0.8)
    ax.set_title("Random Forest Feature Importance", fontsize=12)
    ax.set_xlabel("Gini importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("  Saved feature_importance.png")

    # 3. Drawdown comparison
    fig, ax = plt.subplots(figsize=(12, 4))
    strat_dd = (df["strat_cum"] - df["strat_cum"].cummax()) / df["strat_cum"].cummax() * 100
    bench_dd = (df["bench_cum"] - df["bench_cum"].cummax()) / df["bench_cum"].cummax() * 100
    ax.fill_between(df.index, strat_dd, 0, alpha=0.4, label="Strategy DD")
    ax.fill_between(df.index, bench_dd, 0, alpha=0.3, label="B&H DD")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Comparison (out-of-sample)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drawdown.png"), dpi=150)
    plt.close(fig)
    print("  Saved drawdown.png")

    # 4. Rolling directional accuracy — is any edge stable, or a lucky streak?
    correct = (df["pred"] == df["target"]).astype(float)
    roll_acc = correct.rolling(63).mean()   # ~one quarter
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, roll_acc, linewidth=1.0, label="63-day rolling accuracy")
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--", label="coin flip")
    ax.axhline(report["accuracy"], color="green", linewidth=0.8,
               alpha=0.7, label=f"overall = {report['accuracy']:.3f}")
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Rolling Directional Accuracy", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rolling_accuracy.png"), dpi=150)
    plt.close(fig)
    print("  Saved rolling_accuracy.png")


# --------------------------------------------------------------------------- #
# 7. Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Week 7 — Random Forest return-direction model")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--horizon", type=int, default=1, help="prediction horizon in days")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-leaf", type=int, default=50)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--long-short", action="store_true",
                   help="short on down predictions instead of going to cash")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default=None, help="directory to save outputs (default: ./outputs/<ticker>)")
    args = p.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Week 7 — Random Forest direction model: {args.ticker}")
    print(f"{'='*60}")

    # 1. Data
    df = fetch_prices(args.ticker, args.start, args.end)
    print(f"  Fetched {len(df)} rows: {df.index[0].date()} -> {df.index[-1].date()}")

    # 2. Features + label
    df, feature_cols = build_features(df, horizon=args.horizon)
    print(f"  Built {len(feature_cols)} features, {len(df)} usable rows "
          f"(horizon = {args.horizon}d)")

    # 3. Train / predict (out-of-sample)
    test, model, split_date, report = train_predict(
        df, feature_cols,
        train_frac=args.train_frac,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
        seed=args.seed,
    )
    print(f"\n  Train: {report['train_start']} -> {report['train_end']} "
          f"({report['n_train']} rows)")
    print(f"  Test:  {report['test_start']} -> {report['test_end']} "
          f"({report['n_test']} rows, OUT-OF-SAMPLE)")
    print(f"  Directional accuracy: {report['accuracy']:.4f} "
          f"(base rate up = {report['base_rate_up']:.4f})")
    print(f"  Precision on 'up':    {report['precision_up']:.4f}")

    pd.DataFrame([report]).to_csv(os.path.join(outdir, "classification_report.csv"),
                                  index=False)

    # 4. Backtest
    bt = backtest(test, cost_bps=args.cost_bps, long_short=args.long_short)

    # 5. Metrics
    strat_m = compute_metrics(bt, "RF Strategy", "strat_ret")
    bench_m = compute_metrics(bt, "Buy & Hold", "spy_ret")
    pd.DataFrame([strat_m, bench_m]).to_csv(os.path.join(outdir, "metrics.csv"),
                                            index=False)

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
            print(f"    Switches:         {m['num_switches']:>8d}")

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

    # 8. Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp.sort_values(ascending=False).to_csv(
        os.path.join(outdir, "feature_importance.csv"), header=["importance"])
    print(f"\n  Top features:")
    print(imp.sort_values(ascending=False).head(6).to_string())

    # 9. Save daily out-of-sample data
    export_cols = ["close", "ret", "pred", "pred_proba", "position",
                   "strat_ret", "strat_cum", "bench_cum"]
    bt[export_cols].to_csv(os.path.join(outdir, "backtest.csv"))
    print(f"\n  Saved backtest.csv ({len(bt)} rows)")

    # 10. Plots
    print(f"\n  Generating plots ...")
    plot_results(bt, model, feature_cols, report, outdir)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to {outdir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
