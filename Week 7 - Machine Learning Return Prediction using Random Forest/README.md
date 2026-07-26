# Week 7 — Machine-Learning Return Prediction (Random Forest)

Out-of-sample test of whether a Random Forest, trained only on backward-looking technical features from OHLCV, can predict the direction of SPY's next-day return well enough to beat buy-and-hold (2020–2026). Part of the [10-week quant strategy series](../).

## Result

**Lost to buy-and-hold — and the model's directional accuracy is *below* the naive base rate.** The strategy earns a positive absolute return but is beaten on every risk-adjusted measure by simply holding SPY. More tellingly, it predicts the correct direction on 51.7% of out-of-sample days, while the market rose on 54.8% of them — so a coin that *always* guessed "up" would have been right more often than the model.

| Metric            | RF Strategy | Buy & Hold |
|-------------------|-------------|------------|
| Total return      | +38.6%      | +141.8%    |
| CAGR              | 5.20%       | 14.72%     |
| Ann. volatility   | 16.36%      | 20.35%     |
| Sharpe            | 0.39        | 0.78       |
| Max drawdown      | −25.60%     | −33.72%    |
| Calmar            | 0.20        | 0.44       |
| Time in market    | 61.6%       | 100%       |
| Position switches | 497         | —          |

**Classification (out-of-sample, 1,621 days):**

| Metric                     | Value |
|----------------------------|-------|
| Directional accuracy       | 51.7% |
| Base rate (always-long)    | 54.8% |
| Precision on "up" calls    | 55.3% |

The only thing the strategy improves is max drawdown (−25.6% vs −33.7%), and that is a side effect of sitting in cash 38% of the time, not of well-timed exits. See `equity_curve.png` (strategy tracks the benchmark until 2022, then flatlines while SPY runs away) and `rolling_accuracy.png` (accuracy swings from 0.37 to 0.70 and mean-reverts to a coin flip).

## Hypothesis

A Random Forest fed lagged returns, moving-average ratios, RSI, realised volatility, a volume ratio, daily range, and momentum can classify the sign of the next day's return with enough edge to beat buy-and-hold after transaction costs.

## Method

- **Features (15, all strictly backward-looking):** returns at lags 1/2/3/5/10; close vs SMA-5/10/20; RSI-14; realised vol over 10 and 21 days; volume ÷ 20-day average volume; normalised daily range `(high−low)/close`; momentum over 10 and 20 days.
- **Target:** sign of the next day's return (`fwd_ret > 0`). Used only as a training label — never as a feature.
- **Split:** chronological 70/30, **no shuffling**. Train 2005-02-02 → 2020-02-10 (3,781 rows); test 2020-02-11 → 2026-07-24 (1,621 rows). The model never sees the future.
- **Model:** `RandomForestClassifier`, heavily regularised (`max_depth=5`, `min_samples_leaf=50`, `max_features="sqrt"`, `class_weight="balanced"`) because daily equity noise overfits trivially.
- **Signal:** long when the model predicts "up", cash otherwise (long/short optional via `--long-short`).
- **P&L alignment:** `position.shift(1) * ret` — a decision made at the close of day *t* earns the return of day *t+1*. No look-ahead.
- **Cost:** 10 bps per position switch (Week 4+ convention).
- **Benchmark:** buy-and-hold SPY over the identical out-of-sample window.
- **Data:** `yfinance` daily OHLCV, auto-adjusted.

## Run

```bash
pip install scikit-learn yfinance pandas numpy matplotlib
python strategy.py

python strategy.py --ticker QQQ                       # different underlying
python strategy.py --horizon 5                        # 5-day-ahead direction
python strategy.py --long-short                       # short instead of cash
python strategy.py --cost-bps 0                        # isolate the cost drag
python strategy.py --n-estimators 500 --max-depth 4   # tune the forest
```

Outputs are written to the folder `strategy.py` lives in.

## Files

| File | Description |
|---|---|
| `strategy.py` | End-to-end pipeline: OHLCV → features → chronological RF → OOS backtest |
| `metrics.csv` | Strategy vs. buy-and-hold summary (return, Sharpe, drawdown, etc.) |
| `classification_report.csv` | Train/test dates, accuracy, precision, base rate, confusion matrix |
| `annual_returns.csv` | Year-by-year strategy vs. benchmark |
| `worst_drawdowns.csv` | Five deepest strategy drawdowns with start/trough/recovery |
| `feature_importance.csv` | Gini importance per feature |
| `backtest.csv` | Daily out-of-sample series: prediction, probability, position, equity |
| `equity_curve.png` | Growth of $1, strategy vs. buy-and-hold |
| `feature_importance.png` | Ranked feature importances |
| `drawdown.png` | Underwater curves, strategy vs. buy-and-hold |
| `rolling_accuracy.png` | 63-day rolling directional accuracy vs. the coin-flip line |

## Notes & limitations

- **Accuracy is below the always-long base rate.** The single most important number here: 51.7% < 54.8%. The model has a whisper of signal over a 50/50 coin, but it is beaten by the dumbest possible baseline in a market that mostly went up.
- **The exit calls are worse than a coin flip.** Of the 622 days the model went to cash, the market rose on 336 of them (~54%). Every time it steps aside it is more likely than not throwing away an up day — this is the mechanism behind the flat equity curve.
- **It didn't even earn its keep as a defense.** In 2022, the one bear year, the strategy lost *more* than buy-and-hold (−20.8% vs −18.2%). The shallower max drawdown comes from time in cash, not skilful timing.
- **Overtrading.** 497 switches ≈ one flip every three days. Run with `--cost-bps 0` to size the transaction-cost bill.
- **No stable edge.** Feature importances are bunched between 0.05 and 0.09 with nothing dominating, and rolling accuracy oscillates around 0.517 with no persistent regime — noise, not a tradable signal.
- **Single train/test split, not walk-forward.** A rolling / expanding-window retrain would be more realistic and is the natural next iteration; it is unlikely to change the conclusion on daily index direction.
- **Directional ≠ profitable, and vice versa.** Accuracy and P&L diverge because a few large-move days dominate returns; both are reported so neither can flatter the result.

## References

- Marcos López de Prado (2018), *Advances in Financial Machine Learning* — on look-ahead bias, chronological splitting, and why naive ML backtests overstate performance.
