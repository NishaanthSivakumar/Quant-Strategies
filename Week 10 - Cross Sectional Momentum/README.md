# Week 10 — Cross-Sectional Momentum (Jegadeesh–Titman 12-1)

**Result: −11.70% CAGR against +7.97% for SPY. The loss is not statistically significant (t = −1.26), and survivorship bias in the universe acts directly against the short leg.**

Long top decile by 12-1 momentum, short bottom decile, monthly rebalance, 137 large-cap US equities, March 2000 – December 2024 (298 months).

| Metric | Momentum L/S (net) | Long leg | Equal-weight universe | SPY |
|---|---|---|---|---|
| CAGR | −11.70% | 10.98% | 13.96% | 7.97% |
| Total return | −95.46% | +1,230.6% | +2,469.3% | +571.5% |
| Annualised vol | 27.14% | 19.04% | 15.76% | 15.29% |
| Sharpe | −0.31 | 0.64 | 0.91 | 0.58 |
| Max drawdown | −94.95% | −48.45% | −47.28% | −50.78% |
| Monthly win rate | 51.34% | 61.41% | 66.78% | 63.42% |
| Skew | −1.15 | −0.13 | −0.43 | −0.48 |

Gross CAGR is −10.35%; costs account for 1.35pp of the loss.

## Hypothesis

Stocks that outperformed over the prior twelve months, excluding the most recent month, continue to outperform over the following month. A long-short decile spread on that ranking should earn a positive return.

Reference: Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*.

## Method

| Parameter | Value |
|---|---|
| Universe | 137 large-cap US equities (140 requested, 3 unavailable) |
| Formation window | 12 months, skipping the most recent month |
| Ranking | cross-sectional, deciles |
| Portfolio | long top decile, short bottom decile, equal weight within leg |
| Exposure | 100% long / 100% short, zero-cost, 200% gross |
| Holding period | 1 month |
| Rebalance | monthly, at month-end |
| Names per leg | 13 average |
| Transaction cost | 10 bps one-way per dollar traded |
| Benchmarks | SPY buy-and-hold, equal-weight universe |
| Random control | 50 seeds, same leg sizes, drawn from the same eligible set |
| Sample | March 2000 – December 2024, 298 months |

Parameters are textbook defaults, fixed before the run and not adjusted afterwards.

At each month-end `t` the signal uses prices through `t` only; the position is held over month `t+1` and earns `r[t+1]`. Turnover is measured against the previous month's drifted book and cost charged on dollars traded. Names without a computable signal or a tradable price at `t` are excluded for that month.

## Diagnostics

### Significance of the spread

| Window | Gross spread / month | t |
|---|---|---|
| Full sample | −0.57% | −1.26 |
| Excluding 2009 | −0.25% | −0.58 |
| 2010–2024 | +0.10% | +0.24 |

Two months carry most of the loss: November 2002 (−36.95%) and April 2009 (−34.90%).

### Selection quality, scored independently of P&L

| Measure | Value |
|---|---|
| Mean rank IC | −0.0083 (t = −0.63) |
| Months with positive IC | 50.7% |
| Decile monotonicity (Spearman) | −0.94 |
| D1 mean forward return (lowest momentum) | +1.60% / month |
| D10 mean forward return (highest momentum) | +1.00% / month |

Decile ordering is near-monotone in the opposite direction to the hypothesis. Monotonicity carries no standard error; the underlying slope is the IC above.

### Leg decomposition

| Leg | CAGR | Vol | Max DD |
|---|---|---|---|
| Long (winners), long-only | +10.98% | 19.04% | −48.45% |
| Short (losers), short-only | −21.58% | 30.14% | −99.72% |

Short-leg detail for the two worst months:

| Month | Shorted basket return | Largest contributors |
|---|---|---|
| Nov 2002 | +39.8% | AMT +176%, GLW +137% |
| Apr 2009 | +33.6% | AXP +88%, PRU +52%, AFL +49%, AIG +38% |

### Random control

| Basis | Random mean CAGR | sd | Momentum | Percentile | z |
|---|---|---|---|---|---|
| Net | −4.93% | 1.74% | −11.70% | 0 | −3.89 |
| Gross | −0.67% | 1.81% | −10.35% | 0 | −5.35 |

Random L/S vol is 10.3% against 27.1% for momentum, so volatility drag differs by ~3.1pp/yr (0.5σ²). The cross-seed sd measures selection sampling noise on a shared price path and is not a standard error; z here is not a t-statistic.

### Turnover and costs

| Measure | Value |
|---|---|
| Average monthly turnover | 1.25x |
| Annual cost drag | 1.50% |
| Random-control turnover | 3.64x |

## Run

```bash
pip install yfinance pandas numpy matplotlib
python strategy.py
```

Prices cache to `results/prices_cache.csv` on first run; `--refresh` forces re-download.

| Argument | Default | Description |
|---|---|---|
| `--tickers` | 140 large caps | Universe |
| `--benchmark` | `SPY` | Benchmark ticker |
| `--start` / `--end` | `1999-01-01` / `2024-12-31` | Sample window |
| `--formation` | `12` | Formation window in months |
| `--skip` | `1` | Months skipped before the formation window |
| `--n-portfolios` | `10` | Ranked buckets; 10 = deciles |
| `--cost-bps` | `10` | One-way cost per dollar traded |
| `--random-seeds` | `50` | Random-control draws |
| `--refresh` | off | Force price re-download |

## Files

| File | Contents |
|---|---|
| `strategy.py` | Full implementation |
| `results/metrics.csv` | Performance for both legs, combined, and both benchmarks |
| `results/diagnostics.csv` | IC, monotonicity, turnover, random-control statistics |
| `results/annual_returns.csv` | Calendar-year returns by leg and benchmark |
| `results/worst_drawdowns.csv` | Deepest drawdown episodes |
| `results/backtest.csv` | Monthly leg returns, turnover, cost, equity |
| `results/decile_returns.csv` | Mean forward return by momentum decile |
| `results/ic_by_month.csv` | Monthly rank information coefficient |
| `results/random_control.csv` | Per-seed CAGR, Sharpe, turnover — net and gross |
| `results/equity_curves.png` | Momentum vs SPY vs equal-weight universe |
| `results/decile_returns.png` | Mean next-month return by decile |
| `results/ic_series.png` | Monthly IC with 12-month rolling mean |
| `results/leg_decomposition.png` | Long leg vs short leg vs combined |
| `results/random_control.png` | Momentum against the 50-seed random distribution |

## Notes and limitations

- **Survivorship bias.** The universe is a fixed list of stocks listed at the time of writing, so companies that were delisted or acquired are absent. This acts specifically against the short leg, since names that went to zero are exactly what belongs there. Magnitude: equal-weight universe 13.96%/yr vs SPY 7.97%/yr — a ~6pp/yr gap for US large caps. Correcting it requires a point-in-time constituent database, which this series does not have.
- **The result is not statistically significant.** t = −1.26 gross, −1.54 net, +0.24 over 2010–2024. This is not evidence that momentum is inverted.
- **Three tickers were unavailable at run time.** MRO (acquired by ConocoPhillips, ceased trading Nov 2024) and K (Kellanova, delisted Dec 2025) are no longer retrievable from the data source. BK remains listed but changed ticker to BNY in May 2026; it is missing here and should be restored on any re-run.
- **Exposure is not risk-matched to the benchmarks.** A 200% gross, 27%-vol long-short book is compared on CAGR to 15%-vol indices, which mixes mean return with volatility drag.
- **Large-cap universe only.** Momentum is documented as weaker among the largest and most liquid names, making this close to the least favourable cross-section for the test.
- **No borrow costs or short-availability constraints.** The short leg is frictionless beyond the 10 bps trading cost. In 2002 and 2009 several of those names would have been costly or impossible to borrow.
- **Single formation window.** Only 12-1 is reported; no window scan was run, per the series convention against post-hoc parameter selection.
- **The random control is a selection test, not a significance test.** See the diagnostics table above.
