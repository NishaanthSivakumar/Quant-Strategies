# Week 9 — Pairs Trading / Cointegration

**Result: lost. −17.31% total return (−1.30% CAGR) against +681.40% for buy-and-hold SPY over the same period. Market-neutrality worked (beta 0.020). The trading rule did not.**

The cointegration filter genuinely finds spreads that revert more often than randomly chosen ones — 30.74% convergence versus 23.89% ± 3.23% for random pairs, more than two standard deviations above the random distribution. It also stops out more and trades 27% more often, and the two cancel exactly. Net return lands in the lower third of the random-selection band.

| Metric | Pairs (net) | Buy & hold SPY |
|---|---|---|
| Total return | −17.31% | +681.40% |
| CAGR | −1.30% | +15.16% |
| Annualised volatility | 3.36% | 16.56% |
| Sharpe | −0.372 | 0.935 |
| Sortino | −0.360 | 1.152 |
| Max drawdown | −21.35% | −33.72% |
| Calmar | −0.061 | 0.450 |
| Beta to SPY | 0.020 | 1.000 |
| Correlation to SPY | 0.099 | 1.000 |

Period: 2012-01-03 to 2026-08-10 (3,671 trading days). Gross Sharpe is −0.167, so roughly half the loss is transaction costs and half is the strategy.

## Hypothesis

If two assets are cointegrated, the spread between them is stationary. When the spread moves far from its mean it should revert: long the cheap leg, short the expensive leg, collect the convergence. Returns should be close to market-neutral because the market exposure of the two legs largely cancels.

The question this week is not whether mean reversion exists — it does — but whether *selecting pairs by cointegration* adds anything over selecting them some other way, once costs are paid. The strategy is therefore run three times with identical machinery and only the selection rule changed: lowest Engle-Granger p-value, highest return correlation, and random draw from the same universe.

## Method

Universe is 20 US-listed ETFs chosen ex ante for plausible economic linkage (energy, precious metals, country funds, US sectors). Duplicate share classes of the same index are excluded — they cointegrate trivially and the spread is smaller than the cost of trading it.

Walk-forward, no overlap between estimation and trading:

1. Every 63 trading days, look back over a 252-day formation window.
2. Test all 190 candidate pairs for cointegration on log prices (Engle-Granger, constant trend, fixed lag 1). Keep the 5 lowest p-values below 0.05.
3. Estimate the hedge ratio β by OLS on the formation window; record the formation-window spread mean and standard deviation.
4. Trade the following 63 days out of sample. The z-score uses formation β, mean and sd — nothing from the trading window enters the signal.
5. Entry at |z| ≥ 2.0, exit at |z| ≤ 0.5, stop at |z| ≥ 4.0 (pair disabled for the rest of the block). Positions are liquidated at block end because β is re-estimated.
6. Signal on day *t*, exposure on day *t+1*. Costs charged on turnover at 10 bps per unit, consistent with Weeks 4–8.

Sizing is dollar-neutral per pair: weights +1/(1+β) and −β/(1+β), so gross exposure per pair is 1. Capital is allocated at 1/`top_n`, not 1/(pairs found) — when fewer than 5 pairs clear the filter the shortfall stays in cash. Portfolio gross exposure never exceeds 1.0 and no leverage is used.

All thresholds are textbook defaults. Nothing was tuned on the reported results.

## Decomposition

Same machinery, selection rule swapped. The random control is a sample, not a number, so it runs 50 times and is reported as a distribution.

| Variant | Total return | CAGR | Volatility | Sharpe | Max DD | Ann. turnover |
|---|---|---|---|---|---|---|
| Cointegration (strategy) | −17.31% | −1.30% | 3.36% | −0.372 | −21.35% | 6.90 |
| Cointegration, zero cost | −8.57% | −0.61% | 3.35% | −0.167 | −17.34% | 6.90 |
| Correlation-selected | −14.67% | −1.08% | 2.32% | −0.458 | −16.70% | 5.50 |
| Random pairs (single seed) | −1.29% | −0.09% | 3.14% | −0.013 | −12.26% | 5.65 |
| Buy & hold SPY | +681.40% | +15.16% | 16.56% | 0.935 | −33.72% | 0.00 |

The single-seed random figure is a sampling artifact and should not be read as "random beat the strategy." Across 50 seeds:

| Random-pair control (50 seeds) | Value |
|---|---|
| CAGR mean | −0.88% |
| CAGR sd | 0.77% |
| CAGR range | −2.38% to +0.47% |
| Strategy CAGR percentile | 28th |
| Seeds the strategy beat | 14 of 50 |

The percentile has a standard error of roughly 6 points at 50 draws, so read it as "lower third of the band," not as 28%.

Where the strategy *does* separate from random is in trade mechanics, not returns:

| | Strategy | Random (50 seeds) | z |
|---|---|---|---|
| Convergence rate | 30.74% | 23.89% ± 3.23% | +2.12 |
| Stop-out rate | 38.52% | 29.51% ± 2.62% | +3.44 |
| Trades | 257 | 202 ± 9 | +5.92 |
| Ann. turnover | 6.90 | 5.47 ± 0.25 | +5.63 |
| Avg trade return | −0.16% | −0.10% ± 0.28% | −0.20 |

Cointegration selection improves convergence by 6.9 points and is above 47 of the 50 random draws. It also stops out more than every random draw, because tighter spreads that revert faster also breach 4z faster, and it pays 26% more turnover for the trades it generates. The selection works; the trading rule spends the benefit.

## Signal quality, scored separately from P&L

Following the Week 8 convention, relationship quality is measured independently of whether the trades made money.

| Diagnostic | Value |
|---|---|
| Mean formation cointegration p-value | 0.0153 |
| Mean out-of-sample forward ADF p-value | 0.4732 |
| Out-of-sample stationarity rate (252d forward window) | 5.36% |
| Median formation half-life | 6.5 days |
| Median out-of-sample half-life | 31.9 days |
| Spreads wider out of sample than in formation | 59.8% |

Two things follow.

**The selected spreads are not stationary out of sample.** The 5.36% rejection rate is the ADF test's own nominal size — indistinguishable from never rejecting. This is not a power problem: the forward window is 252 days, where the test rejects a genuinely stationary spread with an 11-day half-life about 61% of the time. `oos_adf_pvalue_block` is also reported, but the 63-day block has only ~10% power and should not be interpreted.

**The selected p-values match what pure multiple testing would produce.** 190 pairs tested per block at p < 0.05 means ~9.5 pass by chance alone even under a null of no cointegration anywhere, so the book fills every block regardless. Comparing the 5 selected p-values against the order statistics of 190 uniform draws (*k*/191):

| Rank | Observed mean p | Null expectation |
|---|---|---|
| 1 | 0.0039 | 0.0052 |
| 2 | 0.0088 | 0.0105 |
| 3 | 0.0120 | 0.0157 |
| 4 | 0.0156 | 0.0209 |
| 5 | 0.0199 | 0.0262 |

Marginally better than noise, and not by much. No Bonferroni or FDR correction is applied — that is the point being demonstrated, not an oversight.

## Why it loses

The payoff geometry is adverse before any trade is placed. Entry at 2.0z, exit at 0.5z and stop at 4.0z means a win is worth 1.5 z-units and a loss costs 2.0. Break-even convergence rate is 2.0/3.5 = **57.1%**. Realised convergence is **30.7%**.

| Exit reason | Trades | Avg return (unit gross) |
|---|---|---|
| Converged | 79 | +2.57% |
| Stopped out | 99 | −2.32% |
| Block end / time | 79 | −0.18% |

Expectancy: 0.307 × (+2.57%) + 0.385 × (−2.32%) + 0.307 × (−0.18%) = **−0.16% per trade**. Median holding period is 14 days.

The mechanism is the half-life gap. Formation spreads revert with a 6.5-day half-life; out of sample that stretches to 31.9 days. With a 63-day maximum hold and a 32-day half-life, most positions structurally cannot reach 0.5z before the block ends, which is why 38.5% stop out and 30.7% expire.

## Annual returns

| Year | Strategy (net) | Strategy (gross) | SPY |
|---|---|---|---|
| 2012 | 0.00% | 0.00% | +14.17% |
| 2013 | −2.85% | −2.20% | +32.31% |
| 2014 | −1.57% | −0.88% | +13.46% |
| 2015 | +1.16% | +2.06% | +1.23% |
| 2016 | +2.44% | +3.45% | +12.00% |
| 2017 | −1.06% | −0.42% | +21.71% |
| 2018 | −4.30% | −3.35% | −4.57% |
| 2019 | −3.71% | −3.15% | +31.22% |
| 2020 | −3.16% | −2.22% | +18.33% |
| 2021 | −5.48% | −4.81% | +28.73% |
| 2022 | −3.91% | −3.43% | −18.18% |
| 2023 | +1.60% | +1.82% | +26.18% |
| 2024 | +1.60% | +2.32% | +24.89% |
| 2025 | +1.20% | +2.08% | +17.72% |
| 2026 | −0.42% | +0.27% | +14.11% |

2012 is the formation year — no positions taken. Nine losing years out of fourteen traded.

## Run

```bash
pip install yfinance pandas numpy statsmodels matplotlib
python strategy.py
```

Roughly 5 minutes on the default settings: 190 cointegration tests × 54 formation windows, plus 50 random-control draws at ~0.4s each.

```bash
python strategy.py --skip-controls        # main variant only, much faster
python strategy.py --cache                # reuse results/prices_cache.csv
python strategy.py --top-n 10 --entry-z 1.5
python strategy.py --random-seeds 100     # tighter percentile estimate
```

Key arguments: `--formation 252`, `--trading 63`, `--top-n 5`, `--entry-z 2.0`, `--exit-z 0.5`, `--stop-z 4.0`, `--pval 0.05`, `--cost-bps 10`, `--diag-window 252`, `--random-seeds 50`, `--cash-yield 0.0`.

## Files

| File | Contents |
|---|---|
| `strategy.py` | Full implementation |
| `results/metrics.csv` | Headline metrics, strategy and benchmark |
| `results/annual_returns.csv` | Year-by-year net, gross and benchmark |
| `results/worst_drawdowns.csv` | Five worst drawdown episodes |
| `results/backtest.csv` | Daily returns, costs, turnover, exposure, equity |
| `results/decomposition.csv` | Selection-rule variants side by side |
| `results/random_control.csv` | Full metrics for each of the 50 random draws |
| `results/random_control_summary.csv` | Random distribution and strategy percentile |
| `results/pair_selection.csv` | Pairs chosen per block, with β, p-value, half-life |
| `results/trades.csv` | Every round trip: entry/exit z, days held, exit reason |
| `results/signal_quality.csv` | Per-pair OOS stationarity and half-life diagnostics |
| `results/equity_curve.png` | Strategy vs buy-and-hold, log scale |
| `results/drawdown.png` | Underwater curves, both series |
| `results/selection_decomposition.png` | 50 random paths with both selection rules overlaid |
| `results/exposure_and_costs.png` | Active pairs and rolling 1-year cost drag |
| `results/spread_example.png` | One pair's z-score across a single out-of-sample block |

## Notes and limitations

**Idle capital earns nothing.** Average gross exposure is 24.83% and the book holds no position on 34.87% of days. A real market-neutral book earns the cash rate on the balance, which at 2012–2026 short rates would be a material addition. `--cash-yield` exists but defaults to 0 for consistency with earlier weeks. The reported return understates a live implementation on this axis.

**No short-borrow costs or shorting constraints.** Every pair assumes the short leg is freely borrowable at zero fee. For liquid sector ETFs this is close to true but not free, so the reported return overstates a live implementation on this axis.

**Execution is at daily closes with no slippage model.** The 10 bps per unit turnover is a flat assumption, not a modelled spread. Pairs trading is more spread-sensitive than the single-asset strategies in Weeks 1–8, because every entry and exit crosses two spreads rather than one.

**Survivorship and selection in the universe.** The 20 ETFs were chosen ex ante for economic linkage, but they were chosen in 2026 with knowledge of which ETFs still exist. All 20 traded continuously across the whole window, so no fund failures are captured.

**The random control is one universe, not one strategy.** All 50 draws pick from the same 20 tickers, so the control isolates the selection *rule*, not the universe. A different universe would shift the whole band.

**Block-end liquidation is a design choice with a cost.** Positions are closed every 63 days because β is re-estimated, which forces 79 trades to exit at block end for an average −0.18%. Carrying positions across re-estimation would avoid that but introduces ambiguity about which β the open position is held against.

**Three bugs were found and fixed before these results were generated**, all of which materially affected the first run: the hedge-ratio filter tested |β| and admitted negative β (22 selections where both legs went long, which is not a hedge); position sizing divided by pairs found rather than `top_n`, concentrating the full book into a single spread in three blocks; and `active_pairs` was inferred from non-zero weight columns, which undercounts when two pairs share a leg. The first run reported −31.46% and a −36.07% drawdown; roughly 14 points of that was defect rather than strategy. No thresholds were changed at any point.

**Validation.** Truncating the price history and re-running reproduces the earlier weight path to 0.00e+00, confirming no look-ahead. Gross exposure is verified never to exceed 1.0.
