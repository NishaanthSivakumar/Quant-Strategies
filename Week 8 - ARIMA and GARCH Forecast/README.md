# Week 8 — ARIMA + GARCH Forecast Strategy

**Result: lost badly to buy-and-hold — but only half the model failed.** The GARCH(1,1) volatility forecast genuinely beat a naive 21-day benchmark, and vol-target sizing on its own would have beaten buy-and-hold on a risk-adjusted basis. The ARIMA mean forecast was worse than useless, and its churn generated 97.5% in cumulative trading costs, turning a +445% gross return into +109% net.

SPY, 2008-12-22 → 2026-07-30 (4,427 out-of-sample days), 10 bps per unit of turnover.

## Metrics

| Metric | ARIMA+GARCH (combo) | Buy & Hold SPY |
|---|---|---|
| Total return | 109.07% | **1052.99%** |
| CAGR | 4.29% | **14.93%** |
| Ann. volatility | **11.24%** | 17.79% |
| Sharpe | 0.430 | **0.872** |
| Sortino | 0.523 | **1.084** |
| Max drawdown | **-19.87%** | -33.72% |
| Calmar | 0.216 | **0.443** |
| Win rate | 40.73% | 55.39% |
| Time in market | 74.7% | 100% |
| Avg exposure | 0.678 | 1.000 |
| Switches | 1,109 | 0 |
| Total turnover | 975.0 | 0 |
| Total cost | 97.50% | 0 |

## Forecast quality

Scored independently of P&L, because a strategy can lose money with a good model and make money with a bad one.

| Mean model (ARIMA) | Value |
|---|---|
| Out-of-sample R² vs zero-forecast | **-0.00991** |
| Directional accuracy | 0.5267 |
| Naive always-up base rate | **0.5540** |
| Days predicted up | 74.76% |

| Volatility model (GARCH) | GARCH(1,1) | Naive 21-day |
|---|---|---|
| QLIKE (lower is better) | **1.567** | 1.732 |
| MSE vs squared returns | **1.63e-07** | 1.81e-07 |
| Correlation with abs. return | 0.4638 | — |
| Mean forecast ann. vol | 15.88% | — |
| Realised ann. vol | 17.79% | — |

Mean GARCH persistence (α+β) across 211 refits: 0.9652. All 211 converged.

## Where the return went

Decomposing the combo strategy isolates which half of the model is responsible for what:

| Variant | Sharpe | Total return |
|---|---|---|
| Vol-target sizing only, no costs | **1.005** | 735.8% |
| Buy & Hold | 0.867 | 1034.0% |
| ARIMA gate only, no costs | 0.812 | 633.7% |
| Combo, gross of costs | 0.917 | 445.5% |
| Combo, net of costs | 0.422 | 105.7% |

Vol targeting beats buy-and-hold on Sharpe. The direction gate loses to it. And the gate flips on 25.0% of days, which is what produces the turnover — costs alone cost 0.495 of Sharpe.

## Worst drawdowns (strategy)

| Start | Trough | End | Depth | Length |
|---|---|---|---|---|
| 2021-08-17 | 2022-07-14 | 2024-11-29 | -19.87% | 1,200d |
| 2014-12-08 | 2016-02-11 | 2017-01-05 | -18.59% | 759d |
| 2011-03-14 | 2012-06-04 | 2013-10-24 | -18.06% | 955d |
| 2018-09-21 | 2018-12-24 | 2019-07-02 | -16.42% | 284d |
| 2020-02-20 | 2020-03-23 | 2020-08-31 | -15.88% | 193d |

## Hypothesis

The classical time-series toolkit splits a return series into a conditional mean and a conditional variance. The literature is blunt about which half is forecastable, so the strategy was built to test both separately:

- **ARIMA (mean).** Daily equity returns are close to a martingale. An ARIMA mean forecast should be near-useless — but "should be" is not evidence, so it gets measured with out-of-sample R² and directional accuracy against the base rate.
- **GARCH (variance).** Volatility clusters, and GARCH(1,1) is a genuinely good one-step-ahead variance forecaster. If there is an edge here, it should show up in **sizing**, not **direction**.

## Method

1. Fetch daily SPY closes, compute log returns.
2. Walk forward: every 21 days, refit ARIMA(1,0,1) on the trailing 1,000 days, then fit GARCH(1,1) to that ARIMA's residuals.
3. Freeze the parameters and filter them forward across the block to produce one-step-ahead forecasts of the mean (μ) and volatility (σ) for each day. Filtering with fixed parameters uses only past observations.
4. Map (μ, σ) onto a portfolio weight. Three modes:
   - `mean` — direction only: long if μ > threshold, else cash.
   - `voltarget` — size only: weight = target_vol / forecast_vol, capped.
   - `combo` (default) — vol-target size, gated by the ARIMA direction.
5. Charge 10 bps on turnover, benchmark against buy-and-hold over the identical window.

### Guarding against look-ahead

- Parameters are estimated only on data ending at the **start** of each block, then frozen. The model never re-estimates using the days it is used to trade.
- The backtest applies `position.shift(1) * ret`: a weight chosen at the close of day *t* earns day *t+1*'s return.
- The first 1,000 days are burned on the initial fit and excluded from both the strategy and the benchmark.
- Validated offline on synthetic data with a **truncation-invariance test**: recomputing forecasts on a series cut 300 days short produces bit-identical overlapping forecasts. On i.i.d. simulated returns, directional accuracy landed at 0.507.

## Run

```bash
python strategy.py                                     # defaults: SPY, combo mode
python strategy.py --mode voltarget --target-vol 0.15  # sizing only
python strategy.py --mode mean                         # direction only
python strategy.py --ticker QQQ --p 2 --q 2 --window 750 --refit-every 42
```

Requires `statsmodels`, `arch`, `yfinance`, `pandas`, `numpy`, `matplotlib`. Runtime ~45–60s at defaults (211 walk-forward refits over 20 years).

Key flags: `--window` (fit window), `--refit-every` (refit cadence), `--mode`, `--threshold` (entry threshold in bps), `--target-vol`, `--max-weight` (exposure cap, default 1.0 = no leverage), `--rebal-band` (default 0.05), `--cost-bps` (default 10).

## Files

| File | Contents |
|---|---|
| `strategy.py` | Full implementation |
| `results/metrics.csv` | Headline metrics, strategy vs benchmark |
| `results/forecast_diagnostics.csv` | Mean and vol forecast scoring |
| `results/annual_returns.csv` | Year-by-year returns and average exposure |
| `results/worst_drawdowns.csv` | Five deepest drawdown episodes |
| `results/fit_log.csv` | Per-refit ARIMA AIC and GARCH parameters |
| `results/backtest.csv` | Daily forecasts, weights, turnover, returns |
| `results/equity_curve.png` | Equity curves with drawdown panel |
| `results/volatility_forecast.png` | GARCH forecast vs 21-day realised vol |
| `results/exposure.png` | Position size through time |
| `results/forecast_quality.png` | Mean-forecast scatter, GARCH persistence by refit |

## Notes

- **The ARIMA mean forecast has negative out-of-sample R².** A constant zero forecast would have been more accurate. Directional accuracy of 0.5267 also sits below the 0.5540 always-up base rate, so the second week running, a model beat a coin flip and still lost to the naive benchmark.
- **The forecast is biased long.** The model predicted up on 74.8% of days against a 55.4% base rate — the ARIMA constant is picking up the equity drift, which is real but not tradeable information.
- **The GARCH forecast is legitimately good.** It beats the 21-day trailing benchmark on both QLIKE and MSE, correlates 0.46 with absolute returns, and tracks realised vol closely through 2020 and 2022. It also runs low: 15.88% mean forecast against 17.79% realised, because the vol clip and the one-step horizon both smooth the tails.
- **Costs are the mechanism of failure, but not the cause.** Gross Sharpe was 0.917, roughly buy-and-hold. Net was 0.422. The 975 units of turnover come almost entirely from the direction gate flipping on a quarter of all days — the useless half of the model is the half generating the bill.
- **Risk control worked.** Volatility 11.24% vs 17.79%, max drawdown -19.87% vs -33.72%. The strategy sidestepped the worst of March 2020 (average exposure 0.462 across 2020) and beat the benchmark in 2018 and 2022, the only two down years in the sample. It just gave up far too much upside everywhere else.
- **The honest conclusion:** run `--mode voltarget` and drop the ARIMA gate entirely. Sizing on a GARCH forecast produced Sharpe 1.005 gross against buy-and-hold's 0.867, on a fraction of the turnover. This week's headline number is bad because the strategy insisted on using a forecast that doesn't work.
- **Caveat on the vol-target variant.** The 1.005 Sharpe figure above is gross of costs and uses no rebalance band. It is a diagnostic decomposition, not a backtested strategy — a proper run of `--mode voltarget` with costs is the obvious next test, not a result already claimed.
- A diagnostics bug was found and fixed after the first run: twelve days in the sample closed exactly flat, making the squared-return proxy zero and sending QLIKE to `inf` for both models. The fix filters to strictly positive proxy values. Backtest results were unaffected.
