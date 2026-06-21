# Week 3 — Day-of-Week (Calendar) Effect

Does the day of the week predict stock returns? This week tests the classic
"Monday effect" / "Friday effect" claim from the finance literature — not on
one ticker, but across five — and asks whether any effect that shows up is
big enough to survive transaction costs.

## The strategy

The calendar effect is one of the oldest documented "anomalies" in finance:

> Average returns are not uniform across the trading week. Older studies
> claim Mondays are unusually weak (the "Monday effect") and the day before
> the weekend is unusually strong.

This week tests that directly:

1. Pull daily returns for a basket of tickers.
2. Group returns by day of week (Monday–Friday) and compare means.
3. Run a one-way ANOVA across the five groups, plus a t-test on the
   best-vs-worst day, to check whether any spread is statistically real
   or just noise.
4. If a "best day" exists in-sample, build the simplest possible rule
   around it — long only on that day, flat otherwise — and backtest it
   against buy-and-hold, with costs included.

The point isn't to find a profitable trick. It's to test a textbook claim
honestly: is the effect real, is it consistent across assets, and does it
survive the cost of acting on it.

## Methodology

Same discipline as Weeks 1–2, applied to a statistical question instead of
a technical indicator:

- **Signal timing.** The day-of-week label for day *t* is known in advance
  (it's a calendar fact, not a lagging indicator), but the position is still
  shifted one day before being multiplied by returns, so the backtest never
  uses day *t*'s return to decide day *t*'s position contamination from
  same-day execution assumptions.
- **Transaction costs.** 5 bps per trade, same as Week 1 and Week 2.
- **Statistical significance, not just point estimates.** A "best day" with
  a 0.3% higher mean return than the worst day means nothing if a one-way
  ANOVA across all five days returns a p-value of 0.6. This week reports the
  ANOVA F-stat and p-value alongside the raw means, and flags whether the
  result clears p < 0.05.
- **Multiple tickers, tested independently.** A calendar effect that only
  shows up in one ticker is more likely sampling noise than a real anomaly.
  Testing SPY, QQQ, AAPL, MSFT, and GLD together checks whether any pattern
  recurs across assets or is ticker-specific.
- **No parameter optimization.** The "best day" is whatever the data says
  in-sample — there's no search over thresholds or lookback windows.
- **Benchmark comparison is mandatory.** Strategy returns are reported
  against buy-and-hold on the same ticker, not in isolation.
- **Limitations stated explicitly** — see below.

## Results

> **⚠️ Numbers below are placeholders.** This sandbox's network egress
> doesn't allow outbound calls to Yahoo Finance, so I verified the pipeline
> end-to-end against synthetic price data (confirmed: data loads, the ANOVA
> test runs, the backtest produces sane equity curves, metrics compute
> without errors) but didn't generate real results. Run
> `python strategy.py` locally and replace this table and the two PNGs per
> ticker with the actual output.

| Ticker | Best day (in-sample) | ANOVA p-value | Significant? | Strategy total return | B&H total return | Strategy Sharpe | B&H Sharpe |
|--------|----------------------|----------------|---------------|------------------------|-------------------|------------------|------------|
| SPY    | TBD                  | TBD            | TBD           | TBD                    | TBD                | TBD              | TBD        |
| QQQ    | TBD                  | TBD            | TBD           | TBD                    | TBD                | TBD              | TBD        |
| AAPL   | TBD                  | TBD            | TBD           | TBD                    | TBD                | TBD              | TBD        |
| MSFT   | TBD                  | TBD            | TBD           | TBD                    | TBD                | TBD              | TBD        |
| GLD    | TBD                  | TBD            | TBD           | TBD                    | TBD                | TBD              | TBD        |

> **[Insert per-ticker day-of-week bar chart and equity curve here]**

## What I expect, going in

Worth writing down a prediction *before* running it, so the writeup isn't
just retrofitted to whatever number comes out:

- I expect the ANOVA to come back **not significant** for most or all of
  these five tickers over a 2015–2024 window. The Monday effect was
  documented mainly in pre-1990s US equity data; multiple studies since
  have found it's weakened or disappeared, plausibly because once an
  anomaly is published, capital flows in to arbitrage it away.
- If something *does* come back significant, I'd treat that with more
  suspicion than excitement — five tickers tested independently means a
  roughly 1-in-4 chance that at least one clears p < 0.05 purely by chance,
  even if there's no real effect anywhere (multiple-comparisons problem).
  A single significant ticker out of five is weaker evidence than it looks.
- Even in a world where the ANOVA is significant, the long-only-one-day
  rule only trades ~20% of the time (1 of 5 weekdays), so costs are low,
  but so is the return ceiling — this isn't a strategy designed to beat
  buy-and-hold on raw return, it's designed to test whether the underlying
  statistical claim is real.

*(This section gets replaced with what actually happened once the script
is run with real data — keeping the prediction visible either way, since a
miss is as informative as a hit.)*

## Limitations

- **Multiple comparisons.** Testing 5 tickers × 1 ANOVA each raises the
  chance of a false positive somewhere in the table. A Bonferroni-style
  correction (effectively requiring p < 0.01 instead of p < 0.05 to call
  something significant) would be the more rigorous bar; not applied here,
  flagged instead.
- **In-sample day selection.** The "best day" is chosen and traded on the
  same data — there's no train/test split. A real test of whether this is
  tradeable would fit the best day on the first half of the window and
  check whether it still holds in the second half. Worth doing as a
  follow-up if any ticker's ANOVA comes back significant.
- **Survivorship and adjustment.** Prices are split/dividend-adjusted via
  `yfinance`'s `auto_adjust=True`, which handles the obvious data issue but
  doesn't address survivorship bias for the equity tickers (AAPL, MSFT)
  over a 10-year window — both happen to still exist and be index
  constituents today, which is itself a selection effect.
- **US market hours / holidays.** Day-of-week labels come from the
  Yahoo Finance trading calendar; short weeks around holidays aren't
  treated specially, which slightly thins some Monday/Friday samples.

## Repository layout

```
week-03-calendar-effects/
├── README.md                      ← this file
├── strategy.py                    ← data, stats test, backtest, plots
├── {TICKER}_dow_summary.csv       ← per-ticker mean/std/n by day of week
├── {TICKER}_dow_means.png         ← bar chart of annualized mean return by day
├── {TICKER}_backtest.csv          ← full daily backtest output
├── {TICKER}_equity_curve.png      ← strategy vs. buy-and-hold
└── overview_all_tickers.csv       ← one row per ticker, the summary table above
```

No `notebook.ipynb` and no separate `results/` folder this week — consistent
with the script-only, flat-output convention adopted from Week 2 onward.

## Running this week

```bash
cd week-03-calendar-effects
python strategy.py
```

Optional flags to test a different basket or date range:

```bash
python strategy.py --tickers SPY IWM EFA --start 2010-01-01 --end 2024-12-31
```

Price data is fetched on demand via `yfinance` — nothing large is committed
to the repo.

## Tech stack

Same as the rest of the series, with `scipy` actually put to use for the
first time (the ANOVA and t-tests):

- **pandas, numpy** — data manipulation
- **yfinance** — price data
- **matplotlib** — bar charts and equity curves
- **scipy** — `f_oneway` (ANOVA) and `ttest_ind` for significance testing

---

*Part of the [Quant Strategies](../README.md) series — one backtest a week,
honest results either way.*
