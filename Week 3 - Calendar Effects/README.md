# Week 3 — Day-of-Week (Calendar) Effect

Does the day of the week predict stock returns? Tests the classic Monday/Friday effect claim across five tickers and asks whether anything found is real enough to trade.

## The strategy

The calendar effect is one of the oldest documented "anomalies" in finance — older studies claim Mondays are unusually weak and Fridays unusually strong.

This week tests it directly:

1. Pull daily returns for SPY, QQQ, AAPL, MSFT, GLD (2015–2024).
2. Group returns by day of week and compare means.
3. Run a one-way ANOVA + best-vs-worst t-test to check whether any spread is statistically real.
4. Build the simplest possible rule (long on the best day only, flat otherwise), backtest with costs.

## Methodology

- 5 bps per trade transaction cost, same as Weeks 1–2.
- ANOVA p-value reported alongside raw means — point estimates without a significance test are meaningless.
- Five tickers tested independently — a calendar effect that only appears in one is more likely noise than a real anomaly.
- No parameter optimization. "Best day" is whatever the data says in-sample.
- Benchmark: buy-and-hold the same ticker over the same window.

## Results

| Ticker | Best day  | ANOVA p-value | Significant? | Strategy total return | B&H total return | Strategy Sharpe | B&H Sharpe |
|--------|-----------|---------------|---------------|------------------------|-------------------|------------------|------------|
| SPY    | Wednesday | 0.99          | No            | -16.5%                 | +240.8%           | -0.20            | 0.79       |
| QQQ    | Wednesday | 0.77          | No            | +2.3%                  | +441.2%           | 0.07             | 0.89       |
| AAPL   | Monday    | 0.17          | No            | +69.1%                 | +935.9%           | 0.47             | 0.97       |
| MSFT   | Wednesday | 0.78          | No            | +43.2%                 | +958.0%           | 0.35             | 1.01       |
| GLD    | Friday    | 0.72          | No            | -18.9%                 | +110.9%           | -0.28            | 0.60       |

> **[Insert per-ticker day-of-week bar charts and equity curves here]**

**No ticker showed a statistically significant day-of-week effect.** AAPL had the lowest p-value at 0.17 — still well above the 0.05 threshold. The "best day" varies across the basket (Wednesday for three tickers, Monday for one, Friday for one), which is the pattern you'd expect from sampling noise, not a real anomaly.

The strategy underperformed buy-and-hold on every ticker, in both total return and Sharpe. Three of five tickers (QQQ, AAPL, MSFT) produced positive strategy returns in absolute terms, but the opportunity cost of sitting out 80% of trading days crushed them all — AAPL's +69.1% shrinks next to buy-and-hold's +935.9%. That's the arithmetically expected outcome of a 20%-time-in-market rule with no statistically significant edge.

The Monday effect that 1970s finance papers found in US equities doesn't show up in 2015–2024 data — at least not on this basket and not at p < 0.05. Detailed analysis in the Medium write-up.

## Limitations

- **Multiple comparisons.** 5 ANOVAs inflates the chance of a false positive. Moot here since nothing was significant, but worth flagging.
- **In-sample day selection.** "Best day" is chosen and traded on the same data. A train/test split would be the more rigorous test.
- **Basket isn't independent.** Four of five tickers are US equities; SPY/QQQ overlap heavily with AAPL/MSFT.

## Repository layout

```
week-03-calendar-effects/
├── README.md
├── strategy.py
├── {TICKER}_dow_summary.csv      ← mean/std/n return by day of week
├── {TICKER}_dow_means.png        ← bar chart of annualized mean by day
├── {TICKER}_backtest.csv         ← full daily backtest output
├── {TICKER}_equity_curve.png     ← strategy vs. buy-and-hold
└── overview_all_tickers.csv      ← the summary table above
```

## Running this week

```bash
cd week-03-calendar-effects
python strategy.py
```

Optional flags to test a different basket or window:

```bash
python strategy.py --tickers SPY IWM EFA --start 2010-01-01 --end 2024-12-31
```

---

*Part of the [Quant Strategies](../README.md) series — one backtest a week, honest results either way.*