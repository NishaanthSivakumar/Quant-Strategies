# Moving Average Crossover — SPY Backtest

A clean, honest backtest of one of the most famous strategies in trading: the **10/50 moving average crossover** on the SPY ETF, 2019–2024.

> **TL;DR** — The strategy underperformed buy-and-hold by 92 percentage points (67% vs 158% total return). More interesting: it also underperformed a naive "constant 71% SPY / 29% cash" portfolio by ~30 percentage points, which means the timing signal is *actively destroying value*, not just reducing exposure. Its only genuine achievement: cutting max drawdown roughly in half (-15% vs -34%). An expensive way to buy downside protection.

---

## Hypothesis

> When the 10-day moving average crosses **above** the 50-day moving average, the asset is in an uptrend — go long. When it crosses back below, exit to cash.

This is a textbook trend-following rule. The intuition is sound: in a real uptrend, short-term price action should lead the longer-term average. The empirical question is whether the signal is strong enough — and clean enough — to make money after costs.

## Methodology

| Choice | Value | Why |
|---|---|---|
| Asset | SPY | Liquid, well-known, transaction costs are realistic |
| Period | 2019-01-01 to 2024-12-31 | 6 years: COVID, 2022 drawdown, 2023–24 AI rally |
| Signal | 10-day MA > 50-day MA → long, else flat | Classic short/long pair |
| Execution | Signal at close of t, traded at t+1 | No look-ahead bias |
| Costs | 5 bps per trade | Conservative retail estimate |
| Position sizing | All-in or all-out | Keeps the test about the signal, not sizing |

## How to run

```bash
pip install yfinance pandas numpy matplotlib
python ma_crossover_backtest.py
```

The script prints a metrics table and saves `ma_crossover_results.png`.

## Results

| Metric | Buy & Hold | MA Crossover |
|---|---|---|
| Total Return | 158.4% | 66.7% |
| CAGR | 17.2% | 8.9% |
| Annualised Volatility | 19.8% | 12.3% |
| Sharpe Ratio | 0.90 | 0.76 |
| Max Drawdown | -33.7% | -15.1% |
| Time in market | 100% | 71.2% |
| Number of trades | — | 33 |

## Analysis — why it underperformed

**1. Missing 29% of an asset compounding at 17%/year is the dominant cost.**
The strategy was out of the market about 29% of the time. On an index that compounded at 17.2% per year, that's the single biggest source of underperformance — far larger than transaction costs or whipsaws. Trend-following earns its keep only when the asset has genuine downtrends worth timing. SPY in 2019–2024 had two sharp drawdowns and one major one (COVID), but the rest of the time it was rising.

**2. The signal is worse than no signal at the same exposure level.**
This is the sharpest finding from the test. The strategy is in the market 71% of the time. A naive portfolio that is simply 71% SPY and 29% cash (daily-rebalanced) would have returned approximately **96%** over the same period. The MA strategy returned **67%**. **The timing signal isn't just reducing exposure — it's actively destroying ~30 percentage points of return relative to the same exposure held constantly.**

The mechanism: the 50-day moving average is a lagging indicator by construction. A 50-day MA is, by definition, the mean of the last 50 days of prices. By the time it confirms a trend reversal, ~25 days of the reversal have already happened. The COVID exit in March 2020 happened near the bottom; the 2020 re-entry was well into the recovery. Both ends of every major move got chopped off.

**3. Transaction costs are NOT the main story here.**
33 trades over 6 years at 5 bps per trade ≈ 165 basis points of total drag, or about 1.65 percentage points of total return. That's meaningful but tiny against the 92-point gap. Most beginner write-ups blame trend-following failure on costs and whipsaws. The real culprit is structural: timing a strongly uptrending asset is a losing proposition no matter how cheap your costs are.

**4. Drawdown reduction is real — but it has a cheaper competitor.**
The strategy did cut max drawdown from -33.7% to -15.1%, more than half. That's a genuine outcome. But you could achieve similar drawdown reduction more cheaply by simply holding less SPY. The MA strategy buys downside protection at a price of ~30 percentage points of return relative to constant partial exposure — a poor trade for almost any investor.

**5. The Sharpe gap confirms the verdict.**
The strategy's lower volatility (12.3% vs 19.8%) makes the return underperformance look "less bad" on an absolute basis. But the Sharpe ratio (0.76 vs 0.90) shows that the volatility reduction didn't compensate for the lost return. On a risk-adjusted basis, the strategy is just a worse buy-and-hold.

## What this teaches about strategy design

**Average exposure isn't enough — timing has to add value over and above it.** The right benchmark for *any* timing strategy isn't only buy-and-hold; it's also a constant-exposure portfolio matched to the strategy's average exposure. If the signal can't beat that, it's destroying value rather than adding it.

**Sharpe captures what total return hides.** Lower returns can look acceptable if they come with lower risk. The Sharpe ratio is the honest tally. Always look at it, not just absolute return.

**Trend-following needs an asset that trends both ways.** Persistent upward drifters are the worst possible domain. Better candidates: commodities, currencies, single stocks with idiosyncratic momentum, or assets with bigger regime shifts.

**Don't blame what's small when something larger is in plain sight.** Costs mattered here (~1.65pp), but they were a sideshow. The dominant problem was the signal itself.

## Limitations

- **One asset, one period.** Results don't generalise.
- **No parameter optimisation.** I picked 10 and 50 because they're the textbook defaults. Deliberately didn't search for the "best" window pair, which would have produced a better-looking but overfit result.
- **Daily close data only.** Ignores intraday slippage and bid-ask spread; the 5-bp cost assumption tries to bundle these in.
- **Long-only.** No short on the bearish signal, which would change the picture in 2022.

## Next steps

- Add a constant-exposure benchmark (e.g., 70/30 SPY/cash) directly into the code so the comparison is automated
- Test the strategy on a more volatile, less persistently-trending asset (oil, gold, BTC) to see if the signal is picking up trend information or just generating noise
- Test different window pairs (5/20, 20/100, 50/200) and report all of them honestly — not only the best

---

**Built as the first project in my path toward quant trading internships.** The point isn't to find an alpha source — it's to demonstrate I can implement a strategy cleanly, test it without fooling myself, and analyse the results critically.