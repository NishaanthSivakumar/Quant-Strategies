# Quant Strategies — 10 Weeks, 10 Backtests (Summer 2026)

A weekly series of quantitative trading strategies, implemented from scratch in Python, with honest write-ups of what worked and what didn't.

Built as preparation for quant trading internships. The goal isn't to find alpha — it's to demonstrate clean implementation, rigorous testing, and critical analysis of results, including the losers.

---

## The series

| #  | Strategy                          | Asset(s)     | Result                    |
|----|-----------------------------------|--------------|---------------------------|
| 01 | Moving Average Crossover (10/50)  | SPY          | Lost vs B&H               |
| 02 | RSI Mean Reversion                | AAPL         | High win rate, lost vs B&H|
| 03 | Day-of-Week (Calendar) Effect     | 5 tickers    | Null; lost B&H            |
| 04 | Dual Momentum (cross-asset)       | SPY, VEU     | Won risk-adj.             |
| 05 | Post-Earnings Announcement Drift  | 15 mega-caps | Null; lost bench.         |
| 06 | Volatility Risk Premium           | SPY, VIX     | Won risk-adj.             |
| 07 | ML Return Prediction (Random Forest) | SPY       | Lost vs B&H               |
| 08 | TBD                               | —            | —                         |
| 09 | TBD                               | —            | —                         |
| 10 | TBD                               | —            | —                         |

*"Result" reports honest outcomes vs. a sensible benchmark — not whether I'd trade it. Week 07's Random Forest cleared a coin flip (51.7% out-of-sample directional accuracy) but fell short of the naive always-long base rate (54.8%) and lost to buy-and-hold on every risk-adjusted measure — the honest "ML doesn't time daily index direction" result.*

---

## Repository layout

```
quant-strategies/
├── README.md              ← you are here
├── requirements.txt       ← shared dependencies
├── week-01-ma-crossover/  ← one folder per strategy
│   ├── README.md          ← writeup
│   ├── strategy.py        ← clean implementation
│   └── results/           ← equity curve, trades, metrics
└── ...
```

Each week's folder is self-contained — clone the repo, install requirements, and run `python strategy.py` from inside the folder.

---

## Methodology

Every backtest in this repo follows the same discipline:

- **Transaction costs are always included.** A "free" backtest is fiction. Earlier weeks use 5 bps per trade; from Week 04 onward, costs are charged at 10 bps one-way and only when the position actually changes, which is the more honest model for a rotation strategy.
- **Benchmark comparison is mandatory.** Strategy returns are reported against buy-and-hold on the same asset — or, for multi-asset strategies, against an equal-weight, monthly-rebalanced basket of the same assets — not in isolation.
- **Risk-adjusted metrics over raw returns.** Sharpe ratio, max drawdown, and Calmar are reported alongside total return.
- **No parameter optimisation by default.** Textbook defaults are used unless robustness across windows is explicitly tested.
- **Out-of-sample where possible.** For any strategy that learns from data (Week 07 onward), the train/test split is strictly chronological — the model is fit on an early window and evaluated only on a held-out later window, so results never reflect look-ahead.
- **Limitations are stated explicitly.** Survivorship bias, look-ahead, data quality, and sample-window bias — flagged, not hidden.

If a strategy loses, the writeup explains *why* — late entries, whipsaw losses, regime dependence, cost sensitivity, or an edge too thin to survive transaction costs.

---

## Tech stack

Python · pandas · numpy · yfinance · matplotlib · scipy · statsmodels · scikit-learn (Week 07+)

---

## About

I'm building toward quant trading internships and publishing one strategy a week — code, write-up, and honest results either way. Feedback from practitioners is genuinely welcome; if you spot a methodological error, please open an issue or reach out.

---

*Last updated: Week 7 complete. Series in progress.*