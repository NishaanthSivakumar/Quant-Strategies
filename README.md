# Quant Strategies — 10 Weeks, 10 Backtests (Summer 2026)

A weekly series of quantitative trading strategies, implemented from scratch in Python, with honest write-ups of what worked and what didn't.

Built as preparation for quant trading internships. The goal isn't to find alpha — it's to demonstrate clean implementation, rigorous testing, and critical analysis of results, including the losers.

---

## The series

| #  | Strategy                        | Asset(s) | Result          |
|----|---------------------------------|----------|-----------------|
| 01 | Moving Average Crossover (10/50)| SPY      | Lost vs B&H     |
| 02 | RSI Mean Reversion              | TBD      | —               |
| 03 | TBD                             | —        | —               |
| 04 | TBD                             | —        | —               |
| 05 | TBD                             | —        | —               |
| 06 | TBD                             | —        | —               |
| 07 | TBD                             | —        | —               |
| 08 | TBD                             | —        | —               |
| 09 | TBD                             | —        | —               |
| 10 | TBD                             | —        | —               |

*"Result" reports honest outcomes vs. a sensible benchmark — not whether I'd trade it.*

---

## Repository layout

```
quant-strategies/
├── README.md              ← you are here
├── requirements.txt       ← shared dependencies
├── week-01-ma-crossover/  ← one folder per strategy
│   ├── README.md          ← writeup
│   ├── notebook.ipynb     ← exploration
│   ├── strategy.py        ← clean implementation
│   └── results/           ← equity curve, trades, metrics
└── ...
```

Each week's folder is self-contained — clone the repo, install requirements, and any week's notebook should run end-to-end.

---

## Methodology

Every backtest in this repo follows the same discipline:

- **Transaction costs are always included** (default: 5 bps per trade). A "free" backtest is fiction.
- **Benchmark comparison is mandatory.** Strategy returns are reported against buy-and-hold on the same asset, not in isolation.
- **Risk-adjusted metrics over raw returns.** Sharpe ratio, max drawdown, and Calmar are reported alongside total return.
- **No parameter optimisation by default.** Textbook defaults are used unless robustness across windows is explicitly tested.
- **Out-of-sample where possible.** When optimisation is done, results are reported on a held-out test period.
- **Limitations are stated explicitly.** Survivorship bias, look-ahead, data quality — flagged, not hidden.

If a strategy loses, the writeup explains *why* — late entries, whipsaw losses, regime dependence, cost sensitivity. A clean negative result with a clear diagnosis is more useful than an overfit positive one.

---

## Tech stack

- **Python 3.11+**
- **pandas, numpy** — data manipulation
- **yfinance** — price data
- **matplotlib** — equity curves and diagnostics
- **statsmodels, scipy** — statistical tests where relevant
- **Jupyter** — exploration

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Running a week

```bash
git clone https://github.com/<your-username>/quant-strategies.git
cd quant-strategies
pip install -r requirements.txt
```

Price data is fetched on demand via `yfinance`, so no large datasets are committed to the repo.

---

## About

I'm building toward quant trading internships and publishing one strategy a week — code, write-up, and honest results either way. Feedback from practitioners is genuinely welcome; if you spot a methodological error, please open an issue or reach out.

---

*Last updated: Week 1 complete. Series in progress.*

