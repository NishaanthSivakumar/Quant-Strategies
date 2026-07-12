# Quant Strategies — 10 Weeks, 10 Backtests (Summer 2026)

A weekly series of quantitative trading strategies, implemented from scratch in Python, with honest write-ups of what worked and what didn't.

Built as preparation for quant trading internships. The goal isn't to find alpha — it's to demonstrate clean implementation, rigorous testing, and critical analysis of results, including the losers.

---

## The series

| #  | Strategy                         | Asset(s)      | Result                              |
|----|----------------------------------|---------------|-------------------------------------|
| 01 | Moving Average Crossover (10/50) | SPY           | Lost vs B&H                         |
| 02 | RSI Mean Reversion               | AAPL          | High win rate, lost vs B&H          |
| 03 | Day-of-Week (Calendar) Effect    | 5 tickers     | Null; lost B&H                      |
| 04 | Dual Momentum (cross-asset)      | SPY, VEU      | Won risk-adj.                       |
| 05 | Post-Earnings Announcement Drift | 15 mega-caps  | Null / reversal; lost benchmark     |
| 06 | Volatility Risk Premium          | SPY, VIX      | Won risk-adj.; tail risk intact     |
| 07 | TBD                              | —             | —                                   |
| 08 | TBD                              | —             | —                                   |
| 09 | TBD                              | —             | —                                   |
| 10 | TBD                              | —             | —                                   |

*"Result" reports honest outcomes vs. a sensible benchmark — not whether I'd trade it.*

Week 3's day-of-week effect showed no statistically significant return difference on any ticker (p > 0.05). Week 5's PEAD — one of the most replicated anomalies in finance — failed to appear in mega-caps and reversed in the tails, consistent with the anomaly being arbitraged away in the most efficient universe. Week 6's VRP strategy won on risk-adjusted terms (Sharpe 0.683 vs 0.543, max DD −45.9% vs −55.2%) but the backward-looking signal couldn't dodge the worst of the GFC — the premium is real, the protection is late.

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

- **Transaction costs are always included.** Weeks 1–3 use 5 bps per trade. Week 4 onward uses 10 bps per switch/event — a higher bar, reflecting the more realistic cost model as strategies got more complex. A "free" backtest is fiction.
- **Benchmark comparison is mandatory.** Strategy returns are reported against buy-and-hold on the same asset(s), not in isolation. Sharpe ratio, max drawdown, and Calmar are reported alongside total return.
- **No parameter optimisation by default.** Textbook defaults are used unless robustness across windows is explicitly tested.
- **Out-of-sample where possible.** When optimisation is done, results are reported on a held-out test period.
- **Limitations are stated explicitly.** Survivorship bias, look-ahead, data quality, and sample-window bias — flagged, not hidden.

If a strategy loses, the writeup explains *why* — late entries, whipsaw losses, regime dependence, cost sensitivity, or (Week 5) testing an anomaly in the universe where it's most arbitraged away. A clean negative result with a clear diagnosis is more useful than an overfit positive one. And when a strategy "wins," the writeup is just as explicit about *where* the edge came from (Week 4 won on risk reduction, not return; Week 6 won on risk-adjusted terms but still had a −45.9% drawdown because the signal is backward-looking).

---

## Tech stack

- **Python 3.11+**
- **pandas, numpy** — data manipulation
- **yfinance** — price, earnings, and volatility data
- **matplotlib** — equity curves and diagnostics
- **statsmodels, scipy** — statistical tests where relevant

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Running a week

```bash
git clone https://github.com/<NishaanthSivakumar>/quant-strategies.git
cd quant-strategies
pip install -r requirements.txt
cd week-06-vrp
python strategy.py
```

Price and volatility data are fetched on demand via `yfinance`, so no large datasets are committed to the repo.

---

## About

I'm building toward quant trading internships and publishing one strategy a week — code, write-up, and honest results either way. Feedback from practitioners is genuinely welcome; if you spot a methodological error, please open an issue or reach out.

---

*Last updated: Weeks 1–6 done. Series in progress.*