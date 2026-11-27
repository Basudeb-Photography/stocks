# 🎯 Deep Dive Stock Predictor & P&L Simulator

An interactive Indian stock market (NSE/BSE) technical analysis, target validation, backtesting, and position-sizing dashboard built with Streamlit.

---

## 🚀 Key Features

1. **Universal NSE/BSE Search & URL Persistence**:
   * Choose from 45+ liquid Indian equities or enter **any** custom NSE/BSE ticker (e.g., `SUZLON`, `IREDA`, `TATAPOWER`, `BSE`, `500325.BO`).
   * Selected stock, timeframe (`1d`, `1h`, `15m`), Risk-Reward multiplier, investment capital, and margin type sync to URL query parameters so browser refreshes stay on your active stock (with no forced default).
   * Quick-pick chips and an on-demand **Live NSE Breakout Radar** scan stocks for active breakouts and volume surges.

2. **🎯 Target Touch Logic & Optimal R:R Advisor**:
   * Evaluates whether **Target 1** and **Target 2** are realistic before hitting Stop Loss by combining walk-forward historical hit rates, ATR swing-expansion envelopes, `ADX` trend strength, `RSI` extremes, and structural support/resistance barriers between CMP and your targets.
   * Compares your active Risk-Reward slider (`1.5x–3.5x`) against the stock's optimal historical expectancy R:R.

3. **⏳ Logical Holding Plan**:
   * Provides a tailored strategy mode (*Momentum Runner*, *Scale-Out 50% at T1 + Trail*, *Book Majority at T1*, or *Wait for Confirmation*), empirical median bars to T1/T2, max-hold time-stop rules, and `EMA 10` / `EMA 20` trailing rules.

4. **📜 Past Performances & Walk-Forward Strategy Backtest**:
   * Displays multi-horizon stock returns (`1W`, `1M`, `3M`, `6M`, `1Y` for `1d`), drawdown from period high, max drawdown, and annualized volatility.
   * Runs causal, non-overlapping signal backtests across `1.5x–3.5x` R:R multipliers comparing full-target vs. 50/50 scale-out expectancy and Buy & Hold returns.

5. **⚡ Breakout Signal Confirmation & Interactive Candlestick Chart**:
   * Audits 20/50-bar price triggers, relative volume surge, Bollinger squeeze coiling, and candle close quality, with CMP, SL, T1/T2 touch odds, and S/R zones plotted on Plotly candlesticks.

6. **💰 Capital Allocation & P&L Simulator + Catalyst News**:
   * Computes share quantity, margin outlay (`1x` Delivery or `5x` Intraday MIS), risk at SL, gains at T1/T2, and live Google News RSS sentiment.

---

## 🛠️ How to Run Locally

```powershell
.\.venv\Scripts\streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```
*(Or double-click `run_app.bat`)*

## 🧪 Run Automated Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests/test_app_and_engine.py
```
