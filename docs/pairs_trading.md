
# 📊 Advanced Mean Reversion Pairs Trading

A **market-neutral trading dashboard** built with Streamlit for analyzing and backtesting mean reversion pairs trading strategies.

---

## Overview

This application allows traders and analysts to:

- Identify **highly correlated stock pairs**.
- Evaluate **pair quality metrics** including correlation, cointegration, half-life, and Hurst exponent.
- Generate **mean reversion trading signals**.
- Backtest pairs trading strategies with **Z-score-based entry and exit rules**.
- Visualize **price action, ratio, Z-score, and cumulative returns** interactively.

---

## Features

### Sidebar Configuration

- **Stock Selection**: Choose from preset sectors or enter custom tickers.
- **Time Period**: Select years of historical data to download.
- **Strategy Parameters**:
  - Z-score thresholds for entry/exit.
  - Lookback window for rolling statistics.
  - Minimum correlation for pair selection.
- **Advanced Options**:
  - Stop loss percentage.
  - Half-life thresholds.
  - Option to use log price ratios.

### Data Handling

- Fetches price data from **Yahoo Finance** with caching.
- Automatically filters tickers with insufficient data.
- Adjusts for stock splits and dividends.

### Pair Quality Metrics

- **Correlation** of daily returns.
- **Mean Reversion Half-life** (Ornstein-Uhlenbeck).
- **Hurst Exponent** to check mean-reverting behavior.
- **Cointegration Test** (Engle-Granger).

### Trading Logic

- Calculate **ratio** (or log ratio) of stock pair prices.
- Rolling mean and standard deviation for Z-score calculation.
- **Entry signals**:
  - Z > entry threshold → Short first stock, Long second.
  - Z < -entry threshold → Long first stock, Short second.
- **Exit signals**:
  - Z crosses exit threshold → exit trade.
  - Stop-loss management.
- Tracks trades with timestamps, action type, and returns.

### Backtesting and Performance Metrics

- Calculates:
  - Total and average return
  - Standard deviation
  - Sharpe ratio
  - Win rate, average win/loss
  - Profit factor
  - Maximum drawdown

### Visualizations

1. **Normalized Price Chart**
2. **Price Ratio with Bollinger Bands**
3. **Z-Score with Entry/Exit Thresholds**
4. **Cumulative Returns**
5. **Correlation Heatmap**
6. **Top correlated pairs table**

---

## How to Run

1. Install dependencies:

```bash
pip install streamlit pandas numpy yfinance plotly scipy statsmodels
````

2. Save the script as `app.py`.

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Use the sidebar to configure tickers, dates, and strategy parameters.

---

## Educational Section

The app includes an interactive **learning section**:

* Core concept of pairs trading.
* Math behind ratio and Z-score.
* Example trade scenarios with Coke vs Pepsi.
* Risks and margin considerations.

---

## Notes

* Recommended at least **2 tickers** for analysis.
* Strategy assumes **highly correlated stocks**.
* Mean reversion characteristics are evaluated using **half-life** and **Hurst exponent**.
* Backtest results are **illustrative** and do not represent financial advice.

---
