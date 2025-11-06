# Momentum Trading Dashboard

A **Streamlit app** to visualize and analyze momentum trading indicators, providing live buy/sell signals, charts, and a small back-test summary.

---

## 🚀 Overview

**App Title:** Momentum Trading – From Theory to Signals  
**Purpose:** Learn momentum trading, visualize key indicators, and get actionable signals.

- Fetches historical stock data from Yahoo Finance
- Calculates momentum indicators: ROC, % Change, Z-Score, Stochastic, RSI, MACD
- Generates buy/sell signals based on MACD crossovers
- Provides interactive charts and a back-test summary
- Allows CSV download of all signals

---

## 🖥️ Features

### 1. Configuration (Sidebar)

Users can configure:

- **Tickers**: Comma-separated (e.g., `GME, AMC, AAPL`)
- **Years of history**: 1–5 years
- **Indicator windows**:
  - ROC window (days)
  - Stochastic %K / %D periods
  - RSI window
  - MACD fast EMA, slow EMA, signal EMA
- **Run Analysis** button

---

### 2. Indicators Calculated

| Indicator | Description |
|-----------|-------------|
| **ROC** | Rate of Change – speed of price movement |
| **% Change** | Daily percentage change |
| **Z-Score** | Measures extremity of daily move |
| **Stochastic (%K / %D)** | Overbought/oversold detection |
| **RSI** | Relative Strength Index; overbought (>70) / oversold (<30) |
| **MACD** | Trend strength + crossover signals |

---

### 3. Momentum Trading Concept

- **Momentum Trading:** Ride trends instead of predicting reversals.
- **Strategy:** 
  - Buy stocks moving up, expecting continuation
  - Sell stocks falling, expecting continuation
- **Motto:** "Buy high, sell higher"
- **Risk:** Trends may reverse suddenly; use stop-losses and manage risk.

---

### 4. Charts

For each ticker, the app shows:

1. **Price with Buy/Sell signals** (green triangle-up / red triangle-down)
2. **ROC & % Change**
3. **Stochastic %K / %D** with overbought (80) and oversold (20) levels
4. **RSI** with 70/30 thresholds
5. **MACD** with Signal line and Histogram

All charts are interactive using Plotly.

---

### 5. Back-Test Summary

For each ticker:

- Total trades executed
- Win rate (%)
- Total return (%)
- Average trade return
- Trade log expandable

---

### 6. CSV Download

Download a CSV file with **all calculated indicators and signals**.

---

## 📈 Quick Start

1. Enter tickers in the sidebar (e.g., `GME, AMC, TSLA`)
2. Adjust look-back windows for indicators
3. Click **Run Analysis**
4. View interactive charts, signals, and back-test metrics
5. Optionally, download CSV of all signals

---

## ⚙️ Requirements

Python libraries:

```text
streamlit
pandas
numpy
yfinance
plotly
matplotlib
````

---

## 🔗 How to Run

```bash
streamlit run your_app.py
```

---

## 📝 Notes

* Data is fetched using Yahoo Finance API with `group_by='ticker'`
* MACD crossovers generate **buy/sell signals**
* Indicators are calculated per ticker
* Charts and summaries update dynamically
* App supports multiple tickers simultaneously

```

---
