# Multi-Stock Trading Strategy Optimizer

A **Streamlit app** to optimize Bollinger Band + RSI trading strategies across multiple stocks using **Genetic Algorithms**.

---

## 🚀 Overview

**App Title:** Multi-Stock Trading Strategy Optimizer  
**Purpose:** Optimize trading strategies on multiple stocks, backtest them, and visualize performance.

- Fetches historical stock data from Yahoo Finance
- Preprocesses data, calculates Bollinger Bands and RSI
- Backtests strategies with entry/exit, trailing stop, and stop-loss
- Optimizes strategy parameters using a **Genetic Algorithm (GA)**
- Displays **individual stock performance** and **cumulative returns**
- Interactive charts with Plotly for strategy analysis

---

## 🖥️ Features

### 1. Configuration (Sidebar)

Users can configure:

- **Stock tickers:** Comma-separated (e.g., `AAPL, MSFT, GOOGL`)
- **Date range:** Start and end dates
- **Interval:** `1h`, `1d`, `5m`, `15m`, `30m`
- **Genetic Algorithm parameters:**
  - Population size
  - Generations
  - Crossover probability (`cxpb`)
  - Mutation probability (`mutpb`)
  - Tournament size (`tournsize`)
- **Parameter ranges for optimization:**
  - Bollinger Bands entry/exit multipliers
  - RSI entry/exit levels
  - Trailing stop and stop-loss percentages
- **Backtest options:**
  - Minimum trades filter per ticker
  - Risk-free rate
  - Show individual plots
  - Max tickers to plot

---

### 2. Data Processing

- Fetch historical stock data using `yfinance`
- Preprocess data:
  - Calculate daily returns
  - Handle missing data
- Compute trading indicators:
  - **Bollinger Bands**
  - **RSI**

---

### 3. Backtesting

- Strategy considers:
  - Entry and exit signals based on Bollinger Bands & RSI
  - Trailing stop and stop-loss protection
- Computes:
  - Position (1 = long, 0 = no position)
  - Strategy returns
  - Performance metrics (Sharpe ratio, cumulative returns)
- Filters strategies based on minimum trades per ticker

---

### 4. Genetic Algorithm Optimization

- Optimizes six strategy parameters:
  1. BB Entry multiplier
  2. BB Exit multiplier
  3. RSI Entry level
  4. RSI Exit level
  5. Trailing stop %
  6. Stop-loss %
- Uses DEAP library for GA:
  - Population initialization
  - Selection (Tournament)
  - Crossover (Blend)
  - Mutation (Gaussian)
- Fitness function maximizes **Sharpe ratio** and **return**, penalizes excessive trades
- Progress shown via Streamlit progress bar and status text
- User can stop optimization early

---

### 5. Visualization

**Individual stock plots:**

- **Trading strategy plot:**
  - Close price
  - Bollinger Bands
  - Buy & Sell signals
  - RSI chart with thresholds
- **Cumulative returns plot:**
  - Strategy vs Buy & Hold returns
- **Summary metrics per stock:**
  - Number of trades
  - Total return
  - Sharpe ratio

---

### 6. Output

- **Optimized strategy parameters**
- **Performance metrics**
  - Avg Sharpe ratio
  - Avg total return
  - Avg number of trades
- **Detailed backtest results per stock**
- Interactive charts
- Summary table for all tickers

---

## 📈 Quick Start

1. Enter stock tickers in the sidebar (comma-separated)
2. Set start/end dates and interval
3. Configure GA parameters and strategy ranges
4. Click **Run Optimization**
5. Explore:
   - Optimized parameters
   - Individual stock strategy charts
   - Cumulative returns
   - Summary statistics

---

## ⚙️ Requirements

Python libraries:

```text
streamlit
yfinance
pandas
numpy
deap
plotly
````

---

## 🔗 How to Run

```bash
streamlit run multi_stock_strategy_optimizer.py
```

---

## 📝 Notes

* Backtest uses **daily returns** if `1d` interval, **hourly returns** if `1h` interval, etc.
* Strategies are constrained by user-defined parameter ranges
* GA optimization allows exploration of high-performing strategies across multiple stocks
* Plots are interactive with Plotly for hover details and zoom

```

---


