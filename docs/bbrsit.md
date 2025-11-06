
# 📊 BBRSIT / Trading Utilities Module

This Python module provides **reusable functions and classes** for fetching market data, calculating indicators, backtesting strategies, and optionally optimizing parameters.  
It is derived from `TradingBasics/strategies.ipynb` and is designed for programmatic usage outside of Jupyter notebooks.

---

## ⚡ Features

- **Data Fetching Helpers**
  - Fetch by period (e.g., 1y, 5y)
  - Fetch by explicit date range
- **Indicators**
  - Bollinger Bands
  - RSI
  - Combined indicator calculation
- **Backtesting**
  - Long-only strategy using BB + RSI
  - Trailing stop support
  - Strategy returns computation
- **Paper Trading**
  - Generates list of trades and daily returns
- **Metrics**
  - Total return
  - Sharpe ratio
- **Strategy Optimization (Optional)**
  - `OptimalStrategy` class uses **DEAP** for parameter optimization
- **Streamlit UI**
  - Quick backtest or paper-trade interface
  - Interactive charts of indicators and strategy performance

---

## 📦 Installation

```bash
pip install pandas numpy matplotlib yfinance
````

For optimization functionality:

```bash
pip install deap
```

For interactive UI:

```bash
pip install streamlit
```

---

## 🛠 Usage

### Importing the module

```python
from trading_utils import (
    fetch_data_period,
    fetch_data_range,
    preprocess_data,
    calculate_indicators,
    backtest,
    paper_trade,
    calculate_returns,
    calculate_sharpe_ratio,
    OptimalStrategy
)
```

### Quick Example

```python
import pandas as pd

# Fetch data
data = fetch_data_period("AAPL", period="1y", interval="1d")

# Preprocess & calculate indicators
data = preprocess_data(data)
data = calculate_indicators(data)

# Run backtest
bt = backtest(data, bb_entry=1.02, bb_exit=0.98, rsi_entry=30, rsi_exit=70, trailing_stop_pct=0.05)

# Paper trades
trades, daily_returns = paper_trade(data, 1.02, 0.98, 30, 70, 0.05)

# Calculate metrics
total_return = calculate_returns(trades)
sharpe = calculate_sharpe_ratio(daily_returns)

print(f"Total return: {total_return:.2%}, Sharpe ratio: {sharpe:.2f}")
```

---

### Optimizing a Strategy

```python
from trading_utils import OptimalStrategy

opt = OptimalStrategy("AAPL", period="1y")
opt.fetch_and_prepare_data()

best_params = opt.optimize_strategy(population_size=100, ngen=50)
print("Best strategy parameters:", best_params)

opt.plot_best_strategy(best_params)
```

---

### Running Streamlit UI

```bash
streamlit run trading_utils.py
```

The sidebar allows:

* Choosing ticker and mode (quick backtest or paper trade)
* Setting strategy parameters (BB entry/exit, RSI entry/exit, trailing stop)
* Visualizing indicator plots and cumulative returns
* Listing trades and computing metrics

---

## ⚖️ Notes / Disclaimer

* This module is for **research and educational purposes only**.
* **Not financial advice**. Trading involves real risk.
* Some features (e.g., DEAP optimization) require optional dependencies.

---

## 📂 Module Contents

* `fetch_data_period`, `fetch_data_range` – data retrieval
* `preprocess_data` – cleans and prepares OHLCV data
* `calculate_bollinger_bands`, `calculate_rsi`, `calculate_indicators` – technical indicators
* `backtest`, `paper_trade` – strategy simulation
* `calculate_returns`, `calculate_sharpe_ratio` – performance metrics
* `OptimalStrategy` – class for optimization and plotting

---
