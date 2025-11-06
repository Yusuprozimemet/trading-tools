# Market Making Strategy – Volatility-Scaled Spreads

**Streamlit-ready:**
Run with:

```bash
streamlit run path_to_script.py
```

---

## Overview

This application simulates and backtests a **market-making strategy** using **15-minute historical data**. The strategy:

* Quotes **bid** and **ask** around the mid-price
* Adjusts spreads based on **volatility**
* Uses **Moving Averages** and **Bollinger Bands** for entry/exit signals
* Supports **manual parameters** or **grid-based optimization**

Key features:

* Fetch OHLCV data via Yahoo Finance
* Calculate rolling indicators (MA, Bollinger Bands, Volatility, Support, Bid/Ask Spread)
* Generate buy/sell signals with stop-loss and take-profit
* Backtest with portfolio tracking
* Parameter optimization with coarse → fine progressive search
* Metrics display: Total Return, Sharpe Ratio, Max Drawdown, Win Rate, etc.
* Interactive **Plotly dashboard**
* Export full backtest CSV

---

## Dependencies

Add to `requirements.txt`:

```
streamlit
pandas
numpy
yfinance
plotly
```

Optional:

* `matplotlib` or `seaborn` if extending plots
* Standard Python libraries (`datetime`, `random`, `os`, `concurrent.futures`, `warnings`, `traceback`, `time`)

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run market_making.py
```

3. Configure the **sidebar inputs**:

### Core Inputs

* **Ticker:** e.g., `AAPL`
* **Days of 15-min data:** 3–60
* **Initial Capital ($)**
* **Transaction Cost (%):** includes commission + slippage
* **Base Spread (bps):** base bid-ask spread in basis points

### Strategy Parameters

* Short and Long Moving Average windows
* Shares per trade
* Max position
* Support window (for rolling support calculation)
* Stop-loss / Take-profit (%)
* Bollinger Band standard deviation

### Optimization (optional)

* Toggle **Run Parameter Optimization**
* Set min/max trials for **coarse** and **fine** grid search

4. Click **Run Market Maker** to:

* Fetch data
* Calculate indicators
* Run backtest
* (Optional) Optimize parameters

---

## Core Functions

### Data Fetching

```python
fetch_data(symbol, start, end, interval)
```

* Returns OHLCV DataFrame for ticker
* Handles MultiIndex and missing columns

### Indicators

```python
calculate_indicators(df, short_window, long_window, boll_std, support_window, base_spread)
```

* Computes:

  * Short & Long MA
  * Volatility
  * Bollinger Bands
  * Support (rolling min)
  * Bid / Ask / Spread

### Signal Generation

```python
generate_signals(df, stop_loss, take_profit)
```

* Generates **Buy** and **Sell** signals based on support, stop-loss, take-profit

### Backtesting

```python
run_backtest(df, capital, shares_per_trade, cost, max_pos, stop_loss, take_profit, short_w, long_w, boll_std, supp_w, base_spread)
```

* Tracks **Holdings**, **Cash**, **Total Portfolio Value**, **Returns**

### Metrics

```python
calc_metrics(portfolio)
```

* Calculates:

  * Total Return
  * CAGR
  * Volatility
  * Sharpe Ratio
  * Sortino Ratio
  * Max Drawdown
  * Win Rate
  * Profit Factor

### Optimization

```python
optimize_params(data, capital, cost, base_spread, min_coarse_trials, max_coarse_trials, min_fine_trials, max_fine_trials)
```

* Progressive **coarse → fine grid search** to maximize Sharpe ratio
* Uses **parallel processing** for efficiency
* Returns best parameter set

---

## Dashboard

### Visualizations

* **Price & Quotes:** Close, Short/Long MA, Bollinger Bands, Support, Bid/Ask
* **Spread ($)**
* **Portfolio Value**
* **Shares Held**

### Metrics & Parameters

* Key metrics displayed as **Streamlit metrics**
* Detailed metrics in **expander**
* Parameters table shows **used settings** (manual or optimized)

### Export

* Full backtest with Bid/Ask/Spread available as **CSV download**

---

## Educational Section

* **Market Making:** Earn the spread, not predict direction
* **Volatility-Scaled Spread:** Wider spread in volatile markets, narrower in calm
* **Entry/Exit:** Buy on support, Sell on stop-loss or take-profit
* **Risks:** Inventory risk, adverse selection, slippage

---

## Tips

* Use **liquid tickers**: `AAPL`, `NVDA`, `SPY`
* Start with **7–21 days** of 15-min data
* Base spread: 10–20 bps
* Keep **stop-loss < take-profit**

---

This Markdown serves as a **full reference guide** for your Market Making strategy Streamlit app.

---

