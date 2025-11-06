

# BFIT.AS — Bollinger + RSI + Trailing-Stop Optimiser

**Streamlit-ready version:**
Run with:

```bash
streamlit run bfit_strategy.py
```

---

## Overview

This script implements a **Bollinger–RSI Adaptive Trailing Reversion** strategy, combining:

* Bollinger Bands
* Relative Strength Index (RSI)
* Trailing-stop logic

The strategy supports:

* Backtesting
* Paper-trading simulation
* Multi-objective optimization using **DEAP Genetic Algorithm (GA)**
* Streamlit-based interactive UI for parameter tuning and visualization

---

## Features

* Fetches OHLCV data from Yahoo Finance while handling window limits:

  * Hourly data → max 60 days
  * Daily data → max 730 days
* Computes technical indicators:

  * 20-period Bollinger Bands
  * 14-period RSI
* Backtests long-only strategy with:

  * Entry/exit logic
  * Trailing-stop
* Evaluates performance metrics:

  * Sharpe ratio
  * Max drawdown
  * Total returns
* DEAP GA optimization for:

  * BB entry/exit multipliers
  * RSI entry/exit thresholds
  * Trailing-stop %
* Generates plots:

  * Price chart with buy/sell signals
  * RSI chart with thresholds
  * Equity curve vs buy-and-hold
* Extracts trades and allows CSV download
* Non-blocking Streamlit progress bar for optimization

---

## Dependencies

Add to `requirements.txt`:

```
streamlit
pandas
numpy
matplotlib
yfinance
deap
tqdm
```

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run bfit_strategy.py
```

3. Use the sidebar to:

   * Input ticker symbol (default `BFIT.AS`)
   * Select start/end dates
   * Choose data interval (`1h` or `1d`)
   * Configure GA parameters: population size, generations, patience, random seed

4. Click **Run optimisation** to:

   * Backtest the strategy
   * Optimize parameters
   * Display charts and metrics

5. Download detected trades as CSV.

---

## Key Functions

### Data Fetching

```python
fetch_data(ticker: str, start: str, end: str, interval: str = "1h")
```

Fetches OHLCV data from Yahoo Finance with window protection.

### Indicators

```python
add_indicators(df: pd.DataFrame) -> pd.DataFrame
```

Adds:

* Bollinger Bands: `MA20`, `UpperBB`, `LowerBB`
* RSI: `RSI`

### Backtesting

```python
backtest(df, bb_entry, bb_exit, rsi_entry, rsi_exit, trail_pct)
```

Calculates:

* Strategy positions
* Strategy returns
* Cumulative equity

### Metrics

* `sharpe(returns: pd.Series, rf: float = 0.02)` — Annualized Sharpe ratio
* `max_drawdown(cumulative: pd.Series)` — Maximum drawdown

### DEAP Optimization

```python
optimise(df, pop_size=200, ngen=80, seed=None, progress_callback=None, patience=0)
```

Returns:

* `best_params` — Best GA parameters
* `best_fit` — Fitness tuple (Sharpe, TotalRet, Calmar, Trades, MaxDD)
* Number of generations executed

### Plotting

```python
plot_optimal(df, params, ticker="")
```

Returns matplotlib Figures:

* Price + signals
* RSI chart
* Cumulative equity

### Trade Extraction

```python
extract_trades(bt: pd.DataFrame) -> pd.DataFrame
```

Returns:

* Buy/sell trade list with entry/exit, pct return, duration

---

## Notes

* Optimization can be slow for large populations and generations.
* Yahoo Finance limits hourly data to ~60 days.
* DEAP's creator objects are reused to avoid session errors.
* Non-deterministic results unless a seed is specified.

---

This documentation covers the **BFIT.AS** strategy script and its Streamlit interface, providing instructions to run, optimize, and visualize results.

---

