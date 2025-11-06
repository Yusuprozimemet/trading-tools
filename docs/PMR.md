# Pairs Mean-Reversion (Streamlit Wrapper)

This Streamlit app provides a **lightweight UI** for running the `pairs_mean_reversion` strategy.  
It calls the core functions from `strategies.PMR` and allows interactive configuration and backtesting.

---

## Overview

- **Purpose**: Enable traders to run the pairs mean-reversion strategy with a visual interface.
- **Core Functions Used**:
  - `fetch_prices` – Download historical or intraday price data.
  - `calculate_hedge_ratio` – Compute the hedge ratio (beta) for the pair.
  - `bollinger_bands` – Calculate rolling mean and Bollinger Bands for spread.
  - `generate_signals` – Generate long/short signals based on spread deviations.
  - `backtest_pair` – Simulate trades and calculate PnL.

- **Modes Supported**:
  - `swing` – Daily price data analysis.
  - `intraday` – Intraday data analysis (15m/1h intervals).
  - `both` – Run both swing and intraday analyses simultaneously.

---

## Usage

Run the Streamlit app:

```bash
streamlit run strategies/PMR.py
````

### Sidebar Parameters

1. **Pair Selection**

   * Choose from `RECOMMENDED_PAIRS` or enter a custom ticker pair.

2. **Mode**

   * `both`, `swing`, `intraday`.

3. **Historical Data**

   * `History (days, swing)` – number of days for daily price history.
   * `Intraday days` – number of days for intraday data.

4. **Trade Parameters**

   * Notional per trade.
   * Bollinger window and standard deviation.
   * Maximum concurrent trades.

5. **Portfolio Settings**

   * Portfolio balance.
   * Risk tolerance for swing and intraday trades.
   * Maximum loss per share.

---

## Analysis Workflow

1. **Fetch price data** (daily or intraday).
2. **Compute hedge ratio** (beta) for the selected pair.
3. **Calculate spread** and Bollinger Bands.
4. **Generate trading signals** based on deviation from the moving average.
5. **Backtest trades**:

   * Track entries/exits.
   * Compute equity curve and PnL.
   * Suggest optimal number of shares based on portfolio balance and risk tolerance.

---

## Results

### Swing (Daily) Analysis

* Displays hedge ratio.
* Visualizes spread with Bollinger Bands.
* Lists closed trades and total backtest PnL.
* Suggests max shares and potential per-share profit.

### Intraday Analysis

* Similar outputs as swing analysis, but using 15m/1h intraday prices.

---

## Visualizations

* **Spread vs Bollinger Bands**
* **Closed trades table**
* **Equity / PnL summary**
* Interactive charts using **Plotly** (falls back to `st.line_chart` if unavailable).

---

## Notes

* Requires Python packages:

  * `streamlit`, `pandas`, `numpy`, `plotly`
* Assumes `strategies.PMR` is available in the project root.
* Provides robust handling for missing or empty data.
* Designed for **educational and analytical purposes**, not financial advice.

---

