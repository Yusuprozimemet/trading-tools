# Pro Chart Viewer – Streamlit App

A **Streamlit app** for advanced stock charting and spread trading with robust Yahoo Finance integration. No more Yahoo errors, smart date limits, and interactive indicators.

---

## Features

- **Chart Modes**:
  - **Single Stock**: Plot individual stock price data.
  - **Spread Trading**: Analyze custom or preset pairs of stocks, including ratio-adjusted spreads.

- **Intervals & Smart Date Limits**:
  - Supports multiple intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo.
  - Auto-limits historical data based on Yahoo Finance restrictions.

- **Indicators**:
  - **Moving Averages**: SMA 20/50, EMA 9/21
  - **Bollinger Bands**: Upper, lower, optional fill
  - **RSI**: Relative Strength Index
  - **Volume & Volume MA**
  - **Spread Mode**: Z-score, long/short signals

- **Visualization**:
  - Candlestick chart
  - Close price line overlay
  - Subplots for RSI, Z-Score (spread mode), and volume
  - Dynamic horizontal lines for key price levels
  - Dark/Light mode toggle
  - Perfect gap handling for intraday intervals

- **Live Signals Table**:
  - Displays latest long/short signals in spread trading mode
  - Shows price, Z-score, and timestamp

---

## Installation

Install required packages:

```bash
pip install streamlit pandas numpy yfinance plotly
````

Run the app:

```bash
streamlit run charts.py
```

---

## Usage

### Sidebar Options

1. **Chart Mode**:

   * **Single Stock**: Input ticker symbol (e.g., `ASML.AS`)
   * **Spread Trading**: Choose a preset pair or define a custom pair with ratio.

2. **Interval**: Select chart interval (e.g., 1h, 1d).

3. **Days Back**: Slider auto-adjusts based on interval limits.

4. **Indicators**:

   * Toggle candlesticks and close line.
   * Enable/disable SMA, EMA, Bollinger Bands, RSI, Volume.
   * Configure Z-score window and signal threshold for spreads.

5. **H-Lines**: Add horizontal lines at specific price levels (comma-separated).

6. **Dark Mode**: Switch between dark and light Plotly templates.

---

### Features Overview

* **Candlestick + Line Overlay**: Visualize open, high, low, close prices with optional line chart.
* **Moving Averages**: SMA & EMA with customizable periods and colors.
* **Bollinger Bands**: Upper and lower bands with optional shaded fill.
* **RSI**: Relative Strength Index for momentum analysis.
* **Volume**: Volume bars with 20-period moving average overlay.
* **Z-Score (Spread Trading)**: Identify long/short signals based on spread deviations.
* **Signals Table**: Last 10 active long/short signals in spread mode.

---

### Presets for Spread Trading

| Preset    | Stock 1 | Stock 2 | Ratio  |
| --------- | ------- | ------- | ------ |
| AGN/ASML  | AGN.AS  | ASML.AS | 0.0072 |
| WKL/REN   | WKL.AS  | REN.AS  | 3.6    |
| HEIA/ASML | HEIA.AS | ASML.AS | 0.084  |

Custom spreads can be defined with any two tickers and ratio.

---

### Notes

* **Data Source**: Yahoo Finance
* **Caching**: Stock data cached for 60 seconds
* **Time Zones**: Localized for correct intraday plotting
* **Error Handling**: Robust checks for missing or invalid data

---

### Quick Start

1. Open the app:

```bash
streamlit run charts.py
```

2. Select chart mode and configure options in the sidebar.
3. View interactive charts and indicators.
4. In spread mode, monitor live signals for trading insights.

---

### Footer

> No more Yahoo errors • Smart date limits • Perfect gaps • Dutch engineering 🚀

````
Run with:
```bash
streamlit run chart/charts.py
````

