
# Interactive Spread Trading Analyzer

A **Streamlit app** for analyzing the spread between two stocks with multiple technical indicators and trading signals.  
It supports interactive visualization using Plotly and provides actionable metrics like Z-Score, RSI, MACD, Bollinger Bands, and ATR.

---

## Features

- Analyze **spread between any two stocks** with a configurable ratio.
- Calculate **technical indicators** on the spread:
  - Z-Score and trading signals (long/short)
  - MACD and MACD Histogram
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - ATR (Average True Range)
- Visualize spread and indicators interactively with Plotly.
- Filter **non-trading hours** for intraday data.
- Supports multiple **timeframes**:
  - Daily (1 Year)
  - Hourly (1 Month)
  - 15-Minute (5 Days)
  - All timeframes at once.
- Recent trading signals table with spread and Z-Score.

---

## Installation

Install the required packages:

```bash
pip install streamlit yfinance pandas numpy plotly
````

Run the app:

```bash
streamlit run app.py
```

---

## Usage

### Sidebar Configuration

1. **Select Preset Pair**

   * Choose a preset stock pair (e.g., `AGN/ASML`, `WKL/REN`) or `Custom`.

2. **Stock Pair Inputs**

   * Stock 1 and Stock 2 tickers and optional names.
   * Ratio multiplier for Stock 2 in the spread calculation.

3. **Analysis Parameters**

   * **Z-Score Window**: Rolling window size for mean-reversion signals.

4. **Timeframe Selection**

   * Daily, Hourly, 15-Minute, or All Timeframes.

5. **Analyze Spread Button**

   * Click to fetch data, calculate indicators, and display results.

---

### Interpreting Outputs

* **Current Spread**: Difference between Stock 1 and Stock 2 × ratio.
* **Z-Score**: Measures deviation from the rolling mean of the spread.
* **Signal**:

  * **LONG (Green ▲)**: Z-Score < -2 → Spread is below mean, expect reversion up.
  * **SHORT (Red ▼)**: Z-Score > 2 → Spread is above mean, expect reversion down.
  * **NEUTRAL**: No trading signal.
* **RSI**: Momentum indicator (Overbought > 70, Oversold < 30).
* **MACD & Histogram**: Trend-following indicator for spread.
* **Bollinger Bands**: Volatility bands around the spread.
* **ATR**: Average True Range, measures volatility.

---

### Visualization

The app generates an interactive Plotly chart with:

1. **Spread Price & Bollinger Bands**
2. **Z-Score with thresholds**
3. **Volume & MACD**
4. **RSI with reference lines**
5. **ATR**

Recent trading signals are displayed in a table with:

* Date
* Signal (LONG/SHORT)
* Spread value
* Z-Score

---

## Notes

* **Trading Hours Filtering**: Ensures intraday data only includes trading hours (9:00–17:30 CET/CEST, Mon–Fri).
* **Error Handling**: Displays errors if stock data cannot be retrieved.
* Supports multiple **intervals**: Daily (`1d`), Hourly (`1h`), 15-Minutes (`15m`).

---

## Authors

* Developed as an interactive tool for pairs/spread analysis with technical indicators.
* Designed for **educational and analytical purposes**, not financial advice.

---

## Example Presets

| Preset   | Stock 1 | Stock 2 | Ratio  |
| -------- | ------- | ------- | ------ |
| AGN/ASML | AGN.AS  | ASML.AS | 0.0072 |
| WKL/REN  | WKL.AS  | REN.AS  | 3.6    |

---

## How to Extend

* Add more technical indicators.
* Integrate portfolio backtesting with position sizing.
* Add alert notifications for signals.

