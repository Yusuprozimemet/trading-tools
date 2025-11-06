

# Stock Correlation Utility

**Streamlit-ready version:**
Run with:

```bash
streamlit run strategies/correlation.py
```

---

## Overview

This module provides a **stock correlation analysis tool** with optional technical indicator calculations. It was converted from `TradingBasics/stockcorrelation.ipynb` and includes a **Streamlit UI** for interactive use.

Key features:

* Compute common technical indicators per ticker:

  * Bollinger Bands (BB)
  * Relative Strength Index (RSI)
  * Average True Range (ATR)
  * SuperTrend
  * MACD (with signal and histogram)
* Fetch historical OHLCV data from Yahoo Finance
* Compute correlation matrix of ticker Close prices
* Display correlation matrix and heatmap
* Streamlit UI for easy interaction and CSV download
* Per-ticker indicator data with CSV export

---

## Dependencies

Add to `requirements.txt` if needed:

```
streamlit
pandas
numpy
matplotlib
yfinance
seaborn  # optional, for heatmap visualization
```

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run strategies/correlation.py
```

3. Use the sidebar to:

   * Enter tickers (comma-separated, e.g., `SPY, AAPL, MSFT`)
   * Select period (`1y`, `2y`, `5y`, `6mo`, `3mo`)
   * Choose interval (`1d`, `1wk`, `1mo`)
   * Toggle indicator calculation (slower if enabled)

4. Press **Run** to:

   * Fetch and process data
   * Compute correlation matrix
   * Display heatmap
   * Download correlation CSV and per-ticker CSVs

---

## Key Functions

### Data Normalization

```python
_normalize_columns(df: pd.DataFrame) -> pd.DataFrame
```

Converts MultiIndex columns (from grouped Yahoo downloads) to single-level.

### Bollinger Bands

```python
BBand(df, base="Close", period=20, multiplier=2, multiplier3=3)
```

Adds columns:

* `UpperBB`, `LowerBB`, `MiddleBB`, `UpperBB3`, `LowerBB3`

### RSI

```python
RSI(df, base="Close", period=14)
```

Adds column:

* `RSI`

### ATR

```python
calculate_atr(df, period=14) -> pd.Series
```

### SuperTrend

```python
calculate_supertrend(df, period=10, multiplier=3) -> pd.Series
```

### MACD

```python
calculate_macd(df, fast_period=12, slow_period=26, signal_period=9)
```

Returns `(macd, signal, histogram)`.

### Compute All Indicators

```python
calculate_indicators(df: pd.DataFrame) -> pd.DataFrame
```

Adds BB, RSI, ATR, SuperTrend, MACD, Signal, Histogram.

### Correlation Matrix

```python
calculate_correlation(ticker_data: dict, tickers: List[str]) -> pd.DataFrame
```

Returns correlation of Close prices for selected tickers.

---

## Streamlit UI Features

* Input tickers, period, and interval via sidebar
* Option to calculate indicators
* Displays:

  * Correlation matrix
  * Heatmap (using Seaborn if available, fallback to Matplotlib)
  * Sample per-ticker data
* CSV download options:

  * Correlation matrix
  * Per-ticker OHLCV + indicators

---

## Notes

* Indicator calculation can be slower for many tickers.
* Multi-ticker downloads use Yahoo Finance's grouped requests.
* Streamlit must be installed to run the UI.
* Heatmap is enhanced if Seaborn is installed; otherwise, Matplotlib fallback is used.

---

This documentation covers the **Stock Correlation Utility**, highlighting the Streamlit workflow, indicator calculations, and correlation analysis.

---
