# 📊 Comprehensive Pair Trading Analysis App

This Streamlit app provides a complete workflow for discovering, analyzing, and backtesting **pair trading strategies** on stock data using historical prices, cointegration, and Bollinger Bands.

---

## **Features**

1. **Pair Discovery**

   * Identify cointegrated stock pairs.
   * Calculate correlations and cointegration p-values.
   * Display top pairs with normalized price charts.
   * User-configurable significance level for cointegration tests.

2. **Pair Analysis & Trading**

   * Analyze a selected pair with hedge ratio and spread calculations.
   * Apply Bollinger Bands to the spread for trading signals.
   * Simulate a trading strategy with risk management.
   * Visualize spread, entry/exit points, and portfolio performance.
   * Trade log and key metrics (final value, total return, etc.).

---

## **Installation**

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install required packages:

```bash
pip install streamlit yfinance pandas numpy statsmodels plotly
```

3. Run the app:

```bash
streamlit run app.py
```

---

## **Usage**

### **Sidebar Configuration**

* **Mode Selection**

  * `Pair Discovery`: Find cointegrated pairs.
  * `Pair Analysis & Trading`: Analyze a specific pair and simulate a strategy.

### **Pair Discovery Settings**

* Enter tickers (comma-separated).
* Select date range (`Start Date` and `End Date`).
* Set cointegration significance level.
* Click **Find Pairs**.

### **Pair Analysis & Trading Settings**

* Enter two ticker symbols.
* Select date range for analysis.
* Configure trading parameters:

  * Initial Balance
  * Risk Tolerance (%)
  * Max Loss per Share (€)
* Configure Bollinger Bands:

  * Window size
  * Standard deviation multiplier
* Click **Run Analysis**.

---

## **Technical Details**

* **Data Source**: Yahoo Finance (`yfinance`).
* **Statistical Methods**:

  * Cointegration test (`coint`) to identify tradable pairs.
  * Augmented Dickey-Fuller test (`adfuller`) for spread stationarity.
  * Hedge ratio computed via Ordinary Least Squares regression.
* **Indicators**:

  * Spread between two stocks
  * Bollinger Bands applied to spread
* **Trading Simulation**:

  * Long/short entry and exit based on Bollinger Bands.
  * Portfolio value updated with simulated trades.
  * Risk management via position sizing.

---

## **Output Visualizations**

1. **Pair Discovery**

   * Normalized price charts of top cointegrated pairs.
   * Correlation coefficient displayed for each pair.

2. **Pair Analysis**

   * Spread with Bollinger Bands and entry/exit signals.
   * Portfolio value over time.
   * Trade log table with executed trades.

---

## **Example Workflow**

1. Start with **Pair Discovery** to identify cointegrated stocks.
2. Select a promising pair and switch to **Pair Analysis & Trading**.
3. Set trading parameters and Bollinger Bands.
4. Run the analysis to simulate the strategy.
5. Review portfolio performance, metrics, and trade log.

---

## **Tips**

* Start with a small set of tickers to quickly identify pairs.
* Adjust Bollinger Bands and risk parameters based on volatility.
* Review cointegration and ADF p-values to ensure stationarity of spread.

---


