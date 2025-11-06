
# 📈 BBRSI Strategy Backtester + Smart Debug

This Streamlit application allows you to **backtest a Bollinger Bands + RSI (BBRSI) trading strategy** with **automatic debugging features** that explain *why* trades did or did not occur.

If the strategy generates **0 trades**, the app will:
- Diagnose whether Bollinger Band or RSI triggers never occurred
- Show how close price and RSI came to triggering trades
- Suggest corrected parameter values automatically

This makes it extremely useful for **strategy tuning**, **education**, and **practical trading research**.

---

## 🔥 Key Features

| Feature | Description |
|--------|-------------|
| **BB + RSI Strategy Logic** | Entry when price is oversold & breaks lower band; exit on overbought signal or trailing stop |
| **Auto-Debug Mode** | If 0 trades occur, system explains why and provides guided parameter fixes |
| **Trailing Stop** | Automatically protects profits after entry |
| **Performance Metrics** | Sharpe, Max Drawdown, Total Return, Trades Count |
| **Visualizations** | Price Bands + Buy/Sell points, RSI chart shading |
| **One-Click Safe Parameters** | Quickly generate workable trades |
| **Data Download** | Export backtest results as CSV |

---

## 🛠 Installation

Install dependencies:

```bash
pip install streamlit yfinance pandas numpy matplotlib
````

---

## ▶️ Run the App

```
streamlit run bbrsi_backtest.py
```

The app will open automatically in your browser at:

```
http://localhost:8501
```

---

## 🎛 Strategy Inputs (Sidebar)

| Parameter           | Meaning                                               |
| ------------------- | ----------------------------------------------------- |
| **BB Entry**        | Multiplier below Lower Bollinger Band to trigger buy  |
| **BB Exit**         | Multiplier above Upper Bollinger Band to trigger exit |
| **RSI Entry**       | Oversold threshold for entry (e.g., 30)               |
| **RSI Exit**        | Overbought threshold for exit (e.g., 70)              |
| **Trailing Stop %** | Protects profits after entry by following price       |

Also configurable:

* Ticker
* Date range
* Interval (1h, 1d, 1wk, 1mo)

---

## 🧠 Debug Mode (When Trades = 0)

If no trades occur, the app will show:

| Diagnostic Displayed           | Meaning                                             |
| ------------------------------ | --------------------------------------------------- |
| Price touched BB Entry level?  | Was price *ever* low enough to qualify as oversold? |
| RSI < Entry threshold count    | Was RSI ever oversold?                              |
| Entry conditions overlap count | Did both conditions ever occur on same bar?         |
| Closest misses                 | How close signals came to triggering                |

It also suggests parameter adjustments.

---

## 📊 Performance Metrics Displayed

* Total Strategy Return
* Buy & Hold Return (for comparison)
* Sharpe Ratio
* Maximum Drawdown (Risk)
* Calmar Ratio
* Total Number of Trades

---

## 📈 Visualizations

* **Price + Bollinger Bands + Buy/Sell Markers**
* Highlights **BB-only touches** and **RSI-only touches**
* **RSI Chart** with oversold/overbought zones

---

## 💾 Export Data

You can download all processed results:

```
debug_<TICKER>.csv
```

Contains:

* Entry/Exit Prices
* Signals
* Strategy Equity Curve
* Buy & Hold Comparison

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.
It **does not constitute financial advice**.
Trading involves risk.

---

## 🧩 Future Enhancements (Optional)

* Parameter optimization via Genetic Algorithm
* Multi-asset portfolio mode
* Sharpe-maximizing auto-tuning mode

---

**Enjoy your research — and may your signals be sharp and your drawdowns shallow.** 🚀📉📈

