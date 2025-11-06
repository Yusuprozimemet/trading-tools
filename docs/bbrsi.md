
# 🎯 BBRSI Trading Strategy Optimizer

This Streamlit application optimizes a trading strategy based on **Bollinger Bands**, **RSI**, and **Trailing Stop** signals using a **Genetic Algorithm (GA)**.  
The tool backtests strategies, evaluates performance metrics (Sharpe, Return, Drawdown, etc.), and visualizes trading signals and performance vs Buy & Hold.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| **Technical Indicators** | Bollinger Bands (MA20 ± 2σ) + RSI (14) |
| **Trading Logic** | Entry/Exit signals based on BB + RSI conditions and trailing stop |
| **Backtesting** | Computes trade signals, returns, cumulative performance |
| **Genetic Optimization** | GA searches for optimal parameters (BB multipliers, RSI levels, trailing stop) |
| **Visualizations** | Price chart with Buy/Sell signals, RSI indicator, cumulative return comparison |
| **Performance Metrics** | Sharpe, Total Return, Calmar Ratio, Trade Count, Max Drawdown |
| **Download Results** | Export backtest results as CSV |

---

## 📦 Requirements

Install dependencies:

```bash
pip install streamlit yfinance pandas numpy matplotlib deap
````

Optional (recommended):

```bash
pip install ipykernel
```

---

## ▶️ Running the App

From the project directory:

```bash
streamlit run BBRSI.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 🧠 Strategy Logic

### **Entry Condition**

Enter long when:

```
Close < LowerBB × BB_Entry  AND  RSI < RSI_Entry
```

### **Exit Condition**

Exit position when:

```
Close > UpperBB × BB_Exit  AND  RSI > RSI_Exit
```

### **Risk Management**

* A **Trailing Stop** follows price upward to secure gains.

---

## 🧬 Genetic Algorithm Optimization

The GA evolves these parameters:

| Parameter           | Range       | Meaning                                             |
| ------------------- | ----------- | --------------------------------------------------- |
| BB Entry Multiplier | 0.80 → 1.05 | Controls how aggressively to enter below lower band |
| BB Exit Multiplier  | 0.95 → 1.25 | Controls how quickly to exit at upper band          |
| RSI Entry Threshold | 10 → 45     | Oversold threshold                                  |
| RSI Exit Threshold  | 55 → 90     | Overbought threshold                                |
| Trailing Stop       | 0.5% → 15%  | Tightness of profit lock                            |

Fitness rewards:

* High Sharpe Ratio
* High Total Return
* High Calmar Ratio
  While penalizing:
* Excessive trades
* Deep drawdowns

---

## 🖥 UI Overview

### Sidebar Controls

* **Ticker** selection (e.g., AAPL, TSLA, BTC-USD)
* **Date range** + interval (`1h`, `1d`, `1wk`)
* **GA Parameters**: Population size, Generations, Random seed
* **Run Optimization** button

### Output

* Best strategy parameter values
* Performance metrics (Sharpe, Return, Drawdown)
* Trading signal chart
* RSI visualization
* Strategy vs Buy & Hold comparison
* CSV download button

---

## 📊 Example Output

| Metric           | Example  |
| ---------------- | -------- |
| Sharpe Ratio     | `1.42`   |
| Total Return     | `+24.5%` |
| Calmar Ratio     | `0.78`   |
| Number of Trades | `14`     |
| Max Drawdown     | `-8.3%`  |

---

## 📁 Project File Structure

```
BBRSI.py          # Main Streamlit app and optimizer
```

---

## 📝 Notes

* Hourly data is limited by Yahoo Finance to recent days.
* Optimization is CPU-intensive—higher GA population or generations → slower run.
* This is a **research / backtesting tool**, not investment advice.

---

## ⚖️ Disclaimer

This software is provided for academic and research purposes only.
It **does not** constitute financial advice. Trade responsibly.

---

