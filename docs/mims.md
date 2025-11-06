
# Trading-Analysis Dashboard — Multi-Factor Intrinsic Momentum Strategy (MIMS)

A comprehensive **Streamlit dashboard** to analyze stocks and ETFs using a **6-step multi-factor framework**: Screening → Sector Benchmarking → Fundamentals → Valuation → Technicals → Decision Matrix & Portfolio Allocation.

---

## Features

- ✅ Multi-ticker input (stocks & ETFs)  
- ✅ ETF relative strength comparison  
- ✅ Fundamental analysis (Revenue, Net Income, FCF, ROE, Debt/Equity, Moat)  
- ✅ Intrinsic value estimation (DCF & Discounted Earnings)  
- ✅ Technical analysis (SMA, Bollinger Bands, RSI, MACD)  
- ✅ Pair analysis with ratio, z-score, and rolling correlation  
- ✅ Decision matrix & suggested portfolio allocation  
- ✅ Interactive Plotly charts  
- ✅ Downloadable CSV report  

---

## Installation

1. **Clone the repository**:

```bash
git clone <repo_url>
cd <repo_folder>
````

2. **Create and activate a Python environment** (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**:

```bash
streamlit run BBRSI.py
```

---

## Usage

1. Open the **sidebar** and configure tickers:

   * **Stocks:** Comma-separated (e.g., `ASML.AS, INTC`)
   * **ETFs:** Comma-separated (e.g., `SMH, SPY`)

2. Optional inputs:

   * Run **Pair Analysis** for custom ticker comparison
   * Choose **lookback period** for pairs
   * Select **method description** to see step-by-step logic

3. Navigate through sections:

   * **1️⃣ Stock & ETF Screening** — Price, 52W high/low, P/E, volatility
   * **2️⃣ ETF Relative Strength** — Compare ETFs vs benchmark
   * **3️⃣ Fundamental Snapshot** — Revenue, Net Income, FCF, ratios
   * **4️⃣ Intrinsic Value** — DCF / Discounted Earnings valuation
   * **5️⃣ Technical Analysis** — SMA, RSI, MACD, support/resistance
   * **6️⃣ Decision Matrix** — BUY / HOLD / SELL recommendation
   * **7️⃣ Suggested Portfolio Allocation** — Allocation % with Pie chart
   * **8️⃣ Download CSV Report** — Export all data

---

## Methodology

The dashboard implements a **6-step Multi-Factor Intrinsic Momentum Strategy (MIMS)**:

1. **Screening** — Identify liquid, fundamentally strong stocks
2. **Sector Benchmarking** — Evaluate sector performance vs market
3. **Fundamentals** — Assess growth, profitability, financial strength
4. **Valuation Analysis** — Estimate intrinsic value and margin of safety
5. **Technical Analysis** — Detect trend, momentum, support/resistance
6. **Decision Matrix** — Combine all scores into actionable recommendations

**Optional Step 0:** Ticker selection logic (Top-down macro → Bottom-up micro → quantitative filters).

---

## Example

**Input:**

* Stocks: `ASML.AS, INTC`
* ETFs: `SMH, SPY`

**Output:**

* Screening table with price, 52W High/Low, P/E, market cap, volatility
* ETF ratio chart
* Fundamental snapshot
* Intrinsic value heatmap & margin of safety
* Technical charts (Candlestick + SMA/BB/RSI/MACD)
* Decision matrix & portfolio allocation
* Downloadable CSV report

---

## Dependencies

* `streamlit`
* `yfinance`
* `pandas`
* `numpy`
* `plotly`
* `matplotlib`

---

