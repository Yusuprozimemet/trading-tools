# Monte Carlo Portfolio Optimizer

A **Streamlit app** to construct and analyze investment portfolios using Monte Carlo simulations and the efficient frontier.

---

## 🚀 Overview

**App Title:** Monte Carlo Portfolio Optimization with Efficient Frontier  
**Purpose:** Find optimal portfolios, visualize risk-return trade-offs, and explore the efficient frontier.

- Simulates thousands of random portfolios
- Calculates expected return, risk (volatility), and Sharpe ratio
- Displays the **efficient frontier**
- Highlights **Max Sharpe Ratio** and **Min Volatility** portfolios
- Allows CSV download of all simulation results

---

## 🖥️ Features

### 1. Configuration (Sidebar)

Users can configure:

- **Stock tickers:** Comma-separated (e.g., `AAPL, MSFT, TSLA`)
- **Historical period:** 1–10 years
- **Number of simulations:** 1,000–50,000
- **Risk-free rate (%)**
- **Run Simulation** button

Default tickers: `AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, XOM, PG, KO, WMT`

---

### 2. Portfolio Calculations

**Key computations:**

- Daily returns and covariance matrix from historical data
- Portfolio metrics:
  - Expected annual return
  - Annualized risk (standard deviation)
  - Sharpe ratio
- Monte Carlo simulation of random portfolios
- Efficient frontier calculation
- Optimal portfolios:
  - Max Sharpe Ratio
  - Min Volatility

---

### 3. Interactive Plots

- **Efficient Frontier:** Red line represents optimal risk-return trade-offs
- **Simulated portfolios:** Colored by Sharpe ratio (Viridis scale)
- **Optimal points:**
  - **Gold star:** Maximum Sharpe ratio
  - **Green diamond:** Minimum volatility

Hover tooltips display risk, return, and Sharpe ratio.

---

### 4. Optimal Portfolios

Displays **two key portfolios**:

1. **Max Sharpe Ratio** (best risk-adjusted return)
2. **Min Volatility** (lowest risk portfolio)

For each portfolio:

- Return (%)
- Risk (%)
- Sharpe ratio
- Weight allocation per ticker (sorted descending)

---

### 5. CSV Download

Download a CSV containing:

- Risk (σ)
- Expected Return
- Sharpe ratio
- Portfolio weights for all tickers

---

## 📈 Quick Start

1. Enter stock tickers in the sidebar (comma-separated)
2. Set historical period, number of simulations, and risk-free rate
3. Click **Run Simulation**
4. Explore:
   - Efficient Frontier
   - Max Sharpe and Min Volatility portfolios
5. Download CSV with all simulations and weights

---

## ⚙️ Requirements

Python libraries:

```text
streamlit
numpy
pandas
yfinance
plotly
scipy
````

---

## 🔗 How to Run

```bash
streamlit run monte_carlo_app.py
```

---

## 📝 Notes

* Data is fetched from Yahoo Finance using `Adj Close` if available
* Portfolios are constrained to **weights between 0%–100%** and sum to 100%
* Efficient frontier traces minimum-risk portfolios for target returns
* Monte Carlo results include simulated portfolio cloud for visualization

```

---

