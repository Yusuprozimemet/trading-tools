
# Portfolio Tracker Streamlit App

A **Streamlit app** for tracking portfolio holdings, transactions, and dividends.  
The app provides summary statistics, visualizations, and allows exporting the portfolio to Excel.

---

## Features

- View **sample holdings**, transactions, and dividends.
- Compute **summary statistics**:
  - Total Portfolio Value
  - Total Cost Basis
  - Total Unrealized Gain/Loss
  - Total Dividends Received
- Visualize **allocation by asset type** using an interactive Plotly pie chart.
- **Export portfolio** data to an Excel workbook with multiple sheets:
  - Holdings
  - Transactions
  - Dividends
  - Summary
- Lightweight and user-friendly interface with Streamlit.

---

## Installation

Install required packages:

```bash
pip install streamlit pandas plotly
````

Run the app:

```bash
streamlit run portofolio/portofolio.py
```

---

## Usage

### Sidebar Configuration

* **Use sample data**: Checkbox to load pre-defined sample portfolio.
* **Run**: Button to generate the portfolio view.

---

### Main App Sections

1. **Holdings**
   Displays a table of portfolio holdings with:

   * Ticker, Asset Name, Asset Type
   * Shares Owned, Purchase Price, Current Price
   * Total Cost Basis, Current Value, Unrealized Gain/Loss
   * Dividends Received, Sector, Brokerage, Currency

2. **Summary**
   Provides key metrics about the portfolio.

3. **Transactions**
   Shows all recorded buy/sell transactions including:

   * Date, Ticker, Transaction Type
   * Quantity, Price per Share, Total Amount
   * Fees and Notes

4. **Dividends**
   Displays dividend records:

   * Date, Ticker, Amount, Reinvested, Tax Withheld

5. **Allocation**
   Pie chart showing allocation by **Asset Type**.

6. **Export**
   Download the portfolio as an Excel file with multiple sheets.

---

### Sample Data

| Ticker  | Asset Name       | Asset Type | Shares Owned | Purchase Price | Current Price |
| ------- | ---------------- | ---------- | ------------ | -------------- | ------------- |
| AAPL    | Apple Inc.       | Stock      | 100          | 150.00         | 220.00        |
| MSFT    | Microsoft Corp.  | Stock      | 50           | 300.00         | 350.00        |
| SPY     | SPDR S&P 500 ETF | ETF        | 200          | 400.00         | 450.00        |
| BTC-USD | Bitcoin          | Crypto     | 0.5          | 40000.00       | 60000.00      |

---

## Notes

* **Excel Export**: Uses `xlsxwriter` or `openpyxl` if installed, otherwise falls back to CSV.
* Only **sample data** is supported in this lightweight version.
* Plotly charts are interactive and responsive.
* Streamlit is required to run the app.

---

## Authors

* Converted from Jupyter Notebook: `TradingBasics/portofolio_tracker.ipynb`.
* Provides a quick UI for portfolio tracking and visualization.

---

## Quick Start

1. Open the app:

```bash
streamlit run portofolio/portofolio.py
```

2. Use the sidebar to **load sample data** and click **Run**.
3. Explore holdings, transactions, dividends, allocation pie chart, and download the Excel report.

```
