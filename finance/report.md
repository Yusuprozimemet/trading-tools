# Dutch Financial Markets Report – Streamlit App

A **Streamlit app** providing real-time analysis of Dutch financial markets, including AEX stocks, news, and market trends.

---

## Features

- **Stock Metrics**:
  - Fetches stock data for Dutch/Euronext tickers.
  - Calculates key metrics: Current Price, Market Cap, P/E Ratio, Dividend Yield, Volatility, Beta, 52-week High/Low, Moving Averages (MA50, MA200).
  - Displays top stocks by Market Cap.

- **Sector Analysis**:
  - Aggregates market cap, dividend yield, volatility, and stock count by sector.
  - Visualizes sector market share with interactive Plotly pie chart.

- **Correlation Analysis**:
  - Computes 1-year price correlation matrix between selected stocks.
  - Heatmap visualization for identifying co-movements.

- **News Feed**:
  - Scrapes Dutch financial and political news from Google News RSS feeds.
  - Filters by keywords, lookback hours, and max results per query.
  - Expandable news items with summary and link to the full article.

- **Export Options**:
  - Download stock metrics and news feed as CSV files.
  
- **Interactive Sidebar**:
  - Configure tickers, news queries, lookback hours, max news per query.
  - Generate report manually or enable auto-refresh every 5 minutes.

---

## Installation

Install required packages:

```bash
pip install streamlit pandas numpy yfinance feedparser plotly pytz python-dateutil
````

Run the app:

```bash
streamlit run financialReport.py
```

---

## Usage

### Sidebar Configuration

1. **Tickers**: Enter Dutch/Euronext stock symbols (`.AS` suffix).
2. **News Keywords**: Comma-separated terms for Dutch financial or political news.
3. **News Lookback (hours)**: How far back to fetch news articles.
4. **Max News per Query**: Limit the number of news items per keyword.
5. **Generate Report**: Fetch latest stock data and news.
6. **Auto-refresh**: Optional 5-minute automatic updates.

---

### Main App Sections

1. **Market Overview**:

   * Displays metrics such as total stocks, total market cap, average volatility, and news count.

2. **Top 10 by Market Cap**:

   * Shows the largest Dutch stocks with key financial indicators.

3. **Sector Analysis**:

   * Sector-level summary table and pie chart visualization.

4. **Price Correlation Matrix**:

   * Heatmap showing correlations between stock price changes over 1 year.

5. **Dutch Financial & Political News**:

   * News articles relevant to selected keywords, sorted by published time.

6. **Export Report**:

   * Download CSVs for stock metrics and news feed.

---

### Sample Metrics Table

| Ticker  | Company      | Price € | Market Cap €B | P/E  | Div Yield % | Volatility % | Beta | 52W High | 52W Low | MA50  | MA200 |
| ------- | ------------ | ------- | ------------- | ---- | ----------- | ------------ | ---- | -------- | ------- | ----- | ----- |
| ASML.AS | ASML Holding | 720.50  | 300.00        | 35.5 | 0.8         | 22.5         | 1.10 | 750.00   | 600.00  | 710.0 | 695.0 |

---

### Notes

* **Data Sources**:

  * Stock data: Yahoo Finance
  * News: Google News RSS
* **Time Zone**: CET (Central European Time)
* **Caching**:

  * Stock data cached for 1 hour
  * News data cached for 30 minutes
* Only tickers with valid Yahoo Finance data are displayed.

---

### Footer

Generated with real-time data:

```
Data: Yahoo Finance • News: Google News RSS • Generated: <current date and time> CET
```

---

## Quick Start

1. Open the app:

```bash
streamlit run financialReport.py
```

2. Configure your tickers, news keywords, and parameters in the sidebar.
3. Click **Generate Report** to view metrics, visualizations, and news feed.
4. Export CSV reports if needed.

```
