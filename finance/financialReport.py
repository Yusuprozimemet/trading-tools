# financialReport.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from dateutil import parser
import os
import base64
from typing import List, Dict, Any

# --------------------------------------------------------------
# Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="Dutch Financial Report",
    layout="wide",
    page_icon="Chart",
    initial_sidebar_state="expanded"
)

st.title("Dutch Financial Markets Report")
st.markdown("**Real-time analysis of AEX stocks, news, and market trends**")

# --------------------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------------------
st.sidebar.header("Report Settings")

# Tickers (Dutch/Euronext focus)
default_tickers = [
    'ASML.AS', 'INGA.AS', 'PRX.AS', 'ADYEN.AS', 'UMG.AS',
    'HEIA.AS', 'KPN.AS', 'PHIA.AS', 'AKZA.AS', 'MT.AS',
    'WKL.AS', 'BESI.AS', 'TKWY.AS', 'EXO.AS', 'AGN.AS'
]

tickers_input = st.sidebar.text_area(
    "Tickers (one per line)",
    value="\n".join(default_tickers),
    height=150,
    help="Use .AS for Euronext Amsterdam"
)
tickers = [t.strip() for t in tickers_input.split("\n") if t.strip()]

news_queries = st.sidebar.text_input(
    "News Keywords",
    value="asielzoekers, nederlandse economie, AEX, inflatie, ECB",
    help="Comma-separated Dutch financial/political terms"
).split(",")
news_queries = [q.strip() for q in news_queries if q.strip()]

hours_ago = st.sidebar.slider("News Lookback (hours)", 1, 72, 24)
max_news_per_query = st.sidebar.slider("Max News per Query", 3, 15, 5)

refresh = st.sidebar.button("Generate Report", type="primary")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 min", value=False)

if auto_refresh:
    st.sidebar.info("Auto-refresh enabled. Report updates every 5 minutes.")
    time.sleep(300)
    st.rerun()

# --------------------------------------------------------------
# Caching Functions
# --------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(tickers: List[str]) -> Dict[str, Any]:
    data = {}
    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            if hist.empty:
                continue

            close = hist['Close']
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

            data[ticker] = {
                'info': info,
                'hist': hist,
                'current_price': close.iloc[-1],
                'ma50': close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan,
                'ma200': close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan,
                'volatility': volatility,
                'year_high': hist['High'].max(),
                'year_low': hist['Low'].min(),
                'volume_avg': hist['Volume'].mean(),
                'beta': info.get('beta', np.nan),
                'pe': info.get('trailingPE', np.nan),
                'div_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'market_cap': info.get('marketCap', 0) / 1e9,
                'sector': info.get('sector', 'Unknown'),
                'company_name': info.get('longName', ticker),
            }
        except:
            pass
        progress.progress((i + 1) / len(tickers))
    return data


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_dutch_news(queries: List[str], hours: int, max_per_query: int) -> pd.DataFrame:
    base_url = "https://news.google.com/rss"
    tz = pytz.timezone('Europe/Amsterdam')
    news_items = []

    for query in queries:
        try:
            url = f"{base_url}/search?q={query.replace(' ', '+')}&hl=nl&gl=NL&ceid=NL:nl"
            feed = feedparser.parse(url)

            count = 0
            for entry in feed.entries:
                if count >= max_per_query:
                    break
                try:
                    pub_dt = parser.parse(entry.published)
                    if pub_dt.tzinfo is None:
                        pub_dt = pytz.utc.localize(pub_dt)
                    pub_dt = pub_dt.astimezone(tz)

                    if (datetime.now(tz) - pub_dt).total_seconds() > hours * 3600:
                        continue

                    news_items.append({
                        'query': query,
                        'title': entry.title,
                        'source': entry.source.title if hasattr(entry, 'source') else 'Unknown',
                        'link': entry.link,
                        'published': pub_dt.strftime('%Y-%m-%d %H:%M'),
                        'summary': entry.summary if hasattr(entry, 'summary') else ''
                    })
                    count += 1
                except:
                    continue
        except:
            continue

    return pd.DataFrame(news_items).sort_values('published', ascending=False)


# --------------------------------------------------------------
# Generate Report
# --------------------------------------------------------------
if refresh or 'first_run' not in st.session_state:
    st.session_state.first_run = True

    with st.spinner("Fetching stock data..."):
        stock_data = fetch_stock_data(tickers)

    with st.spinner("Scraping Dutch news..."):
        news_df = fetch_dutch_news(news_queries, hours_ago, max_news_per_query)

    if not stock_data:
        st.error("No stock data fetched. Check ticker symbols.")
        st.stop()

    # --------------------------------------------------------------
    # Metrics DataFrame
    # --------------------------------------------------------------
    metrics = []
    for t, d in stock_data.items():
        metrics.append({
            'Ticker': t,
            'Company': d['company_name'],
            'Sector': d['sector'],
            'Price €': round(d['current_price'], 2),
            'Market Cap €B': round(d['market_cap'], 2),
            'P/E': round(d['pe'], 2) if d['pe'] else 'N/A',
            'Div Yield %': round(d['div_yield'], 2),
            'Volatility %': round(d['volatility'] * 100, 2),
            'Beta': round(d['beta'], 2) if d['beta'] else 'N/A',
            '52W High': round(d['year_high'], 2),
            '52W Low': round(d['year_low'], 2),
            'MA50': round(d['ma50'], 2) if not np.isnan(d['ma50']) else 'N/A',
            'MA200': round(d['ma200'], 2) if not np.isnan(d['ma200']) else 'N/A',
        })

    df_metrics = pd.DataFrame(metrics).sort_values(
        'Market Cap €B', ascending=False)

    # --------------------------------------------------------------
    # Display Report
    # --------------------------------------------------------------
    st.markdown("---")
    st.header(
        f"Market Overview – {datetime.now().strftime('%d %B %Y, %H:%M')} CET")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AEX Giants", len(df_metrics))
    with col2:
        total_cap = df_metrics['Market Cap €B'].sum()
        st.metric("Total Market Cap", f"€{total_cap:,.0f}B")
    with col3:
        avg_vol = df_metrics['Volatility %'].mean()
        st.metric("Avg Volatility", f"{avg_vol:.1f}%")
    with col4:
        st.metric("News Items", len(news_df))

    # --------------------------------------------------------------
    # Top 10 Table
    # --------------------------------------------------------------
    st.subheader("Top 10 by Market Cap")
    st.dataframe(
        df_metrics.head(10).style.format({
            'Price €': '€{:.2f}',
            'Market Cap €B': '€{:.2f}B',
            'Div Yield %': '{:.2f}%',
            'Volatility %': '{:.1f}%',
        }),
        use_container_width=True
    )

    # --------------------------------------------------------------
    # Sector Summary
    # --------------------------------------------------------------
    st.subheader("Sector Analysis")
    sector_df = df_metrics.groupby('Sector').agg({
        'Market Cap €B': 'sum',
        'Div Yield %': 'mean',
        'Volatility %': 'mean',
        'Ticker': 'count'
    }).round(2).rename(columns={'Ticker': 'Count'})
    sector_df = sector_df.sort_values('Market Cap €B', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(sector_df, use_container_width=True)
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=sector_df.index,
            values=sector_df['Market Cap €B'],
            textinfo='label+percent',
            hole=0.3
        )])
        fig.update_layout(title="Market Cap by Sector", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------------
    st.subheader("Price Correlation Matrix")
    prices = pd.DataFrame({t: d['hist']['Close']
                          for t, d in stock_data.items()})
    corr = prices.pct_change().corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10}
    ))
    fig.update_layout(height=600, title="Stock Price Correlations (1Y)")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------
    # News Feed
    # --------------------------------------------------------------
    st.subheader("Dutch Financial & Political News")
    if not news_df.empty:
        for _, row in news_df.iterrows():
            with st.expander(f"**{row['title']}** – {row['source']} • {row['published']}"):
                st.caption(row['summary'])
                st.markdown(f"[Read more]({row['link']})")
    else:
        st.info("No recent news found for the selected queries.")

    # --------------------------------------------------------------
    # Export Options
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Export Report")

    col1, col2 = st.columns(2)
    with col1:
        csv_metrics = df_metrics.to_csv(index=False).encode()
        st.download_button(
            "Download Stock Metrics (CSV)",
            data=csv_metrics,
            file_name=f"dutch_stocks_{datetime.now():%Y%m%d}.csv",
            mime="text/csv"
        )
    with col2:
        csv_news = news_df.to_csv(index=False).encode()
        st.download_button(
            "Download News Feed (CSV)",
            data=csv_news,
            file_name=f"dutch_news_{datetime.now():%Y%m%d}.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------------
    # Footer
    # --------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Data: Yahoo Finance • News: Google News RSS • "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CET"
    )

else:
    st.info("Click **Generate Report** to load the latest data.")
