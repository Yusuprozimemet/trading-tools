import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------
st.set_page_config(page_title="Momentum Trading Dashboard", layout="wide")
st.title("Momentum Trading – From Theory to Signals")
st.markdown(
    """
    **Learn momentum, see the indicators, and get live buy/sell signals**
    All data is fetched exactly like the BB-RSI app (`group_by='ticker'`).
    """
)

# --------------------------------------------------------------
# Sidebar – user inputs
# --------------------------------------------------------------
st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="GME, AMC, AAPL"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

years_back = st.sidebar.slider("Years of history", 1, 5, 2)
end_date = datetime.today()
start_date = end_date - timedelta(days=years_back * 365)

roc_window = st.sidebar.slider("ROC window (days)", 5, 30, 12)
stoch_k = st.sidebar.slider("%K period", 5, 30, 14)
stoch_d = st.sidebar.slider("%D period (SMA of %K)", 1, 10, 3)
rsi_window = st.sidebar.slider("RSI window", 5, 30, 14)
macd_fast = st.sidebar.slider("MACD fast EMA", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD slow EMA", 20, 50, 26)
macd_signal = st.sidebar.slider("MACD signal EMA", 5, 20, 9)

run = st.sidebar.button("Run Analysis", type="primary")

# --------------------------------------------------------------
# Robust data download (same as bbrsi.py)
# --------------------------------------------------------------


@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        group_by='ticker',
        auto_adjust=False
    )
    if raw is None or raw.empty:
        st.error("No data returned from Yahoo Finance.")
        return None

    price_dict = {}
    for t in tickers:
        df = raw if len(tickers) == 1 else raw[t]
        if 'Adj Close' in df.columns:
            price_dict[t] = df['Adj Close']
        elif 'Close' in df.columns:
            price_dict[t] = df['Close']
            st.warning(f"Using un-adjusted Close for {t}")
        else:
            st.warning(f"No price column for {t}")
    if not price_dict:
        return None
    prices = pd.DataFrame(price_dict).dropna(how='all')
    return prices if not prices.empty else None


# --------------------------------------------------------------
# Indicator calculations
# --------------------------------------------------------------
def add_indicators(df):
    for col in df.columns:
        p = df[col]

        # ---- basic momentum -------------------------------------------------
        df[f"{col}_ROC"] = p.pct_change(periods=roc_window)
        df[f"{col}_PctChg"] = p.pct_change()

        # ---- Z-Score of daily % change --------------------------------------
        daily_chg = p.pct_change()
        df[f"{col}_Z"] = (daily_chg - daily_chg.mean()) / daily_chg.std()

        # ---- Stochastic ------------------------------------------------------
        low_min = p.rolling(window=stoch_k).min()
        high_max = p.rolling(window=stoch_k).max()
        df[f"{col}_%K"] = 100 * (p - low_min) / (high_max - low_min)
        df[f"{col}_%D"] = df[f"{col}_%K"].rolling(window=stoch_d).mean()

        # ---- RSI -------------------------------------------------------------
        delta = p.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window=rsi_window).mean()
        ma_down = down.rolling(window=rsi_window).mean()
        rs = ma_up / ma_down
        df[f"{col}_RSI"] = 100 - (100 / (1 + rs))

        # ---- MACD ------------------------------------------------------------
        ema_fast = p.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = p.ewm(span=macd_slow, adjust=False).mean()
        df[f"{col}_MACD"] = ema_fast - ema_slow
        df[f"{col}_Signal"] = df[f"{col}_MACD"].ewm(
            span=macd_signal, adjust=False).mean()
        df[f"{col}_Hist"] = df[f"{col}_MACD"] - df[f"{col}_Signal"]

        # ---- Signals ---------------------------------------------------------
        df[f"{col}_Buy"] = (df[f"{col}_MACD"] > df[f"{col}_Signal"]) & (
            df[f"{col}_MACD"].shift(1) <= df[f"{col}_Signal"].shift(1)
        )
        df[f"{col}_Sell"] = (df[f"{col}_MACD"] < df[f"{col}_Signal"]) & (
            df[f"{col}_MACD"].shift(1) >= df[f"{col}_Signal"].shift(1)
        )
    return df


# --------------------------------------------------------------
# Back-test helper
# --------------------------------------------------------------
def backtest_signals(df, ticker):
    price = df[ticker]
    buy = df[f"{ticker}_Buy"]
    sell = df[f"{ticker}_Sell"]

    position = 0
    entry_price = 0
    trades = []

    for date, row in df.iterrows():
        if position == 0 and row[f"{ticker}_Buy"]:
            position = 1
            entry_price = row[ticker]
            trades.append({"Date": date, "Type": "BUY", "Price": entry_price})
        elif position == 1 and row[f"{ticker}_Sell"]:
            position = 0
            exit_price = row[ticker]
            ret = (exit_price - entry_price) / entry_price
            trades.append({"Date": date, "Type": "SELL",
                          "Price": exit_price, "Return": ret})
    return pd.DataFrame(trades)


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------
if run:
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    with st.spinner("Downloading data …"):
        prices = fetch_data(tickers, start_date, end_date)
    if prices is None:
        st.stop()

    df = add_indicators(prices.copy())

    # --------------------------------------------------------------
    # Educational markdown (exactly what you supplied)
    # --------------------------------------------------------------
    st.markdown("---")
    st.header("Momentum Trading – Theory")
    st.markdown(
        """
        ## 1. What is Momentum Trading?
        - It's like catching a wave in surfing. You try to ride a stock's price movement as it's going up or down quickly.
        - The idea is to **buy stocks that are already going up**, hoping they'll continue to rise.
        - Or **sell stocks that are falling**, expecting them to keep falling.

        ## 2. The Basic Idea
        > **"Buy high, sell higher."** – follow the trend, not fight it.

        ## 3. Tools We Use
        | Indicator | What it tells you |
        |-----------|-------------------|
        | **ROC** | Speed of price change |
        | **% Change** | Daily move |
        | **Z-Score** | How extreme the move is |
        | **Stochastic (%K / %D)** | Overbought / oversold |
        | **RSI** | Overbought (>70) / oversold (<30) |
        | **MACD** | Trend strength + crossovers |

        ## 4. Risks
        - Trends can reverse **suddenly**.
        - You can buy at the top or sell at the bottom.
        - **Always use stop-losses** and never risk more than you can lose.
        """
    )

    # --------------------------------------------------------------
    # Interactive chart per ticker
    # --------------------------------------------------------------
    for ticker in tickers:
        st.subheader(f"{ticker} – Price + All Indicators")
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "Price", "ROC & % Change", "Stochastic", "RSI", "MACD"
            ),
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
        )

        # 1. Price + buy/sell markers
        fig.add_trace(go.Scatter(x=df.index, y=df[ticker],
                                 name="Price", line=dict(color="royalblue")), row=1, col=1)
        buys = df[df[f"{ticker}_Buy"]]
        sells = df[df[f"{ticker}_Sell"]]
        fig.add_trace(go.Scatter(x=buys.index, y=buys[ticker],
                                 mode="markers", name="Buy",
                                 marker=dict(symbol="triangle-up", size=12, color="lime")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells[ticker],
                                 mode="markers", name="Sell",
                                 marker=dict(symbol="triangle-down", size=12, color="red")), row=1, col=1)

        # 2. ROC & % Change
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_ROC"],
                                 name="ROC", line=dict(color="orange")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_PctChg"],
                                 name="%Chg", line=dict(color="purple")), row=2, col=1)

        # 3. Stochastic
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_%K"],
                                 name="%K", line=dict(color="cyan")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_%D"],
                                 name="%D", line=dict(color="magenta")), row=3, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

        # 4. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_RSI"],
                                 name="RSI", line=dict(color="gold")), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

        # 5. MACD
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_MACD"],
                                 name="MACD", line=dict(color="blue")), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_Signal"],
                                 name="Signal", line=dict(color="red")), row=5, col=1)
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[f"{ticker}_Hist"],
                name="Histogram",
                marker_color=np.where(
                    df[f"{ticker}_Hist"] > 0, "green",
                    np.where(df[f"{ticker}_Hist"] < 0, "red", "gray")
                )
            ),
            row=5, col=1
        )

        fig.update_layout(height=900, showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------
        # Back-test summary
        # --------------------------------------------------------------
        trades = backtest_signals(df, ticker)
        if not trades.empty:
            returns = trades[trades["Type"] == "SELL"]["Return"]
            win_rate = (returns > 0).mean() * 100 if not returns.empty else 0
            total_ret = ((1 + returns).prod() - 1) * \
                100 if not returns.empty else 0
            n_trades = len(returns)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trades", n_trades)
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Total Return", f"{total_ret:.2f}%")
            col4.metric(
                "Avg Trade", f"{returns.mean()*100:.2f}%" if n_trades else "—")

            # Show trade log
            with st.expander("Trade Log"):
                st.dataframe(trades.style.format(
                    {"Price": "${:,.2f}", "Return": "{:.2%}"}))

    # --------------------------------------------------------------
    # CSV download of *all* signals
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Download Full Signal Table")
    signal_cols = [c for c in df.columns if any(
        x in c for x in ["Buy", "Sell", "MACD", "Hist", "RSI", "%K"])]
    export = df[signal_cols].copy()
    export = export.round(4)
    csv = export.to_csv().encode()
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"momentum_signals_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv"
    )

else:
    st.info("Enter tickers and click **Run Analysis** to start.")
    st.markdown(
        """
        ### Quick Start
        1. **Type tickers** – e.g. `GME, AMC, TSLA`
        2. Adjust **look-back windows** (ROC, Stochastic, RSI, MACD)
        3. Press **Run Analysis**

        You’ll see:
        * All classic momentum indicators
        * **Buy** (green triangle-up) / **Sell** (red triangle-down) signals from MACD crossovers
        * A tiny back-test summary per ticker
        * Full CSV export
        """
    )
