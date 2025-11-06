import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Page configuration
st.set_page_config(page_title="Pair Trading Analysis", layout="wide")

# Title
st.title("📊 Comprehensive Pair Trading Analysis")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Pair Discovery", "Pair Analysis & Trading"]
)

if mode == "Pair Discovery":
    st.header("🔍 Pair Discovery")

    # Input for tickers
    default_tickers = ['ASML.AS', 'INGA.AS', 'PRX.AS', 'ADYEN.AS', 'ASM.AS',
                       'DSFIR.AS', 'WKL.AS', 'AD.AS', 'UMG.AS', 'HEIA.AS',
                       'MT.AS', 'NN.AS', 'KPN.AS', 'PHIA.AS', 'AKZA.AS',
                       'BESI.AS', 'AGN.AS', 'ARCAD.AS']

    tickers_input = st.sidebar.text_area(
        "Enter tickers (comma-separated)",
        value=", ".join(default_tickers)
    )

    tickers = [t.strip() for t in tickers_input.split(",")]

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2023, 10, 20))

    significance = st.sidebar.slider(
        "Cointegration Significance", 0.01, 0.10, 0.05, 0.01)

    if st.sidebar.button("Find Pairs"):
        with st.spinner("Fetching data and analyzing pairs..."):
            # Fetch data
            price_data = {}
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist_data = stock.history(start=start_date, end=end_date)
                    if hist_data is not None and not hist_data.empty:
                        price_data[ticker] = hist_data['Close']
                except Exception as e:
                    st.warning(f"Error fetching {ticker}: {str(e)}")

            # Find pairs
            n = len(price_data)
            keys = list(price_data.keys())
            pairs = []
            correlations = {}

            for i in range(n):
                for j in range(i+1, n):
                    stock1 = price_data[keys[i]]
                    stock2 = price_data[keys[j]]

                    # Align the two series on common dates to avoid different-length arrays
                    s1, s2 = stock1.align(stock2, join='inner')
                    s1 = s1.dropna()
                    s2 = s2.dropna()

                    if len(s1) > 0 and len(s2) > 0:
                        correlation = s1.corr(s2)
                        correlations[(keys[i], keys[j])] = correlation

                        # Check cointegration (use aligned series). Wrap in try/except to be robust.
                        try:
                            _, p_value, _ = coint(s1, s2)
                        except Exception as e:
                            logging.warning(
                                f"Cointegration test failed for {keys[i]} and {keys[j]}: {e}")
                            continue

                        if p_value < significance:
                            spread = s1 - s2 * (s1.mean() / s2.mean())
                            adf_result = adfuller(spread.dropna())
                            if adf_result[1] < 0.05:
                                pairs.append({
                                    'Stock 1': keys[i],
                                    'Stock 2': keys[j],
                                    'Correlation': correlation,
                                    'Cointegration p-value': p_value,
                                    'ADF p-value': adf_result[1]
                                })

            if pairs:
                st.success(f"Found {len(pairs)} cointegrated pairs!")
                pairs_df = pd.DataFrame(pairs)
                st.dataframe(pairs_df, use_container_width=True)

                # Visualize top pairs immediately (no click required)
                for idx, pair in enumerate(pairs[:5]):  # Show top 5 pairs
                    stock1 = pair['Stock 1']
                    stock2 = pair['Stock 2']

                    try:
                        # Align on common dates before normalizing/plotting
                        s1, s2 = price_data[stock1].align(
                            price_data[stock2], join='inner')
                        s1 = s1.dropna()
                        s2 = s2.dropna()

                        if len(s1) == 0 or len(s2) == 0:
                            st.warning(
                                f"Not enough overlapping data to plot {stock1} vs {stock2}.")
                            continue

                        # Avoid division by zero if series is constant
                        if s1.max() == s1.min() or s2.max() == s2.min():
                            st.warning(
                                f"One of the series for {stock1} or {stock2} is constant; cannot normalize.")
                            continue

                        norm_stock1 = (s1 - s1.min()) / (s1.max() - s1.min())
                        norm_stock2 = (s2 - s2.min()) / (s2.max() - s2.min())

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=norm_stock1.index, y=norm_stock1, name=stock1))
                        fig.add_trace(go.Scatter(
                            x=norm_stock2.index, y=norm_stock2, name=stock2))
                        fig.update_layout(
                            title=f"Normalized Prices: {stock1} vs {stock2} (Corr: {pair['Correlation']:.2f})",
                            xaxis_title="Date",
                            yaxis_title="Normalized Price"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        logging.exception(
                            f"Failed to plot normalized chart for {stock1} vs {stock2}: {e}")
                        st.warning(
                            f"Could not plot normalized chart for {stock1} vs {stock2}: {e}")
                        continue
            else:
                st.warning(
                    "No cointegrated pairs found with the given parameters.")

else:  # Pair Analysis & Trading
    st.header("💹 Pair Analysis & Trading Strategy")

    # Input for pair
    col1, col2 = st.sidebar.columns(2)
    with col1:
        stock1 = st.text_input("Stock 1", value="AGN.AS")
    with col2:
        stock2 = st.text_input("Stock 2", value="ASML.AS")

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 6, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    # Trading parameters
    st.sidebar.subheader("Trading Parameters")
    initial_balance = st.sidebar.number_input(
        "Initial Balance (€)", value=1000, step=100)
    risk_tolerance = st.sidebar.slider("Risk Tolerance (%)", 1, 10, 2) / 100
    max_loss_per_share = st.sidebar.number_input(
        "Max Loss per Share (€)", value=0.5, step=0.1)

    # Bollinger Bands parameters
    bb_window = st.sidebar.number_input(
        "Bollinger Bands Window", value=20, step=1)
    bb_std = st.sidebar.number_input(
        "Bollinger Bands Std Dev", value=2.0, step=0.1)

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing pair..."):
            # Fetch data
            tickers = [stock1, stock2]
            raw = yf.download(tickers, start=start_date, end=end_date)

            # Helper to extract adjusted close regardless of yfinance's returned layout
            def _extract_adj_close(df, tickers_list):
                # Single series (one ticker)
                if isinstance(df, pd.Series):
                    return df.to_frame(name=tickers_list[0])

                # MultiIndex columns (common case for multiple tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    # Typical layout: top-level ['Open','High','Low','Close','Adj Close','Volume']
                    if 'Adj Close' in df.columns.get_level_values(0):
                        return df['Adj Close']

                    # Sometimes the levels are reversed or labeled differently; search columns
                    cols = [col for col in df.columns if (
                        len(col) > 1 and col[1] == 'Adj Close')]
                    if cols:
                        return pd.DataFrame({col[0]: df[col] for col in cols})

                # Single-level columns where 'Adj Close' is a column
                if 'Adj Close' in df.columns:
                    return df['Adj Close']

                # Fallback: if the dataframe already uses tickers as columns (already adjusted/close)
                found = [t for t in tickers_list if t in df.columns]
                if found:
                    return df[found]

                # Try 'Close' as another fallback
                if isinstance(df.columns, pd.MultiIndex) and 'Close' in df.columns.get_level_values(0):
                    return df['Close']
                if 'Close' in df.columns:
                    return df['Close']

                raise KeyError(
                    "'Adj Close' not found in downloaded data. Available columns: {}".format(df.columns))

            try:
                data = _extract_adj_close(raw, tickers)
            except KeyError as e:
                st.error(
                    f"Could not find adjusted close in downloaded data: {e}")
                # Stop processing this run
                st.stop()

            data = data.dropna()

            if data.empty:
                st.error(
                    "No data available for the selected stocks and date range.")
            else:
                # Calculate hedge ratio
                def calculate_hedge_ratio(y, x):
                    x_with_const = sm.add_constant(x)
                    model = sm.OLS(y, x_with_const).fit()
                    return model.params[1]

                hedge_ratio = calculate_hedge_ratio(data[stock1], data[stock2])

                # Calculate spread
                spread = data[stock1] - data[stock2] * hedge_ratio

                # Calculate Bollinger Bands
                def calculate_bollinger_bands(series, window=20, num_std=2):
                    rolling_mean = series.rolling(window=window).mean()
                    rolling_std = series.rolling(window=window).std()
                    upper_band = rolling_mean + (rolling_std * num_std)
                    lower_band = rolling_mean - (rolling_std * num_std)
                    return rolling_mean, upper_band, lower_band

                middle_band, upper_band, lower_band = calculate_bollinger_bands(
                    spread, window=bb_window, num_std=bb_std)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
                with col2:
                    correlation = data[stock1].corr(data[stock2])
                    st.metric("Correlation", f"{correlation:.4f}")
                with col3:
                    _, coint_pvalue, _ = coint(data[stock1], data[stock2])
                    st.metric("Cointegration p-value", f"{coint_pvalue:.4f}")
                with col4:
                    adf_result = adfuller(spread.dropna())
                    st.metric("ADF p-value", f"{adf_result[1]:.4f}")

                # Find trade points
                def find_trade_points(spread, lower_band, middle_band, upper_band):
                    entry_points = []
                    exit_points = []

                    for i in range(1, len(spread)):
                        # Long entry
                        if spread.iloc[i] < lower_band.iloc[i] and spread.iloc[i-1] >= lower_band.iloc[i-1]:
                            entry_points.append(
                                (spread.index[i], spread.iloc[i], 'long'))

                        # Long exit
                        if entry_points and spread.iloc[i] > middle_band.iloc[i]:
                            exit_points.append(
                                (spread.index[i], spread.iloc[i], 'long'))

                        # Short entry
                        if spread.iloc[i] > upper_band.iloc[i] and spread.iloc[i-1] <= upper_band.iloc[i-1]:
                            entry_points.append(
                                (spread.index[i], spread.iloc[i], 'short'))

                        # Short exit
                        if exit_points and spread.iloc[i] < middle_band.iloc[i]:
                            exit_points.append(
                                (spread.index[i], spread.iloc[i], 'short'))

                    return entry_points, exit_points

                entry_points, exit_points = find_trade_points(
                    spread, lower_band, middle_band, upper_band)

                # Plot spread with Bollinger Bands
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=spread,
                              name='Spread', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=middle_band,
                              name='Middle Band', line=dict(dash='dash', color='orange')))
                fig.add_trace(go.Scatter(x=data.index, y=upper_band,
                              name='Upper Band', line=dict(dash='dash', color='green')))
                fig.add_trace(go.Scatter(x=data.index, y=lower_band,
                              name='Lower Band', line=dict(dash='dash', color='red')))

                # Add entry/exit markers
                for entry_date, entry_val, entry_type in entry_points:
                    color = 'green' if entry_type == 'long' else 'red'
                    fig.add_trace(go.Scatter(
                        x=[entry_date], y=[entry_val],
                        mode='markers',
                        marker=dict(color=color, size=10,
                                    symbol='triangle-up'),
                        name=f'{entry_type.capitalize()} Entry',
                        showlegend=False
                    ))

                for exit_date, exit_val, exit_type in exit_points:
                    color = 'blue' if exit_type == 'long' else 'orange'
                    fig.add_trace(go.Scatter(
                        x=[exit_date], y=[exit_val],
                        mode='markers',
                        marker=dict(color=color, size=10, symbol='circle'),
                        name=f'{exit_type.capitalize()} Exit',
                        showlegend=False
                    ))

                fig.update_layout(
                    title=f"Spread with Bollinger Bands: {stock1} vs {stock2}",
                    xaxis_title="Date",
                    yaxis_title="Spread Value",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Simulate trading strategy
                entry_dict = {date: entry_type for date,
                              _, entry_type in entry_points}
                exit_dict = {date: exit_type for date,
                             _, exit_type in exit_points}

                portfolio_values = []
                trades_log = []
                balance = initial_balance
                shares = 0

                for date in data.index:
                    if date in entry_dict:
                        entry_type = entry_dict[date]
                        entry_price = data[stock1][date]
                        shares = int((balance * risk_tolerance) /
                                     max_loss_per_share)

                        if entry_type == 'long':
                            balance -= shares * entry_price
                            trades_log.append(
                                (date, 'BUY', shares, entry_price, balance))
                        elif entry_type == 'short':
                            balance += shares * entry_price
                            trades_log.append(
                                (date, 'SELL', shares, entry_price, balance))

                    if date in exit_dict:
                        exit_type = exit_dict[date]
                        exit_price = data[stock1][date]

                        if exit_type == 'long':
                            balance += shares * exit_price
                            trades_log.append(
                                (date, 'SELL', shares, exit_price, balance))
                            shares = 0
                        elif exit_type == 'short':
                            balance -= shares * exit_price
                            trades_log.append(
                                (date, 'BUY', shares, exit_price, balance))
                            shares = 0

                    total_value = balance + shares * \
                        data[stock1][date] if shares > 0 else balance
                    portfolio_values.append(total_value)

                # Plot portfolio value
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=data.index,
                    y=portfolio_values,
                    name='Portfolio Value',
                    line=dict(color='blue')
                ))
                fig_portfolio.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (€)",
                    height=400
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)

                # Display results
                final_value = portfolio_values[-1]
                total_return = (
                    (final_value - initial_balance) / initial_balance) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Balance", f"€{initial_balance:.2f}")
                with col2:
                    st.metric("Final Portfolio Value", f"€{final_value:.2f}")
                with col3:
                    st.metric("Total Return", f"{total_return:.2f}%",
                              delta=f"€{final_value - initial_balance:.2f}")

                # Trade log
                if trades_log:
                    st.subheader("Trade Log")
                    trades_df = pd.DataFrame(trades_log, columns=[
                                             'Date', 'Action', 'Shares', 'Price (€)', 'Balance (€)'])
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades executed in the selected period.")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: Start with Pair Discovery to find cointegrated pairs, then use Pair Analysis to backtest your strategy.")
