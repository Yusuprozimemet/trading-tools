import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import itertools
from scipy import stats

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Advanced Mean Reversion Pairs Trading",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Advanced Mean Reversion Pairs Trading")
st.markdown(
    """
    **Market-neutral strategy:** Profit from temporary divergences between correlated stocks.
    When pairs drift apart, bet on them returning to their historical relationship.
    """
)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.header("⚙️ Configuration")

# Ticker input with presets
st.sidebar.subheader("Stock Selection")
preset = st.sidebar.selectbox(
    "Preset sectors",
    ["Custom", "Tech Giants", "Oil & Gas", "Consumer Goods", "Big Banks", "Pharma"]
)

preset_tickers = {
    "Custom": "KO, PEP, JNJ, PG, XOM, CVX",
    "Tech Giants": "AAPL, MSFT, GOOGL, META, NVDA",
    "Oil & Gas": "XOM, CVX, BP, SHEL, COP",
    "Consumer Goods": "KO, PEP, PG, CL, KMB",
    "Big Banks": "JPM, BAC, WFC, C, GS",
    "Pharma": "JNJ, PFE, MRK, ABBV, LLY"
}

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=preset_tickers.get(preset, preset_tickers["Custom"])
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Date range
st.sidebar.subheader("📅 Time Period")
years_back = st.sidebar.slider("Years of history", 1, 10, 3)
end_date = datetime.today()
start_date = end_date - timedelta(days=years_back * 365)

st.sidebar.markdown(f"**From:** {start_date.strftime('%Y-%m-%d')}")
st.sidebar.markdown(f"**To:** {end_date.strftime('%Y-%m-%d')}")

# Strategy parameters
st.sidebar.subheader("📈 Strategy Parameters")
z_entry = st.sidebar.slider("Z-Score entry threshold", 1.5, 3.5, 2.0, 0.1)
z_exit = st.sidebar.slider("Z-Score exit threshold", 0.0, 1.5, 0.5, 0.1)
lookback = st.sidebar.slider("Lookback window (days)", 20, 252, 60)
min_correlation = st.sidebar.slider(
    "Min correlation for pairs", 0.5, 0.95, 0.7, 0.05)

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    stop_loss = st.slider("Stop loss (%)", 0, 20, 10)
    min_half_life = st.slider("Min half-life (days)", 1, 60, 1)
    max_half_life = st.slider("Max half-life (days)", 1, 365, 60)
    use_log_ratio = st.checkbox("Use log ratio", value=False)

run = st.sidebar.button("🚀 Run Pairs Analysis", type="primary")

# ==============================================================================
# DATA FETCHING
# ==============================================================================


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(tickers, start, end):
    """Download price data with robust error handling"""
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )

        if raw is None or raw.empty:
            return None, "No data returned from Yahoo Finance"

        price_dict = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    df = raw
                else:
                    df = raw[t]

                if 'Close' in df.columns:
                    series = df['Close'].dropna()
                    if len(series) > 50:  # Minimum data requirement
                        price_dict[t] = series
                    else:
                        st.warning(
                            f"⚠️ {t}: Insufficient data ({len(series)} days)")
            except Exception as e:
                st.warning(f"⚠️ {t}: {str(e)}")

        if not price_dict:
            return None, "No valid price data found"

        prices = pd.DataFrame(price_dict)
        prices = prices.dropna(thresh=len(prices) * 0.7,
                               axis=1)  # Keep if >70% data

        return prices, None

    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


# ==============================================================================
# PAIR STATISTICS
# ==============================================================================
def calculate_half_life(ratio):
    """Calculate mean reversion half-life using Ornstein-Uhlenbeck"""
    try:
        ratio_lag = ratio.shift(1).dropna()
        ratio_ret = ratio.diff().dropna()

        # Align the series
        ratio_lag = ratio_lag[ratio_ret.index]

        # OLS regression
        ratio_lag = ratio_lag - ratio_lag.mean()
        slope = np.polyfit(ratio_lag, ratio_ret, 1)[0]

        if slope >= 0:
            return np.inf

        half_life = -np.log(2) / slope
        return half_life if half_life > 0 else np.inf
    except:
        return np.inf


def calculate_hurst_exponent(ratio, max_lag=20):
    """Calculate Hurst exponent (H < 0.5 = mean reverting)"""
    try:
        lags = range(2, max_lag)
        tau = []

        for lag in lags:
            pp = np.subtract(ratio[lag:], ratio[:-lag])
            tau.append(np.std(pp))

        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return 0.5


def cointegration_test(s1, s2):
    """Perform Engle-Granger cointegration test"""
    try:
        # Run OLS regression
        slope = np.polyfit(s1, s2, 1)[0]
        spread = s2 - slope * s1

        # ADF test on residuals
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(spread.dropna(), maxlag=1,
                          regression='c', autolag=None)

        # p-value < 0.05 suggests cointegration
        return result[1]
    except:
        return 1.0


def analyze_pair_quality(df, stock1, stock2, use_log=False):
    """Comprehensive pair quality metrics"""
    s1 = df[stock1].dropna()
    s2 = df[stock2].dropna()

    # Align series
    common_idx = s1.index.intersection(s2.index)
    s1 = s1[common_idx]
    s2 = s2[common_idx]

    if len(s1) < 50:
        return None

    # Correlation
    correlation = s1.pct_change().corr(s2.pct_change())

    # Price ratio
    if use_log:
        ratio = np.log(s1) - np.log(s2)
    else:
        ratio = s1 / s2

    # Half-life
    half_life = calculate_half_life(ratio)

    # Hurst exponent
    hurst = calculate_hurst_exponent(ratio.values)

    # Cointegration
    coint_pvalue = cointegration_test(s1, s2)

    return {
        'correlation': correlation,
        'half_life': half_life,
        'hurst': hurst,
        'cointegration_pvalue': coint_pvalue,
        'mean_reverting': hurst < 0.5 and half_life < 365,
        'cointegrated': coint_pvalue < 0.05
    }


# ==============================================================================
# TRADING LOGIC
# ==============================================================================
def analyze_pair(df, stock1, stock2, lookback, z_entry, z_exit, use_log=False, stop_loss_pct=10):
    """Analyze pair and generate trading signals"""
    s1 = df[stock1].dropna()
    s2 = df[stock2].dropna()

    # Align series
    common_idx = s1.index.intersection(s2.index)
    s1 = s1[common_idx]
    s2 = s2[common_idx]

    # Calculate ratio
    if use_log:
        ratio = np.log(s1) - np.log(s2)
    else:
        ratio = s1 / s2

    # Rolling statistics
    rolling_mean = ratio.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = ratio.rolling(window=lookback, min_periods=lookback).std()
    zscore = (ratio - rolling_mean) / rolling_std

    # Trading signals
    in_long = False
    in_short = False
    trades = []
    entry_price1, entry_price2 = 0, 0

    for i, date in enumerate(zscore.index):
        if i < lookback:
            continue

        z = zscore.iloc[i]
        price1 = s1.iloc[i]
        price2 = s2.iloc[i]

        if np.isnan(z):
            continue

        # Entry logic
        if not in_long and not in_short:
            if z > z_entry:  # Stock1 overvalued
                in_short = True
                entry_price1, entry_price2 = price1, price2
                trades.append({
                    "Date": date,
                    "Action": "SHORT",
                    "Stock1": stock1,
                    "Stock2": stock2,
                    "Z": round(z, 3),
                    "Price1": price1,
                    "Price2": price2,
                    "Ratio": ratio.iloc[i]
                })
            elif z < -z_entry:  # Stock1 undervalued
                in_long = True
                entry_price1, entry_price2 = price1, price2
                trades.append({
                    "Date": date,
                    "Action": "LONG",
                    "Stock1": stock1,
                    "Stock2": stock2,
                    "Z": round(z, 3),
                    "Price1": price1,
                    "Price2": price2,
                    "Ratio": ratio.iloc[i]
                })

        # Exit logic - mean reversion
        elif in_long and abs(z) <= z_exit:
            in_long = False
            # Long stock1, short stock2
            ret = (price1 - entry_price1) / entry_price1 - \
                (price2 - entry_price2) / entry_price2
            trades.append({
                "Date": date,
                "Action": "EXIT_LONG",
                "Z": round(z, 3),
                "Return": ret,
                "Reason": "Mean Reversion"
            })

        elif in_short and abs(z) <= z_exit:
            in_short = False
            # Short stock1, long stock2
            ret = (entry_price1 - price1) / entry_price1 + \
                (price2 - entry_price2) / entry_price2
            trades.append({
                "Date": date,
                "Action": "EXIT_SHORT",
                "Z": round(z, 3),
                "Return": ret,
                "Reason": "Mean Reversion"
            })

        # Stop loss
        elif in_long:
            ret = (price1 - entry_price1) / entry_price1 - \
                (price2 - entry_price2) / entry_price2
            if ret < -stop_loss_pct / 100:
                in_long = False
                trades.append({
                    "Date": date,
                    "Action": "STOP_LONG",
                    "Z": round(z, 3),
                    "Return": ret,
                    "Reason": "Stop Loss"
                })

        elif in_short:
            ret = (entry_price1 - price1) / entry_price1 + \
                (price2 - entry_price2) / entry_price2
            if ret < -stop_loss_pct / 100:
                in_short = False
                trades.append({
                    "Date": date,
                    "Action": "STOP_SHORT",
                    "Z": round(z, 3),
                    "Return": ret,
                    "Reason": "Stop Loss"
                })

    return pd.DataFrame(trades), ratio, zscore, rolling_mean, rolling_std


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================
def calculate_metrics(returns):
    """Calculate comprehensive performance metrics"""
    if len(returns) == 0:
        return {}

    total_return = ((1 + returns).prod() - 1) * 100
    avg_return = returns.mean() * 100
    std_return = returns.std() * 100
    sharpe = (returns.mean() / returns.std() *
              np.sqrt(252)) if returns.std() > 0 else 0

    win_rate = (returns > 0).mean() * 100
    avg_win = returns[returns > 0].mean() * 100 if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() * 100 if (returns < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    max_drawdown = 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'avg_return': avg_return,
        'std_return': std_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if run:
    if len(tickers) < 2:
        st.error("❌ Please enter at least 2 tickers for pairs trading.")
        st.stop()

    # Fetch data
    with st.spinner("📥 Downloading price data..."):
        prices, error = fetch_data(tickers, start_date, end_date)

    if error:
        st.error(f"❌ {error}")
        st.stop()

    if prices is None or prices.empty:
        st.error("❌ No valid price data available.")
        st.stop()

    tickers = prices.columns.tolist()
    if len(tickers) < 2:
        st.error("❌ Not enough valid tickers after data cleaning.")
        st.stop()

    st.success(
        f"✅ Downloaded data for {len(tickers)} stocks: {', '.join(tickers)}")

    # ==============================================================================
    # EDUCATIONAL SECTION
    # ==============================================================================
    st.markdown("---")
    with st.expander("📚 **Learn: Mean Reversion Pairs Trading**", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 Core Concept
            - Find **two stocks that move together** (high correlation)
            - When they **diverge**, bet on **convergence**
            - Market-neutral: profit regardless of market direction

            ### 📊 The Math
            1. **Ratio**: Stock A / Stock B
            2. **Z-Score**: (Ratio - Mean) / StdDev
            3. **Entry**: |Z| > 2.0 (divergence)
            4. **Exit**: |Z| < 0.5 (convergence)

            ### ✅ Good Pairs
            - High correlation (>0.7)
            - Same sector/industry
            - Cointegrated (statistical test)
            - Mean-reverting (Hurst < 0.5)
            """)

        with col2:
            st.markdown("""
            ### 📈 Trading Example
            **Setup**: KO vs PEP (Coke vs Pepsi)

            **Scenario 1: Z-Score = +2.5**
            - KO overvalued vs PEP
            - **Action**: Short KO, Long PEP
            - **Exit**: When Z → 0

            **Scenario 2: Z-Score = -2.5**
            - KO undervalued vs PEP
            - **Action**: Long KO, Short PEP
            - **Exit**: When Z → 0

            ### ⚠️ Risks
            - Pairs can **break down permanently**
            - Requires **margin** for short selling
            - **Transaction costs** matter
            - Need **tight risk management**
            """)

    # ==============================================================================
    # CORRELATION ANALYSIS
    # ==============================================================================
    st.markdown("---")
    st.header("🔗 Correlation Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Calculate correlation
        returns = prices.pct_change().dropna()
        corr = returns.corr()

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdYlGn',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="Correlation Heatmap (Daily Returns)",
            height=500,
            xaxis_title="Stock",
            yaxis_title="Stock"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📋 Top Pairs")
        st.markdown(f"*Correlation ≥ {min_correlation}*")

        # Find top pairs
        pairs_quality = []
        for stock1, stock2 in itertools.combinations(tickers, 2):
            corr_val = corr.loc[stock1, stock2]
            if corr_val >= min_correlation:
                pairs_quality.append({
                    'Pair': f"{stock1}/{stock2}",
                    'Correlation': corr_val
                })

        if pairs_quality:
            pairs_df = pd.DataFrame(pairs_quality).sort_values(
                'Correlation', ascending=False)
            st.dataframe(
                pairs_df.style.format({'Correlation': '{:.3f}'})
                .background_gradient(subset=['Correlation'], cmap='RdYlGn', vmin=0.5, vmax=1.0),
                height=400,
                hide_index=True
            )
        else:
            st.warning(f"No pairs found with correlation ≥ {min_correlation}")

    # ==============================================================================
    # PAIR SELECTION
    # ==============================================================================
    st.markdown("---")
    st.header("🎯 Pair Analysis")

    # Get all valid pairs
    valid_pairs = []
    for s1, s2 in itertools.combinations(tickers, 2):
        if corr.loc[s1, s2] >= min_correlation:
            valid_pairs.append(f"{s1} / {s2}")

    if not valid_pairs:
        st.error(
            f"❌ No pairs meet the minimum correlation threshold of {min_correlation}")
        st.stop()

    selected = st.selectbox(
        f"**Select a pair** ({len(valid_pairs)} available)",
        valid_pairs,
        index=0
    )
    stock1, stock2 = selected.split(" / ")

    # ==============================================================================
    # PAIR QUALITY METRICS
    # ==============================================================================
    with st.spinner(f"🔍 Analyzing {stock1} vs {stock2}..."):
        quality = analyze_pair_quality(prices, stock1, stock2, use_log_ratio)

        if quality:
            st.markdown("### 📊 Pair Quality Metrics")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Correlation", f"{quality['correlation']:.3f}")
            mcol2.metric(
                "Half-Life (days)", f"{quality['half_life']:.1f}" if quality['half_life'] < 999 else "∞")
            mcol3.metric("Hurst Exponent", f"{quality['hurst']:.3f}")
            mcol4.metric("Cointegration p-value",
                         f"{quality['cointegration_pvalue']:.4f}")

            # Quality assessment
            quality_score = 0
            quality_msgs = []

            if quality['correlation'] > 0.8:
                quality_score += 1
                quality_msgs.append("✅ Strong correlation")
            elif quality['correlation'] > 0.7:
                quality_msgs.append("⚠️ Moderate correlation")
            else:
                quality_msgs.append("❌ Weak correlation")

            if quality['cointegrated']:
                quality_score += 1
                quality_msgs.append(
                    "✅ Cointegrated (statistically significant)")
            else:
                quality_msgs.append("⚠️ Not cointegrated")

            if quality['hurst'] < 0.5:
                quality_score += 1
                quality_msgs.append("✅ Mean-reverting behavior")
            else:
                quality_msgs.append("⚠️ Trending behavior")

            if min_half_life <= quality['half_life'] <= max_half_life:
                quality_score += 1
                quality_msgs.append(
                    f"✅ Good half-life ({quality['half_life']:.1f} days)")
            elif quality['half_life'] > max_half_life:
                quality_msgs.append(
                    f"⚠️ Slow mean reversion ({quality['half_life']:.1f} days)")
            else:
                quality_msgs.append(
                    f"⚠️ Very fast mean reversion ({quality['half_life']:.1f} days)")

            # Display quality assessment
            if quality_score >= 3:
                st.success("🌟 **Excellent pair!** " + " | ".join(quality_msgs))
            elif quality_score >= 2:
                st.info("👍 **Good pair.** " + " | ".join(quality_msgs))
            else:
                st.warning("⚠️ **Questionable pair.** " +
                           " | ".join(quality_msgs))

    # ==============================================================================
    # RUN BACKTEST
    # ==============================================================================
    with st.spinner("⚙️ Running backtest..."):
        trades_df, ratio, zscore, rolling_mean, rolling_std = analyze_pair(
            prices, stock1, stock2, lookback, z_entry, z_exit, use_log_ratio, stop_loss
        )

    # ==============================================================================
    # VISUALIZATION
    # ==============================================================================
    st.markdown("### 📈 Price & Signal Analysis")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            f"{stock1} vs {stock2} - Normalized Prices",
            "Price Ratio with Bollinger Bands",
            "Z-Score with Entry/Exit Thresholds",
            "Cumulative Returns"
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # Row 1: Normalized prices
    norm1 = prices[stock1] / prices[stock1].iloc[0] * 100
    norm2 = prices[stock2] / prices[stock2].iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=prices.index, y=norm1, name=stock1,
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=prices.index, y=norm2, name=stock2,
        line=dict(color='#ff7f0e', width=2)
    ), row=1, col=1)

    # Row 2: Ratio with Bollinger Bands
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std

    fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio, name="Ratio",
        line=dict(color='purple', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=rolling_mean, name="Mean",
        line=dict(color='black', width=1, dash='dash')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=upper_band.index, y=upper_band, name="Upper BB",
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=lower_band.index, y=lower_band, name="Lower BB",
        line=dict(color='red', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
    ), row=2, col=1)

    # Row 3: Z-Score with thresholds
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore, name="Z-Score",
        line=dict(color='orange', width=2)
    ), row=3, col=1)

    fig.add_hline(y=z_entry, line_dash="dash",
                  line_color="red", line_width=2, row=3, col=1)
    fig.add_hline(y=-z_entry, line_dash="dash",
                  line_color="red", line_width=2, row=3, col=1)
    fig.add_hline(y=z_exit, line_dash="dot", line_color="gray",
                  line_width=1, row=3, col=1)
    fig.add_hline(y=-z_exit, line_dash="dot",
                  line_color="gray", line_width=1, row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black",
                  line_width=1, row=3, col=1)

    # Add trade markers
    if not trades_df.empty:
        long_entries = trades_df[trades_df["Action"] == "LONG"]
        short_entries = trades_df[trades_df["Action"] == "SHORT"]
        exits = trades_df[trades_df["Action"].str.contains("EXIT|STOP")]

        if not long_entries.empty:
            fig.add_trace(go.Scatter(
                x=long_entries["Date"], y=long_entries["Z"],
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=12,
                            color="lime", line=dict(width=1, color='darkgreen'))
            ), row=3, col=1)

        if not short_entries.empty:
            fig.add_trace(go.Scatter(
                x=short_entries["Date"], y=short_entries["Z"],
                mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=12,
                            color="red", line=dict(width=1, color='darkred'))
            ), row=3, col=1)

        if not exits.empty:
            exit_z = zscore.reindex(exits["Date"]).fillna(0)
            fig.add_trace(go.Scatter(
                x=exits["Date"], y=exit_z,
                mode="markers", name="Exit",
                marker=dict(symbol="x", size=10,
                            color="black", line=dict(width=2))
            ), row=3, col=1)

    # Row 4: Cumulative returns
    if not trades_df.empty:
        exit_trades = trades_df[trades_df["Action"].str.contains(
            "EXIT|STOP")].copy()
        if not exit_trades.empty and "Return" in exit_trades.columns:
            exit_trades['Cumulative'] = (1 + exit_trades['Return']).cumprod()
            fig.add_trace(go.Scatter(
                x=exit_trades["Date"], y=(exit_trades['Cumulative'] - 1) * 100,
                name="Cumulative Return",
                line=dict(color='green', width=2),
                fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'
            ), row=4, col=1)
            fig.add_hline(y=0, line_dash="solid",
                          line_color="black", line_width=1, row=4, col=1)

    fig.update_layout(
        height=1200,
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)
    # ==============================================================================
    # TRADE LOG & METRICS
    # ==============================================================================
    st.markdown("### 📝 Trade Log & Performance Metrics")
    if not trades_df.empty:
        st.write(trades_df)
    else:
        st.write("No trades executed.")
