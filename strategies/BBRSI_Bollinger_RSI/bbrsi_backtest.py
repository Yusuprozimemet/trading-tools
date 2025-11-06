# bbrsi_backtest.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="BBRSI Backtester + DEBUG", layout="wide")
st.title("BBRSI Strategy Backtester + Smart Debug")
st.markdown("**Now with auto-debug when 0 trades!**")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End", value=datetime.now())
    interval = st.selectbox("Interval", ["1h", "1d", "1wk", "1mo"], index=1)

    st.divider()
    st.subheader("Optimized Parameters")
    colA, colB = st.columns(2)
    with colA:
        bb_entry = st.number_input(
            "BB Entry", 0.80, 1.05, 0.95, 0.005, format="%.4f")
        bb_exit = st.number_input(
            "BB Exit", 0.95, 1.25, 1.10, 0.005, format="%.4f")
        rsi_entry = st.number_input("RSI Entry", 10.0, 45.0, 30.0, 0.5)
    with colB:
        rsi_exit = st.number_input("RSI Exit", 55.0, 90.0, 70.0, 0.5)
        trailing_stop_pct = st.number_input(
            "Trailing Stop %", 0.5, 20.0, 7.0, 0.5) / 100

    if st.button("Apply Safe Defaults (Get Trades Fast)", type="secondary"):
        bb_entry = 0.98
        bb_exit = 1.02
        rsi_entry = 35.0
        rsi_exit = 65.0
        trailing_stop_pct = 0.08
        st.success("Safe parameters applied!")

    run = st.button("Run Backtest", type="primary", use_container_width=True)

# ---------------------------
# Core Functions
# ---------------------------


@st.cache_data
def fetch_data(ticker, start, end, interval):
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
        if df.empty:
            return None, "No data"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df, None
    except Exception as e:
        return None, str(e)


def add_indicators(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20, min_periods=10).mean()
    df['STD20'] = df['Close'].rolling(20, min_periods=10).std()
    df['UpperBB'] = df['MA20'] + 2 * df['STD20']
    df['LowerBB'] = df['MA20'] - 2 * df['STD20']

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - 100 / (1 + rs)
    df['RSI'] = df['RSI'].fillna(50)

    df['Returns'] = df['Close'].pct_change().fillna(0)
    df.dropna(subset=['MA20', 'RSI'], inplace=True)
    return df


def backtest(df, bb_entry, bb_exit, rsi_entry, rsi_exit, trail_pct):
    df = df.copy()
    df['Position'] = 0
    df['Entry_Price'] = np.nan
    df['Exit_Price'] = np.nan

    # Debug columns
    df['BB_Entry_Trigger'] = (df['Close'] < df['LowerBB'] * bb_entry)
    df['RSI_Entry_Trigger'] = (df['RSI'] < rsi_entry)
    df['Entry_Signal'] = df['BB_Entry_Trigger'] & df['RSI_Entry_Trigger']

    df['BB_Exit_Trigger'] = (df['Close'] > df['UpperBB'] * bb_exit)
    df['RSI_Exit_Trigger'] = (df['RSI'] > rsi_exit)
    df['Trail_Stop_Price'] = np.nan

    in_pos = False
    highest = 0
    trail_stop = 0

    for i in range(1, len(df)):
        close = df['Close'].iloc[i]
        lower = df['LowerBB'].iloc[i]
        upper = df['UpperBB'].iloc[i]
        rsi = df['RSI'].iloc[i]

        if not in_pos:
            if close < lower * bb_entry and rsi < rsi_entry:
                df.iat[i, df.columns.get_loc('Position')] = 1
                df.iat[i, df.columns.get_loc('Entry_Price')] = close
                in_pos = True
                highest = close
                trail_stop = close * (1 - trail_pct)
                df.iat[i, df.columns.get_loc('Trail_Stop_Price')] = trail_stop
        else:
            if close > highest:
                highest = close
                trail_stop = highest * (1 - trail_pct)
            df.iat[i, df.columns.get_loc('Trail_Stop_Price')] = trail_stop

            exit_bb_rsi = close > upper * bb_exit and rsi > rsi_exit
            exit_trail = close <= trail_stop

            if exit_bb_rsi or exit_trail:
                df.iat[i, df.columns.get_loc('Position')] = 0
                df.iat[i, df.columns.get_loc('Exit_Price')] = close
                in_pos = False
            else:
                df.iat[i, df.columns.get_loc('Position')] = 1

    df['Strategy_Returns'] = df['Position'].shift(1).fillna(0) * df['Returns']
    df['Cum_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cum_BuyHold'] = (1 + df['Returns']).cumprod()

    return df


# ---------------------------
# Main Execution
# ---------------------------
if run:
    with st.spinner("Fetching data..."):
        data, err = fetch_data(ticker, start_date, end_date, interval)

    if err or data is None:
        st.error(f"Data error: {err}")
    else:
        data = add_indicators(data)
        if len(data) < 50:
            st.error("Not enough data after indicators.")
        else:
            result = backtest(data, bb_entry, bb_exit,
                              rsi_entry, rsi_exit, trailing_stop_pct)
            stats = {
                "Total Return": result['Cum_Strategy'].iloc[-1] - 1,
                "Buy & Hold": result['Cum_BuyHold'].iloc[-1] - 1,
                "Trades": int((result['Position'].diff().abs() == 1).sum() // 2)
            }

            # === DEBUG DASHBOARD WHEN 0 TRADES ===
            if stats["Trades"] == 0:
                st.error("NO TRADES GENERATED → Opening Debug Mode")
                st.markdown("### Why No Trades? Let's Find Out")

                col1, col2, col3, col4 = st.columns(4)
                bb_touch = (result['Close'] <
                            result['LowerBB'] * bb_entry).sum()
                rsi_low = (result['RSI'] < rsi_entry).sum()
                both = result['Entry_Signal'].sum()

                col1.metric("Price touched BB × Entry", bb_touch)
                col2.metric("RSI < Entry Level", rsi_low)
                col3.metric("Both conditions met", both)
                col4.metric("Total days", len(result))

                st.markdown("#### Diagnosis:")
                if bb_touch == 0:
                    st.warning(
                        "Price NEVER went below LowerBB × BB_Entry\nYour BB_Entry is too low (e.g. 0.88)\nTry 0.97–1.00")
                if rsi_low == 0:
                    st.warning(
                        "RSI never went below your entry level\nTry RSI Entry = 35–40")
                if both == 0 and bb_touch > 0 and rsi_low > 0:
                    st.info(
                        "Conditions happened but never on the same day\nSlightly loosen one parameter")

                # Show closest misses
                result['BB_Distance'] = result['Close'] / \
                    (result['LowerBB'] * bb_entry)
                closest_bb = result['BB_Distance'].nsmallest(5)
                st.write("Closest BB touches (ratio < 1 = trigger):")
                st.dataframe(closest_bb)

                result['RSI_Distance'] = result['RSI'] - rsi_entry
                closest_rsi = result['RSI_Distance'].nsmallest(5)
                st.write("Closest RSI touches:")
                st.dataframe(closest_rsi)

            else:
                st.success(
                    f"Backtest Complete! {stats['Trades']} trades found")

            # === Normal Metrics ===
            total_ret = result['Cum_Strategy'].iloc[-1] - 1
            bh_ret = result['Cum_BuyHold'].iloc[-1] - 1
            sharpe = (result['Strategy_Returns'].mean() * 252) / (result['Strategy_Returns'].std()
                                                                  * np.sqrt(252)) if result['Strategy_Returns'].std() > 0 else 0
            max_dd = (result['Cum_Strategy'] /
                      result['Cum_Strategy'].cummax() - 1).min()

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Total Return", f"{total_ret:.2%}")
            c2.metric("Buy & Hold", f"{bh_ret:.2%}")
            c3.metric("Sharpe", f"{sharpe:.3f}")
            c4.metric("Max DD", f"{max_dd:.2%}")
            c5.metric(
                "Calmar", f"{total_ret/abs(max_dd):.2f}" if max_dd != 0 else "∞")
            c6.metric("Trades", stats["Trades"])

            st.divider()

            # === Charts ===
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(result.index, result['Close'],
                    label='Price', linewidth=1.5)
            ax.plot(result.index, result['LowerBB'] * bb_entry,
                    label=f'Entry Level (×{bb_entry})', color='green', linestyle='--', alpha=0.8)
            ax.plot(result.index, result['UpperBB'] * bb_exit,
                    label=f'Exit Level (×{bb_exit})', color='red', linestyle='--', alpha=0.8)

            buys = result[result['Position'].diff() == 1]
            sells = result[result['Position'].diff() == -1]
            ax.scatter(buys.index, buys['Close'], marker='^',
                       color='lime', s=120, label='BUY', zorder=10)
            ax.scatter(sells.index, sells['Close'], marker='v',
                       color='red', s=120, label='SELL', zorder=10)

            # Highlight almost-signals
            almost = result[result['Entry_Signal'] == False]
            almost_bb = almost[almost['BB_Entry_Trigger']]
            almost_rsi = almost[almost['RSI_Entry_Trigger']]
            if len(almost_bb) > 0:
                ax.scatter(almost_bb.index, almost_bb['Close'], marker='o',
                           color='orange', s=30, alpha=0.6, label='BB touch only')
            if len(almost_rsi) > 0:
                ax.scatter(almost_rsi.index, almost_rsi['Close'], marker='x',
                           color='purple', s=30, alpha=0.6, label='RSI only')

            ax.set_title(f"{ticker} - BBRSI Debug Chart")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            # RSI Chart
            fig2, ax2 = plt.subplots(figsize=(15, 4))
            ax2.plot(result.index, result['RSI'], color='purple')
            ax2.axhline(rsi_entry, color='green', linestyle='--',
                        linewidth=2, label=f'Entry {rsi_entry}')
            ax2.axhline(rsi_exit, color='red', linestyle='--',
                        linewidth=2, label=f'Exit {rsi_exit}')
            ax2.fill_between(
                result.index, 0, 100, where=result['RSI'] < rsi_entry, color='green', alpha=0.2)
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            # Download
            csv = result[['Close', 'RSI', 'LowerBB', 'UpperBB', 'Position',
                          'Entry_Price', 'Exit_Price', 'Cum_Strategy', 'Cum_BuyHold']].copy()
            st.download_button("Download Full Data", data=csv.to_csv(
            ), file_name=f"debug_{ticker}.csv", mime="text/csv")

else:
    st.info("Enter parameters → Click **Run Backtest**")
    st.markdown("""
    ### Getting 0 trades? This version will:
    - Tell you **exactly** which condition failed
    - Show **closest misses**
    - Suggest working values
    - Let you fix with **one click**
    """)
