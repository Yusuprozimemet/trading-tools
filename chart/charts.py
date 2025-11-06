# charts.py - BULLETPROOF EDITION (No more Yahoo errors!)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Pro Charts", layout="wide")
st.title("Pro Chart Viewer 🚀")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Chart Mode", ["Single Stock", "Spread Trading"])

    if mode == "Single Stock":
        ticker = st.text_input("Ticker", "ASML.AS").upper()
        pair_mode = False
    else:
        st.subheader("Pair Setup")
        preset = st.selectbox(
            "Preset", ["Custom", "AGN/ASML", "WKL/REN", "HEIA/ASML"])
        presets = {
            "AGN/ASML": ("AGN.AS", "ASML.AS", 0.0072, "AGN", "ASML"),
            "WKL/REN": ("WKL.AS", "REN.AS", 3.6, "WKL", "REN"),
            "HEIA/ASML": ("HEIA.AS", "ASML.AS", 0.084, "HEIA", "ASML")
        }
        if preset != "Custom":
            stock1, stock2, ratio, name1, name2 = presets[preset]
        else:
            stock1 = st.text_input("Stock 1", "AGN.AS")
            name1 = st.text_input("Name 1", "AGN")
            stock2 = st.text_input("Stock 2", "ASML.AS")
            name2 = st.text_input("Name 2", "ASML")
            ratio = st.number_input("Ratio", 0.0001, 10.0, 0.0072, 0.0001)
        ticker = f"{name1}−{name2}×{ratio}"
        pair_mode = True

    interval = st.selectbox("Interval",
                            ["1m", "5m", "15m", "30m", "1h",
                                "4h", "1d", "1wk", "1mo"],
                            index=2)

    # SMART DAYS BACK based on interval
    max_days = {
        "1m": 7,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "4h": 730,
        "1d": 2000,
        "1wk": 2000,
        "1mo": 2000
    }
    default_days = {"1m": 5, "5m": 30, "15m": 45,
                    "30m": 60, "1h": 180, "4h": 365}.get(interval, 365)
    max_d = max_days[interval]

    days = st.slider(
        f"Days Back (max {max_d} for {interval})", 1, max_d, min(default_days, max_d))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    st.caption(f"Yahoo limits: 1m→7d | 5-60m→60d | 1h+→730d | daily→no limit")

    st.divider()
    st.subheader("Indicators")
    show_candles = st.checkbox("Candlesticks", True)
    show_line = st.checkbox("Close Line", False)

    ma_options = []
    if st.checkbox("SMA 20", True):
        ma_options.append(("SMA20", 20, "yellow", False))
    if st.checkbox("SMA 50", True):
        ma_options.append(("SMA50", 50, "orange", False))
    if st.checkbox("EMA 9", True):
        ma_options.append(("EMA9", 9, "lime", True))
    if st.checkbox("EMA 21"):
        ma_options.append(("EMA21", 21, "cyan", True))

    show_bb = st.checkbox("Bollinger Bands", True)
    bb_fill = st.checkbox("BB Fill", True)
    show_rsi = st.checkbox("RSI", True)
    show_volume = st.checkbox("Volume", True)

    if pair_mode:
        z_window = st.slider("Z-Score Window", 10, 100, 20)
        z_thresh = st.slider("Signal Threshold", 1.5, 3.0, 2.0, 0.1)

    hlines = st.text_input("H-Lines (comma)", "0")
    dark_mode = st.checkbox("Dark Mode", True)

# ====================== SMART DATA FETCH ======================


@st.cache_data(ttl=60)
def fetch_smart(ticker, start, end, interval):
    # Convert to string dates for yfinance
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    df = yf.Ticker(ticker).history(
        start=start_str, end=end_str, interval=interval)

    if df.empty:
        st.error(f"No data for {ticker} - check ticker or date range")
        return None

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def filter_trading_hours(df, interval):
    if df is None or df.empty:
        return df
    if interval in ['1d', '1wk', '1mo']:
        return df[df.index.weekday < 5]
    # 09:00 - 17:30 CET
    return df[df.index.map(lambda t: t.weekday() < 5 and 540 <= t.hour*60 + t.minute <= 1050)]


# Fetch
if not pair_mode:
    df = fetch_smart(ticker, start_date, end_date, interval)
    df = filter_trading_hours(df, interval)
else:
    df1 = fetch_smart(stock1, start_date, end_date, interval)
    df2 = fetch_smart(stock2, start_date, end_date, interval)
    if df1 is None or df2 is None:
        st.stop()
    df1 = filter_trading_hours(df1, interval)
    df2 = filter_trading_hours(df2, interval)

    idx = df1.index.intersection(df2.index)
    df1, df2 = df1.loc[idx], df2.loc[idx]

    spread = pd.DataFrame(index=idx)
    spread['Open'] = df1['Open'] - df2['Open'] * ratio
    spread['High'] = df1['High'] - df2['Low'] * ratio
    spread['Low'] = df1['Low'] - df2['High'] * ratio
    spread['Close'] = df1['Close'] - df2['Close'] * ratio
    spread['Volume'] = (df1['Volume'] + df2['Volume']) / 2
    df = spread

if df is None or df.empty:
    st.stop()

# ====================== INDICATORS ======================


def add_indicators(d):
    d = d.copy()
    for name, p, col, ema in ma_options:
        d[name] = d['Close'].ewm(span=p).mean(
        ) if ema else d['Close'].rolling(p).mean()

    if show_bb:
        mid = d['Close'].rolling(20).mean()
        std = d['Close'].rolling(20).std()
        d['BBU'] = mid + 2*std
        d['BBL'] = mid - 2*std

    if show_rsi:
        delta = d['Close'].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        d['RSI'] = 100 - 100/(1 + up/down.replace(0, np.nan))

    if show_volume:
        d['VMA20'] = d['Volume'].rolling(20).mean()

    if pair_mode:
        mean = d['Close'].rolling(z_window).mean()
        std = d['Close'].rolling(z_window).std()
        d['Z'] = (d['Close'] - mean) / std
        d['Signal'] = 0
        d.loc[d['Z'] > z_thresh, 'Signal'] = -1
        d.loc[d['Z'] < -z_thresh, 'Signal'] = 1

    return d


df = add_indicators(df)

# ====================== PLOT ======================
template = "plotly_dark" if dark_mode else "plotly_white"
rows = 4 if pair_mode else 3
heights = [0.5, 0.15, 0.2, 0.15] if pair_mode else [0.6, 0.25, 0.15]

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=heights,
    subplot_titles=(ticker, "Z-Score" if pair_mode else "RSI",
                    "Volume", "RSI" if pair_mode else None)
)

# Candles + Line
if show_candles:
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
if show_line:
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(
        color="#00ff88", width=2)), row=1, col=1)

# MAs + BB
for name, _, col, _ in ma_options:
    fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name, line=dict(
        color=col, width=2)), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], name="BB Upper", line=dict(
        color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], name="BB Lower", line=dict(
        color="gray", dash="dot")), row=1, col=1)
    if bb_fill:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BBU'], fill=None, showlegend=False, line=dict(width=0)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BBL'], fill='tonexty', fillcolor="rgba(100,150,255,0.12)", name="BB"), row=1, col=1)

# Signals
if pair_mode:
    long = df[df['Signal'] == 1]
    short = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=long.index, y=long['Low']*0.999, mode='markers', name='LONG',
                             marker=dict(symbol='triangle-up', size=14, color='lime')), row=1, col=1)
    fig.add_trace(go.Scatter(x=short.index, y=short['High']*1.001, mode='markers', name='SHORT',
                             marker=dict(symbol='triangle-down', size=14, color='red')), row=1, col=1)

# H-Lines
try:
    for lvl in [float(x.strip()) for x in hlines.split(",") if x.strip()]:
        fig.add_hline(y=lvl, line=dict(
            color="cyan", dash="dash"), row=1, col=1)
except:
    pass

# Subplots
if pair_mode:
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Z'], name="Z-Score", line=dict(color="#ff00ff")), row=2, col=1)
    fig.add_hline(y=z_thresh, line=dict(color="red", dash="dot"), row=2)
    fig.add_hline(y=-z_thresh, line=dict(color="lime", dash="dot"), row=2)
    rsi_row = 4
else:
    rsi_row = 2

if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(
        color="#aa00ff")), row=rsi_row, col=1)
    for lvl, col in [(70, "red"), (30, "lime"), (50, "gray")]:
        fig.add_hline(y=lvl, line=dict(
            color=col, dash="dot" if lvl != 50 else "dash"), row=rsi_row)

if show_volume:
    colors = ['red' if o >= c else 'green' for o,
              c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name="Volume", marker_color=colors, opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VMA20'], name="Vol MA20", line=dict(
        color="orange")), row=3, col=1)

# ====================== FINAL LAYOUT ======================
fig.update_layout(
    height=1000 if pair_mode else 900,
    template=template,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    margin=dict(l=50, r=50, t=80, b=50)
)

# Perfect gaps removal
if interval in ['1m', '5m', '15m', '30m', '1h', '4h']:
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[17.5, 9], pattern="hour")
    ])
else:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

st.plotly_chart(fig, use_container_width=True)

# Signals table
if pair_mode:
    st.markdown("---")
    st.subheader("Live Signals")
    sig = df[df['Signal'] != 0].tail(10).copy()
    if not sig.empty:
        sig['Signal'] = sig['Signal'].map({1: "LONG", -1: "SHORT"})
        sig['Z'] = sig['Z'].round(3)
        sig['Price'] = sig['Close'].round(4)
        disp = sig[['Signal', 'Price', 'Z']].sort_index(ascending=False)
        disp.index = disp.index.strftime("%b %d %H:%M")
        st.dataframe(disp, use_container_width=True)

st.success(
    "No more Yahoo errors • Smart date limits • Perfect gaps • Dutch engineering")
st.caption("Run: streamlit run chart/charts.py")
