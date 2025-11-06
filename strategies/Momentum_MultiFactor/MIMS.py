# ─────────────────────────────────────────────────────────────────────────────
#  Trading-Analysis Dashboard
#  Multi-Factor Intrinsic Momentum Strategy (MIMS)
#  ------------------------------------------------
#  • Multi-ticker input
#  • ETF comparison
#  • Fundamental, valuation, technical analysis
#  • Interactive Plotly charts
#  • Decision matrix + portfolio allocation
#  • Downloadable CSV report
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings

warnings.filterwarnings("ignore")

# ────────────────────── Page config ──────────────────────
st.set_page_config(page_title="Multi-Factor Intrinsic Momentum Strategy (MIMS)",
                   layout="wide", page_icon="Chart")

# ────────────────────── Sidebar – user input ──────────────────────
st.sidebar.header("Configuration")

# ---- Step 0: Ticker Selection (Top-down -> Bottom-up) ----------
# (Step 0 guidance moved to the method details; sidebar only contains manual ticker inputs)

# ---- Tickers -------------------------------------------------
default_stocks = "ASML.AS, INTC"          # change defaults here
default_etfs = "SMH, SPY"
stock_input = st.sidebar.text_input(
    "Stocks (comma-separated)", value=default_stocks
)
etf_input = st.sidebar.text_input(
    "ETFs (comma-separated)", value=default_etfs
)

stocks = [t.strip().upper() for t in stock_input.split(",") if t.strip()]
etfs = [t.strip().upper() for t in etf_input.split(",") if t.strip()]

# ---- Pair analysis inputs (uses tickers from Configuration) ------
st.sidebar.markdown("---")
run_pair = st.sidebar.button("Run Pair Analysis")
pair_period = st.sidebar.selectbox(
    "Pair lookback", ["3mo", "6mo", "1y", "2y"], index=2)

# ---- Method description selector --------------------------------
method_desc_options = [
    "Overview",
    "Step 0 — Ticker Selection Logic",
    "Screening",
    "Sector Benchmarking",
    "Fundamentals",
    "Intrinsic Value (DCF)",
    "Technical Analysis",
    "ETF/Pair Relative Strength",
]
selected_method = st.sidebar.selectbox(
    "Select method to view details", method_desc_options)

# ---- Global constants ----------------------------------------
DKK_TO_USD = 0.15          # keep for NVO if you ever add it
DISCOUNT_RATE_DEFAULT = 0.0537
TERMINAL_RATE = 0.04

# ────────────────────── Helper functions ──────────────────────


@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_stock_data(ticker, period="2y"):
    """Return pickle-serializable data for a ticker.

    Returns:
        hist (pd.DataFrame) or None,
        info (dict) or None,
        financials (pd.DataFrame) or None,
        cashflow (pd.DataFrame) or None
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info or {}
        # financials and cashflow are pandas objects and pickle-serializable
        fin = stock.financials if hasattr(
            stock, "financials") else pd.DataFrame()
        cf = stock.cashflow if hasattr(stock, "cashflow") else pd.DataFrame()
        return hist, info, fin, cf
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None, None, None, None


def screen_stocks_enhanced(tickers):
    data, price_data = [], {}
    for t in tickers:
        hist, info, fin, cf = fetch_stock_data(t)
        if hist is None or hist.empty:
            continue
        close = hist["Close"]
        current = close.iloc[-1]
        high52 = hist["High"].max()
        low52 = hist["Low"].min()
        decl = (high52 - current) / high52 * 100
        rise = (current - low52) / low52 * 100
        pe = info.get("trailingPE", np.nan)
        cap = info.get("marketCap", 0) / 1e9
        vol = hist["Volume"].mean()
        vola = close.pct_change().std() * np.sqrt(252) * 100

        price_data[t] = close
        data.append({
            "Ticker": t,
            "Price": round(current, 2),
            "52W High": round(high52, 2),
            "52W Low": round(low52, 2),
            "Decline %": round(decl, 2),
            "Rise %": round(rise, 2),
            "P/E": round(pe, 2) if not np.isnan(pe) else "N/A",
            "Cap $B": round(cap, 2) if cap else "N/A",
            "Avg Vol M": f"{vol/1e6:.1f}" if vol else "N/A",
            "Vol %": round(vola, 2) if vola else "N/A",
        })
    return pd.DataFrame(data), price_data


def plot_price_comparison(price_data):
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (t, ser) in enumerate(price_data.items()):
        norm = (ser / ser.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=ser.index, y=norm, mode="lines",
            name=t, line=dict(color=colors[i % len(colors)], width=2)
        ))
    fig.update_layout(
        title="Normalized Performance (from first day)",
        xaxis_title="Date", yaxis_title="Return %",
        hovermode="x unified", template="plotly_white",
        height=450, legend=dict(x=0.01, y=0.99)
    )
    return fig


def compare_etfs_enhanced():
    if len(etfs) < 2:
        return None, None, None
    a, b = etfs[0], etfs[1]
    ha, _, _, _ = fetch_stock_data(a)
    hb, _, _, _ = fetch_stock_data(b)
    if ha is None or hb is None:
        return None, None, None
    common = ha.index.intersection(hb.index)
    ratio = (ha.loc[common, "Close"] / hb.loc[common, "Close"]) * 100
    return ratio.iloc[-1], ratio.min(), ratio


def compare_pair(a, b, period="1y"):
    """Compare two tickers and return latest ratio, min ratio, and series (a/b * 100)."""
    if not a or not b:
        return None, None, None
    ha, _, _, _ = fetch_stock_data(a, period=period)
    hb, _, _, _ = fetch_stock_data(b, period=period)
    if ha is None or hb is None:
        return None, None, None, None, None
    common = ha.index.intersection(hb.index)
    if common.empty:
        return None, None, None, None, None
    a_close = ha.loc[common, "Close"]
    b_close = hb.loc[common, "Close"]
    ratio = (a_close / b_close) * 100

    # Spread and z-score (normalized spread)
    spread = a_close - b_close
    zscore = (spread - spread.mean()) / \
        spread.std() if spread.std() != 0 else spread * 0

    # Rolling correlation (30-day default)
    rolling_corr = a_close.rolling(window=30).corr(b_close)

    return ratio.iloc[-1], ratio.min(), ratio, zscore, rolling_corr


def benchmark_sectors(sector_etfs, benchmark="SPY", period="1y"):
    """Return list of sector names whose ETF outperforms the benchmark over the period."""
    strong = []
    for sector_name, etf in sector_etfs.items():
        try:
            ha, _, _, _ = fetch_stock_data(etf, period=period)
            hb, _, _, _ = fetch_stock_data(benchmark, period=period)
            if ha is None or hb is None:
                continue
            common = ha.index.intersection(hb.index)
            if common.empty:
                continue
            ratio = (ha.loc[common, "Close"] / hb.loc[common, "Close"]) * 100
            latest = ratio.iloc[-1]
            if latest > 100:
                strong.append(sector_name)
        except Exception:
            continue
    return strong


def select_tickers_from_sectors(sectors, sector_pools, min_vol=1e6, min_mcap=5e9, period="1y"):
    """Pick tickers from chosen sectors and apply simple liquidity & size filters."""
    selected = []
    selected_etfs = []
    for s in sectors:
        pool = sector_pools.get(s, [])
        # first element expected to be ETF symbol
        if pool:
            etf = pool[0]
            selected_etfs.append(etf)
        for tk in pool:
            try:
                hist, info, fin, cf = fetch_stock_data(tk, period=period)
                if hist is None or hist.empty:
                    continue
                avg_vol = hist["Volume"].mean()
                mcap = info.get("marketCap", 0)
                if avg_vol >= min_vol and mcap >= min_mcap:
                    selected.append(tk)
            except Exception:
                continue
    # de-duplicate preserving order
    seen = set()
    sel_unique = []
    for t in selected:
        if t not in seen:
            seen.add(t)
            sel_unique.append(t)
    return sel_unique, selected_etfs


def plot_etf_comparison(ratio):
    if ratio is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio,
                             mode="lines", name="Ratio",
                             line=dict(color="purple")))
    fig.add_hline(y=100, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"Relative Strength",
        xaxis_title="Date", yaxis_title="Ratio",
        template="plotly_white", height=350
    )
    return fig


def plot_pair_comparison(ratio, a, b):
    if ratio is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio,
                             mode="lines", name=f"{a}/{b}",
                             line=dict(color="purple")))
    fig.add_hline(y=100, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"{a} vs {b} Relative Strength",
        xaxis_title="Date", yaxis_title="Ratio",
        template="plotly_white", height=350
    )
    return fig


def plot_pair_advanced(ratio, zscore, rolling_corr, a, b):
    """Plot ratio, z-score of spread and rolling correlation in a 3-row subplot."""
    if ratio is None:
        return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=(f"{a}/{b} Ratio", "Spread z-score", "Rolling Correlation (30d)"))

    fig.add_trace(go.Scatter(x=ratio.index, y=ratio,
                  name=f"{a}/{b} Ratio", line=dict(color="purple")), row=1, col=1)
    if zscore is not None:
        fig.add_trace(go.Scatter(x=zscore.index, y=zscore,
                      name="Z-Score", line=dict(color="orange")), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
        fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
    if rolling_corr is not None:
        fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr,
                      name="Rolling Corr", line=dict(color="blue")), row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)

    fig.update_layout(height=900, template="plotly_white")
    return fig


def check_fundamentals_enhanced(ticker):
    hist, info, fin, cf = fetch_stock_data(ticker)
    if info is None:
        return {"Ticker": ticker, "Error": "No data"}
    try:
        # financials and cashflow were retrieved in fetch_stock_data
        fin = fin if isinstance(fin, pd.DataFrame) else pd.DataFrame()
        cf = cf if isinstance(cf, pd.DataFrame) else pd.DataFrame()

        # ---- Revenue -------------------------------------------------
        rev_col = next(
            (c for c in ["Total Revenue", "Revenue"] if c in fin.index), None)
        if not rev_col:
            return {"Ticker": ticker, "Error": "Revenue missing"}
        rev = fin.loc[rev_col].iloc[0] / 1e9
        rev_prev = fin.loc[rev_col].iloc[1] / 1e9 if fin.shape[1] > 1 else rev
        rev_g = (rev - rev_prev) / rev_prev * 100 if rev_prev else 0

        # ---- Net Income -----------------------------------------------
        ni_col = next((c for c in ["Net Income", "Net Income Common Stockholders"]
                       if c in fin.index), None)
        ni_g = np.nan
        if ni_col:
            ni = fin.loc[ni_col].iloc[0] / 1e9
            ni_prev = fin.loc[ni_col].iloc[1] / 1e9 if fin.shape[1] > 1 else ni
            ni_g = (ni - ni_prev) / ni_prev * 100 if ni_prev else 0
        else:
            ni = np.nan

        # ---- Free Cash Flow -------------------------------------------
        fcf = np.nan
        fcf_g = "N/A"
        if not cf.empty and "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow"].iloc[0] / 1e9
            fcf_prev = cf.loc["Free Cash Flow"].iloc[1] / \
                1e9 if cf.shape[1] > 1 else fcf
            fcf_g = (fcf - fcf_prev) / fcf_prev * 100 if fcf_prev else 0

        # ---- Health ---------------------------------------------------
        d2e = info.get("debtToEquity", np.nan)
        roe = info.get("returnOnEquity", np.nan)
        cr = info.get("currentRatio", np.nan)

        # ---- Moat -----------------------------------------------------
        sector = info.get("sector", "")
        cap = info.get("marketCap", 0)
        moat = "High" if sector == "Healthcare" and cap > 100e9 else "Medium/Low"

        return {
            "Ticker": ticker,
            "Revenue $B": round(rev, 2),
            "Rev Growth %": round(rev_g, 2),
            "Net Income $B": round(ni, 2) if not np.isnan(ni) else "N/A",
            "NI Growth %": round(ni_g, 2) if not np.isnan(ni_g) else "N/A",
            "FCF $B": round(fcf, 2) if not np.isnan(fcf) else "N/A",
            "FCF Growth %": round(fcf_g, 2) if isinstance(fcf_g, (int, float)) else "N/A",
            "Debt/Equity": round(d2e, 2) if not np.isnan(d2e) else "N/A",
            "ROE %": round(roe * 100, 2) if not np.isnan(roe) else "N/A",
            "Current Ratio": round(cr, 2) if not np.isnan(cr) else "N/A",
            "Moat": moat,
        }
    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}


def calculate_intrinsic_value_enhanced(ticker, model_type="DCF"):
    hist, info, fin, cf = fetch_stock_data(ticker)
    if info is None:
        return {}, []
    try:
        fin = fin if isinstance(fin, pd.DataFrame) else pd.DataFrame()
        cf = cf if isinstance(cf, pd.DataFrame) else pd.DataFrame()
        discount_rates = [0.08, DISCOUNT_RATE_DEFAULT, 0.10]
        shares = info.get("sharesOutstanding", 1e9) / 1e9

        # growth scenarios – you can tweak per ticker if needed
        scenarios = {
            "Conservative": {"y1_5": 0.02 if ticker == "UNH" else 0.05, "y6_10": 0.03},
            "Default":      {"y1_5": 0.0241 if ticker == "UNH" else 0.1447, "y6_10": 0.04},
            "Optimistic":   {"y1_5": 0.1047 if ticker == "UNH" else 0.20, "y6_10": 0.077},
        }

        results, sens = {}, []
        for sc, gr in scenarios.items():
            sc_res = {}
            for dr in discount_rates:
                if model_type == "DCF":
                    fcf0 = cf.loc["Free Cash Flow"].iloc[0] / \
                        1e9 if not cf.empty and "Free Cash Flow" in cf.index else 13
                    if ticker == "NVO":
                        fcf0 *= DKK_TO_USD
                    fcf = fcf0
                    pv = []
                    for y in range(1, 11):
                        g = gr["y1_5"] if y <= 5 else gr["y6_10"]
                        fcf *= (1 + g)
                        pv.append(fcf / (1 + dr) ** y)
                    tv = (fcf * (1 + TERMINAL_RATE)) / (dr - TERMINAL_RATE)
                    tv_pv = tv / (1 + dr) ** 10
                    iv = (sum(pv) + tv_pv) / shares
                else:   # Discounted Net Income (UNH style)
                    ni0 = fin.loc["Net Income"].iloc[0] / \
                        1e9 if "Net Income" in fin.index else 20
                    ni = ni0
                    pv = []
                    for y in range(1, 11):
                        g = gr["y1_5"] if y <= 5 else gr["y6_10"]
                        ni *= (1 + g)
                        pv.append(ni / (1 + dr) ** y)
                    tv = (ni * (1 + TERMINAL_RATE)) / (dr - TERMINAL_RATE)
                    tv_pv = tv / (1 + dr) ** 10
                    iv = (sum(pv) + tv_pv) / shares
                sc_res[f"{dr:.1%}"] = round(iv, 2)
                sens.append(
                    {"Scenario": sc, "Discount": f"{dr:.1%}", "IV": round(iv, 2)})
            results[sc] = sc_res
        return results, sens
    except Exception as e:
        st.warning(f"Valuation error for {ticker}: {e}")
        return {}, []


def plot_intrinsic_value_analysis(ticker, iv_res, sens, cur_price):
    if not iv_res:
        return None
    df = pd.DataFrame(sens)
    heat = df.pivot(index="Scenario", columns="Discount", values="IV")
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"{ticker} – Intrinsic Value Sensitivity",
                        f"{ticker} – Margin of Safety (5.4% rate)"),
        specs=[[{"type": "heatmap"}], [{"type": "bar"}]]
    )
    fig.add_trace(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index,
                             colorscale="RdYlGn", text=heat.values,
                             texttemplate="%{text}", textfont_size=10), row=1, col=1)

    default_vals = [iv_res[s]["5.4%"] for s in iv_res]
    mos = [((v - cur_price) / v * 100) for v in default_vals]
    colors = ["red" if m < 0 else "orange" if m < 20 else "green" for m in mos]
    fig.add_trace(go.Bar(x=list(iv_res.keys()), y=mos,
                         marker_color=colors, text=[f"{m:.1f}%" for m in mos],
                         textposition="auto"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.update_layout(height=650, title_text=f"{ticker} Valuation")
    return fig


def technical_analysis_enhanced(ticker):
    hist, info, fin, cf = fetch_stock_data(ticker, period="1y")
    if hist is None or hist.empty:
        return None, None
    h = hist.copy()
    h["SMA20"] = h["Close"].rolling(20).mean()
    h["SMA50"] = h["Close"].rolling(50).mean()
    h["SMA200"] = h["Close"].rolling(200).mean()

    # Bollinger
    h["BBmid"] = h["Close"].rolling(20).mean()
    bb_std = h["Close"].rolling(20).std()
    h["BBup"] = h["BBmid"] + bb_std * 2
    h["BBdn"] = h["BBmid"] - bb_std * 2

    # RSI
    delta = h["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn
    h["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = h["Close"].ewm(span=12, adjust=False).mean()
    ema26 = h["Close"].ewm(span=26, adjust=False).mean()
    h["MACD"] = ema12 - ema26
    h["MACD_sig"] = h["MACD"].ewm(span=9, adjust=False).mean()
    h["MACD_hist"] = h["MACD"] - h["MACD_sig"]

    cur = h["Close"].iloc[-1]
    sma50 = h["SMA50"].iloc[-1]
    sma200 = h["SMA200"].iloc[-1]
    rsi = h["RSI"].iloc[-1]
    macd = h["MACD"].iloc[-1]
    support = h["Low"].tail(60).min()
    resist = h["High"].tail(60).max()

    trend = ("Strong Uptrend" if cur > sma50 > sma200 else
             "Strong Downtrend" if cur < sma50 < sma200 else
             "Short-term Uptrend" if cur > sma50 else "Sideways/Weak")

    summary = {
        "Ticker": ticker,
        "Price": round(cur, 2),
        "SMA50": round(sma50, 2),
        "SMA200": round(sma200, 2),
        "RSI": round(rsi, 2),
        "MACD": round(macd, 2),
        "Support": round(support, 2),
        "Resistance": round(resist, 2),
        "Trend": trend,
    }
    return h, summary


def plot_technical_analysis(hist, summary):
    if hist is None:
        return None
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{summary['Ticker']} Price & MAs", "RSI", "MACD"),
        row_heights=[0.6, 0.2, 0.2]
    )
    # Candles
    fig.add_trace(go.Candlestick(x=hist.index,
                                 open=hist["Open"], high=hist["High"],
                                 low=hist["Low"], close=hist["Close"],
                                 name="Price"), row=1, col=1)
    # MAs
    for col, name, color in [("SMA20", "SMA20", "orange"),
                             ("SMA50", "SMA50", "blue"),
                             ("SMA200", "SMA200", "red")]:
        fig.add_trace(go.Scatter(x=hist.index, y=hist[col],
                                 name=name, line=dict(color=color)), row=1, col=1)
    # Bollinger
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BBup"], name="BB Up",
                             line=dict(color="gray", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["BBdn"], name="BB Dn",
                             line=dict(color="gray", dash="dash"),
                             fill="tonexty", fillcolor="rgba(200,200,200,0.2)"), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI",
                             line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD",
                             line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_sig"], name="Signal",
                             line=dict(color="red")), row=3, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_hist"], name="Hist",
                         marker_color="gray"), row=3, col=1)

    fig.update_layout(height=800, showlegend=True,
                      title_text=f"{summary['Ticker']} Technicals")
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ────────────────────── MAIN APP ──────────────────────
def main():
    st.title("Trading-Analysis Dashboard")
    st.caption(
        "Enter tickers → get screening, fundamentals, valuation, technicals, decision matrix & allocation")

    # Method description text mapping (detailed 6-step protocol)
    method_descriptions = {
        "Overview": '''
**Six-Step Investment Analysis Protocol **

This app implements a 6-step analysis flow: Screening → Sector Benchmarking → Fundamentals → Valuation → Technicals → Decision Matrix. Select any step below to view the full process, inputs, outputs and the logical flow.

Logical flow diagram:
```
        ┌────────────────────────┐
        │ 1. Stock Screening      │
        │  • Price/PE/Volatility  │
        └──────────┬──────────────┘
               │
               ▼
        ┌────────────────────────┐
        │ 2. Sector Benchmarking  │
        │  • ETF vs SPY Ratio     │
        └──────────┬──────────────┘
               │
               ▼
        ┌────────────────────────┐
        │ 3. Fundamental Analysis │
        │  • Growth / ROE / FCF   │
        └──────────┬──────────────┘
               │
               ▼
        ┌────────────────────────┐
        │ 4. Valuation Analysis   │
        │  • DCF / Margin Safety  │
        └──────────┬──────────────┘
               │
               ▼
        ┌────────────────────────┐
        │ 5. Technical Analysis   │
        │  • SMA / RSI / MACD     │
        └──────────┬──────────────┘
               │
               ▼
        ┌────────────────────────┐
        │ 6. Decision Matrix      │
        │  • BUY / HOLD / SELL    │
        └────────────────────────┘
```
''',

        "Screening": '''
1️⃣ Stock Screening — “Find Potential Winners"

Objective: Identify companies worth analyzing.
Inputs: Market data, price history, valuation ratios.
Process:
- Collect price and volume data (via yfinance).
- Calculate 52-week highs/lows, % change, volatility, P/E, market cap.
- Rank or filter by performance and stability.
Output: Shortlist of fundamentally strong, liquid stocks for deeper study.
''',

        "Sector Benchmarking": '''
2️⃣ Sector Benchmarking — “Check the Macro Context"

Objective: Understand if the sector is outperforming or lagging.
Inputs: Sector ETF (e.g., XLV) and market benchmark (e.g., SPY).
Process:
- Compute relative performance ratio (Sector ETF / SPY).
- Visualize the trend to see sector strength or weakness.
Output: Sector performance insight to guide allocation bias.
''',

        "Fundamentals": '''
3️⃣ Fundamental Analysis — “Measure the Business Engine"

Objective: Evaluate financial health and growth.
Inputs: Financial statements (income statement, cash flow, balance sheet).
Process:
- Compute revenue & earnings growth, free cash flow, ROE, current ratio, debt/equity.
- Assess profitability, efficiency, and leverage.
- Visualize growth trends.
Output: Business-quality scorecard (Growth / Profitability / Financial Strength).
''',

        "Intrinsic Value (DCF)": '''
4️⃣ Valuation Analysis — “Estimate Intrinsic Value"

Objective: Determine if the stock is fairly priced.
Inputs: Current earnings or FCF, growth assumptions, discount rates.
Process:
- Project 10 years of cash flows under multiple scenarios.
- Discount them to present value (multiple discount rates).
- Compute intrinsic value per share and margin of safety.
Output: Intrinsic value vs. market price table + valuation heatmap.
''',

        "Technical Analysis": '''
5️⃣ Technical Analysis — “Time the Entry"

Objective: Identify optimal buy or sell points.
Inputs: Daily price data.
Process:
- Compute SMA (20/50/200), Bollinger Bands, RSI, MACD.
- Detect uptrend/downtrend, support/resistance.
- Generate Buy/Hold/Sell signals.
Output: Technical summary of trend and momentum.
''',

        "ETF/Pair Relative Strength": '''
6️⃣ Integrated Decision Matrix — “Synthesize Everything"

Objective: Make an actionable investment call.
Inputs: Scores from fundamentals, valuation, and technicals.
Process:
- Combine weighted scores.
- Classify stock as BUY / HOLD / SELL.
- Optionally rank by conviction or margin of safety.
Output: Final decision with supporting evidence.
''',
        "Step 0 — Ticker Selection Logic": '''
🧩 Step 0 — “Ticker Selection Logic"

Before Step 1 (screening), there must be a clear selection protocol that defines what universe of symbols enters your system.

We’ll look at it in two layers:

Layer	What It Means	Example Source
A. Strategic Universe (Macro level)	Which sectors or themes you care about.	e.g., “Healthcare”, “AI & Semiconductors”, “Dividend Growth”
B. Tactical Symbols (Micro level)	Which individual tickers represent those areas.	e.g., UNH, JNJ, LLY, XLV, SPY

⚙️ How Yusup Selects Tickers

From the transcript and his broader framework, he chooses in three layers of logic:

1️⃣ Top-Down (Macro Trend / Sector Strength)

Start with sector ETFs (e.g., XLV, XLF, XLK).

Compare them to the market benchmark (SPY) using relative strength.

Focus only on outperforming sectors.

2️⃣ Bottom-Up (Fundamental Quality within Sector)

Within that strong sector, identify:

Large-cap leaders with stable earnings.

Consistent FCF, strong ROE, low debt.

e.g., In Healthcare → UNH, LLY, NVO, PFE.

3️⃣ Watchlist Refinement (Quantitative Filter)

Use metrics like revenue CAGR > 8 %, ROE > 15 %, low debt/equity.

Drop those failing basic quality criteria.

So, he starts broad (sector), then zooms into quality leaders — macro → micro → fundamentals.
'''
    }

    # Show the selected method description in an expander
    with st.expander(f"Method: {selected_method}"):
        st.markdown(method_descriptions.get(
            selected_method, "No description available."))

    # Note: Step 0 guidance is included in the method descriptions above.
    # Active universe is always taken from manual inputs (for clarity and control)
    active_stocks = stocks
    active_etfs = etfs
    if not active_stocks:
        st.warning("Add at least one stock ticker in the sidebar.")
        return

    # ------------------------------------------------------------------
    # 1. Screening
    # ------------------------------------------------------------------
    st.subheader("1. Stock & ETF Screening")
    screen_df, price_data = screen_stocks_enhanced(active_stocks + active_etfs)
    st.dataframe(screen_df.style.format(
        {"Price": "${:,.2f}"}), use_container_width=True)

    if price_data:
        st.plotly_chart(plot_price_comparison(
            price_data), use_container_width=True)

    # ------------------------------------------------------------------
    # 2. ETF relative strength
    # ------------------------------------------------------------------
    if len(etfs) >= 2:
        st.subheader(f"2. {etfs[0]} vs {etfs[1]} Relative Strength")
        latest, low, series = compare_etfs_enhanced()
        if latest:
            col1, col2 = st.columns(2)
            col1.metric("Latest Ratio", f"{latest:.2f}")
            col2.metric("1-Year Low", f"{low:.2f}")
            st.plotly_chart(plot_etf_comparison(
                series), use_container_width=True)

    # ------------------------------------------------------------------
    # Pair analysis (user-run)
    # ------------------------------------------------------------------
    st.subheader("Pair Analysis (custom)")
    if run_pair:
        combined = stocks + etfs
        if len(combined) < 2:
            st.warning(
                "Add at least two tickers in the Configuration to run pair analysis.")
        else:
            a = combined[0].strip().upper()
            b = combined[1].strip().upper()
            latest_p, low_p, series_p, z_p, corr_p = compare_pair(
                a, b, period=pair_period)
            if latest_p is None:
                st.warning(
                    f"No overlapping data for {a} and {b} or fetch failed.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Latest Ratio", f"{latest_p:.2f}")
                c2.metric("Period Low", f"{low_p:.2f}")
                # show advanced pair plot (ratio, z-score, rolling corr)
                st.plotly_chart(plot_pair_advanced(
                    series_p, z_p, corr_p, a, b), use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Fundamentals
    # ------------------------------------------------------------------
    st.subheader("3. Fundamental Snapshot")
    fund_list = [check_fundamentals_enhanced(t) for t in stocks]
    fund_df = pd.DataFrame(fund_list)
    st.dataframe(fund_df, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Intrinsic Value
    # ------------------------------------------------------------------
    st.subheader("4. Intrinsic Value (DCF / Discounted Earnings)")
    iv_tabs = st.tabs([f"{t} Valuation" for t in stocks])
    for tab, ticker in zip(iv_tabs, stocks):
        with tab:
            model = "Net Income" if ticker == "UNH" else "DCF"
            iv_res, sens = calculate_intrinsic_value_enhanced(ticker, model)
            if iv_res:
                hist, info, fin, cf = fetch_stock_data(ticker)
                cur_price = hist["Close"].iloc[-1] if hist is not None and not hist.empty else np.nan
                st.write(f"**Current price:** ${cur_price:,.2f}")
                for sc, rates in iv_res.items():
                    st.write(f"**{sc}**")
                    for r, v in rates.items():
                        st.write(f"- {r} → **${v:,.0f}**")
                st.plotly_chart(plot_intrinsic_value_analysis(ticker, iv_res, sens, cur_price),
                                use_container_width=True)

    # ------------------------------------------------------------------
    # 5. Technicals
    # ------------------------------------------------------------------
    st.subheader("5. Technical Analysis")
    tech_tabs = st.tabs([f"{t} Chart" for t in stocks])
    for tab, ticker in zip(tech_tabs, stocks):
        with tab:
            hist, summ = technical_analysis_enhanced(ticker)
            if summ:
                col1, col2, col3 = st.columns(3)
                col1.metric("Price", f"${summ['Price']}")
                col2.metric("RSI", summ["RSI"])
                col3.metric("Trend", summ["Trend"])
                st.plotly_chart(plot_technical_analysis(
                    hist, summ), use_container_width=True)

    # ------------------------------------------------------------------
    # 6. Decision Matrix
    # ------------------------------------------------------------------
    st.subheader("6. Decision Matrix")
    decision = []
    for t in stocks:
        # price
        hist, info, fin, cf = fetch_stock_data(t)
        price = hist["Close"].iloc[-1] if hist is not None and not hist.empty else np.nan

        # fundamental score (revenue growth)
        fg = next((f for f in fund_list if f["Ticker"] == t), {})
        rev_g = fg.get("Rev Growth %", 0) if isinstance(
            fg.get("Rev Growth %"), (int, float)) else 0
        fund_score = max(min(rev_g / 10, 2), -2)

        # technical score
        _, ts = technical_analysis_enhanced(t)
        trend_score = 1 if ts and "Uptrend" in ts["Trend"] else - \
            1 if ts and "Downtrend" in ts["Trend"] else 0
        rsi = ts["RSI"] if ts else 50
        rsi_score = 1 if rsi < 35 else -1 if rsi > 65 else 0
        tech_score = trend_score + rsi_score

        # valuation score
        iv_res, _ = calculate_intrinsic_value_enhanced(
            t, "Net Income" if t == "UNH" else "DCF")
        val_score = 0
        if iv_res and "5.4%" in iv_res.get("Default", {}):
            iv = iv_res["Default"]["5.4%"]
            mos = (iv - price) / iv * 100
            if mos >= 30:
                val_score = 2
            elif mos >= 20:
                val_score = 1
            elif mos >= 0:
                val_score = 0
            else:
                val_score = -1

        total = fund_score + tech_score + val_score
        rec = ("STRONG BUY" if total >= 3 else
               "BUY" if total >= 1 else
               "HOLD" if total >= -1 else
               "AVOID")
        decision.append({
            "Ticker": t,
            "Price": f"${price:,.2f}",
            "Fund": f"{fund_score:.1f}",
            "Tech": f"{tech_score:.1f}",
            "Val": f"{val_score:.1f}",
            "Total": f"{total:.1f}",
            "Recommendation": rec,
        })
    dec_df = pd.DataFrame(decision)
    st.dataframe(dec_df, use_container_width=True)

    # ------------------------------------------------------------------
    # 7. Portfolio Allocation (only BUY/STRONG BUY)
    # ------------------------------------------------------------------
    st.subheader("7. Suggested Allocation (max 50 % of equity)")
    alloc = {}
    total_alloc = 0
    for row in decision:
        if row["Recommendation"] in ("STRONG BUY", "BUY"):
            score = float(row["Total"])
            pct = max(min((score / 6) * 20, 25), 5)
            alloc[row["Ticker"]] = pct
            total_alloc += pct
    if total_alloc > 50:
        factor = 50 / total_alloc
        alloc = {k: v * factor for k, v in alloc.items()}
    alloc_df = pd.DataFrame([
        {"Ticker": k, "Allocation %": f"{v:.1f}"} for k, v in alloc.items()
    ])
    if not alloc_df.empty:
        st.dataframe(alloc_df, use_container_width=True)
        pie = go.Figure(data=[go.Pie(labels=list(alloc.keys()) + ["Other"],
                                     values=list(alloc.values()) +
                                     [100 - sum(alloc.values())],
                                     hole=0.4)])
        pie.update_layout(title="Portfolio Allocation")
        st.plotly_chart(pie, use_container_width=True)

    # ------------------------------------------------------------------
    # Monitoring: Stop-loss & Take-profit levels
    # ------------------------------------------------------------------
    st.subheader("Monitoring: Stop-loss & Take-profit")
    stop_list = []
    for t in stocks:
        hist, info, fin, cf = fetch_stock_data(t)
        if hist is None or hist.empty:
            continue
        cur = hist["Close"].iloc[-1]
        stop = cur * 0.85
        take = cur * 1.30
        stop_list.append({"Ticker": t, "Current": f"${cur:.2f}",
                         "Stop Loss (15%)": f"${stop:.2f}", "Take Profit (30%)": f"${take:.2f}"})
    if stop_list:
        stop_df = pd.DataFrame(stop_list)
        st.dataframe(stop_df, use_container_width=True)

    # ------------------------------------------------------------------
    # 8. Download report
    # ------------------------------------------------------------------
    st.subheader("8. Download CSV Report")
    report = {
        "Screening": screen_df,
        "Fundamentals": fund_df,
        "Decision": dec_df,
    }
    csv_buffer = io.StringIO()
    for name, df in report.items():
        csv_buffer.write(f"--- {name} ---\n")
        csv_buffer.write(df.to_csv(index=False))
        csv_buffer.write("\n\n")
    csv_bytes = csv_buffer.getvalue().encode()
    st.download_button(
        label="Download Full Report (CSV)",
        data=csv_bytes,
        file_name=f"trading_analysis_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    st.success(
        "Analysis complete! Adjust inputs in the sidebar to explore other tickers.")


if __name__ == "__main__":
    main()
