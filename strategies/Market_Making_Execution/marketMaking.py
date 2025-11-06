import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
import concurrent.futures
import traceback
import time
import random

# --------------------------------------------------------------
# In-memory indicator cache
# --------------------------------------------------------------
_INDICATOR_CACHE = {}


def progress_with_rate(pbar, status, done, total, start_time, label_text="Optimizing"):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else float("inf")
    frac = min(done / total, 1.0) if total > 0 else 1.0
    pbar.progress(frac)
    if remaining == float("inf"):
        status.text(f"{label_text} — {done}/{total}  |  ETA: calculating...")
    else:
        status.text(
            f"{label_text} — {done}/{total}  |  ETA: {remaining:,.1f}s")


def get_indicators_cached(data, sw, lw, bstd, supw, base_spread):
    key = (int(sw), int(lw), float(bstd), int(supw))
    if key in _INDICATOR_CACHE:
        return _INDICATOR_CACHE[key]
    dfi = calculate_indicators(data, sw, lw, bstd, supw, base_spread)
    _INDICATOR_CACHE[key] = dfi
    return dfi


def _eval_single(params, data, capital, cost, base_spread):
    try:
        sw, lw, spt, mp, supw, sl, tp, bstd = params
        if sw >= lw:
            return (params, -np.inf, None)

        dfi = get_indicators_cached(data, sw, lw, bstd, supw, base_spread)
        if len(dfi) < 10:
            return (params, -np.inf, None)

        port = run_backtest(dfi, capital, spt, cost, mp, sl, tp,
                            sw, lw, bstd, supw, base_spread)
        mets = calc_metrics(port)
        sharpe = mets.get("Sharpe", -np.inf)
        if sharpe is None:
            sharpe = -np.inf
        return (params, sharpe, mets)
    except Exception:
        return (params, -np.inf, {"error": traceback.format_exc()})


warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------
st.set_page_config(page_title="Market Making Strategy", layout="wide")
st.title("Market Making with Volatility-Scaled Spreads")
st.markdown(
    """
    **Quote bid and ask around the mid-price. Earn the spread when filled.**  
    Uses volatility to widen/narrow the spread and MA + Bollinger for entry/exit.
    """
)

# --------------------------------------------------------------
# Sidebar – fully configurable
# --------------------------------------------------------------
st.sidebar.header("Configuration")

# Core inputs
symbol = st.sidebar.text_input("Ticker", value="AAPL")
days_back = st.sidebar.slider(
    "Days of 15-min data", 3, 60, 14,
    help="More data = slower optimization")
interval = "15m"
end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
    value=100_000, step=10_000)
transaction_cost_pct = st.sidebar.slider(
    "Transaction Cost (%)", 0.0, 1.0, 0.10, 0.01,
    help="Per trade commission + slippage")
transaction_cost = transaction_cost_pct / 100

base_spread_bps = st.sidebar.slider(
    "Base Spread (bps)", 5, 50, 10, 1,
    help="Base spread in basis points (0.01% = 1 bp). Final = base × (1 + volatility)"
) / 10_000

# Manual strategy parameters
st.sidebar.subheader("Strategy Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    short_window = st.slider("Short MA Window", 1, 30, 7)
    long_window = st.slider("Long MA Window", 5, 100, 20,
                            help="Must be > Short MA")
    shares_per_trade = st.slider("Shares per Trade", 1, 50, 5)
    max_position = st.slider("Max Position", 1, 100, 20)
with col2:
    support_window = st.slider("Support Window", 5, 100, 20)
    stop_loss_pct = st.slider("Stop Loss (%)", 0.1, 10.0, 1.0, 0.1)
    take_profit_pct = st.slider("Take Profit (%)", 0.1, 15.0, 2.0, 0.1)
    boll_std = st.slider("Bollinger Std Dev", 0.5, 3.0, 2.0, 0.1)

if short_window >= long_window:
    st.sidebar.error("Short MA must be < Long MA")
    long_window = short_window + 1

stop_loss = stop_loss_pct / 100
take_profit = take_profit_pct / 100

# Optimization toggle + trial sliders
optimize = st.sidebar.checkbox("Run Parameter Optimization", value=False)
if optimize:
    st.sidebar.info(
        "Grid search will override manual settings. Takes 1–5 min.")
    st.sidebar.subheader("Optimization Settings")

    # Coarse
    min_coarse_trials = st.sidebar.slider(
        "Min Coarse Trials", 100, 2000, 500, 100,
        help="Minimum random coarse-grid trials (start fast)")
    max_coarse_trials = st.sidebar.slider(
        "Max Coarse Trials", 500, 5000, 2000, 100,
        help="Maximum coarse-grid trials (more thorough)")

    # Fine
    min_fine_trials = st.sidebar.slider(
        "Min Fine Trials", 100, 2000, 500, 100,
        help="Minimum random fine-grid trials around best coarse")
    max_fine_trials = st.sidebar.slider(
        "Max Fine Trials", 500, 5000, 1500, 100,
        help="Maximum fine-grid trials (refine the optimum)")

run = st.sidebar.button("Run Market Maker", type="primary")

# --------------------------------------------------------------
# Data download
# --------------------------------------------------------------


@st.cache_data(show_spinner=False)
def fetch_data(symbol, start, end, interval):
    data = yf.download(symbol, start=start, end=end,
                       interval=interval, progress=False, auto_adjust=False)
    if data.empty:
        st.error("No data returned. Check ticker or date range.")
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    return data[required].dropna()


# --------------------------------------------------------------
# Indicator calculation
# --------------------------------------------------------------
def calculate_indicators(df, short_window, long_window, boll_std,
                         support_window, base_spread):
    df = df.copy()
    close = df["Close"]
    df["Short_MA"] = close.rolling(window=short_window).mean()
    df["Long_MA"] = close.rolling(window=long_window).mean()
    df["Volatility"] = close.rolling(window=long_window).std()
    df["Bollinger_Upper"] = df["Short_MA"] + boll_std * df["Volatility"]
    df["Bollinger_Lower"] = df["Short_MA"] - boll_std * df["Volatility"]
    df["Support"] = df["Bollinger_Lower"].rolling(window=support_window).min()
    df["Spread"] = base_spread * (1 + df["Volatility"] / close)
    df["Bid"] = close - df["Spread"] / 2
    df["Ask"] = close + df["Spread"] / 2
    return df.dropna()


# --------------------------------------------------------------
# Signal generation
# --------------------------------------------------------------
def generate_signals(df, stop_loss, take_profit):
    signals = pd.DataFrame(index=df.index, columns=["Buy", "Sell"]).fillna(0)
    in_position = False
    entry_price = 0.0
    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        support = df["Support"].iloc[i]
        if not in_position and price <= support:
            signals.loc[signals.index[i], "Buy"] = 1
            in_position = True
            entry_price = price
        elif in_position:
            if price >= entry_price * (1 + take_profit) or price <= entry_price * (1 - stop_loss):
                signals.loc[signals.index[i], "Sell"] = 1
                in_position = False
    return signals


# --------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------
def run_backtest(df, capital, shares_per_trade, cost, max_pos,
                 stop_loss, take_profit, short_w, long_w,
                 boll_std, supp_w, base_spread):
    portfolio = pd.DataFrame(index=df.index,
                             columns=["Holdings", "Cash", "Total", "Returns"])
    portfolio["Holdings"] = 0.0
    portfolio["Cash"] = capital
    portfolio["Total"] = capital
    portfolio["Returns"] = 0.0

    signals = generate_signals(df, stop_loss, take_profit)

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        bid = df["Bid"].iloc[i]
        ask = df["Ask"].iloc[i]

        portfolio.loc[portfolio.index[i],
                      "Holdings"] = portfolio["Holdings"].iloc[i-1]
        portfolio.loc[portfolio.index[i], "Cash"] = portfolio["Cash"].iloc[i-1]

        # ENTRY
        if signals["Buy"].iloc[i] and portfolio["Holdings"].iloc[i] < max_pos:
            buy_shares = min(
                shares_per_trade,
                max_pos - portfolio["Holdings"].iloc[i],
                portfolio["Cash"].iloc[i] // (ask * (1 + cost))
            )
            if buy_shares > 0:
                portfolio.loc[portfolio.index[i], "Holdings"] += buy_shares
                portfolio.loc[portfolio.index[i],
                              "Cash"] -= buy_shares * ask * (1 + cost)

        # EXIT
        if signals["Sell"].iloc[i] and portfolio["Holdings"].iloc[i] > 0:
            sell_shares = min(shares_per_trade, portfolio["Holdings"].iloc[i])
            if sell_shares > 0:
                portfolio.loc[portfolio.index[i], "Holdings"] -= sell_shares
                portfolio.loc[portfolio.index[i],
                              "Cash"] += sell_shares * bid * (1 - cost)

        portfolio.loc[portfolio.index[i], "Total"] = (
            portfolio.loc[portfolio.index[i], "Cash"] +
            portfolio.loc[portfolio.index[i], "Holdings"] * price
        )

    portfolio["Returns"] = portfolio["Total"].pct_change().fillna(0)
    return portfolio


# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------
def calc_metrics(portfolio, rf=0.02):
    returns = portfolio["Returns"].dropna()
    if len(returns) == 0:
        return {k: 0.0 for k in
                "Total Return,CAGR,Volatility,Sharpe,Sortino,Max DD,Win Rate,Profit Factor".split(",")}

    excess = returns - rf / (252 * 24 * 4)
    total_ret = (portfolio["Total"].iloc[-1] / portfolio["Total"].iloc[0]) - 1
    periods_per_year = 252 * 24 * 4
    cagr = (1 + total_ret) ** (periods_per_year / len(portfolio)) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = np.sqrt(periods_per_year) * excess.mean() / \
        returns.std() if returns.std() != 0 else 0
    sortino = (np.sqrt(periods_per_year) * excess.mean() /
               returns[returns < 0].std()) if len(returns[returns < 0]) > 0 else 0
    max_dd = (portfolio["Total"] / portfolio["Total"].cummax() - 1).min()
    win_rate = (returns > 0).mean()
    profit_factor = (abs(returns[returns > 0].sum() / returns[returns < 0].sum())
                     if (returns < 0).any() else np.inf)

    return {
        "Total Return": total_ret, "CAGR": cagr, "Volatility": vol,
        "Sharpe": sharpe, "Sortino": sortino, "Max DD": max_dd,
        "Win Rate": win_rate, "Profit Factor": profit_factor
    }


# --------------------------------------------------------------
# OPTIMIZATION – progressive coarse → fine (both trial-controlled)
# --------------------------------------------------------------
def optimize_params(data, capital, cost, base_spread,
                    min_coarse_trials, max_coarse_trials,
                    min_fine_trials, max_fine_trials):
    # ---------- Coarse grid ----------
    sw_range = range(max(1, short_window - 5), min(30, short_window + 6), 3)
    lw_range = range(max(short_window + 5, long_window - 20),
                     min(120, long_window + 25), 10)
    spt_range = [max(1, shares_per_trade // 2), shares_per_trade,
                 shares_per_trade * 2]
    mp_range = [max(1, max_position // 2), max_position,
                min(100, max_position * 2)]
    supw_range = range(max(5, support_window - 20),
                       min(150, support_window + 25), 10)
    sl_range = [stop_loss * 0.5, stop_loss, stop_loss * 1.5]
    tp_range = [take_profit * 0.5, take_profit, take_profit * 1.5]
    bstd_range = [max(0.5, boll_std - 1), boll_std, min(3.0, boll_std + 1)]

    full_coarse = [(sw, lw, spt, mp, supw, sl, tp, bstd)
                   for sw in sw_range for lw in lw_range
                   for spt in spt_range for mp in mp_range
                   for supw in supw_range for sl in sl_range
                   for tp in tp_range for bstd in bstd_range
                   if sw < lw]

    if not full_coarse:
        st.warning("No valid coarse-grid combinations.")
        return None

    max_coarse_trials = min(max_coarse_trials, len(full_coarse))
    random.shuffle(full_coarse)

    coarse_grid = full_coarse[:min_coarse_trials]
    remaining_coarse = full_coarse[min_coarse_trials:]

    best_sharpe = -np.inf
    best = None
    done = 0
    pbar = st.progress(0)
    status = st.empty()
    start_time = time.time()

    # ---------- Helper for parallel evaluation ----------
    def evaluate(grid, label_suffix="", total_for_progress=None):
        nonlocal best, best_sharpe, done
        cpu = os.cpu_count() or 1
        workers = min(max(1, cpu - 1), len(grid))
        total = total_for_progress or len(grid)

        try:
            with concurrent.futures.ProcessPoolExecutor(workers) as exec:
                futures = {exec.submit(_eval_single, p, data, capital,
                                       cost, base_spread): p for p in grid}
                for f in concurrent.futures.as_completed(futures):
                    params, sharpe, _ = f.result()
                    done += 1
                    progress_with_rate(pbar, status, done, total,
                                       start_time, f"Coarse Search{label_suffix}")
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best = params
        except Exception:
            # fallback to threads
            with concurrent.futures.ThreadPoolExecutor(workers) as exec:
                futures = {exec.submit(_eval_single, p, data, capital,
                                       cost, base_spread): p for p in grid}
                for f in concurrent.futures.as_completed(futures):
                    params, sharpe, _ = f.result()
                    done += 1
                    progress_with_rate(pbar, status, done, total,
                                       start_time, f"Coarse Search (threads){label_suffix}")
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best = params

    # ---------- Coarse progressive ----------
    evaluate(coarse_grid, f" [initial {len(coarse_grid)}]")

    trials_done = len(coarse_grid)
    while remaining_coarse and trials_done < max_coarse_trials:
        batch = min(
            max(50, int(0.15 * trials_done) + 50),
            max_coarse_trials - trials_done,
            len(remaining_coarse)
        )
        next_batch = remaining_coarse[:batch]
        remaining_coarse = remaining_coarse[batch:]
        evaluate(next_batch, f" [+{batch}]")
        coarse_grid.extend(next_batch)
        trials_done += batch

    pbar.empty()
    status.empty()
    if best is None:
        st.warning("Coarse search yielded no valid result.")
        return None

    # ---------- Fine grid around best coarse ----------
    sw, lw, spt, mp, supw, sl, tp, bstd = best
    fine_sw = range(max(1, sw - 2), sw + 3)
    fine_lw = range(max(sw + 1, lw - 4), lw + 5, 2)
    fine_spt = [max(1, spt - 1), spt, spt + 1]
    fine_mp = [max(1, mp - 5), mp, mp + 5]
    fine_supw = range(max(5, supw - 5), supw + 6, 2)
    fine_sl = [sl * 0.9, sl, sl * 1.1]
    fine_tp = [tp * 0.9, tp, tp * 1.1]
    fine_bstd = [max(0.5, bstd - 0.2), bstd, min(3.0, bstd + 0.2)]

    full_fine = [(sw, lw, spt, mp, supw, sl, tp, bstd)
                 for sw in fine_sw for lw in fine_lw
                 for spt in fine_spt for mp in fine_mp
                 for supw in fine_supw for sl in fine_sl
                 for tp in fine_tp for bstd in fine_bstd
                 if sw < lw]

    if not full_fine:
        st.info("Fine grid empty – using best coarse.")
        return best

    max_fine_trials = min(max_fine_trials, len(full_fine))
    random.shuffle(full_fine)

    fine_grid = full_fine[:min_fine_trials]
    remaining_fine = full_fine[min_fine_trials:]

    done = 0
    pbar = st.progress(0)
    status = st.empty()
    start_time = time.time()

    # ---------- Fine progressive ----------
    evaluate(fine_grid, f" → Fine Search [initial {len(fine_grid)}]",
             total_for_progress=max_fine_trials)

    fine_done = len(fine_grid)
    while remaining_fine and fine_done < max_fine_trials:
        batch = min(
            max(30, int(0.2 * fine_done) + 30),
            max_fine_trials - fine_done,
            len(remaining_fine)
        )
        next_batch = remaining_fine[:batch]
        remaining_fine = remaining_fine[batch:]
        evaluate(next_batch, f" → Fine Search [+{batch}]",
                 total_for_progress=max_fine_trials)
        fine_grid.extend(next_batch)
        fine_done += batch

    pbar.empty()
    status.empty()

    st.info(
        f"**Optimization finished!**  "
        f"Coarse: {trials_done:,}/{max_coarse_trials:,}  |  "
        f"Fine: {fine_done:,}/{max_fine_trials:,}  |  "
        f"Best Sharpe: {best_sharpe:.2f}"
    )
    return best


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------
if run:
    with st.spinner("Fetching 15-min data..."):
        data = fetch_data(symbol, start_date, end_date, interval)
    if data is None:
        st.stop()

    best_params = None
    if optimize:
        with st.spinner("Optimizing parameters..."):
            best_params = optimize_params(
                data, initial_capital, transaction_cost, base_spread_bps,
                min_coarse_trials, max_coarse_trials,
                min_fine_trials, max_fine_trials
            )
        if best_params:
            sw, lw, spt, mp, supw, sl, tp, bstd = best_params
            dfi_opt = calculate_indicators(
                data, sw, lw, bstd, supw, base_spread_bps)
            port_opt = run_backtest(dfi_opt, initial_capital, spt,
                                    transaction_cost, mp, sl, tp,
                                    sw, lw, bstd, supw, base_spread_bps)
            final_sharpe = calc_metrics(port_opt)["Sharpe"]
            st.success(
                f"Optimization complete – Final Sharpe: {final_sharpe:.2f}")
        else:
            st.warning("Optimization failed. Falling back to manual settings.")
            optimize = False

    # Use optimized or manual parameters
    if best_params:
        sw, lw, spt, mp, supw, sl, tp, bstd = best_params
        source = "Optimized"
    else:
        sw, lw, spt, mp, supw, sl, tp, bstd = (
            short_window, long_window, shares_per_trade, max_position,
            support_window, stop_loss, take_profit, boll_std)
        source = "Manual"

    dfi = calculate_indicators(data, sw, lw, bstd, supw, base_spread_bps)
    portfolio = run_backtest(dfi, initial_capital, spt, transaction_cost,
                             mp, sl, tp, sw, lw, bstd, supw, base_spread_bps)
    metrics = calc_metrics(portfolio)

    # --------------------------------------------------------------
    # Educational section
    # --------------------------------------------------------------
    st.markdown("---")
    st.header("Market Making – Theory")
    st.markdown(
        """
        ## 1. What is Market Making?
        - Quote **bid** and **ask** around the mid-price.
        - Earn the **spread** when both sides are filled.
        - Goal: capture **bid-ask bounce**, not directional prediction.

        ## 2. Volatility-Scaled Spread
        - `Spread = base_bps × (1 + σ/price)`
        - Wider in volatile markets → higher profit per fill
        - Narrower in calm markets → more fills

        ## 3. Entry / Exit
        - **Buy** when price touches **support** (rolling min of lower Bollinger)
        - **Sell** on **take-profit** or **stop-loss**

        ## 4. Risks
        - Inventory risk (holding during a crash)
        - Adverse selection (filled only on losing side)
        - Slippage in live execution
        """
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    st.subheader(f"{symbol} – Market Making Dashboard")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=("Price & Quotes", "Bid/Ask Spread",
                        "Portfolio Value", "Holdings"),
        row_heights=[0.5, 0.15, 0.2, 0.15],
    )
    # price + indicators
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Close"], name="Close",
                             line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Short_MA"], name="Short MA",
                             line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Long_MA"], name="Long MA",
                             line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Bollinger_Upper"], name="BB Upper",
                             line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Bollinger_Lower"], name="BB Lower",
                             line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Support"], name="Support",
                             line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Bid"], name="Bid",
                             line=dict(color="red", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Ask"], name="Ask",
                             line=dict(color="lime", dash="dash")), row=1, col=1)
    # spread
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi["Spread"], name="Spread ($)",
                             line=dict(color="purple")), row=2, col=1)
    # equity & holdings
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["Total"],
                             name="Portfolio Value", line=dict(color="blue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["Holdings"],
                             name="Shares Held", line=dict(color="orange")), row=4, col=1)

    fig.update_layout(height=900, showlegend=True, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{metrics['Total Return']:.2%}")
    c2.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{metrics['Max DD']:.2%}")
    c4.metric("Win Rate", f"{metrics['Win Rate']:.1%}")

    with st.expander("All Metrics"):
        for k, v in metrics.items():
            fmt = "{:.2%}" if any(
                x in k for x in ["Return", "DD", "Rate", "Factor"]) else "{:.2f}"
            st.write(f"**{k}**: `{fmt.format(v)}`")

    # --------------------------------------------------------------
    # Parameters table
    # --------------------------------------------------------------
    st.subheader("Parameters Used")
    param_df = pd.DataFrame([[
        sw, lw, spt, mp, supw,
        f"{sl:.1%}", f"{tp:.1%}", bstd,
        f"{base_spread_bps*10000:.0f} bps"
    ]], columns=[
        "Short MA", "Long MA", "Shares/Trade", "Max Pos", "Support Win",
        "Stop Loss", "Take Profit", "BB Std Dev", "Base Spread"
    ])
    param_df.index = [source]
    st.table(param_df)

    # --------------------------------------------------------------
    # CSV export
    # --------------------------------------------------------------
    export = portfolio.copy()
    export["Bid"] = dfi["Bid"]
    export["Ask"] = dfi["Ask"]
    export["Spread"] = dfi["Spread"]
    csv = export.to_csv().encode()
    st.download_button(
        "Download Full Backtest",
        data=csv,
        file_name=f"marketmaking_{symbol}_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )
else:
    st.info("Configure parameters in the sidebar and click **Run Market Maker**.")
    st.markdown(
        """
        ### Quick Tips
        - Use **liquid tickers** (`AAPL`, `NVDA`, `SPY`)
        - Start with **7–21 days** of 15-min data
        - Try **base spread = 10–20 bps**
        - Keep **stop-loss < take-profit**
        """
    )
