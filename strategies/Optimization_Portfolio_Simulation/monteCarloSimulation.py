import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize

# --------------------------------------------------------------
# Page config
# --------------------------------------------------------------
st.set_page_config(page_title="Monte Carlo Portfolio Optimizer", layout="wide")
st.title("Monte Carlo Portfolio Optimization with Efficient Frontier")
st.markdown("Find the optimal investment portfolio using Monte Carlo simulations")

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
st.sidebar.header("Portfolio Configuration")

default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM',
                   'JNJ', 'XOM', 'PG', 'KO', 'WMT']

tickers_input = st.sidebar.text_area(
    "Enter stock tickers (comma-separated)",
    value=", ".join(default_tickers),
    help="e.g. AAPL, MSFT, TSLA"
)

# ---- safe ticker parsing -------------------------------------------------
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

years_back = st.sidebar.slider("Years of historical data", 1, 10, 3)
end_date = datetime.now()
start_date = end_date - timedelta(days=years_back * 365)

num_simulations = st.sidebar.slider(
    "Number of simulations", 1000, 50000, 10000, step=1000)

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)", 0.0, 10.0, 4.0, 0.1) / 100

run_simulation = st.sidebar.button("Run Simulation", type="primary")

# --------------------------------------------------------------
# Robust data download – same pattern as bbrsi.py
# --------------------------------------------------------------


@st.cache_data(show_spinner=False)
def fetch_stock_data(tickers, start, end):
    """
    Returns a DataFrame with columns = tickers and rows = dates.
    Uses **Adj Close** when available, otherwise **Close**.
    """
    try:
        # group_by='ticker' → each ticker becomes a top-level key
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            group_by='ticker',
            auto_adjust=False          # keep Adj Close separate
        )

        if raw is None or raw.empty:
            st.error("Yahoo Finance returned no data.")
            return None

        price_dict = {}
        for ticker in tickers:
            # single-ticker case: raw is a normal DataFrame
            if len(tickers) == 1:
                df = raw.copy()
            else:
                # multi-ticker → raw[ticker] is the sub-DataFrame
                if ticker not in raw:
                    st.warning(f"{ticker} not found in downloaded data.")
                    continue
                df = raw[ticker].copy()

            # prefer Adj Close, fall back to Close
            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
                st.warning(f"Using unadjusted Close for {ticker}")
            else:
                st.warning(f"No price column for {ticker}")
                continue

            # drop NaNs and store
            series = series.dropna()
            if series.empty:
                st.warning(f"All prices NaN for {ticker}")
                continue
            price_dict[ticker] = series

        if not price_dict:
            st.error("No usable price series after processing.")
            return None

        # Align all series on the same index (inner join)
        prices = pd.DataFrame(price_dict)
        prices = prices.dropna(how='all')          # safety
        if prices.empty:
            st.error("All rows became NaN after alignment.")
            return None

        return prices

    except Exception as e:
        st.error(f"Download error: {e}")
        return None


# --------------------------------------------------------------
# Portfolio maths
# --------------------------------------------------------------
def portfolio_metrics(weights, mean_ret, cov_mat, rf):
    ann_ret = np.sum(mean_ret * weights) * 252
    ann_std = np.sqrt(weights.T @ (cov_mat * 252) @ weights)
    sharpe = (ann_ret - rf) / ann_std if ann_std > 0 else 0
    return ann_ret, ann_std, sharpe


def monte_carlo(mean_ret, cov_mat, n_sims, rf):
    n = len(mean_ret)
    results = np.zeros((3, n_sims))          # std, ret, sharpe
    weight_records = []

    prog = st.progress(0)
    for i in range(n_sims):
        w = np.random.random(n)
        w /= w.sum()
        ret, std, shp = portfolio_metrics(w, mean_ret, cov_mat, rf)
        results[:, i] = std, ret, shp
        weight_records.append(w)

        if i % max(1, n_sims // 100) == 0:
            prog.progress(i / n_sims)
    prog.empty()
    return results, weight_records


def min_vol_portfolio(mean_ret, cov_mat, rf):
    n = len(mean_ret)
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    bnds = tuple((0, 1) for _ in range(n))
    res = minimize(lambda w: portfolio_metrics(w, mean_ret, cov_mat, rf)[1],
                   n * [1. / n],
                   method='SLSQP',
                   bounds=bnds,
                   constraints=cons)
    return res


def max_ret_portfolio(mean_ret):
    n = len(mean_ret)
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    bnds = tuple((0, 1) for _ in range(n))
    res = minimize(lambda w: -np.sum(mean_ret * w),
                   n * [1. / n],
                   method='SLSQP',
                   bounds=bnds,
                   constraints=cons)
    return res


def efficient_frontier(mean_ret, cov_mat, rf, n_points=100):
    min_res = min_vol_portfolio(mean_ret, cov_mat, rf)
    min_ret = np.sum(mean_ret * min_res.x) * 252

    max_res = max_ret_portfolio(mean_ret)
    max_ret = np.sum(mean_ret * max_res.x) * 252

    target_rets = np.linspace(min_ret, max_ret * 1.05, n_points)
    frontier = []

    for tr in target_rets:
        cons = (
            {'type': 'eq', 'fun': lambda w: w.sum() - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(mean_ret * w) * 252 - tr}
        )
        bnds = tuple((0, 1) for _ in range(len(mean_ret)))
        res = minimize(lambda w: portfolio_metrics(w, mean_ret, cov_mat, rf)[1],
                       len(mean_ret) * [1. / len(mean_ret)],
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
        if res.success:
            ret, std, _ = portfolio_metrics(res.x, mean_ret, cov_mat, rf)
            frontier.append([std, ret])
    return np.array(frontier)


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------
if run_simulation:
    with st.spinner("Downloading price data…"):
        price_df = fetch_stock_data(tickers, start_date, end_date)

    if price_df is None or price_df.empty:
        st.stop()

    # ----- returns -------------------------------------------------
    daily_ret = price_df.pct_change().dropna()
    mean_ret = daily_ret.mean()
    cov_mat = daily_ret.cov()

    # ----- quick stats ---------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Stocks", len(tickers))
    col2.metric("Observations", len(daily_ret))
    col3.metric("Simulations", f"{num_simulations:,}")

    # ----- Monte Carlo ---------------------------------------------
    with st.spinner(f"Running {num_simulations:,} simulations…"):
        sim_results, sim_weights = monte_carlo(
            mean_ret, cov_mat, num_simulations, risk_free_rate)

    # ----- Efficient frontier --------------------------------------
    with st.spinner("Tracing the efficient frontier…"):
        frontier = efficient_frontier(mean_ret, cov_mat, risk_free_rate)

    # ----- optimal points -------------------------------------------
    max_sharpe_idx = np.argmax(sim_results[2])
    min_vol_idx = np.argmin(sim_results[0])

    max_sharpe_w = sim_weights[max_sharpe_idx]
    min_vol_w = sim_weights[min_vol_idx]

    # ----- Plot ----------------------------------------------------
    st.subheader("Efficient Frontier")
    fig = go.Figure()

    # simulated cloud
    fig.add_trace(go.Scatter(
        x=sim_results[0], y=sim_results[1],
        mode='markers',
        marker=dict(size=4, color=sim_results[2],
                    colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Sharpe")),
        name='Simulated',
        hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>"
    ))

    # frontier line
    fig.add_trace(go.Scatter(
        x=frontier[:, 0], y=frontier[:, 1],
        mode='lines', line=dict(color='red', width=3),
        name='Efficient Frontier'
    ))

    # max Sharpe
    fig.add_trace(go.Scatter(
        x=[sim_results[0, max_sharpe_idx]],
        y=[sim_results[1, max_sharpe_idx]],
        mode='markers',
        marker=dict(size=14, color='gold', symbol='star',
                    line=dict(color='black', width=2)),
        name='Max Sharpe',
        hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{text:.2f}<extra></extra>",
        text=[sim_results[2, max_sharpe_idx]]
    ))

    # min volatility
    fig.add_trace(go.Scatter(
        x=[sim_results[0, min_vol_idx]],
        y=[sim_results[1, min_vol_idx]],
        mode='markers',
        marker=dict(size=14, color='lime', symbol='diamond',
                    line=dict(color='black', width=2)),
        name='Min Volatility',
        hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{text:.2f}<extra></extra>",
        text=[sim_results[2, min_vol_idx]]
    ))

    fig.update_layout(
        height=650,
        xaxis_title="Risk (σ)",
        yaxis_title="Expected Return",
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(tickformat=".1%"),
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----- Optimal portfolio tables ---------------------------------
    st.subheader("Optimal Portfolios")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Max Sharpe Ratio")
        ret, std, shp = portfolio_metrics(
            max_sharpe_w, mean_ret, cov_mat, risk_free_rate)
        st.metric("Return", f"{ret:.2%}")
        st.metric("Risk", f"{std:.2%}")
        st.metric("Sharpe", f"{shp:.3f}")

        df = pd.DataFrame({"Ticker": tickers, "Weight": max_sharpe_w})
        df = df.sort_values("Weight", ascending=False)
        df["Weight"] = df["Weight"].map("{:.2%}".format)
        st.dataframe(df, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("### Min Volatility")
        ret, std, shp = portfolio_metrics(
            min_vol_w, mean_ret, cov_mat, risk_free_rate)
        st.metric("Return", f"{ret:.2%}")
        st.metric("Risk", f"{std:.2%}")
        st.metric("Sharpe", f"{shp:.3f}")

        df = pd.DataFrame({"Ticker": tickers, "Weight": min_vol_w})
        df = df.sort_values("Weight", ascending=False)
        df["Weight"] = df["Weight"].map("{:.2%}".format)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # ----- Download -------------------------------------------------
    st.subheader("Download")
    out = pd.DataFrame({
        "Risk": sim_results[0],
        "Return": sim_results[1],
        "Sharpe": sim_results[2]
    })
    for i, t in enumerate(tickers):
        out[t] = [w[i] for w in sim_weights]

    csv = out.to_csv(index=False).encode()
    st.download_button(
        "Download full simulation (CSV)",
        data=csv,
        file_name=f"monte_carlo_{datetime.now():%Y%m%d}.csv",
        mime="text/csv"
    )

else:
    st.info("Configure the portfolio on the left and click **Run Simulation**.")
    st.markdown("""
    ### How to use
    1. **Enter tickers** (comma-separated)  
    2. Choose **historical period** and **# of simulations**  
    3. Press **Run Simulation**  
    4. Explore the **Efficient Frontier** and the two optimal portfolios  

    The **gold star** = best risk-adjusted return (max Sharpe).  
    The **green diamond** = lowest possible risk.
    """)
