"""
Bollinger–RSI Adaptive Trailing Reversion

BFIT.AS – Bollinger + RSI + Trailing-Stop optimisation
Streamlit-ready version: run with `streamlit run bfit_strategy.py`

Dependencies (add to requirements.txt if needed):
- streamlit
- pandas
- numpy
- matplotlib
- yfinance
- deap
- tqdm

This file adapts the reference script into a Streamlit app. It:
 - fetches data (with Yahoo window protection)
 - computes Bollinger bands and RSI
 - runs a DEAP GA multi-objective optimisation
 - plots results in Streamlit (matplotlib -> st.pyplot)

Notes:
 - If DEAP's creator objects already exist (repeat runs in the same session), the app handles that gracefully.
 - Long optimisations can take time; the UI exposes population and generation sizes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
import io
import traceback

# ------------------------------------------------------------------
# 1. Data fetching (handles Yahoo-Finance limits)
# ------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: str, end: str, interval: str = "1h"):
    """
    Yahoo Finance rules (as of 2025):
        • hourly  → max 60 days
        • daily   → max 730 days
    The function automatically shrinks the window if the request is too big.
    Returns a DataFrame with OHLCV.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end,   "%Y-%m-%d")

    if interval == "1h":
        max_days = 60
    else:
        max_days = 730

    # adjust window if requested window exceeds limits
    if (end_dt - start_dt).days > max_days:
        start_dt = end_dt - timedelta(days=max_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    st.info(f"Downloading {ticker} [{start_str} → {end_str}] @ {interval}")
    data = yf.download(ticker, start=start_str, end=end_str,
                       interval=interval, progress=False)

    if data.empty:
        raise ValueError(f"No data for {ticker} in the selected window")
    return data

# ------------------------------------------------------------------
# 2. Indicators
# ------------------------------------------------------------------


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Bollinger Bands
    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["UpperBB"] = df["MA20"] + 2 * df["STD20"]
    df["LowerBB"] = df["MA20"] - 2 * df["STD20"]

    # RSI (14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    df["RSI"] = 100 - 100 / (1 + rs)
    return df.dropna()

# ------------------------------------------------------------------
# 3. Back-test engine
# ------------------------------------------------------------------


def backtest(df: pd.DataFrame, bb_entry: float, bb_exit: float, rsi_entry: float, rsi_exit: float, trail_pct: float) -> pd.DataFrame:
    df = df.copy()
    df["Position"] = 0
    in_trade = False
    entry_price = highest = trail_stop = 0.0
    # Precompute integer column positions for robust scalar access

    def _col_index(df_obj, col_name: str):
        try:
            loc = df_obj.columns.get_loc(col_name)
        except Exception:
            # fallback: case-insensitive match or substring
            for idx, c in enumerate(df_obj.columns):
                name = c if not isinstance(c, tuple) else c[-1]
                if str(name).lower() == col_name.lower():
                    return idx
                if col_name.lower() in str(name).lower():
                    return idx
            raise KeyError(f"Column '{col_name}' not found in DataFrame")

        # Normalize possible return types from get_loc
        # get_loc can return an int, slice, boolean mask array, or an array of indices
        if isinstance(loc, (int, np.integer)):
            return int(loc)
        if isinstance(loc, slice):
            # choose the start of the slice if available
            if loc.start is not None:
                return int(loc.start)
            # otherwise fall back to 0
            return 0
        # array-like (e.g., boolean mask, list of positions)
        try:
            # convert indexer to numpy array of positions
            arr = np.asarray(loc)
            if arr.size > 0:
                return int(np.flatnonzero(arr)[0]) if arr.dtype == bool else int(arr.reshape(-1)[0])
        except Exception:
            pass
        # if we reach here, fall back to searching columns by name
        for idx, c in enumerate(df_obj.columns):
            name = c if not isinstance(c, tuple) else c[-1]
            if str(name).lower() == col_name.lower() or col_name.lower() in str(name).lower():
                return idx
        raise KeyError(
            f"Column '{col_name}' not found in DataFrame (after normalization)")

    idx_close = int(_col_index(df, "Close"))
    idx_lower = int(_col_index(df, "LowerBB"))
    idx_upper = int(_col_index(df, "UpperBB"))
    idx_rsi = int(_col_index(df, "RSI"))
    idx_pos = int(_col_index(df, "Position"))

    for i in range(1, len(df)):
        # scalar access via iat (row, col)
        price = float(df.iat[i, idx_close])

        # ---------- ENTRY ----------
        if not in_trade:
            lowerbb = float(df.iat[i, idx_lower])
            rsi_val = float(df.iat[i, idx_rsi])
            bb_ok = price < (lowerbb * float(bb_entry))
            rsi_ok = rsi_val < float(rsi_entry)
            if bb_ok and rsi_ok:
                df.iat[i, idx_pos] = 1
                in_trade = True
                entry_price = highest = price
                trail_stop = price * (1 - trail_pct)
            else:
                df.iat[i, idx_pos] = 0

        # ---------- EXIT ----------
        else:
            # update trailing stop
            if price > highest:
                highest = price
                trail_stop = price * (1 - trail_pct)

            upperbb = float(df.iat[i, idx_upper])
            rsi_val = float(df.iat[i, idx_rsi])
            bb_exit_ok = price > (upperbb * float(bb_exit))
            rsi_exit_ok = rsi_val > float(rsi_exit)
            trail_ok = price <= trail_stop

            if (bb_exit_ok and rsi_exit_ok) or trail_ok:
                df.iat[i, idx_pos] = 0
                in_trade = False
            else:
                df.iat[i, idx_pos] = 1

    df["Returns"] = df["Close"].pct_change()
    df["Strat_Returns"] = df["Position"].shift(1) * df["Returns"]
    df["Strat_Returns"] = df["Strat_Returns"].fillna(0)
    df["Cum_Strat"] = (1 + df["Strat_Returns"]).cumprod()
    df["Cum_BuyHold"] = (1 + df["Returns"]).cumprod()
    return df

# ------------------------------------------------------------------
# 4. Performance metrics
# ------------------------------------------------------------------


def sharpe(returns: pd.Series, rf: float = 0.02):
    excess = returns - rf / 252
    if excess.std() == 0:
        return -np.inf
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(cumulative: pd.Series):
    roll_max = np.maximum.accumulate(cumulative)
    dd = cumulative / roll_max - 1
    return dd.min()


def evaluate(individual, df: pd.DataFrame):
    bb_entry, bb_exit, rsi_entry, rsi_exit, trail = individual
    try:
        bt = backtest(df, bb_entry, bb_exit, rsi_entry, rsi_exit, trail)
        strat_ret = bt["Strat_Returns"]

        total_ret = bt["Cum_Strat"].iloc[-1] - 1
        sharpe_val = sharpe(strat_ret)
        mdd = max_drawdown(bt["Cum_Strat"])
        calmar = total_ret / abs(mdd) if mdd != 0 else -np.inf
        trades = int((bt["Position"].diff().abs() == 1).sum() // 2)

        # ---- penalties -------------------------------------------------
        # >100 trades → penalty
        trade_pen = max(0, (trades - 100) / 100)
        dd_pen = max(0, (mdd - 0.2) / 0.2)

        sharpe_val -= (trade_pen + dd_pen)

        # Return tuple matching the creator weights
        return sharpe_val, total_ret, calmar, trades, mdd
    except Exception:
        # Defensive: if evaluation fails for an individual, return a very poor fitness
        return -1e6, -1e6, -1e6, 1e6, 1.0

# ------------------------------------------------------------------
# 5. DEAP optimisation
# ------------------------------------------------------------------


def create_deap_types():
    # Avoid re-creating creators if they already exist in the session
    try:
        creator.FitnessMulti
    except Exception:
        creator.create("FitnessMulti", base.Fitness,
                       # maximise 1-3, minimise 4-5
                       weights=(1.0, 1.0, 1.0, -1.0, -1.0))
    try:
        creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def optimise(df: pd.DataFrame, pop_size: int = 200, ngen: int = 80, seed: int | None = None, progress_callback=None, patience: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    create_deap_types()

    toolbox = base.Toolbox()
    toolbox.register("bb_entry",   random.uniform, 0.90, 1.05)
    toolbox.register("bb_exit",    random.uniform, 0.90, 1.05)
    toolbox.register("rsi_entry",  random.uniform, 20, 40)
    toolbox.register("rsi_exit",   random.uniform, 60, 80)
    toolbox.register("trail",      random.uniform, 0.01, 0.10)

    def make_ind():
        ind = creator.Individual([
            toolbox.bb_entry(),
            toolbox.bb_exit(),
            toolbox.rsi_entry(),
            toolbox.rsi_exit(),
            toolbox.trail()
        ])
        # enforce RSI_exit > RSI_entry
        while ind[3] <= ind[2]:
            ind[3] = toolbox.rsi_exit()
        return ind

    toolbox.register("individual", make_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",   evaluate, df=df)
    toolbox.register("mate",       tools.cxBlend, alpha=0.5)
    toolbox.register("mutate",     tools.mutGaussian,
                     mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select",     tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    best = None
    best_fit = (-np.inf, -np.inf, -np.inf, np.inf, np.inf)
    last_improve_gen = -1

    for gen in range(ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring + pop, k=len(pop))

        gen_best = tools.selBest(pop, k=1)[0]
        if gen_best.fitness.values[0] > best_fit[0]:
            best = list(gen_best)
            best_fit = gen_best.fitness.values
            last_improve_gen = gen

        # report progress via callback if provided (value between 0-1)
        if progress_callback is not None:
            try:
                progress_callback((gen + 1) / ngen, gen + 1, best_fit)
            except Exception:
                pass
        # early stopping if no improvement for `patience` generations
        if patience and last_improve_gen >= 0 and (gen - last_improve_gen) >= patience:
            break

    # return best, fitness tuple, and completed generations
    return best, best_fit, (gen + 1)

# ------------------------------------------------------------------
# 6. Plotting the optimal back-test (matplotlib figures)
# ------------------------------------------------------------------


def plot_optimal(df: pd.DataFrame, params, ticker: str = ""):
    """Return a list of matplotlib Figure objects (price, RSI, equity) and the backtest DataFrame."""
    bb_entry, bb_exit, rsi_entry, rsi_exit, trail = params
    bt = backtest(df, bb_entry, bb_exit, rsi_entry, rsi_exit, trail)

    figs = []

    # ---- price + signals ------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(13, 7))
    ax1.plot(bt.index, bt["Close"], label="Close", color="#1f77b4")
    ax1.plot(bt.index, bt["UpperBB"],
             label="Upper BB", alpha=0.6, color="gray")
    ax1.plot(bt.index, bt["LowerBB"],
             label="Lower BB", alpha=0.6, color="gray")

    buys = bt[bt["Position"].diff() == 1]
    sells = bt[bt["Position"].diff() == -1]
    ax1.scatter(buys.index,  buys["Close"],
                marker="^", color="green", s=120, label="Buy")
    ax1.scatter(sells.index, sells["Close"],
                marker="v", color="red",   s=120, label="Sell")

    ax1.set_title(f"{ticker} – Optimal Strategy (Trail {trail:.1%})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    figs.append(fig1)

    # ---- RSI ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(13, 3))
    ax2.plot(bt.index, bt["RSI"], label="RSI", color="purple")
    ax2.axhline(rsi_entry, color="green", linestyle="--",
                label=f"Entry {rsi_entry:.1f}")
    ax2.axhline(rsi_exit,  color="red",   linestyle="--",
                label=f"Exit  {rsi_exit:.1f}")
    ax2.set_title("RSI")
    ax2.legend()
    ax2.grid(alpha=0.3)
    figs.append(fig2)

    # ---- equity curve ---------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(13, 5))
    ax3.plot(bt.index, bt["Cum_Strat"],   label="Strategy",   color="#2ca02c")
    ax3.plot(bt.index, bt["Cum_BuyHold"], label="Buy & Hold", color="#d62728")
    ax3.set_title("Cumulative Returns")
    ax3.legend()
    ax3.grid(alpha=0.3)
    figs.append(fig3)

    return figs, bt


def extract_trades(bt: pd.DataFrame) -> pd.DataFrame:
    """Extract trade pairs (buy -> sell) from backtest positions.

    Returns a DataFrame with columns: entry_date, entry_price, exit_date, exit_price,
    pct_return, duration_days. If a position is still open, exit_* fields are NaN.
    """
    diffs = bt["Position"].diff().fillna(0)
    buy_idxs = np.where(diffs == 1)[0]
    sell_idxs = np.where(diffs == -1)[0]

    trades = []
    sell_ptr = 0
    for b in buy_idxs:
        entry_date = bt.index[b]
        # use iloc to access by integer row position and avoid Series/DataFrame ambiguity
        entry_price = float(bt.iloc[b]["Close"])

        # find the first sell after this buy
        exit_date = None
        exit_price = None
        pct = None
        duration = None
        while sell_ptr < len(sell_idxs) and sell_idxs[sell_ptr] < b:
            sell_ptr += 1
        if sell_ptr < len(sell_idxs):
            s = sell_idxs[sell_ptr]
            if s > b:
                exit_date = bt.index[s]
                exit_price = float(bt.iloc[s]["Close"])
                pct = exit_price / entry_price - 1
                duration = (exit_date - entry_date).total_seconds() / 86400.0
                sell_ptr += 1

        trades.append({
            "entry_date": pd.to_datetime(entry_date),
            "entry_price": entry_price,
            "exit_date": pd.to_datetime(exit_date) if exit_date is not None else pd.NaT,
            "exit_price": exit_price,
            "pct_return": pct,
            "duration_days": duration,
        })

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades["entry_date"] = df_trades["entry_date"].dt.strftime(
            "%Y-%m-%d %H:%M:%S")
        df_trades["exit_date"] = df_trades["exit_date"].where(
            df_trades["exit_date"].notna(), None)
    return df_trades

    return fig, bt

# ------------------------------------------------------------------
# 7. Streamlit UI / main
# ------------------------------------------------------------------


def main():
    st.set_page_config(page_title="BFIT Strategy Optimiser", layout="wide")
    st.title("BFIT.AS — Bollinger + RSI + Trailing-Stop Optimiser")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker", value="BFIT.AS")
        start = st.date_input("Start date", value=(
            datetime.today() - timedelta(days=120))).strftime("%Y-%m-%d")
        end = st.date_input(
            "End date", value=datetime.today()).strftime("%Y-%m-%d")
        interval = st.selectbox("Interval", options=["1h", "1d"], index=0)
        pop_size = st.slider("Population size", min_value=50,
                             max_value=500, value=200, step=10)
        ngen = st.slider("Generations", min_value=10,
                         max_value=200, value=80, step=10)
        patience = st.slider("Early-stop patience (gens, 0=off)",
                             min_value=0, max_value=50, value=0, step=1)
        seed = st.number_input(
            "Random seed (0 for random)", value=0, min_value=0)

        st.markdown("---")
        st.markdown(
            "Tips: Use smaller pop/gen for quick tests. Hourly requests are limited to ~60 days by Yahoo.")

    # placeholder
    status = st.empty()
    metrics_col = st.columns(3)

    try:
        raw = fetch_data(ticker, start, end, interval)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

    df = add_indicators(raw)
    st.subheader("Data snapshot")
    st.dataframe(df.tail(10))

    run = st.button("Run optimisation")
    if run:
        if seed == 0:
            seed_val = None
        else:
            seed_val = int(seed)

        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        best_fit_box = st.empty()

        def progress_cb(pct, gen_idx, best_fit):
            # pct between 0-1
            try:
                progress_bar.progress(min(max(pct, 0.0), 1.0))
                progress_text.text(
                    f"Gen {gen_idx} / {ngen} — best sharpe {best_fit[0]:.3f} — ret {best_fit[1]:.2%}")
            except Exception:
                pass

        with st.spinner("Optimising — this may take a while (check the progress bar)..."):
            try:
                best_params, best_fit, gens_done = optimise(
                    df, pop_size=pop_size, ngen=ngen, seed=seed_val, progress_callback=progress_cb, patience=patience)
            except Exception as e:
                st.error("Optimisation failed. See traceback in logs.")
                st.text(traceback.format_exc())
                progress_bar.empty()
                st.stop()

        progress_bar.progress(1.0)
        st.success(f"Optimisation finished (gens run: {gens_done})")

        st.subheader("Best parameters")
        if best_params is None:
            st.warning(
                "No best candidate found — try increasing population or generations.")
        else:
            st.write({
                "bb_entry": best_params[0],
                "bb_exit": best_params[1],
                "rsi_entry": best_params[2],
                "rsi_exit": best_params[3],
                "trail_pct": best_params[4],
            })
            st.write(
                f"Metrics (fitness tuple): Sharpe, TotalRet, Calmar, Trades, MaxDD = {best_fit}")

            figs, bt = plot_optimal(df, best_params, ticker=ticker)
            st.subheader("Strategy charts")
            for fig in figs:
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Backtest table (last 20 rows)")
            st.dataframe(bt.tail(20))

            # Trades list (buy/sell pairs)
            trades_df = extract_trades(bt)
            st.subheader("Detected trades (entry → exit)")
            if trades_df.empty:
                st.write("No trades detected for these parameters.")
            else:
                st.dataframe(trades_df)
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    "Download trades CSV", csv, file_name=f"{ticker.replace('.', '_')}_trades.csv", mime="text/csv")

            # small metrics panel
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{(bt['Cum_Strat'].iloc[-1]-1):.2%}")
            col2.metric("Sharpe (approx)",
                        f"{sharpe(bt['Strat_Returns']):.3f}")
            col3.metric("Max Drawdown", f"{max_drawdown(bt['Cum_Strat']):.2%}")

    st.markdown("---")
    st.caption(
        "Note: Optimisation is non-deterministic unless you set a seed. Use small test values for quick iterations.")


if __name__ == "__main__":
    main()
