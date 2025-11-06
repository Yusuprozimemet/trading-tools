# BBRSI.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms
import random
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# Page configuration / Title
# ---------------------------
st.set_page_config(
    page_title="BBRSI Trading Strategy Optimizer", layout="wide")
st.title("🎯 BBRSI Trading Strategy Optimizer")
st.markdown(
    "Optimize trading strategies using Bollinger Bands, RSI, and Trailing Stop with Genetic Algorithm"
)

# ---------------------------
# Sidebar inputs
# ---------------------------
with st.sidebar:
    st.header("📊 Configuration")

    ticker = st.text_input("Stock Ticker", value="AAPL",
                           help="Enter a valid stock ticker (e.g., AAPL, MSFT, TSLA)")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=(
            datetime.now() - timedelta(days=90)).date())
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())

    interval = st.selectbox("Interval", ["1h", "1d", "1wk"], index=0,
                            help="Note: Hourly data limited to recent days by Yahoo Finance")

    st.divider()
    st.subheader("🧬 GA Parameters")
    population_size = st.slider("Population Size", 50, 500, 200, 50)
    generations = st.slider("Generations", 20, 200, 100, 20)
    seed_opt = st.checkbox("Set random seed (reproducible)", value=False)
    if seed_opt:
        seed_value = int(st.number_input("Seed value", value=42, step=1))

    st.divider()
    run_optimization = st.button(
        "🚀 Run Optimization", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### About")
    st.markdown("""
    This tool optimizes trading strategies using:
    - **Bollinger Bands** for entry/exit signals
    - **RSI** for overbought/oversold conditions
    - **Trailing Stop** for risk management
    - **Genetic Algorithm** for parameter optimization
    """)

# ---------------------------
# Helper functions
# ---------------------------


@st.cache_data
def fetch_data(ticker: str, start, end, interval: str = "1h"):
    """
    Fetch historical data using yfinance.Ticker.history.
    Accepts start/end as date objects or 'YYYY-MM-DD' strings.
    """
    try:
        # Normalize start/end to datetime.date
        if isinstance(start, (str,)):
            start_dt = datetime.strptime(start, "%Y-%m-%d")
        elif isinstance(start, datetime):
            start_dt = start
        else:
            # date object
            start_dt = datetime.combine(start, datetime.min.time())

        if isinstance(end, (str,)):
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        elif isinstance(end, datetime):
            end_dt = end
        else:
            end_dt = datetime.combine(end, datetime.min.time())

        # Hourly data limitations: keep to last ~60 days for reliable hourly fetch
        if interval == "1h":
            # cap to last 730 days first (user code had 730), then ensure at most 60 days to be safe for hourly
            max_days = 730
            if (end_dt - start_dt).days > max_days:
                start_dt = end_dt - timedelta(days=max_days)
            start_dt = max(start_dt, end_dt - timedelta(days=60))

        # Convert to strings accepted by yfinance
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(
            start=start_str, end=end_str, interval=interval, auto_adjust=False)

        if data.empty:
            return None, f"No data available for {ticker} between {start_str} and {end_str}"
        # Ensure index is timezone-naive datetime
        data.index = pd.to_datetime(data.index).tz_localize(None)
        return data, None
    except Exception as e:
        return None, str(e)


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    # Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20, min_periods=10).mean()
    data['STD20'] = data['Close'].rolling(window=20, min_periods=10).std()
    data['UpperBB'] = data['MA20'] + (data['STD20'] * 2)
    data['LowerBB'] = data['MA20'] - (data['STD20'] * 2)

    # RSI (classic)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=7).mean()
    loss = -delta.clip(upper=0).rolling(window=14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    # Fill RSI gaps conservatively
    data['RSI'] = data['RSI'].fillna(method='ffill').fillna(50)

    data.dropna(subset=['MA20', 'STD20', 'RSI'], inplace=True)
    # Returns column used by backtest routines
    data['Returns'] = data['Close'].pct_change().fillna(0)

    return data


def backtest_strategy(data: pd.DataFrame, bb_entry: float, bb_exit: float,
                      rsi_entry: float, rsi_exit: float, trailing_stop_pct: float) -> pd.DataFrame:
    data = data.copy()
    data['Position'] = 0
    data['Entry_Price'] = np.nan
    data['Exit_Price'] = np.nan

    in_position = False
    entry_price = 0.0
    highest_price = 0.0
    trailing_stop_price = 0.0

    for i in range(1, len(data)):
        close = data['Close'].iloc[i]
        lowerbb = data['LowerBB'].iloc[i]
        upperbb = data['UpperBB'].iloc[i]
        rsi = data['RSI'].iloc[i]

        if not in_position:
            bb_condition = (not np.isnan(lowerbb)) and (
                close < lowerbb * bb_entry)
            rsi_condition = rsi < rsi_entry
            if bb_condition and rsi_condition:
                data.iat[i, data.columns.get_loc('Position')] = 1
                data.iat[i, data.columns.get_loc('Entry_Price')] = close
                in_position = True
                entry_price = close
                highest_price = entry_price
                trailing_stop_price = entry_price * (1 - trailing_stop_pct)
        else:
            # Update highest price and trailing stop
            if close > highest_price:
                highest_price = close
                trailing_stop_price = highest_price * (1 - trailing_stop_pct)

            bb_condition = (not np.isnan(upperbb)) and (
                close > upperbb * bb_exit)
            rsi_condition = rsi > rsi_exit
            trailing_stop_condition = close <= trailing_stop_price

            if (bb_condition and rsi_condition) or trailing_stop_condition:
                data.iat[i, data.columns.get_loc('Position')] = 0
                data.iat[i, data.columns.get_loc('Exit_Price')] = close
                in_position = False
            else:
                data.iat[i, data.columns.get_loc('Position')] = 1

    # Strategy returns: position applied to previous row's returns (shifted)
    data['Strategy_Returns'] = data['Position'].shift(
        1).fillna(0) * data['Returns']
    data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    data['Buy_Hold_Returns'] = (1 + data['Returns']).cumprod()

    return data


def infer_periods_per_year(df: pd.DataFrame) -> int:
    if len(df.index) < 2:
        return 252
    delta = (df.index[1] - df.index[0]).total_seconds()
    days = delta / (24 * 3600)
    if days <= 0:
        return 252
    ppy = int(round(365 / days))
    # clamp
    if ppy <= 0 or ppy > 365 * 24:
        return 252
    return ppy


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.02) -> float:
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / std


def calculate_max_drawdown(cum_returns: pd.Series) -> float:
    if cum_returns.empty:
        return 0.0
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns / rolling_max) - 1
    return float(np.min(drawdown))


def evaluate_strategy(individual, data: pd.DataFrame, periods_per_year: int):
    try:
        bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct = individual

        if rsi_exit <= rsi_entry:
            return -1e6, -1e6, -1e6, 0, 0

        bt = backtest_strategy(data, bb_entry, bb_exit,
                               rsi_entry, rsi_exit, trailing_stop_pct)

        strategy_returns = bt['Strategy_Returns']

        pos_diff = bt['Position'].diff().fillna(0)
        num_entries = int((pos_diff == 1).sum())
        num_exits = int((pos_diff == -1).sum())
        num_trades = num_entries + num_exits

        if num_trades == 0:
            return -1e5, -1e5, -1e5, 0, 0

        sharpe = calculate_sharpe_ratio(
            strategy_returns, periods_per_year=periods_per_year)
        total_return = float(bt['Cumulative_Returns'].iloc[-1] - 1)
        max_dd = calculate_max_drawdown(bt['Cumulative_Returns'])

        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = -10.0
        if np.isnan(total_return) or np.isinf(total_return):
            total_return = -1.0
        if np.isnan(max_dd):
            max_dd = 0.0

        calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

        trade_penalty = max(0, (num_trades - 200) / 200)
        drawdown_penalty = max(0, (abs(max_dd) - 0.5) / 0.5)

        sharpe_adj = sharpe - (trade_penalty + drawdown_penalty)

        return float(sharpe_adj), float(total_return), float(calmar), float(num_trades), float(max_dd)
    except Exception:
        return -1e6, -1e6, -1e6, 0, 0


def optimize_strategy(data: pd.DataFrame, pop_size: int, ngen: int):
    # Clean DEAP creators if they exist
    try:
        del creator.FitnessMulti
    except Exception:
        pass
    try:
        del creator.Individual
    except Exception:
        pass

    # Create fitness and individual
    creator.create("FitnessMulti", base.Fitness,
                   weights=(1.0, 1.0, 1.0, -0.01, -0.5))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Parameter bounds
    bb_entry_bounds = (0.8, 1.05)
    bb_exit_bounds = (0.95, 1.25)
    rsi_entry_bounds = (10.0, 45.0)
    rsi_exit_bounds = (55.0, 90.0)
    trailing_bounds = (0.005, 0.15)

    toolbox.register("attr_bb_entry", random.uniform, *bb_entry_bounds)
    toolbox.register("attr_bb_exit", random.uniform, *bb_exit_bounds)
    toolbox.register("attr_rsi_entry", random.uniform, *rsi_entry_bounds)
    toolbox.register("attr_rsi_exit", random.uniform, *rsi_exit_bounds)
    toolbox.register("attr_trailing_stop", random.uniform, *trailing_bounds)

    def create_individual():
        bb_e = toolbox.attr_bb_entry()
        bb_x = toolbox.attr_bb_exit()
        rsi_e = toolbox.attr_rsi_entry()
        rsi_x = toolbox.attr_rsi_exit()
        if rsi_x <= rsi_e:
            rsi_x = min(rsi_exit_bounds[1], rsi_e + 8.0)
        tr = toolbox.attr_trailing_stop()
        return creator.Individual([bb_e, bb_x, rsi_e, rsi_x, tr])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    periods_per_year = infer_periods_per_year(data)
    toolbox.register("evaluate", evaluate_strategy, data=data,
                     periods_per_year=periods_per_year)

    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", map)

    population = toolbox.population(n=pop_size)

    best_strategy = None
    best_fitness = (-1e9, -1e9, -1e9, 0, 0)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.4)

        # ensure parameters remain in bounds (repair)
        for ind in offspring:
            ind[0] = float(
                np.clip(ind[0], bb_entry_bounds[0], bb_entry_bounds[1]))
            ind[1] = float(
                np.clip(ind[1], bb_exit_bounds[0], bb_exit_bounds[1]))
            ind[2] = float(
                np.clip(ind[2], rsi_entry_bounds[0], rsi_entry_bounds[1]))
            ind[3] = float(
                np.clip(ind[3], rsi_exit_bounds[0], rsi_exit_bounds[1]))
            if ind[3] <= ind[2]:
                ind[3] = min(rsi_exit_bounds[1], ind[2] + 5.0)
            ind[4] = float(
                np.clip(ind[4], trailing_bounds[0], trailing_bounds[1]))

        fits = list(toolbox.map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring + population, k=len(population))

        gen_best = tools.selBest(population, k=1)[0]
        if gen_best.fitness.values[0] > best_fitness[0]:
            best_strategy = gen_best[:]
            best_fitness = gen_best.fitness.values

        progress_bar.progress((gen + 1) / ngen)
        if best_fitness[0] < -1e4:
            status_text.text(
                f"Generation {gen+1}/{ngen} — searching for valid strategies...")
        else:
            status_text.text(
                f"Generation {gen+1}/{ngen} — Best Sharpe: {best_fitness[0]:.4f} | Return: {best_fitness[1]:.2%} | Trades: {int(best_fitness[3])}")

    progress_bar.empty()
    status_text.empty()

    return best_strategy, best_fitness


# ---------------------------
# Main execution in Streamlit
# ---------------------------
if run_optimization:
    if seed_opt:
        random.seed(seed_value)
        np.random.seed(seed_value)

    with st.spinner("Fetching data..."):
        data, error = fetch_data(ticker, start_date, end_date, interval)

    if error:
        st.error(f"Error fetching data: {error}")
    elif data is None or len(data) < 30:
        st.error("Insufficient data points. Please adjust date range or interval.")
    else:
        st.success(f"Fetched {len(data)} data points")

        with st.spinner("Calculating indicators..."):
            data = calculate_indicators(data)

        with st.spinner("Running genetic algorithm optimization..."):
            best_strategy, best_fitness = optimize_strategy(
                data, population_size, generations)

        if best_strategy is None or best_fitness[0] < -1e4:
            st.error("❌ Optimization failed to find a valid strategy. Try widening parameter ranges, increasing population/generations, or changing the date range/interval.")
        else:
            st.success("✅ Optimization completed!")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Sharpe Ratio", f"{best_fitness[0]:.4f}")
            with col2:
                st.metric("Total Return", f"{best_fitness[1]:.2%}")
            with col3:
                st.metric("Calmar Ratio", f"{best_fitness[2]:.4f}")
            with col4:
                st.metric("Num Trades", f"{int(best_fitness[3])}")
            with col5:
                dd_val = best_fitness[4] if not np.isnan(
                    best_fitness[4]) else 0.0
                st.metric("Max Drawdown", f"{dd_val:.2%}")

            st.divider()
            st.subheader("🎯 Optimized Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**BB Entry:** {best_strategy[0]:.4f}")
                st.info(f"**BB Exit:** {best_strategy[1]:.4f}")
            with col2:
                st.info(f"**RSI Entry:** {best_strategy[2]:.2f}")
                st.info(f"**RSI Exit:** {best_strategy[3]:.2f}")
            with col3:
                st.info(f"**Trailing Stop:** {best_strategy[4]:.2%}")

            # Run backtest with best strategy
            bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct = best_strategy
            backtested_data = backtest_strategy(
                data, bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct)

            # Plots
            st.subheader("📈 Price Chart with Signals")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(backtested_data.index,
                     backtested_data['Close'], label='Close Price', linewidth=2)
            ax1.plot(backtested_data.index,
                     backtested_data['UpperBB'], label='Upper BB', alpha=0.7, linestyle='--')
            ax1.plot(backtested_data.index,
                     backtested_data['LowerBB'], label='Lower BB', alpha=0.7, linestyle='--')
            ax1.fill_between(
                backtested_data.index, backtested_data['LowerBB'], backtested_data['UpperBB'], alpha=0.08)

            buy_signals = backtested_data[backtested_data['Position'].diff(
            ) == 1]
            sell_signals = backtested_data[backtested_data['Position'].diff(
            ) == -1]

            ax1.scatter(
                buy_signals.index, buy_signals['Close'], marker='^', s=80, label='Buy', zorder=5)
            ax1.scatter(
                sell_signals.index, sell_signals['Close'], marker='v', s=80, label='Sell', zorder=5)

            ax1.set_title(f'{ticker} Trading Strategy')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)

            st.subheader("📊 RSI Indicator")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(backtested_data.index,
                     backtested_data['RSI'], label='RSI', linewidth=2)
            ax2.axhline(y=rsi_entry, linestyle='--',
                        label=f'Entry ({rsi_entry:.1f})', alpha=0.7)
            ax2.axhline(y=rsi_exit, linestyle='--',
                        label=f'Exit ({rsi_exit:.1f})', alpha=0.7)
            ax2.axhline(y=30, linestyle=':', alpha=0.5)
            ax2.axhline(y=70, linestyle=':', alpha=0.5)
            ax2.set_title('RSI Indicator')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            st.pyplot(fig2)
            plt.close(fig2)

            st.subheader("💰 Cumulative Returns Comparison")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            strategy_cumulative = backtested_data['Cumulative_Returns']
            buy_hold_cumulative = backtested_data['Buy_Hold_Returns']

            ax3.plot(strategy_cumulative.index, strategy_cumulative,
                     label='Strategy', linewidth=2)
            ax3.plot(buy_hold_cumulative.index, buy_hold_cumulative,
                     label='Buy & Hold', linewidth=2)
            ax3.fill_between(strategy_cumulative.index, 1,
                             strategy_cumulative, alpha=0.1)
            ax3.set_title('Strategy vs Buy & Hold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Cumulative Return')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1, color='black', linestyle='-', alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)

            # Performance summary & download
            total_return = strategy_cumulative.iloc[-1] - 1
            buy_hold_return = buy_hold_cumulative.iloc[-1] - 1
            pos_changes = backtested_data['Position'].diff().fillna(0)
            # entry+exit = 2 state changes
            num_trades = int((pos_changes != 0).sum() / 2)

            st.markdown("### 📋 Performance Summary")
            st.write({
                "Total Return (strategy)": f"{total_return:.2%}",
                "Buy & Hold Return": f"{buy_hold_return:.2%}",
                "Number of Trades (approx)": num_trades,
                "Best Sharpe (fitness)": f"{best_fitness[0]:.4f}"
            })

            download_df = backtested_data[['Close', 'Position', 'Entry_Price', 'Exit_Price', 'UpperBB',
                                           'LowerBB', 'RSI', 'Returns', 'Strategy_Returns', 'Cumulative_Returns', 'Buy_Hold_Returns']].copy()
            csv = download_df.to_csv(index=True)
            st.download_button(
                label="Download Backtest Results (CSV)",
                data=csv,
                file_name=f"{ticker}_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    st.info(
        "👈 Configure parameters in the sidebar and click 'Run Optimization' to start")
    st.markdown("""
    ### How to Use:
    1. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)
    2. Select date range (note: hourly data limited to recent days by Yahoo)
    3. Choose time interval
    4. Adjust genetic algorithm parameters (optional)
    5. Click "Run Optimization"
    
    ### Strategy Logic:
    - **Entry Signal**: Price crosses below Lower Bollinger Band × BB_Entry AND RSI < RSI_Entry
    - **Exit Signal**: Price crosses above Upper Bollinger Band × BB_Exit AND RSI > RSI_Exit, OR Trailing Stop triggered
    - **Risk Management**: Trailing stop dynamically adjusts to lock in profits
    """)
