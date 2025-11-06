import yfinance as yf
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time


class MultiStockOptimalStrategy:
    def __init__(self, tickers, start, end, interval="1h"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval
        self.data = {}

    def fetch_data(self):
        try:
            data = yf.download(self.tickers, start=self.start,
                               end=self.end, interval=self.interval, group_by='ticker')
            if data.empty:
                raise ValueError(
                    f"No data available for {self.tickers} between {self.start} and {self.end}")
            st.success(f"✅ Fetched data for {len(self.tickers)} stocks")
            return data
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return None

    def preprocess_data(self, data):
        if data is None or data.empty:
            return None
        processed_data = {}
        for ticker in self.tickers:
            try:
                if len(self.tickers) == 1:
                    stock_data = data.copy()
                else:
                    stock_data = data[ticker].copy()
                stock_data['Returns'] = stock_data['Close'].pct_change()
                stock_data.dropna(inplace=True)
                processed_data[ticker] = stock_data
            except Exception as e:
                st.warning(f"⚠️ Could not process {ticker}: {str(e)}")
                continue
        st.info(f"📊 Preprocessed data for {len(processed_data)} stocks")
        return processed_data

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        data['MA20'] = data['Close'].rolling(window=window).mean()
        data['STD20'] = data['Close'].rolling(window=window).std()
        data['UpperBB'] = data['MA20'] + (data['STD20'] * num_std)
        data['LowerBB'] = data['MA20'] - (data['STD20'] * num_std)
        return data

    def calculate_rsi(self, data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_indicators(self, data):
        for ticker in data:
            data[ticker] = self.calculate_bollinger_bands(data[ticker])
            data[ticker] = self.calculate_rsi(data[ticker])
        return data

    def backtest(self, data, bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct, stop_loss_pct):
        results = {}
        for ticker, stock_data in data.items():
            stock_data = stock_data.copy()
            stock_data['Position'] = 0

            in_trade = False
            entry_price = 0
            highest_price = 0
            trailing_stop_price = 0
            stop_loss_price = 0

            for i in range(1, len(stock_data)):
                if not in_trade:
                    bb_condition = stock_data['Close'].iloc[i] < stock_data['LowerBB'].iloc[i] * bb_entry
                    rsi_condition = stock_data['RSI'].iloc[i] < rsi_entry
                    if bb_condition and rsi_condition:
                        stock_data.loc[stock_data.index[i], 'Position'] = 1
                        in_trade = True
                        entry_price = stock_data['Close'].iloc[i]
                        highest_price = entry_price
                        trailing_stop_price = entry_price * \
                            (1 - trailing_stop_pct)
                        stop_loss_price = entry_price * (1 - stop_loss_pct)
                elif in_trade:
                    if stock_data['Close'].iloc[i] > highest_price:
                        highest_price = stock_data['Close'].iloc[i]
                        trailing_stop_price = highest_price * \
                            (1 - trailing_stop_pct)

                    bb_condition = stock_data['Close'].iloc[i] > stock_data['UpperBB'].iloc[i] * bb_exit
                    rsi_condition = stock_data['RSI'].iloc[i] > rsi_exit
                    trailing_stop_condition = stock_data['Close'].iloc[i] <= trailing_stop_price
                    stop_loss_condition = stock_data['Close'].iloc[i] <= stop_loss_price

                    if bb_condition or rsi_condition or trailing_stop_condition or stop_loss_condition:
                        stock_data.loc[stock_data.index[i], 'Position'] = 0
                        in_trade = False
                    else:
                        stock_data.loc[stock_data.index[i], 'Position'] = 1

            stock_data['Strategy_Returns'] = stock_data['Position'].shift(
                1) * stock_data['Returns']
            stock_data['Strategy_Returns'] = stock_data['Strategy_Returns'].fillna(
                0)
            results[ticker] = stock_data

        return results

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def evaluate_strategy(self, individual, data, min_trades=5, risk_free_rate=0.02):
        try:
            bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct, stop_loss_pct = individual
            backtested_data = self.backtest(
                data, bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct, stop_loss_pct)

            total_sharpe_ratio = 0
            total_return = 0
            total_trades = 0

            for ticker, stock_data in backtested_data.items():
                num_trades = (stock_data['Position'].diff() != 0).sum()
                total_trades += num_trades

                if num_trades < min_trades:
                    continue

                strategy_returns = stock_data['Strategy_Returns']
                sharpe_ratio = self.calculate_sharpe_ratio(
                    strategy_returns, risk_free_rate)
                stock_return = (1 + strategy_returns).prod() - 1

                if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                    sharpe_ratio = 0
                if np.isnan(stock_return) or np.isinf(stock_return):
                    stock_return = 0

                total_sharpe_ratio += sharpe_ratio
                total_return += stock_return

            avg_sharpe_ratio = total_sharpe_ratio / \
                len(self.tickers) if len(self.tickers) > 0 else 0
            avg_return = total_return / \
                len(self.tickers) if len(self.tickers) > 0 else 0
            avg_trades = total_trades / \
                len(self.tickers) if len(self.tickers) > 0 else 0

            trade_penalty = max(0, (avg_trades - 100) / 100)
            avg_sharpe_ratio -= trade_penalty

            return avg_sharpe_ratio, avg_return, avg_trades
        except Exception as e:
            return -np.inf, -np.inf, 0

    def optimize_strategy(self, progress_bar, status_text, ga_config=None, stop_flag=lambda: False, min_trades=5, risk_free_rate=0.02):
        if not self.data:
            st.error("No data available for optimization.")
            return None

        # Clear any existing fitness classes
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Use configuration ranges if provided, otherwise fall back to defaults
        ga_config = ga_config or {}
        bb_entry_min, bb_entry_max = ga_config.get(
            'bb_entry_range', (0.9, 1.1))
        bb_exit_min, bb_exit_max = ga_config.get('bb_exit_range', (0.9, 1.1))
        rsi_entry_min, rsi_entry_max = ga_config.get(
            'rsi_entry_range', (20, 40))
        rsi_exit_min, rsi_exit_max = ga_config.get('rsi_exit_range', (60, 80))
        trailing_min, trailing_max = ga_config.get(
            'trailing_range', (0.01, 0.1))
        stop_loss_min, stop_loss_max = ga_config.get(
            'stop_loss_range', (0.01, 0.1))

        toolbox.register("attr_bb_entry", random.uniform,
                         bb_entry_min, bb_entry_max)
        toolbox.register("attr_bb_exit", random.uniform,
                         bb_exit_min, bb_exit_max)
        toolbox.register("attr_rsi_entry", random.uniform,
                         rsi_entry_min, rsi_entry_max)
        toolbox.register("attr_rsi_exit", random.uniform,
                         rsi_exit_min, rsi_exit_max)
        toolbox.register("attr_trailing_stop", random.uniform,
                         trailing_min, trailing_max)
        toolbox.register("attr_stop_loss", random.uniform,
                         stop_loss_min, stop_loss_max)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_bb_entry, toolbox.attr_bb_exit,
                          toolbox.attr_rsi_entry, toolbox.attr_rsi_exit,
                          toolbox.attr_trailing_stop, toolbox.attr_stop_loss), n=1)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)

        def evaluate(individual):
            # evaluate uses current data and backtest filters
            return self.evaluate_strategy(individual, self.data, min_trades=min_trades, risk_free_rate=risk_free_rate)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian,
                         mu=0, sigma=0.1, indpb=0.2)

        population = toolbox.population(n=ga_config.get('population', 200))
        NGEN = ga_config.get('generations', 100)

        best_strategy = None
        best_fitness = (-np.inf, -np.inf, np.inf)

        cxpb = ga_config.get('cxpb', 0.7)
        mutpb = ga_config.get('mutpb', 0.3)
        tournsize = ga_config.get('tournsize', 3)
        # register selection using tournament size
        toolbox.register("select", tools.selTournament, tournsize=tournsize)

        for gen in range(NGEN):
            offspring = algorithms.varAnd(
                population, toolbox, cxpb=cxpb, mutpb=mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population = toolbox.select(
                offspring + population, k=len(population))

            gen_best = tools.selBest(population, k=1)[0]
            if gen_best.fitness.values[0] > best_fitness[0]:
                best_strategy = gen_best
                best_fitness = gen_best.fitness.values

            progress_bar.progress((gen + 1) / NGEN)
            if (gen + 1) % max(1, int(NGEN/10)) == 0:
                status_text.text(
                    f"Generation {gen+1}/{NGEN}: Sharpe={best_fitness[0]:.4f}, Return={best_fitness[1]:.2%}")

            # allow user to stop optimization early
            if stop_flag():
                status_text.text("⚠️ Optimization stopped by user.")
                break

        status_text.text("✅ Optimization completed!")
        return best_strategy, best_fitness

    def plot_stock_strategy(self, ticker, stock_data, bb_entry, bb_exit, trailing_stop_pct, stop_loss_pct):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=[f'{ticker} Trading Strategy', 'RSI'])

        # Price and Bollinger Bands
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                                 name='Close Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['UpperBB'],
                                 name='Upper BB', line=dict(color='red', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['LowerBB'],
                                 name='Lower BB', line=dict(color='orange', width=1),
                                 fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)'), row=1, col=1)

        # Buy and Sell signals
        buy_signals = stock_data[stock_data['Position'].diff() == 1]
        sell_signals = stock_data[stock_data['Position'].diff() == -1]

        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                 mode='markers', name='Buy Signal',
                                 marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                 mode='markers', name='Sell Signal',
                                 marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'],
                                 name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=600, showlegend=True, hovermode="x unified")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig

    def plot_cumulative_returns(self, ticker, stock_data):
        cumulative_strategy = (1 + stock_data['Strategy_Returns']).cumprod()
        cumulative_buy_hold = (1 + stock_data['Returns']).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=cumulative_strategy,
                                 name='Strategy Returns', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=stock_data.index, y=cumulative_buy_hold,
                                 name='Buy & Hold Returns', line=dict(color='blue', width=2, dash='dash')))

        fig.update_layout(title=f'{ticker} Cumulative Returns Comparison',
                          xaxis_title='Date', yaxis_title='Cumulative Returns',
                          height=400, hovermode="x unified")
        return fig

    def run_strategy(self, progress_bar, status_text, ga_config=None, stop_flag=lambda: False, min_trades=5, risk_free_rate=0.02):
        try:
            status_text.text("📥 Fetching data...")
            raw_data = self.fetch_data()
            if raw_data is None or raw_data.empty:
                st.error("No data available to run the strategy.")
                return None

            status_text.text("🔧 Preprocessing data...")
            self.data = self.preprocess_data(raw_data)
            self.data = self.calculate_indicators(self.data)

            status_text.text("🧬 Optimizing strategy...")
            result = self.optimize_strategy(progress_bar, status_text, ga_config=ga_config,
                                            stop_flag=stop_flag, min_trades=min_trades, risk_free_rate=risk_free_rate)

            if result is None:
                st.error("Optimization failed.")
                return None

            best_strategy, best_fitness = result

            return best_strategy, best_fitness, self.data
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None


def main():
    st.set_page_config(
        page_title="Multi-Stock Trading Strategy Optimizer", layout="wide")

    st.title("📈 Multi-Stock Trading Strategy Optimizer")
    st.markdown(
        "Optimize Bollinger Band + RSI trading strategies across multiple stocks using genetic algorithms")

    # Sidebar inputs
    st.sidebar.header("⚙️ Configuration")

    tickers_input = st.sidebar.text_input(
        "Stock Tickers (comma-separated)", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper()
               for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 9, 1))

    interval = st.sidebar.selectbox(
        "Interval", ["1h", "1d", "5m", "15m", "30m"], index=0)

    # Genetic algorithm controls
    st.sidebar.subheader("Genetic Algorithm")
    population = st.sidebar.number_input(
        "Population", min_value=10, max_value=5000, value=200, step=10)
    generations = st.sidebar.number_input(
        "Generations", min_value=1, max_value=2000, value=100, step=1)
    cxpb = st.sidebar.slider("Crossover probability", 0.0, 1.0, 0.7, 0.01)
    mutpb = st.sidebar.slider("Mutation probability", 0.0, 1.0, 0.3, 0.01)
    tournsize = st.sidebar.number_input(
        "Tournament size", min_value=2, max_value=100, value=3, step=1)

    # Parameter search ranges
    st.sidebar.subheader("Parameter Ranges")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        bb_entry_min = st.number_input(
            "BB Entry min", value=0.9, step=0.01, format="%.3f")
    with c2:
        bb_entry_max = st.number_input(
            "BB Entry max", value=1.1, step=0.01, format="%.3f")
    c3, c4 = st.sidebar.columns(2)
    with c3:
        bb_exit_min = st.number_input(
            "BB Exit min", value=0.9, step=0.01, format="%.3f")
    with c4:
        bb_exit_max = st.number_input(
            "BB Exit max", value=1.1, step=0.01, format="%.3f")

    r1, r2 = st.sidebar.columns(2)
    with r1:
        rsi_entry_min = st.number_input("RSI Entry min", value=20, step=1)
    with r2:
        rsi_entry_max = st.number_input("RSI Entry max", value=40, step=1)
    r3, r4 = st.sidebar.columns(2)
    with r3:
        rsi_exit_min = st.number_input("RSI Exit min", value=60, step=1)
    with r4:
        rsi_exit_max = st.number_input("RSI Exit max", value=80, step=1)

    t1, t2 = st.sidebar.columns(2)
    with t1:
        trailing_min = st.number_input(
            "Trailing stop min", value=0.01, step=0.01, format="%.2f")
    with t2:
        trailing_max = st.number_input(
            "Trailing stop max", value=0.10, step=0.01, format="%.2f")

    s1, s2 = st.sidebar.columns(2)
    with s1:
        stop_loss_min = st.number_input(
            "Stop loss min", value=0.01, step=0.01, format="%.2f")
    with s2:
        stop_loss_max = st.number_input(
            "Stop loss max", value=0.10, step=0.01, format="%.2f")

    # Backtest / misc options
    st.sidebar.subheader("Backtest & Display")
    min_trades = st.sidebar.number_input(
        "Min trades per ticker (filter)", min_value=1, value=5, step=1)
    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (annual)", value=0.02, step=0.001, format="%.3f")
    show_plots = st.sidebar.checkbox("Show individual plots", True)
    max_tickers_plot = st.sidebar.number_input(
        "Max tickers to plot", min_value=1, value=10, step=1)

    # Run / Stop buttons
    run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary")
    stop_optimization = st.sidebar.button("⏹ Stop Optimization")

    # session flag for stopping optimization
    if 'stop_opt' not in st.session_state:
        st.session_state['stop_opt'] = False
    if stop_optimization:
        st.session_state['stop_opt'] = True

    if run_optimization:
        if not tickers:
            st.error("Please enter at least one ticker symbol")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        # reset stop flag when starting a new optimization
        st.session_state['stop_opt'] = False

        strategy = MultiStockOptimalStrategy(
            tickers=tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval
        )

        # assemble GA config
        ga_config = {
            'population': int(population),
            'generations': int(generations),
            'cxpb': float(cxpb),
            'mutpb': float(mutpb),
            'tournsize': int(tournsize),
            'bb_entry_range': (float(bb_entry_min), float(bb_entry_max)),
            'bb_exit_range': (float(bb_exit_min), float(bb_exit_max)),
            'rsi_entry_range': (float(rsi_entry_min), float(rsi_entry_max)),
            'rsi_exit_range': (float(rsi_exit_min), float(rsi_exit_max)),
            'trailing_range': (float(trailing_min), float(trailing_max)),
            'stop_loss_range': (float(stop_loss_min), float(stop_loss_max)),
        }

        result = strategy.run_strategy(
            progress_bar, status_text,
            ga_config=ga_config,
            stop_flag=lambda: st.session_state.get('stop_opt', False),
            min_trades=int(min_trades),
            risk_free_rate=float(risk_free_rate)
        )

        if result:
            best_strategy, best_fitness, backtested_data = result

            # Display results
            st.success("✅ Optimization Complete!")

            # Strategy parameters
            st.header("📊 Optimized Strategy Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BB Entry", f"{best_strategy[0]:.4f}")
                st.metric("BB Exit", f"{best_strategy[1]:.4f}")
            with col2:
                st.metric("RSI Entry", f"{best_strategy[2]:.2f}")
                st.metric("RSI Exit", f"{best_strategy[3]:.2f}")
            with col3:
                st.metric("Trailing Stop", f"{best_strategy[4]:.2%}")
                st.metric("Stop Loss", f"{best_strategy[5]:.2%}")

            # Performance metrics
            st.header("📈 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Sharpe Ratio", f"{best_fitness[0]:.4f}")
            with col2:
                st.metric("Avg Total Return", f"{best_fitness[1]:.2%}")
            with col3:
                st.metric("Avg Number of Trades", f"{best_fitness[2]:.0f}")

            # Backtest with optimized parameters
            backtested_results = strategy.backtest(
                backtested_data,
                best_strategy[0], best_strategy[1],
                best_strategy[2], best_strategy[3],
                best_strategy[4], best_strategy[5]
            )

            # Display charts for each stock
            st.header("📉 Individual Stock Results")
            # optionally limit plotted tickers
            plot_tickers = tickers[:int(max_tickers_plot)]
            for ticker in plot_tickers:
                if ticker in backtested_results:
                    with st.expander(f"📊 {ticker} - Detailed Analysis", expanded=True):
                        stock_data = backtested_results[ticker]

                        # Calculate metrics for this stock
                        num_trades = (stock_data['Position'].diff() != 0).sum()
                        total_return = (
                            1 + stock_data['Strategy_Returns']).prod() - 1
                        sharpe = strategy.calculate_sharpe_ratio(
                            stock_data['Strategy_Returns'], risk_free_rate)

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Number of Trades", num_trades)
                        col2.metric("Total Return", f"{total_return:.2%}")
                        col3.metric("Sharpe Ratio", f"{sharpe:.4f}")

                        # Trading strategy plot
                        if show_plots:
                            fig1 = strategy.plot_stock_strategy(
                                ticker, stock_data,
                                best_strategy[0], best_strategy[1],
                                best_strategy[4], best_strategy[5]
                            )
                            st.plotly_chart(fig1, use_container_width=True)

                        # Cumulative returns plot
                            fig2 = strategy.plot_cumulative_returns(
                                ticker, stock_data)
                            st.plotly_chart(fig2, use_container_width=True)

            # Summary statistics
            st.header("📋 Summary Statistics")
            summary_data = []
            for ticker in tickers:
                if ticker in backtested_results:
                    stock_data = backtested_results[ticker]
                    num_trades = (stock_data['Position'].diff() != 0).sum()
                    total_return = (
                        1 + stock_data['Strategy_Returns']).prod() - 1
                    sharpe = strategy.calculate_sharpe_ratio(
                        stock_data['Strategy_Returns'])
                    summary_data.append({
                        'Ticker': ticker,
                        'Trades': num_trades,
                        'Return': f"{total_return:.2%}",
                        'Sharpe Ratio': f"{sharpe:.4f}"
                    })

            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        progress_bar.empty()
    else:
        st.info(
            "👈 Configure your parameters in the sidebar and click 'Run Optimization' to start")


if __name__ == "__main__":
    main()
