"""Utilities and strategy classes extracted and consolidated from
`TradingBasics/strategies.ipynb`.

Provides:
- data fetch helpers (period or date-range)
- indicator calculators (Bollinger Bands, RSI)
- backtest / paper-trade functions
- `OptimalStrategy` class that can optimize parameters using DEAP

This module is intended as a drop-in programmatic version of the
notebook logic so strategies can be imported and reused.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Optional heavy dependency used for optimization
try:
    from deap import base, creator, tools, algorithms
    _HAS_DEAP = True
except Exception:
    _HAS_DEAP = False

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize possibly-MultiIndex columns returned by yfinance into a
    single-level column index with familiar names like 'Close', 'Open', etc.

    If the DataFrame has a MultiIndex in columns, this function will try to
    use the first (level 0) labels (e.g. 'Close', 'Open') as column names.
    If that fails it will flatten by joining the tuple with an underscore.
    """
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        try:
            # Prefer level 0 (e.g. 'Close', 'Open') which matches the rest of the
            # code that expects plain column names.
            cols = df.columns.get_level_values(0)
            df = df.copy()
            df.columns = cols
            return df
        except Exception:
            # Fallback to joined names: ('Close', 'ALLFG.AS') -> 'Close_ALLFG.AS'
            df = df.copy()
            df.columns = ["_".join(
                [str(c) for c in col if c is not None and c != ""]) for col in df.columns]
            return df
    return df


def fetch_data_period(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for a ticker using yfinance by period.

    Raises ValueError if returned data is empty.
    """
    data = yf.download(ticker, period=period, interval=interval)
    if data is None or data.empty:
        raise ValueError(
            f"No data available for {ticker} (period={period}, interval={interval})")
    logger.info("Fetched %d data points for %s", len(data), ticker)
    return data


def fetch_data_range(ticker: str, start: str | datetime, end: str | datetime, interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data by explicit date range.

    Start and end may be strings or datetime objects.
    """
    data = yf.download(ticker, start=start, end=end, interval=interval)
    if data is None or data.empty:
        raise ValueError(
            f"No data available for {ticker} between {start} and {end}")
    logger.info("Fetched %d data points for %s between %s and %s",
                len(data), ticker, start, end)
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = _normalize_columns(data.copy())
    data["Returns"] = data["Close"].pct_change()
    data.dropna(inplace=True)
    return data


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    data = data.copy()
    data["MA20"] = data["Close"].rolling(window=window).mean()
    data["STD20"] = data["Close"].rolling(window=window).std()
    data["UpperBB"] = data["MA20"] + (data["STD20"] * num_std)
    data["LowerBB"] = data["MA20"] - (data["STD20"] * num_std)
    return data


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = data.copy()
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = _normalize_columns(data.copy())
    data = calculate_bollinger_bands(data)
    data = calculate_rsi(data)
    return data


def backtest(data: pd.DataFrame, bb_entry: float, bb_exit: float, rsi_entry: float, rsi_exit: float,
             trailing_stop_pct: float) -> pd.DataFrame:
    """Run a simple long-only backtest using Bollinger+RSI entry/exit and a trailing stop.

    Returns a dataframe with Position and Strategy_Returns columns added.
    """
    data = _normalize_columns(data.copy())
    data["Position"] = 0
    in_trade = False
    entry_price = 0.0
    highest_price = 0.0
    trailing_stop_price = 0.0

    for i in range(1, len(data)):
        close = data["Close"].iloc[i]
        if not in_trade:
            bb_condition = close < data["LowerBB"].iloc[i] * bb_entry
            rsi_condition = data["RSI"].iloc[i] < rsi_entry
            if bb_condition and rsi_condition:
                data.at[data.index[i], "Position"] = 1
                in_trade = True
                entry_price = close
                highest_price = entry_price
                trailing_stop_price = entry_price * (1 - trailing_stop_pct)
        else:
            if close > highest_price:
                highest_price = close
                trailing_stop_price = highest_price * (1 - trailing_stop_pct)

            bb_condition = close > data["UpperBB"].iloc[i] * bb_exit
            rsi_condition = data["RSI"].iloc[i] > rsi_exit
            trailing_stop_condition = close <= trailing_stop_price

            if bb_condition or rsi_condition or trailing_stop_condition:
                data.at[data.index[i], "Position"] = 0
                in_trade = False
            else:
                data.at[data.index[i], "Position"] = 1

    data["Strategy_Returns"] = data["Position"].shift(
        1) * data.get("Returns", data["Close"].pct_change())
    data["Strategy_Returns"] = data["Strategy_Returns"].fillna(0)
    return data


def paper_trade(data: pd.DataFrame, bb_entry: float, bb_exit: float, rsi_entry: float, rsi_exit: float,
                trailing_stop_pct: float) -> Tuple[List[dict], List[float]]:
    """Return a list of trades (dicts) and a list of daily returns used in some notebook cells."""
    position = 0
    entry_price = 0.0
    highest_price = 0.0
    trailing_stop_price = 0.0
    trades: List[dict] = []
    daily_returns: List[float] = []

    for i in range(len(data)):
        close = data["Close"].iloc[i]
        if position == 0:
            bb_condition = close < data["LowerBB"].iloc[i] * bb_entry
            rsi_condition = data["RSI"].iloc[i] < rsi_entry
            if bb_condition and rsi_condition:
                position = 1
                entry_price = close
                highest_price = entry_price
                trailing_stop_price = entry_price * (1 - trailing_stop_pct)
                trades.append(
                    {"Date": data.index[i], "Type": "Buy", "Price": entry_price})
            daily_returns.append(0.0)
        else:
            if close > highest_price:
                highest_price = close
                trailing_stop_price = highest_price * (1 - trailing_stop_pct)

            bb_condition = close > data["UpperBB"].iloc[i] * bb_exit
            rsi_condition = data["RSI"].iloc[i] > rsi_exit
            trailing_stop_condition = close <= trailing_stop_price

            if bb_condition or rsi_condition or trailing_stop_condition:
                position = 0
                exit_price = close
                trades.append(
                    {"Date": data.index[i], "Type": "Sell", "Price": exit_price})
                # record trade-level return
                daily_returns.append((exit_price - entry_price) / entry_price)
            else:
                # approximate daily (period) return
                prev_close = data["Close"].iloc[i - 1] if i > 0 else close
                daily_returns.append(
                    (close - prev_close) / prev_close if prev_close != 0 else 0.0)

    return trades, daily_returns


def calculate_returns(trades: List[dict]) -> float:
    if len(trades) < 2:
        return 0.0
    total_return = 1.0
    for i in range(0, len(trades) - 1, 2):
        buy_price = trades[i]["Price"]
        sell_price = trades[i + 1]["Price"]
        trade_return = (sell_price - buy_price) / buy_price
        total_return *= (1 + trade_return)
    return total_return - 1


def calculate_sharpe_ratio(returns: List[float] | pd.Series, risk_free_rate: float = 0.02) -> float:
    arr = np.array(returns)
    excess = arr - (risk_free_rate / 252)
    if arr.size == 0 or np.std(excess) == 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(excess) / np.std(excess))


class OptimalStrategy:
    """Encapsulates data fetching, indicator calculation and DEAP optimization.

    The class mirrors the notebook's `OptimalStrategy` but is more reusable.
    If DEAP is not installed the optimize step will raise RuntimeError.
    """

    def __init__(self, ticker: str, *, period: Optional[str] = None, start: Optional[str] = None,
                 end: Optional[str] = None, interval: str = "1d"):
        self.ticker = ticker
        self.period = period
        self.start = start
        self.end = end
        self.interval = interval
        self.data: Optional[pd.DataFrame] = None

    def fetch_and_prepare_data(self) -> None:
        if self.period is not None:
            raw = fetch_data_period(
                self.ticker, period=self.period, interval=self.interval)
        else:
            if self.start is None or self.end is None:
                raise ValueError(
                    "Either period or (start and end) must be provided")
            raw = fetch_data_range(
                self.ticker, self.start, self.end, interval=self.interval)

        raw = preprocess_data(raw)
        raw = calculate_indicators(raw)
        self.data = raw

    def optimize_strategy(self, population_size: int = 200, ngen: int = 100) -> Optional[List[float]]:
        if not _HAS_DEAP:
            raise RuntimeError(
                "DEAP is required for optimization (install via `pip install deap`).")
        if self.data is None or self.data.empty:
            raise ValueError(
                "No data available for optimization. Call fetch_and_prepare_data first.")

        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_bb_entry", np.random.uniform, 0.9, 1.1)
        toolbox.register("attr_bb_exit", np.random.uniform, 0.9, 1.1)
        toolbox.register("attr_rsi_entry", np.random.uniform, 20, 40)
        toolbox.register("attr_rsi_exit", np.random.uniform, 60, 80)
        toolbox.register("attr_trailing_stop", np.random.uniform, 0.01, 0.1)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_bb_entry, toolbox.attr_bb_exit,
                          toolbox.attr_rsi_entry, toolbox.attr_rsi_exit,
                          toolbox.attr_trailing_stop), n=1)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)

        def evaluate(individual):
            try:
                bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct = individual
                bt = backtest(self.data, bb_entry, bb_exit,
                              rsi_entry, rsi_exit, trailing_stop_pct)
                num_trades = int((bt["Position"].diff() != 0).sum())
                if num_trades < 5:
                    return -np.inf, -np.inf, num_trades
                strategy_returns = bt["Strategy_Returns"].fillna(0)
                sharpe = calculate_sharpe_ratio(strategy_returns)
                total_return = float((1 + strategy_returns).prod() - 1)
                trade_penalty = max(0, (num_trades - 100) / 100)
                sharpe -= trade_penalty
                return sharpe, total_return, num_trades
            except Exception as e:
                logger.exception("Error evaluating individual: %s", e)
                return -np.inf, -np.inf, 0

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian,
                         mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=population_size)
        best_strategy = None
        best_fitness = (-np.inf, -np.inf, np.inf)

        for gen in range(ngen):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            pop = toolbox.select(offspring + pop, k=len(pop))
            gen_best = tools.selBest(pop, k=1)[0]
            if gen_best.fitness.values[0] > best_fitness[0]:
                best_strategy = gen_best
                best_fitness = gen_best.fitness.values

        return list(best_strategy) if best_strategy is not None else None

    def plot_best_strategy(self, best_strategy: List[float]) -> None:
        if self.data is None:
            raise ValueError(
                "No data to plot. Call fetch_and_prepare_data first.")
        bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop_pct = best_strategy
        bt = backtest(self.data, bb_entry, bb_exit,
                      rsi_entry, rsi_exit, trailing_stop_pct)

        plt.figure(figsize=(12, 6))
        plt.plot(bt.index, bt["Close"], label="Close")
        plt.plot(bt.index, bt["UpperBB"], label="Upper BB", alpha=0.5)
        plt.plot(bt.index, bt["LowerBB"], label="Lower BB", alpha=0.5)

        buys = bt[bt["Position"].diff() == 1]
        sells = bt[bt["Position"].diff() == -1]
        plt.scatter(buys.index, buys["Close"],
                    marker="^", color="g", s=80, label="Buy")
        plt.scatter(sells.index, sells["Close"],
                    marker="v", color="r", s=80, label="Sell")

        plt.title(
            f"Best Strategy for {self.ticker} — Trailing stop {trailing_stop_pct:.2%}")
        plt.legend()
        plt.show()


__all__ = [
    "fetch_data_period",
    "fetch_data_range",
    "preprocess_data",
    "calculate_bollinger_bands",
    "calculate_rsi",
    "calculate_indicators",
    "backtest",
    "paper_trade",
    "calculate_returns",
    "calculate_sharpe_ratio",
    "OptimalStrategy",
]


# ---------------------- Streamlit UI (module runnable) ----------------------
try:
    import streamlit as st
except Exception:
    st = None


def main():
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install with `pip install streamlit` to run the UI.")

    st.set_page_config(page_title="BBRSIT Strategy Backtester", layout="wide")
    st.title("BBRSIT Strategy Backtester")

    with st.sidebar:
        st.header("Data")
        ticker = st.text_input("Ticker", value="ALLFG.AS")
        mode = st.radio("Mode", ("Quick backtest (period)",
                        "Paper trade (date range)"))

        if mode == "Quick backtest (period)":
            period = st.selectbox(
                "Period", ("1y", "2y", "5y", "6mo", "3mo"), index=0)
            interval = st.selectbox(
                "Interval (period mode)", ("1d", "1wk"), index=0)
        else:
            today = datetime.today().date()
            end_date = st.date_input("End date", value=today)
            start_date = st.date_input(
                "Start date", value=today - pd.Timedelta(days=90))
            interval = st.selectbox(
                "Interval (range mode)", ("1h", "1d"), index=0)

        st.header("Strategy params")
        bb_entry = st.number_input(
            "BB entry multiplier", value=1.02, format="%.4f")
        bb_exit = st.number_input(
            "BB exit multiplier", value=0.9739, format="%.4f")
        rsi_entry = st.number_input("RSI entry", value=29.12, format="%.4f")
        rsi_exit = st.number_input("RSI exit", value=65.21, format="%.4f")
        trailing_stop = st.number_input(
            "Trailing stop %", value=0.0219, format="%.4f")

    run_btn = st.sidebar.button("Run")

    def _show_indicator_plot(df: pd.DataFrame) -> None:
        df_plot = df[["Close", "UpperBB", "LowerBB"]].dropna()
        st.line_chart(df_plot)

    def _show_cumulative(df: pd.DataFrame) -> None:
        cum = (1 + df["Strategy_Returns"]).cumprod()
        bh = (1 + df["Returns"]).cumprod()
        st.line_chart(pd.DataFrame({"Strategy": cum, "BuyHold": bh}))

    if run_btn:
        try:
            st.info("Fetching data...")
            if mode == "Quick backtest (period)":
                raw = fetch_data_period(
                    ticker, period=period, interval=interval)
            else:
                raw = fetch_data_range(ticker, start_date.isoformat(
                ), end_date.isoformat(), interval=interval)

            df = preprocess_data(raw)
            df = calculate_indicators(df)

            st.success(f"Prepared {len(df)} rows of data")
            st.subheader("Data sample")
            st.dataframe(df.head())

            with st.expander("Indicator plot"):
                _show_indicator_plot(df)

            st.subheader("Run backtest")
            bt = backtest(df, bb_entry, bb_exit, rsi_entry,
                          rsi_exit, trailing_stop)
            st.write("Backtest summary")
            # Show the rows where the position changed (buy/sell events).
            changes = bt[bt["Position"].diff().abs() > 0]
            if not changes.empty:
                st.dataframe(
                    changes[["Position", "Strategy_Returns", "Close"]])
            else:
                # Fallback: show the tail of the series if no explicit trades in view
                st.write(bt[["Position", "Strategy_Returns"]].tail())
            with st.expander("Cumulative returns"):
                _show_cumulative(bt)

            st.subheader("Paper trades (list)")
            trades, daily = paper_trade(
                df, bb_entry, bb_exit, rsi_entry, rsi_exit, trailing_stop)
            if trades:
                st.dataframe(pd.DataFrame(trades))
            else:
                st.write("No trades found with these parameters")

            st.markdown("---")
            st.write(
                f"Total return (paper trades): {calculate_returns(trades):.2%}")
            st.write(
                f"Sharpe (daily returns): {calculate_sharpe_ratio(daily):.4f}")

        except Exception as e:
            st.error(f"Error: {e}")
            raise
    else:
        st.write(
            "Use the sidebar to choose a ticker, mode and parameters, then press Run.")


if __name__ == "__main__":
    if st is None:
        print(
            "Streamlit is not installed. Install via `pip install streamlit` to run the UI.")
    else:
        main()
