"""Stock correlation utility and Streamlit UI converted from
`TradingBasics/stockcorrelation.ipynb`.

Provides functions to compute common indicators (Bollinger Bands, RSI,
ATR, SuperTrend, MACD) for a set of tickers and to compute and display a
correlation matrix of their Close prices. The module is runnable with
Streamlit (e.g. `streamlit run strategies/correlation.py`).
"""
from __future__ import annotations

import io
import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If columns are MultiIndex (from yfinance grouped downloads),
    convert to single-level by taking the second level when appropriate
    or joining the tuple into a single string.
    """
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        try:
            # If columns look like ('Close', 'SPY'), prefer the first level
            cols = df.columns.get_level_values(0)
            new = df.copy()
            new.columns = cols
            return new
        except Exception:
            new = df.copy()
            new.columns = [
                "_".join([str(c) for c in col if c is not None]) for col in df.columns]
            return new
    return df


def BBand(df: pd.DataFrame, base: str = "Close", period: int = 20, multiplier: float = 2, multiplier3: float = 3) -> pd.DataFrame:
    sma = df[base].rolling(window=period, min_periods=period - 1).mean()
    sd = df[base].rolling(window=period).std()
    df = df.copy()
    df["UpperBB"] = sma + (multiplier * sd)
    df["LowerBB"] = sma - (multiplier * sd)
    df["MiddleBB"] = sma
    df["UpperBB3"] = sma + (multiplier3 * sd)
    df["LowerBB3"] = sma - (multiplier3 * sd)
    return df


def RSI(df: pd.DataFrame, base: str = "Close", period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df[base].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rUp = up.ewm(com=period - 1, adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
    df["RSI"] = 100 - 100 / (1 + rUp / rDown)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.Series:
    atr = calculate_atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    # initialize first values to NaN
    for i in range(period, len(df)):
        if df["Close"].iat[i] > upper_band.iat[i - 1]:
            supertrend.iat[i] = lower_band.iat[i]
            direction.iat[i] = 1
        elif df["Close"].iat[i] < lower_band.iat[i - 1]:
            supertrend.iat[i] = upper_band.iat[i]
            direction.iat[i] = -1
        else:
            supertrend.iat[i] = supertrend.iat[i - 1]
            direction.iat[i] = direction.iat[i - 1]
            if direction.iat[i] == 1 and lower_band.iat[i] < supertrend.iat[i]:
                supertrend.iat[i] = lower_band.iat[i]
            elif direction.iat[i] == -1 and upper_band.iat[i] > supertrend.iat[i]:
                supertrend.iat[i] = upper_band.iat[i]

    return supertrend


def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = df["Close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df.copy())
    df = BBand(df)
    df = RSI(df)
    df["ATR"] = calculate_atr(df)
    df["SuperTrend"] = calculate_supertrend(df)
    macd, sig, hist = calculate_macd(df)
    df["MACD"] = macd
    df["Signal"] = sig
    df["Histogram"] = hist
    return df


def calculate_correlation(ticker_data: dict, tickers: List[str]) -> pd.DataFrame:
    """Build a DataFrame of close prices for each ticker and compute correlation."""
    close_prices = pd.DataFrame({t: ticker_data[t]["Close"] for t in tickers})
    return close_prices.corr()


__all__ = [
    "BBand",
    "RSI",
    "calculate_atr",
    "calculate_supertrend",
    "calculate_macd",
    "calculate_indicators",
    "calculate_correlation",
]


# ---------------------- Streamlit UI (module runnable) ----------------------
try:
    import streamlit as st
except Exception:
    st = None

try:
    import seaborn as sns
except Exception:
    sns = None


def main() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is required to run this UI. Install with `pip install streamlit`.")

    st.set_page_config(page_title="Stock Correlation", layout="wide")
    st.title("Stock Correlation — quick UI")

    with st.sidebar:
        st.header("Data")
        tickers_txt = st.text_input(
            "Tickers (comma separated)", value="SPY, AAPL, MSFT, GOOGL, AMZN")
        tickers = [t.strip().upper()
                   for t in tickers_txt.split(",") if t.strip()]

        period = st.selectbox(
            "Period", ("1y", "2y", "5y", "6mo", "3mo"), index=0)
        interval = st.selectbox("Interval", ("1d", "1wk", "1mo"), index=0)

        st.markdown("---")
        st.header("Indicator options")
        calc_ind = st.checkbox(
            "Calculate indicators for each ticker (slower)", value=True)

    run = st.sidebar.button("Run")

    if not run:
        st.write(
            "Use the sidebar and press Run to fetch data and compute correlation.")
        return

    st.info("Fetching data from yfinance...")
    try:
        data = yf.download(tickers, period=period,
                           interval=interval, group_by="ticker", threads=True)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return

    if data is None or data.empty:
        st.error("No data returned for the selected tickers/period.")
        return

    # Process each ticker into a dict of DataFrames
    ticker_data = {}
    for t in tickers:
        try:
            df = data[t].copy() if t in data.columns.get_level_values(
                0) else data[t]
        except Exception:
            # Fallback: try selecting by column labels
            try:
                df = data[t].copy()
            except Exception:
                st.warning(f"No data for {t}")
                continue

        df = _normalize_columns(df)
        if calc_ind:
            df = calculate_indicators(df)

        ticker_data[t] = df

    if not ticker_data:
        st.error("No valid ticker data available after processing.")
        return

    st.success(f"Prepared data for {len(ticker_data)} tickers")

    # Compute correlation matrix
    corr = calculate_correlation(ticker_data, list(ticker_data.keys()))

    st.subheader("Correlation matrix")
    st.dataframe(corr)

    # Download CSV
    csv_bytes = corr.to_csv().encode("utf-8")
    st.download_button("Download correlation CSV", data=csv_bytes,
                       file_name="correlation_matrix.csv", mime="text/csv")

    # Plot heatmap
    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(corr, annot=True, cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, ax=ax)
    else:
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.index)
        for (i, j), val in np.ndenumerate(corr.values):
            ax.text(j, i, f"{val:.2f}", ha="center",
                    va="center", color="black")
        fig.colorbar(im, ax=ax)

    st.pyplot(fig)

    # Show sample indicator CSVs for each ticker and allow download
    st.subheader("Per-ticker data (sample)")
    for t, df in ticker_data.items():
        with st.expander(f"{t} — sample and download"):
            st.write(df.head())
            csv_buf = io.StringIO()
            df.to_csv(csv_buf)
            st.download_button(f"Download {t} CSV", data=csv_buf.getvalue(
            ), file_name=f"{t}_data.csv", mime="text/csv")


if __name__ == "__main__":
    if st is None:
        print(
            "Streamlit is not installed. Install via `pip install streamlit` to run the UI.")
    else:
        main()
