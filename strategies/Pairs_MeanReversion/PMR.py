"""
pairs_mean_reversion

This file provides a minimal Streamlit UI that calls the core functions
from `strategies.pairs_mean_reversion`. It keeps the original logic in place
and only adds a lightweight front-end so the launcher or `streamlit run`
produces a usable page.

Usage:
    streamlit run strategies/PMR.py
"""
from __future__ import annotations
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

# Ensure project root (one level up) is on sys.path so we can import the
# `strategies` package regardless of current working directory when Streamlit
# runs the script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # import core functions from the CLI script
    from strategies.PMR import (
        RECOMMENDED_PAIRS,
        fetch_prices,
        calculate_hedge_ratio,
        bollinger_bands,
        generate_signals,
        backtest_pair,
    )
except Exception as e:
    # as a fallback, try importing the module as a sibling (if cwd==strategies)
    try:
        from PMR import (
            RECOMMENDED_PAIRS,
            fetch_prices,
            calculate_hedge_ratio,
            bollinger_bands,
            generate_signals,
            backtest_pair,
        )
    except Exception:
        st.error(
            f"Failed to import core functions from pairs_mean_reversion: {e}")
        raise


st.set_page_config(page_title="Pairs Mean Reversion", layout="wide")


def choose_pair_ui():
    options = [f"{i+1}. {p[0][0]} <-> {p[0][1]}" for i,
               p in enumerate(RECOMMENDED_PAIRS)]
    options.append("Custom pair...")
    sel = st.selectbox("Choose a pair", options)
    if sel == "Custom pair...":
        t1 = st.text_input("Ticker 1", value="MSFT")
        t2 = st.text_input("Ticker 2", value="AAPL")
        return t1.strip(), t2.strip()
    else:
        idx = int(sel.split('.')[0]) - 1
        return RECOMMENDED_PAIRS[idx][0]


def main():
    st.title("Pairs Mean-Reversion (Streamlit wrapper)")

    with st.sidebar.form("params"):
        pair = choose_pair_ui()
        mode = st.selectbox("Mode", ["both", "swing", "intraday"], index=0)
        start_days = st.number_input(
            "History (days, swing)", value=200, min_value=30)
        intraday_days = st.number_input(
            "Intraday days (for 15m/1h)", value=15, min_value=1)
        notional = st.number_input(
            "Notional per trade", value=1000.0, min_value=1.0)
        boll_window = st.number_input(
            "Bollinger window", value=20, min_value=1)
        boll_std = st.number_input("Bollinger std", value=2.0, format="%.2f")
        max_concurrent = st.number_input(
            "Max concurrent trades", value=4, min_value=1)
        # Portfolio settings (from the notebook): balance and risk tolerances
        st.markdown("---")
        balance = st.number_input(
            "Portfolio balance", value=1000.0, min_value=0.0, step=100.0)
        risk_tolerance_swing = st.number_input(
            "Risk tolerance (swing, fraction)", value=0.04, min_value=0.0, max_value=1.0, format="%.4f")
        risk_tolerance_intraday = st.number_input(
            "Risk tolerance (intraday, fraction)", value=0.01, min_value=0.0, max_value=1.0, format="%.4f")
        max_loss_per_share = st.number_input(
            "Max loss per share (currency)", value=0.5, min_value=0.0, format="%.2f")
        run = st.form_submit_button("Run analysis")

    if not run:
        st.info("Configure parameters in the sidebar and click 'Run analysis'.")
        return

    now = datetime.now()
    results = {}

    with st.spinner("Fetching data and running analysis..."):
        if mode in ("swing", "both"):
            start = now - timedelta(days=int(start_days))
            try:
                y_daily, x_daily = fetch_prices(
                    pair, start=start, end=now, interval="1d")
                beta_daily = calculate_hedge_ratio(y_daily, x_daily)
                spread_daily = y_daily - beta_daily * x_daily
                ma_d, up_d, low_d = bollinger_bands(
                    spread_daily, window=int(boll_window), num_std=float(boll_std))
                sigs_daily = generate_signals(
                    spread_daily, ma_d, up_d, low_d, max_concurrent=int(max_concurrent))
                trades_d, equity_d = backtest_pair(
                    y_daily, x_daily, beta_daily, sigs_daily, notional_per_trade=float(notional))
                results['swing'] = {'beta': beta_daily, 'spread': spread_daily, 'ma': ma_d, 'upper': up_d,
                                    'lower': low_d, 'signals': sigs_daily, 'trades': trades_d, 'equity': equity_d}
            except Exception as e:
                st.error(f"Swing analysis failed: {e}")

        if mode in ("intraday", "both"):
            start_i = now - timedelta(days=int(intraday_days))
            try:
                y_intr, x_intr = fetch_prices(
                    pair, start=start_i, end=now, interval='15m')
            except Exception:
                try:
                    y_intr, x_intr = fetch_prices(
                        pair, start=start_i, end=now, interval='1h')
                except Exception:
                    y_intr = x_intr = pd.Series(dtype=float)

            if not y_intr.empty and not x_intr.empty:
                try:
                    beta_i = calculate_hedge_ratio(y_intr, x_intr)
                    spread_intr = y_intr - beta_i * x_intr
                    ma_i, up_i, low_i = bollinger_bands(
                        spread_intr, window=int(boll_window), num_std=float(boll_std))
                    sigs_i = generate_signals(
                        spread_intr, ma_i, up_i, low_i, max_concurrent=int(max_concurrent))
                    trades_i, equity_i = backtest_pair(
                        y_intr, x_intr, beta_i, sigs_i, notional_per_trade=float(notional))
                    results['intraday'] = {'beta': beta_i, 'spread': spread_intr, 'ma': ma_i, 'upper': up_i,
                                           'lower': low_i, 'signals': sigs_i, 'trades': trades_i, 'equity': equity_i}
                except Exception as e:
                    st.error(f"Intraday analysis failed: {e}")

    # Present results
    if 'swing' in results:
        st.subheader("Swing (daily) results")
        s = results['swing']
        st.metric("Hedge ratio (beta)", f"{s['beta']:.4f}")
        st.write("Spread (last 200 values)")
        df_plot = pd.DataFrame({
            'spread': s['spread'],
            'ma': s['ma'],
            'upper': s['upper'],
            'lower': s['lower']
        }).dropna()
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot.index,
                          y=df_plot['spread'], name='Spread'))
            fig.add_trace(go.Scatter(x=df_plot.index,
                          y=df_plot['ma'], name='MA'))
            fig.add_trace(go.Scatter(x=df_plot.index,
                          y=df_plot['upper'], name='Upper'))
            fig.add_trace(go.Scatter(x=df_plot.index,
                          y=df_plot['lower'], name='Lower'))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.line_chart(df_plot)

        trades_df = s['trades']
        if not trades_df.empty:
            st.write("Closed trades")
            st.dataframe(trades_df)
            st.write(
                f"Total backtest PnL (currency): {trades_df['pnl'].sum():.2f}")

            # Simple portfolio sizing (derived from the notebook)
            def calculate_portfolio(balance, risk_tolerance, max_loss_per_share):
                max_loss_per_trade = balance * risk_tolerance
                shares_to_trade = int(
                    max_loss_per_trade / max_loss_per_share) if max_loss_per_share > 0 else 0
                return shares_to_trade, max_loss_per_trade

            swing_shares, swing_max_loss = calculate_portfolio(
                balance, risk_tolerance_swing, max_loss_per_share)
            st.write(
                f"Suggested max shares (swing): {swing_shares} (max loss per trade €{swing_max_loss:.2f})")

            # Simple per-share profit estimate using the y ticker prices recorded in trades
            try:
                def calculate_profit_from_trades(trades_df, shares, ticker_is_y=True):
                    total_profit = 0.0
                    for _, tr in trades_df.iterrows():
                        if ticker_is_y:
                            entry_price = tr.get('entry_price_y')
                            exit_price = tr.get('exit_price_y')
                            if pd.notna(entry_price) and pd.notna(exit_price):
                                if str(tr.get('type', '')).startswith('long'):
                                    profit = (
                                        exit_price - entry_price) * shares
                                else:
                                    profit = (entry_price -
                                              exit_price) * shares
                                total_profit += profit
                    return total_profit

                est_profit = calculate_profit_from_trades(
                    trades_df, swing_shares, ticker_is_y=True)
                st.write(
                    f"Estimated potential profit (simple per-share estimate on y ticker): €{est_profit:.2f}")
            except Exception:
                # don't fail hard for optional estimate
                pass
        else:
            st.info("No closed swing trades detected with current rules.")

    if 'intraday' in results:
        st.subheader("Intraday results")
        i = results['intraday']
        st.metric("Hedge ratio (intraday)", f"{i['beta']:.4f}")
        trades_df = i['trades']
        if not trades_df.empty:
            st.write("Closed intraday trades")
            st.dataframe(trades_df)
            st.write(
                f"Total backtest PnL (currency): {trades_df['pnl'].sum():.2f}")

            intraday_shares, intraday_max_loss = calculate_portfolio(
                balance, risk_tolerance_intraday, max_loss_per_share)
            st.write(
                f"Suggested max shares (intraday): {intraday_shares} (max loss per trade €{intraday_max_loss:.2f})")

            try:
                est_profit_i = calculate_profit_from_trades(
                    trades_df, intraday_shares, ticker_is_y=True)
                st.write(
                    f"Estimated potential profit (simple per-share estimate on y ticker): €{est_profit_i:.2f}")
            except Exception:
                pass
        else:
            st.info("No closed intraday trades detected with current rules.")


if __name__ == '__main__':
    main()
