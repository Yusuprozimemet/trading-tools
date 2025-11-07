import streamlit as st
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# alpaca-py (new SDK)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, GetOrdersRequest
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
# ----------------------------------------------------------------------

load_dotenv()

ALPACA_ENDPOINT = os.getenv("ALPACA_ENDPOINT")
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")


# ---------- Alpaca Clients ----------
@st.cache_resource
def trading_client():
    paper = (ALPACA_ENDPOINT == "https://paper-api.alpaca.markets")
    return TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=paper)


@st.cache_resource
def data_client():
    return StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)


# ---------- Helpers ----------
def format_currency(value):
    try:
        return f"${float(value):,.2f}"
    except:
        return value


def format_percentage(value):
    try:
        return f"{float(value):.2f}%"
    except:
        return value


def get_color(value):
    try:
        return "green" if float(value) >= 0 else "red"
    except:
        return "gray"


# ---------- Streamlit App ----------
def app():
    st.set_page_config(page_title="Alpaca Trading Dashboard", layout="wide")
    st.title("Alpaca Trading Dashboard")
    st.markdown("Real-time trading data and analytics powered by Alpaca Markets")

    trading_cli = trading_client()
    data_cli = data_client()

    # ---------- Account Overview ----------
    st.header("Account Overview")
    try:
        account = trading_cli.get_account()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Portfolio Value", format_currency(account.portfolio_value))
        with col2:
            st.metric("Cash", format_currency(account.cash))
        with col3:
            st.metric("Buying Power", format_currency(account.buying_power))
        with col4:
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            pl_pct = ((equity - last_equity) / last_equity * 100) if last_equity > 0 else 0
            st.metric("Today's P/L %", format_percentage(pl_pct),
                      delta=format_percentage(pl_pct))
        with col5:
            status_color = "ACTIVE" if account.status == "ACTIVE" else "INACTIVE"
            st.metric("Status", f"{status_color} {account.status}")

        with st.expander("Detailed Account Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Account Number:** {account.account_number}")
                st.write(f"**Pattern Day Trader:** {account.pattern_day_trader}")
                st.write(f"**Trading Blocked:** {account.trading_blocked}")
                st.write(f"**Transfers Blocked:** {account.transfers_blocked}")
                st.write(f"**Daytrade Count:** {account.daytrade_count}")
            with col2:
                st.write(f"**Long Market Value:** {format_currency(account.long_market_value)}")
                st.write(f"**Short Market Value:** {format_currency(account.short_market_value)}")
                st.write(f"**Initial Margin:** {format_currency(account.initial_margin)}")
                st.write(f"**Maintenance Margin:** {format_currency(account.maintenance_margin)}")
                st.write(f"**Last Equity:** {format_currency(account.last_equity)}")
    except Exception as e:
        st.error(f"Error fetching account info: {e}")

    st.markdown("---")

    # ---------- Current Positions ----------
    st.header("Current Positions")
    try:
        positions = trading_cli.get_all_positions()
        if positions:
            positions_data = []
            for p in positions:
                unrealized_pl = float(p.unrealized_pl)
                unrealized_plpc = float(p.unrealized_plpc) * 100
                positions_data.append({
                    "Symbol": p.symbol,
                    "Qty": float(p.qty),
                    "Side": "Long" if float(p.qty) > 0 else "Short",
                    "Avg Entry": format_currency(p.avg_entry_price),
                    "Current Price": format_currency(p.current_price),
                    "Market Value": format_currency(p.market_value),
                    "Unrealized P/L": format_currency(unrealized_pl),
                    "P/L %": format_percentage(unrealized_plpc),
                    "Cost Basis": format_currency(p.cost_basis)
                })
            df_positions = pd.DataFrame(positions_data)

            total_pl = sum(float(p.unrealized_pl) for p in positions)
            total_value = sum(float(p.market_value) for p in positions)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Positions", len(positions))
            with col2:
                st.metric("Total Market Value", format_currency(total_value))
            with col3:
                st.metric("Total Unrealized P/L", format_currency(total_pl),
                          delta=format_currency(total_pl))

            # NOTE: width='stretch' replaces use_container_width=True
            st.dataframe(df_positions, width='stretch', hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[p.symbol for p in positions],
                y=[float(p.unrealized_pl) for p in positions],
                marker_color=[get_color(p.unrealized_pl) for p in positions],
                text=[format_currency(p.unrealized_pl) for p in positions],
                textposition='outside'
            ))
            fig.update_layout(title="Unrealized P/L by Position",
                              xaxis_title="Symbol", yaxis_title="P/L ($)",
                              showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions.")
    except Exception as e:
        st.error(f"Error fetching positions: {e}")

    st.markdown("---")

    # ---------- Recent Orders ----------
    st.header("Recent Orders")
    try:
        orders = trading_cli.get_orders(GetOrdersRequest(status="all", limit=20))

        if orders:
            orders_data = []
            for o in orders:
                orders_data.append({
                    "Symbol": o.symbol,
                    "Side": o.side.upper(),
                    "Type": o.type,
                    "Qty": o.qty,
                    "Filled Qty": o.filled_qty,
                    "Status": o.status,
                    "Limit Price": format_currency(o.limit_price) if o.limit_price else "N/A",
                    "Filled Avg Price": format_currency(o.filled_avg_price) if o.filled_avg_price else "N/A",
                    "Submitted": pd.to_datetime(o.submitted_at).strftime("%Y-%m-%d %H:%M:%S"),
                    # <-- Convert UUID to string
                    "Order ID": str(o.id)
                })
            df_orders = pd.DataFrame(orders_data)

            status_counts = df_orders['Status'].value_counts()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Orders", len(orders))
            with col2:
                st.metric("Filled", status_counts.get('filled', 0))
            with col3:
                pending = status_counts.get('pending_new', 0) + status_counts.get('new', 0)
                st.metric("Pending", pending)
            with col4:
                st.metric("Canceled", status_counts.get('canceled', 0))

            # width='stretch' instead of use_container_width
            st.dataframe(df_orders, width='stretch', hide_index=True)
        else:
            st.info("No recent orders.")
    except Exception as e:
        st.error(f"Error fetching orders: {e}")

    st.markdown("---")

    # ---------- Historical Bars & Chart ----------
    st.header("Historical Price Data")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.text_input("Enter Symbol", "AAPL", key="symbol_input")
    with col2:
        timeframe = st.selectbox(
            "Timeframe", ["1Min", "5Min", "15Min", "1Hour", "1Day"], index=4)
    with col3:
        days_back = st.number_input("Days Back", min_value=1, max_value=365, value=30)

    if st.button("Load Chart", type="primary"):
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days_back)

            tf_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1Day": TimeFrame(1, TimeFrameUnit.Day),
            }
            tf = tf_map[timeframe]

            with st.spinner(f"Loading {symbol} data..."):
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_dt,
                    end=end_dt,
                    feed="iex"
                )
                bars = data_cli.get_stock_bars(req).df

                if not bars.empty and symbol in bars.columns:
                    bars = bars[symbol]
                    bars.index = bars.index.tz_localize(None)

                    latest_price = bars['close'].iloc[-1]
                    prev_price = bars['close'].iloc[0]
                    price_change = latest_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Latest Price", format_currency(latest_price))
                    with col2:
                        st.metric("Change", format_currency(price_change),
                                  delta=format_percentage(price_change_pct))
                    with col3:
                        st.metric("High", format_currency(bars['high'].max()))
                    with col4:
                        st.metric("Low", format_currency(bars['low'].min()))
                    with col5:
                        st.metric("Avg Volume", f"{bars['volume'].mean():,.0f}")

                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(f'{symbol} Price', 'Volume'),
                        row_width=[0.2, 0.7]
                    )
                    fig.add_trace(go.Candlestick(
                        x=bars.index,
                        open=bars['open'],
                        high=bars['high'],
                        low=bars['low'],
                        close=bars['close'],
                        name='Price'
                    ), row=1, col=1)

                    colors = ['red' if row['close'] < row['open'] else 'green'
                              for _, row in bars.iterrows()]
                    fig.add_trace(go.Bar(
                        x=bars.index,
                        y=bars['volume'],
                        marker_color=colors,
                        name='Volume',
                        showlegend=False
                    ), row=2, col=1)

                    fig.update_layout(height=700,
                                      xaxis_rangeslider_visible=False,
                                      showlegend=False)
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("View Raw Data"):
                        st.dataframe(bars.style.format({
                            'open': '${:.2f}',
                            'high': '${:.2f}',
                            'low': '${:.2f}',
                            'close': '${:.2f}',
                            'volume': '{:,.0f}'
                        }), width='stretch')
                else:
                    st.warning(f"No data available for {symbol}")
        except Exception as e:
            st.error(f"Error fetching bars: {e}")
            st.info("Tip: Make sure you're using a valid symbol and that the timeframe is appropriate for your Alpaca subscription (IEX feed).")

    st.markdown("---")

    # ---------- Available Assets ----------
    with st.expander("Browse Available Assets"):
        try:
            col1, col2 = st.columns([1, 3])
            with col1:
                asset_status = st.selectbox("Status", ["active", "inactive"], index=0)
                asset_class = st.selectbox("Class", ["us_equity", "crypto"], index=0)

            assets = trading_cli.get_all_assets(
                GetAssetsRequest(status=asset_status, asset_class=asset_class)
            )

            if assets:
                df_assets = pd.DataFrame([{
                    "Symbol": a.symbol,
                    "Name": a.name,
                    "Exchange": a.exchange,
                    "Tradable": "Yes" if a.tradable else "No",
                    "Shortable": "Yes" if getattr(a, "shortable", False) else "No",
                    "Marginable": "Yes" if getattr(a, "marginable", False) else "No",
                    "Status": a.status
                } for a in assets])

                st.write(f"**Total Assets:** {len(df_assets)}")

                search = st.text_input("Search by symbol or name", "")
                if search:
                    df_assets = df_assets[
                        df_assets['Symbol'].str.contains(search.upper()) |
                        df_assets['Name'].str.contains(search, case=False)
                    ]

                st.dataframe(df_assets, width='stretch', hide_index=True, height=400)
            else:
                st.info("No assets found.")
        except Exception as e:
            st.error(f"Error fetching assets: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Built with Streamlit • Powered by Alpaca Markets API</p>
            <p style='font-size: 0.8em;'>Warning: This dashboard is for informational purposes only. Not financial advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    app()