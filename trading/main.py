import streamlit as st
import yaml
import os
from dotenv import load_dotenv
import requests
import alpaca_trade_api as tradeapi
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading

# ---------- Load environment variables ----------
load_dotenv()
ALPACA_ENDPOINT = os.getenv("ALPACA_ENDPOINT")
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---------- Helper Functions ----------


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config():
    return load_yaml(os.path.join(os.path.dirname(__file__), "config.yaml"))


def load_strategy(strategy_name):
    path = os.path.join(os.path.dirname(__file__),
                        "strategies", f"{strategy_name}.yaml")
    return load_yaml(path)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except:
        pass


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

# ---------- Alpaca Client ----------


@st.cache_resource
def alpaca_client():
    return tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_ENDPOINT, api_version="v2")

# ---------- Algo Trading Function ----------


def run_algo(api, strategy_config):
    for symbol, rules in strategy_config.get("rules", {}).items():
        try:
            last_trade = api.get_latest_trade(symbol)
            latest_price = float(last_trade.price)
        except Exception as e:
            st.warning(f"Cannot fetch price for {symbol}: {e}")
            continue

        # Buy logic
        if latest_price <= rules["buy_below"]:
            try:
                order = api.submit_order(
                    symbol=symbol, qty=rules["qty"], side="buy", type="market", time_in_force="gtc")
                st.success(
                    f"🤖 BUY executed: {order.qty} {symbol} at {latest_price}")
                send_telegram(
                    f"🤖 BUY executed: {order.qty} {symbol} at {latest_price}")
            except Exception as e:
                st.error(f"Error BUY {symbol}: {e}")

        # Sell logic
        elif latest_price >= rules["sell_above"]:
            try:
                order = api.submit_order(
                    symbol=symbol, qty=rules["qty"], side="sell", type="market", time_in_force="gtc")
                st.success(
                    f"🤖 SELL executed: {order.qty} {symbol} at {latest_price}")
                send_telegram(
                    f"🤖 SELL executed: {order.qty} {symbol} at {latest_price}")
            except Exception as e:
                st.error(f"Error SELL {symbol}: {e}")

# ---------- Telegram Command Parser ----------


def parse_telegram_command(text):
    parts = text.strip().split()
    if not parts:
        return None, None
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []
    return cmd, args


def execute_command(api, cmd, args):
    response = ""
    try:
        if cmd == "/menu":
            response = (
                "/menu - Show command menu\n"
                "/buy SYMBOL QTY - Buy stock/crypto\n"
                "/sell SYMBOL QTY - Sell stock/crypto\n"
                "/cancel ORDER_ID - Cancel order\n"
                "/check_positions - Show positions\n"
                "/check_orders - Show recent orders\n"
                "/start_algo - Start Algo Trading\n"
                "/stop_algo - Stop Algo Trading"
            )
        elif cmd == "/buy" and len(args) >= 2:
            symbol = args[0].upper()
            qty = int(args[1])
            order = api.submit_order(
                symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc")
            response = f"🤖 BUY executed: {qty} {symbol}"
        elif cmd == "/sell" and len(args) >= 2:
            symbol = args[0].upper()
            qty = int(args[1])
            order = api.submit_order(
                symbol=symbol, qty=qty, side="sell", type="market", time_in_force="gtc")
            response = f"🤖 SELL executed: {qty} {symbol}"
        elif cmd == "/cancel" and len(args) >= 1:
            order_id = args[0]
            api.cancel_order(order_id)
            response = f"❌ Canceled order {order_id}"
        elif cmd == "/check_positions":
            positions = api.list_positions()
            if positions:
                response = "\n".join(
                    [f"{p.symbol} | Qty: {p.qty} | Side: {p.side}" for p in positions])
            else:
                response = "No open positions."
        elif cmd == "/check_orders":
            orders = api.list_orders(status="all", limit=20)
            if orders:
                response = "\n".join(
                    [f"{o.symbol} | Side: {o.side} | Qty: {o.qty} | Status: {o.status}" for o in orders])
            else:
                response = "No recent orders."
        elif cmd == "/start_algo":
            st.session_state.algo_running = True
            response = "🤖 Algo Trading Started"
        elif cmd == "/stop_algo":
            st.session_state.algo_running = False
            response = "⏹ Algo Trading Stopped"
        else:
            response = "⚠️ Unknown command. Send /menu to see available commands."
    except Exception as e:
        response = f"❌ Error executing {cmd}: {e}"
    return response


def check_telegram_commands(api, last_update_id=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"offset": last_update_id+1} if last_update_id else {}
    try:
        resp = requests.get(url, params=params, timeout=5).json()
    except:
        return last_update_id
    updates = resp.get("result", [])
    for update in updates:
        message = update.get("message", {})
        text = message.get("text")
        update_id = update.get("update_id")
        if text:
            cmd, args = parse_telegram_command(text)
            response = execute_command(api, cmd, args)
            send_telegram(response)
        last_update_id = update_id
    return last_update_id

# ---------- Telegram Polling in Background ----------


def telegram_polling_loop(api, interval=5):
    if "telegram_last_update" not in st.session_state:
        st.session_state.telegram_last_update = None
    while True:
        st.session_state.telegram_last_update = check_telegram_commands(
            api, st.session_state.telegram_last_update)
        time.sleep(interval)

# ---------- Streamlit App ----------


def app():
    st.set_page_config(page_title="Trading Dashboard", layout="wide")
    st.title("📊 Algo Trading Dashboard")
    st.markdown("Real-time trading dashboard with Alpaca & Strategy integration")

    api = alpaca_client()
    config = load_config()

    # Strategy Selector
    strategy_choice = st.sidebar.selectbox(
        "Select Strategy", config["strategies"]["active"])
    strategy_config = load_strategy(strategy_choice)

    with st.expander("View Strategy Config"):
        st.json(strategy_config)

    if "algo_running" not in st.session_state:
        st.session_state.algo_running = False

    # Start Telegram background polling once
    if "telegram_thread_started" not in st.session_state:
        thread = threading.Thread(
            target=telegram_polling_loop, args=(api,), daemon=True)
        thread.start()
        st.session_state.telegram_thread_started = True

    # ---------- Account Overview ----------
    st.markdown("---")
    st.header("💼 Account Overview")
    try:
        account = api.get_account()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Portfolio Value", format_currency(
                account.portfolio_value))
        with col2:
            st.metric("Cash", format_currency(account.cash))
        with col3:
            st.metric("Buying Power", format_currency(account.buying_power))
        with col4:
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            pl_pct = ((equity - last_equity) / last_equity *
                      100) if last_equity > 0 else 0
            st.metric("Today's P/L %", format_percentage(pl_pct),
                      delta=format_percentage(pl_pct))
        with col5:
            status_color = "🟢" if account.status == "ACTIVE" else "🔴"
            st.metric("Status", f"{status_color} {account.status}")
    except Exception as e:
        st.error(f"❌ Error fetching account info: {e}")

    # ---------- Positions ----------
    st.markdown("---")
    st.header("💰 Current Positions")
    try:
        positions = api.list_positions()
        if positions:
            df_positions = pd.DataFrame([{
                "Symbol": p.symbol,
                "Qty": float(p.qty),
                "Side": "Long" if float(p.qty) > 0 else "Short",
                "Avg Entry": format_currency(p.avg_entry_price),
                "Current Price": format_currency(p.current_price),
                "Market Value": format_currency(p.market_value),
                "Unrealized P/L": format_currency(float(p.unrealized_pl)),
                "P/L %": format_percentage(float(p.unrealized_plpc) * 100)
            } for p in positions])
            st.dataframe(df_positions, use_container_width=True,
                         hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[p.symbol for p in positions],
                y=[float(p.unrealized_pl) for p in positions],
                marker_color=[get_color(p.unrealized_pl) for p in positions],
                text=[format_currency(p.unrealized_pl) for p in positions],
                textposition="outside"
            ))
            fig.update_layout(title="Unrealized P/L by Position", xaxis_title="Symbol",
                              yaxis_title="P/L ($)", showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📭 No open positions.")
    except Exception as e:
        st.error(f"❌ Error fetching positions: {e}")

    # ---------- Orders ----------
    st.markdown("---")
    st.header("📝 Orders")
    try:
        orders = api.list_orders(status="all", limit=20)
        if orders:
            df_orders = pd.DataFrame([{
                "Symbol": o.symbol,
                "Side": o.side.upper(),
                "Qty": o.qty,
                "Filled": o.filled_qty,
                "Avg Fill Price": o.filled_avg_price,
                "Status": o.status,
                "Submitted": pd.to_datetime(o.submitted_at).strftime("%Y-%m-%d %H:%M:%S"),
                "Order ID": o.id
            } for o in orders])
            st.dataframe(df_orders, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No recent orders.")
    except Exception as e:
        st.error(f"❌ Error fetching orders: {e}")

    # ---------- Manual Trade ----------
    st.markdown("---")
    st.header("🛒 Manual Trade")
    symbol = st.text_input("Symbol", "AAPL")
    qty = st.number_input("Quantity", min_value=1, value=1)
    side = st.selectbox("Side", ["buy", "sell"])
    if st.button("🚀 Submit Order"):
        try:
            order = api.submit_order(
                symbol=symbol, qty=qty, side=side, type="market", time_in_force="gtc")
            st.success(f"Order sent: {side.upper()} {qty} {symbol}")
            send_telegram(f"📢 ORDER PLACED: {side.upper()} {qty} {symbol}")
        except Exception as e:
            st.error(str(e))

    # ---------- Cancel Orders ----------
    st.markdown("---")
    st.header("❌ Cancel Orders")
    try:
        cancelable_orders = [o for o in orders if o.status in [
            "accepted", "pending", "new"]]
        if cancelable_orders:
            cancel_dict = {
                o.id: f"{o.symbol} ({o.side} {o.qty})" for o in cancelable_orders}
            cancel_selection = st.multiselect("Select orders to cancel", options=list(
                cancel_dict.keys()), format_func=lambda x: cancel_dict[x])
            if st.button("Cancel Selected Orders"):
                for order_id in cancel_selection:
                    try:
                        api.cancel_order(order_id)
                        st.success(f"Canceled order {cancel_dict[order_id]}")
                        send_telegram(
                            f"❌ CANCELED ORDER: {cancel_dict[order_id]}")
                    except Exception as e:
                        st.error(f"Error canceling {order_id}: {str(e)}")
        else:
            st.info("No cancelable orders available.")
    except:
        pass

    # ---------- Algo Trading ----------
    st.markdown("---")
    st.header("🤖 Algo Trading Control")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Algo Trading"):
            st.session_state.algo_running = True
            st.success("Algo Trading Started")
            send_telegram("🤖 Algo Trading Started")
    with col2:
        if st.button("⏹ Stop Algo Trading"):
            st.session_state.algo_running = False
            st.warning("Algo Trading Stopped")
            send_telegram("⏹ Algo Trading Stopped")

    if st.session_state.algo_running:
        st.info("🤖 Running Algo Trading...")
        run_algo(api, strategy_config)

    st.markdown("---")
    st.success("✅ Paper trading environment active using Alpaca")


if __name__ == "__main__":
    app()
