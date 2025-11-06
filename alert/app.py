import streamlit as st
import subprocess
import sys
import os
import time
import yaml
from pathlib import Path
import signal

# Try to import psutil for better process management
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Paths - app.py is inside the alert folder
HERE = Path(__file__).resolve().parent
# Directory used as the working directory for the spawned alert process
ALERT_DIR = HERE

SCRIPT = HERE / "alert.py"
PID_FILE = HERE / "alert.pid"
LOG_FILE = HERE / "alert.log"
CONFIG_FILE = HERE / "config.yaml"

st.set_page_config(page_title="Trading Alert System",
                   layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .status-running {
        color: #28a745;
        font-size: 20px;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-size: 20px;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Trading Alert System")
st.markdown("Monitor spread trading signals and receive Telegram alerts")

# Helper functions


def is_running():
    """Check if alert process is running"""
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return False, None

    if HAS_PSUTIL:
        try:
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                # Check if it's actually our python process
                try:
                    if 'python' in proc.name().lower():
                        return True, proc
                except:
                    return True, proc
            return False, None
        except Exception:
            return False, None

    # Fallback without psutil
    try:
        if os.name == 'nt':
            r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                               capture_output=True, text=True, timeout=5)
            if str(pid) in r.stdout:
                return True, None
            return False, None
        else:
            os.kill(pid, 0)
            return True, None
    except Exception:
        return False, None


def verify_alert_script():
    """Verify that alert.py exists and is readable"""
    if not SCRIPT.exists():
        return False, f"❌ alert.py not found at: {SCRIPT}"

    try:
        with open(SCRIPT, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) < 100:
                return False, "❌ alert.py appears to be empty or corrupted"
        return True, "✅ alert.py found"
    except Exception as e:
        return False, f"❌ Cannot read alert.py: {e}"


def start_alert():
    """Start the alert service"""
    # First verify alert.py exists (but don't block on read errors)
    if not SCRIPT.exists():
        return False, f"❌ alert.py not found at: {SCRIPT}"

    running, _ = is_running()
    if running:
        return False, "❌ Alert system is already running"

    # Ensure config exists
    if not CONFIG_FILE.exists():
        return False, "❌ Please save configuration first"

    # Validate config
    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = yaml.safe_load(f)
            if not cfg:
                return False, "❌ Configuration file is empty"

            # Check critical fields
            if 'telegram' not in cfg or not cfg['telegram'].get('token'):
                return False, "❌ Telegram bot token not configured"
            if not cfg['telegram'].get('chat_id'):
                return False, "❌ Telegram chat ID not configured"

    except Exception as e:
        return False, f"❌ Invalid configuration file: {e}"

    # Ensure log file directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Clear or create log file
    try:
        log_f = open(LOG_FILE, "w", buffering=1, encoding='utf-8')
        log_f.write(
            f"=== Alert System Starting at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_f.flush()
    except Exception as e:
        return False, f"❌ Cannot create log file: {e}"

    # Start subprocess
    args = [sys.executable, str(SCRIPT)]
    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        log_f.write(f"Command: {' '.join(args)}\n")
        log_f.write(f"Working directory: {ALERT_DIR}\n")
        log_f.write(f"Python executable: {sys.executable}\n")
        log_f.flush()

        proc = subprocess.Popen(
            args,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(ALERT_DIR),
            creationflags=creationflags,
        )

        log_f.write(f"Process started with PID: {proc.pid}\n")
        log_f.flush()

        # Write pid
        PID_FILE.write_text(str(proc.pid))

        # Give it time to initialize
        time.sleep(3)

        # Check if process is still running
        ret = proc.poll()

        if ret is not None:
            # Process already exited - read logs for error
            log_f.close()
            tail = tail_log(LOG_FILE, lines=50)

            # Try to extract the actual error
            error_lines = [line for line in tail.split('\n') if 'error' in line.lower(
            ) or 'exception' in line.lower() or 'traceback' in line.lower()]

            if error_lines:
                error_summary = '\n'.join(error_lines[:5])
                return False, f"❌ Process exited immediately (code: {ret})\n\n**Error:**\n```\n{error_summary}\n```\n\n**Full logs:**\n```\n{tail}\n```"
            else:
                return False, f"❌ Process exited immediately (code: {ret})\n\n**Logs:**\n```\n{tail}\n```"

        # Verify process is actually running
        running_check, _ = is_running()
        if not running_check:
            log_f.close()
            tail = tail_log(LOG_FILE, lines=50)
            return False, f"❌ Process started but is not responding\n\n**Logs:**\n```\n{tail}\n```"

        log_f.close()
        return True, f"✅ Alert system started successfully (PID: {proc.pid})"

    except Exception as e:
        try:
            log_f.close()
        except:
            pass
        return False, f"❌ Failed to start process: {e}"


def stop_alert():
    """Stop the alert service"""
    running, proc = is_running()
    if not running:
        if PID_FILE.exists():
            try:
                PID_FILE.unlink()
            except Exception:
                pass
        return False, "⚠️ Alert system is not running"

    try:
        if HAS_PSUTIL and proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
            pid = proc.pid
        else:
            pid = int(PID_FILE.read_text().strip())
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(pid), '/F'], check=False)
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Check if still running
                try:
                    os.kill(pid, 0)
                    # Still running, force kill
                    os.kill(pid, signal.SIGKILL)
                except:
                    pass

        if PID_FILE.exists():
            PID_FILE.unlink()
        return True, f"✅ Alert system stopped (PID: {pid})"
    except Exception as e:
        return False, f"❌ Error stopping: {e}"


def tail_log(log_path, lines=100):
    """Get last N lines from log file"""
    if not log_path.exists():
        return "No logs yet..."
    try:
        with open(log_path, 'rb') as f:
            try:
                f.seek(-1024 * 128, os.SEEK_END)
            except OSError:
                f.seek(0)
            data = f.read().decode(errors='replace')
        return '\n'.join(data.splitlines()[-lines:])
    except Exception as e:
        return f"Error reading logs: {e}"


def load_config():
    """Load config from YAML file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return {}
    return get_default_config()


def get_default_config():
    """Get default configuration"""
    return {
        'trading': {
            'stock1': 'AGN.AS',
            'stock2': 'ASML.AS',
            'ratio': 0.0072,
            'name1': 'AGN',
            'name2': 'ASML',
            'z_score_window': 20
        },
        'telegram': {
            'token': '',
            'chat_id': ''
        },
        'strategy': {
            'name': 'z_score',
            'params': {
                'z_threshold': 2.0,
                'z_exit': 0.5,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'min_strategies': 2
            }
        },
        'alerts': {
            'cooldown_minutes': {
                '15-Minutes': 30,
                'Hourly': 60,
                'Daily': 240
            },
            'min_signal_strength': 0.5,
            'send_test_message': True
        },
        'schedule': {
            'check_15min': True,
            'check_hourly': True,
            'check_daily': True,
            'daily_check_hour': 17
        }
    }


def save_config(config):
    """Save config to YAML file"""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False,
                           sort_keys=False)
        return True, "✅ Configuration saved successfully"
    except Exception as e:
        return False, f"❌ Error saving config: {e}"


# Main UI Layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("🎮 Control Panel")

    # Pre-flight checks
    with st.expander("🔍 System Status", expanded=False):
        script_ok, script_msg = verify_alert_script()
        if script_ok:
            st.success(script_msg)
        else:
            st.error(script_msg)

        if CONFIG_FILE.exists():
            st.success(f"✅ Config found: {CONFIG_FILE}")
        else:
            st.warning("⚠️ No configuration saved yet")

        st.info(f"📁 Working directory: {ALERT_DIR}")
        st.info(f"🐍 Python: {sys.executable}")

    # Status display
    running, proc = is_running()

    if running:
        pid_display = proc.pid if proc is not None else '(unknown)'
        st.markdown(f'<p class="status-running">🟢 RUNNING (PID: {pid_display})</p>',
                    unsafe_allow_html=True)

        if st.button("🛑 Stop Alert System", type="primary", use_container_width=True):
            with st.spinner("Stopping..."):
                success, message = stop_alert()
            if success:
                st.success(message)
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)
    else:
        st.markdown('<p class="status-stopped">🔴 STOPPED</p>',
                    unsafe_allow_html=True)

        if st.button("▶️ Start Alert System", type="primary", use_container_width=True):
            with st.spinner("Starting alert system..."):
                success, message = start_alert()

            if success:
                st.success(message)
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)

    st.markdown("---")

    # Quick actions
    st.subheader("Quick Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            if LOG_FILE.exists():
                LOG_FILE.write_text("")
                st.success("Logs cleared")
                time.sleep(0.5)
                st.rerun()

    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=False)

    if not HAS_PSUTIL:
        st.warning(
            "⚠️ Optional: Install `psutil` for better process management\n```pip install psutil```")

with col2:
    st.header("📝 Recent Logs")
    log_content = tail_log(LOG_FILE, lines=30)
    st.code(log_content, language="", height=300)

    if st.button("📋 View Full Logs", use_container_width=True):
        with st.expander("Full Log Output", expanded=True):
            full_logs = tail_log(LOG_FILE, lines=500)
            st.code(full_logs, language="", height=600)

# Configuration Section
st.markdown("---")
st.header("⚙️ Configuration")

config = load_config()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Trading", "📱 Telegram", "🎯 Strategy", "⏰ Schedule"])

with tab1:
    st.subheader("Trading Pair Configuration")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        stock1 = st.text_input("Stock 1 (Ticker)",
                               value=config['trading'].get('stock1', 'AGN.AS'))
        name1 = st.text_input("Stock 1 (Display Name)",
                              value=config['trading'].get('name1', 'AGN'))

    with col_t2:
        stock2 = st.text_input("Stock 2 (Ticker)",
                               value=config['trading'].get('stock2', 'ASML.AS'))
        name2 = st.text_input("Stock 2 (Display Name)",
                              value=config['trading'].get('name2', 'ASML'))

    ratio = st.number_input("Hedge Ratio",
                            value=float(
                                config['trading'].get('ratio', 0.0072)),
                            format="%.6f", step=0.0001)

    z_score_window = st.slider("Z-Score Window (periods)",
                               min_value=10, max_value=50,
                               value=config['trading'].get('z_score_window', 20))

with tab2:
    st.subheader("Telegram Bot Configuration")
    st.info("💡 Get your bot token from @BotFather on Telegram. Get chat_id by messaging your bot and visiting https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")

    token = st.text_input("Bot Token",
                          value=config['telegram'].get('token', ''),
                          type="password",
                          help="Required: Get from @BotFather on Telegram")
    chat_id = st.text_input("Chat ID",
                            value=config['telegram'].get('chat_id', ''),
                            help="Required: Your Telegram chat ID")

    send_test = st.checkbox("Send test message on startup",
                            value=config['alerts'].get('send_test_message', True))

    # Validation indicators
    if token:
        st.success("✅ Bot token configured")
    else:
        st.warning("⚠️ Bot token required")

    if chat_id:
        st.success("✅ Chat ID configured")
    else:
        st.warning("⚠️ Chat ID required")

with tab3:
    st.subheader("Trading Strategy")

    strategy_name = st.selectbox(
        "Select Strategy",
        options=['z_score', 'bollinger', 'macd', 'combined'],
        index=['z_score', 'bollinger', 'macd', 'combined'].index(
            config['strategy'].get('name', 'z_score')),
        format_func=lambda x: {
            'z_score': '📈 Z-Score Mean Reversion',
            'bollinger': '📊 Bollinger Bands + RSI',
            'macd': '📉 MACD Crossover',
            'combined': '🎯 Combined Multi-Strategy'
        }[x]
    )

    st.markdown("**Strategy Parameters**")
    params = config['strategy'].get('params', {})

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        z_threshold = st.number_input("Z-Score Threshold",
                                      value=float(params.get(
                                          'z_threshold', 2.0)),
                                      min_value=0.5, max_value=5.0, step=0.1)
        rsi_oversold = st.slider("RSI Oversold Level",
                                 min_value=10, max_value=40,
                                 value=params.get('rsi_oversold', 30))

    with col_s2:
        z_exit = st.number_input("Z-Score Exit Level",
                                 value=float(params.get('z_exit', 0.5)),
                                 min_value=0.0, max_value=2.0, step=0.1)
        rsi_overbought = st.slider("RSI Overbought Level",
                                   min_value=60, max_value=90,
                                   value=params.get('rsi_overbought', 70))

    min_signal_strength = st.slider("Minimum Signal Strength",
                                    min_value=0.0, max_value=1.0,
                                    value=float(config['alerts'].get(
                                        'min_signal_strength', 0.5)),
                                    step=0.05, format="%.2f")

    if strategy_name == 'combined':
        min_strategies = st.slider("Minimum Agreeing Strategies",
                                   min_value=1, max_value=3,
                                   value=params.get('min_strategies', 2))
    else:
        min_strategies = 2

with tab4:
    st.subheader("Alert Schedule")

    col_sch1, col_sch2 = st.columns(2)

    with col_sch1:
        check_15min = st.checkbox("Check every 15 minutes",
                                  value=config['schedule'].get('check_15min', True))
        check_hourly = st.checkbox("Check every hour",
                                   value=config['schedule'].get('check_hourly', True))
        check_daily = st.checkbox("Check daily",
                                  value=config['schedule'].get('check_daily', True))

    with col_sch2:
        daily_check_hour = st.slider("Daily check time (hour)",
                                     min_value=0, max_value=23,
                                     value=config['schedule'].get('daily_check_hour', 17))

    st.markdown("**Cooldown Periods (minutes)**")
    cooldown = config['alerts'].get('cooldown_minutes', {})

    col_cd1, col_cd2, col_cd3 = st.columns(3)
    with col_cd1:
        cooldown_15min = st.number_input("15-min signals",
                                         value=cooldown.get('15-Minutes', 30),
                                         min_value=1, max_value=120)
    with col_cd2:
        cooldown_hourly = st.number_input("Hourly signals",
                                          value=cooldown.get('Hourly', 60),
                                          min_value=1, max_value=240)
    with col_cd3:
        cooldown_daily = st.number_input("Daily signals",
                                         value=cooldown.get('Daily', 240),
                                         min_value=1, max_value=1440)

# Save Configuration Button
st.markdown("---")
if st.button("💾 Save Configuration", type="primary", use_container_width=True):
    new_config = {
        'trading': {
            'stock1': stock1,
            'stock2': stock2,
            'ratio': ratio,
            'name1': name1,
            'name2': name2,
            'z_score_window': z_score_window
        },
        'telegram': {
            'token': token,
            'chat_id': chat_id
        },
        'strategy': {
            'name': strategy_name,
            'params': {
                'z_threshold': z_threshold,
                'z_exit': z_exit,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'min_strategies': min_strategies
            }
        },
        'alerts': {
            'cooldown_minutes': {
                '15-Minutes': cooldown_15min,
                'Hourly': cooldown_hourly,
                'Daily': cooldown_daily
            },
            'min_signal_strength': min_signal_strength,
            'send_test_message': send_test
        },
        'schedule': {
            'check_15min': check_15min,
            'check_hourly': check_hourly,
            'check_daily': check_daily,
            'daily_check_hour': daily_check_hour
        }
    }

    success, message = save_config(new_config)
    if success:
        st.success(message)
        running_now, _ = is_running()
        if running_now:
            st.warning(
                "⚠️ Alert system is running. Restart it for changes to take effect.")
        else:
            st.info("ℹ️ Configuration saved. You can now start the alert system.")
    else:
        st.error(message)

# Auto-refresh logic
if auto_refresh:
    time.sleep(10)
    st.rerun()
