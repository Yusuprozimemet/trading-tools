import streamlit as st
import subprocess
import sys
import os
import time
import yaml
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False
import signal
from pathlib import Path
import urllib.request
import urllib.parse
import json

# Paths
HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "alert.py"
PID_FILE = HERE / "alert.pid"
LOG_FILE = HERE / "alert.log"
CONFIG_FILE = Path.cwd() / "config.yaml"

st.set_page_config(page_title="Alert System Control", layout="wide")
st.title("Alert System — Controller")
st.sidebar.markdown("# Alert Controller")
st.sidebar.write(
    "Shows current status and logs of the alert service. Start it to run `alert/alert.py` as a background process.")

# Helper functions


def is_running():
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
                return True, proc
            return False, None
        except Exception:
            return False, None

    # Fallback when psutil is not available
    try:
        if os.name == 'nt':
            # Use tasklist to check for PID on Windows
            r = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True)
            if str(pid) in r.stdout:
                return True, None
            return False, None
        else:
            # On Unix-like systems, os.kill(pid, 0) checks existence
            os.kill(pid, 0)
            return True, None
    except Exception:
        return False, None


def start_alert():
    running, _ = is_running()
    if running:
        return False, "Already running"

    # Ensure log file exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(LOG_FILE, "a", buffering=1)

    # Ensure the alert script can find the user's config.yaml.
    # If the repo-root config.yaml exists and there's no copy in alert/, copy it so alert.py (which runs from alert/) sees it.
    root_cfg = CONFIG_FILE
    alert_cfg = HERE / "config.yaml"
    try:
        if root_cfg.exists() and not alert_cfg.exists():
            alert_cfg.write_text(root_cfg.read_text())
    except Exception:
        # non-fatal; continue starting process
        pass

    # Start subprocess
    args = [sys.executable, str(SCRIPT)]
    # On Windows, use CREATE_NEW_PROCESS_GROUP so we can terminate process group
    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        args,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(HERE),
        creationflags=creationflags,
    )

    # Write pid
    PID_FILE.write_text(str(proc.pid))
    # Give the process a moment to initialize and write logs
    time.sleep(1)
    # Check if it is still running
    alive = False
    try:
        if HAS_PSUTIL:
            alive = psutil.pid_exists(proc.pid)
        else:
            # Fallback: check process.poll()
            alive = (proc.poll() is None)
    except Exception:
        alive = False

    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        log_status = "(no logs yet)"
    else:
        log_status = ""

    if alive:
        return True, f"Started (pid={proc.pid}) {log_status}"
    else:
        return False, f"Process exited prematurely (pid={proc.pid}) {log_status}"


def stop_alert():
    running, proc = is_running()
    if not running:
        if PID_FILE.exists():
            try:
                PID_FILE.unlink()
            except Exception:
                pass
        return False, "Not running"

    try:
        # If we have psutil and a Process object, use it for graceful termination
        if HAS_PSUTIL and proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
            pid = proc.pid
        else:
            # Fallback: read pid and attempt platform-specific termination
            pid = int(PID_FILE.read_text().strip())
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(pid), '/F'], check=False)
            else:
                os.kill(pid, signal.SIGTERM)

        if PID_FILE.exists():
            PID_FILE.unlink()
        return True, f"Stopped (pid={pid})"
    except Exception as e:
        return False, f"Error stopping: {e}"


def tail(log_path, lines=200):
    if not log_path.exists():
        return ""
    with open(log_path, 'rb') as f:
        try:
            f.seek(-1024 * 64, os.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read().decode(errors='replace')
    return '\n'.join(data.splitlines()[-lines:])


# UI layout
col1, col2 = st.columns([1, 2])

with col1:
    running, proc = is_running()
    if not HAS_PSUTIL:
        st.warning("Optional package 'psutil' is not installed — status detection and graceful stopping use fallbacks.\nInstall with: pip install psutil")
    if running:
        pid_display = proc.pid if proc is not None else '(unknown)'
        st.success(f"Alert system is RUNNING (pid={pid_display})")
        if st.button("Stop Alert Service", key="stop_button"):
            ok, msg = stop_alert()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

            # Attempt to send Telegram notification about shutdown using config.yaml
            try:
                cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
                token = cfg.get('telegram', {}).get('token', '')
                chat_id = cfg.get('telegram', {}).get('chat_id', '')
                if token and chat_id:
                    text = f"Alert system stopped by user at {time.strftime('%Y-%m-%d %H:%M:%S')}."

                    def send_telegram(token, chat_id, text):
                        url = f"https://api.telegram.org/bot{token}/sendMessage"
                        data = {'chat_id': str(chat_id), 'text': text}
                        data_bytes = urllib.parse.urlencode(data).encode()
                        req = urllib.request.Request(url, data=data_bytes)
                        try:
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                return True, resp.read().decode()
                        except Exception as e:
                            return False, str(e)

                    sent, resp = send_telegram(token, chat_id, text)
                    if sent:
                        st.info("Telegram stop notification sent")
                    else:
                        st.warning(
                            f"Failed to send Telegram notification: {resp}")
                else:
                    st.info(
                        "Telegram token/chat_id not configured; skip sending stop notification")
            except Exception as e:
                st.warning(
                    f"Could not read config/send telegram notification: {e}")

    else:
        st.error("Alert system is STOPPED")
        if st.button("Start Alert Service", key="start_button"):
            ok, msg = start_alert()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    st.subheader("Config")
    if CONFIG_FILE.exists():
        try:
            cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
        except Exception as e:
            cfg = {}
            st.warning(f"Failed to read config.yaml: {e}")
    else:
        cfg = {}
        st.info(
            "No config.yaml found yet. Run the alert script once to generate a template, or create one here.")

    with st.expander("Edit config.yaml (basic)"):
        trading = cfg.get('trading', {})
        telegram = cfg.get('telegram', {})
        strategy = cfg.get('strategy', {})

        stock1 = st.text_input("Stock 1", trading.get('stock1', 'AGN.AS'))
        stock2 = st.text_input("Stock 2", trading.get('stock2', 'ASML.AS'))
        ratio = st.number_input(
            "Ratio", value=float(trading.get('ratio', 0.0072)))

        token = st.text_input("Telegram token", telegram.get('token', ''))
        chat_id = st.text_input(
            "Telegram chat_id", telegram.get('chat_id', ''))

        if st.button("Save config.yaml"):
            new_cfg = cfg.copy()
            new_cfg.setdefault('trading', {})
            new_cfg.setdefault('telegram', {})
            new_cfg['trading']['stock1'] = stock1
            new_cfg['trading']['stock2'] = stock2
            new_cfg['trading']['ratio'] = ratio
            new_cfg['telegram']['token'] = token
            new_cfg['telegram']['chat_id'] = chat_id
            try:
                CONFIG_FILE.write_text(
                    yaml.safe_dump(new_cfg, sort_keys=False))
                st.success("Saved config.yaml")
            except Exception as e:
                st.error(f"Failed to save config.yaml: {e}")

with col2:
    st.subheader("Log (last 200 lines)")
    st.code(tail(LOG_FILE, lines=200), language='')

    st.markdown("---")
    st.subheader("Actions")
    st.write("Use the buttons on the left to start/stop the alert service. Logs are appended to `alert/alert.log`.")
    if st.button("Refresh status/log"):
        # Use st.query_params (new API) to trigger a rerun by updating a timestamp param.
        try:
            params = dict(st.query_params)
            # st.query_params expects lists of strings for values
            params['_refresh'] = [str(int(time.time()))]
            st.query_params = params
            st.stop()
        except Exception:
            # Fallback: just stop the script so the user can reload the page
            st.stop()


# Auto-refresh every 10 seconds when running
if st.checkbox("Auto refresh every 10s"):
    # Update query params to force a rerun after a short sleep.
    try:
        time.sleep(10)
        params = dict(st.query_params)
        params['_autorefresh'] = [str(int(time.time()))]
        st.query_params = params
        st.stop()
    except Exception:
        # If setting query params isn't available, instruct user to refresh manually
        st.info(
            "Auto-refresh not supported in this Streamlit version; please refresh manually")
