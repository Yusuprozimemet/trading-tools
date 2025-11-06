
# Alert System Documentation

## Overview

The **Alert System** monitors trading signals for a pair of stocks and sends Telegram notifications when conditions are met. It supports multiple trading strategies including:

* Z-Score Mean Reversion
* Bollinger Bands + RSI
* MACD Crossover
* Combined Multi-Strategy

The system can run as a **background service** and is controllable via a **Streamlit UI** (`alert_ui.py`).

---

## Features

* Fetch historical and live stock data using **yfinance**.
* Calculate spread between two stocks.
* Compute technical indicators:

  * Z-Score
  * Bollinger Bands
  * RSI
  * MACD
  * ATR
* Generate trading signals based on configured strategies.
* Send alerts via **Telegram**.
* Configurable cooldowns and minimum signal strength.
* Streamlit interface to start/stop the alert service and edit config.
* Auto-refresh logs in UI.

---

## File Structure

```
alert/
├── alert.py         # Main alert system logic
├── alert_ui.py      # Streamlit interface
├── config.yaml      # Configuration
├── alert.log        # Logs
└── alert.pid        # PID file for running process
```

---

## Configuration (`config.yaml`)

```yaml
trading:
  stock1: "AGN.AS"
  stock2: "ASML.AS"
  ratio: 0.0072
  name1: "AGN"
  name2: "ASML"
  z_score_window: 20

telegram:
  token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"

strategy:
  name: "z_score" # Options: z_score, bollinger, macd, combined
  params:
    z_threshold: 2.0
    z_exit: 0.5

alerts:
  cooldown_minutes:
    "15-Minutes": 30
    "Hourly": 60
    "Daily": 240
  min_signal_strength: 0.5
  send_test_message: True

schedule:
  check_15min: True
  check_hourly: True
  check_daily: True
  daily_check_hour: 17
```

### Key Config Options

* `trading.stock1` / `stock2`: Tickers of the pair to monitor.
* `ratio`: Hedge ratio for spread calculation.
* `strategy.name`: Strategy to use (`z_score`, `bollinger`, `macd`, `combined`).
* `alerts.cooldown_minutes`: Minimum minutes between alerts per timeframe.
* `alerts.min_signal_strength`: Minimum confidence to trigger alert.
* `schedule`: Enables/disables checks on 15-minute, hourly, or daily intervals.

---

## Running the Alert System

### Via Command Line

```bash
python alert.py
```

* The script will read `config.yaml`.
* Sends a **test Telegram message** if `send_test_message: True`.
* Runs continuously, checking signals at scheduled intervals.

### Via Streamlit UI

```bash
streamlit run alert_ui.py
```

* Provides a web-based controller.
* Shows alert system status (running/stopped).
* Start/Stop service buttons.
* Edit basic configuration (stocks, Telegram token/chat ID).
* View last 200 lines of logs.
* Auto-refresh every 10 seconds.

---

## Components

### 1. InteractiveSpreadAnalyzer

* Fetches stock data from `yfinance`.
* Filters non-trading periods.
* Computes spread between two stocks.
* Calculates indicators (Z-Score, MACD, Bollinger Bands, RSI, ATR).

### 2. Trading Strategies

* **ZScoreStrategy:** Mean-reversion signals based on z-score thresholds.
* **BollingerBandsStrategy:** Signals based on Bollinger Bands + RSI.
* **MACDStrategy:** Signals on MACD crossover.
* **CombinedStrategy:** Aggregates multiple strategies; requires agreement.

### 3. SignalAlertSystem

* Loads configuration.
* Sets up Telegram bot.
* Periodically checks signals based on schedule.
* Sends alerts via Telegram.
* Handles cooldowns to avoid alert spamming.
* Provides detailed alert messages with metrics and reasoning.

---

## Telegram Alerts

Example alert message:

```
🚨 15-Minutes Trading Signal 🚨

Signal: 🟢 LONG
Strategy: Z-Score Mean Reversion
Confidence: 75%
Timeframe: 15-Minutes

📊 Current Metrics:
• Spread: 0.1234
• Z-Score: -2.34

💡 Reason: Z-Score -2.34 below -2.0

Pair: AGN - ASML
Time: 2025-11-06 16:30:00
```

---

## Logs

* Logs are saved in `alert/alert.log`.
* Last 200 lines displayed in Streamlit UI.
* Includes info messages, errors, and alert events.

---

## Dependencies

```text
streamlit
yfinance
pandas
numpy
plotly
matplotlib
PyYAML
python-telegram-bot
psutil (optional for graceful stop)
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Notes

* The system is **async-based** to fetch data and send alerts concurrently.
* Streamlit UI ensures non-blocking start/stop control.
* Default configuration is created if `config.yaml` is missing.
* Telegram integration is optional but highly recommended.

---


