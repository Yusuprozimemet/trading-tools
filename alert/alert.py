#!/usr/bin/env python3
"""
Trading Alert System - Background Service
This script runs independently from Streamlit and sends Telegram alerts
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import yaml
import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.yaml"
STATE_FILE = HERE / "alert_state.json"


class TradingAlertSystem:
    def __init__(self, config):
        self.config = config
        self.last_alert_times = {}
        logger.info("Trading Alert System initialized")
        logger.info(
            f"Stock pair: {config['trading']['name1']}/{config['trading']['name2']}")
        logger.info(f"Strategy: {config['strategy']['name']}")

    def send_telegram_message(self, message):
        """Send message via Telegram"""
        token = self.config['telegram']['token']
        chat_id = self.config['telegram']['chat_id']

        if not token or not chat_id:
            logger.error("Telegram credentials not configured")
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✓ Telegram message sent successfully")
                return True
            else:
                logger.error(
                    f"Telegram API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def fetch_stock_data(self, ticker, period="3mo", interval="1d"):
        """Fetch stock data from yfinance"""
        try:
            logger.info(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.error(f"No data returned for {ticker}")
                return None

            logger.info(f"✓ Fetched {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def calculate_spread(self, df1, df2, ratio):
        """Calculate spread between two stocks"""
        # Align the dataframes by date
        df = pd.DataFrame({
            'stock1': df1['Close'],
            'stock2': df2['Close']
        }).dropna()

        # Calculate spread
        df['spread'] = df['stock1'] - ratio * df['stock2']
        return df

    def calculate_z_score(self, spread, window=20):
        """Calculate z-score of spread"""
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score = (spread - rolling_mean) / rolling_std
        return z_score

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_z_score_signal(self, df, params):
        """Generate signal based on z-score strategy"""
        z_score = self.calculate_z_score(df['spread'],
                                         window=self.config['trading']['z_score_window'])

        current_z = z_score.iloc[-1]
        z_threshold = params['z_threshold']

        if current_z > z_threshold:
            return {
                'action': 'SHORT',
                'strength': min(abs(current_z) / (z_threshold * 2), 1.0),
                'reason': f'Z-score: {current_z:.2f} (>{z_threshold})',
                'details': f'Spread is {abs(current_z):.2f} std devs above mean'
            }
        elif current_z < -z_threshold:
            return {
                'action': 'LONG',
                'strength': min(abs(current_z) / (z_threshold * 2), 1.0),
                'reason': f'Z-score: {current_z:.2f} (<-{z_threshold})',
                'details': f'Spread is {abs(current_z):.2f} std devs below mean'
            }

        return None

    def generate_bollinger_signal(self, df, params):
        """Generate signal based on Bollinger Bands + RSI"""
        spread = df['spread']
        rolling_mean = spread.rolling(window=20).mean()
        rolling_std = spread.rolling(window=20).std()

        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)

        current_spread = spread.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]

        # Calculate RSI
        rsi = self.calculate_rsi(spread)
        current_rsi = rsi.iloc[-1]

        if current_spread > current_upper and current_rsi > params['rsi_overbought']:
            distance = (current_spread - current_upper) / rolling_std.iloc[-1]
            return {
                'action': 'SHORT',
                'strength': min(distance / 2, 1.0),
                'reason': f'Above upper band, RSI: {current_rsi:.1f}',
                'details': f'Spread at upper Bollinger band'
            }
        elif current_spread < current_lower and current_rsi < params['rsi_oversold']:
            distance = (current_lower - current_spread) / rolling_std.iloc[-1]
            return {
                'action': 'LONG',
                'strength': min(distance / 2, 1.0),
                'reason': f'Below lower band, RSI: {current_rsi:.1f}',
                'details': f'Spread at lower Bollinger band'
            }

        return None

    def generate_macd_signal(self, df, params):
        """Generate signal based on MACD"""
        spread = df['spread']

        # Calculate MACD
        exp1 = spread.ewm(span=12, adjust=False).mean()
        exp2 = spread.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()

        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd.iloc[-2]
        prev_signal = signal_line.iloc[-2]

        # Bullish crossover
        if prev_macd <= prev_signal and current_macd > current_signal:
            diff = abs(current_macd - current_signal)
            return {
                'action': 'LONG',
                'strength': min(diff * 10, 1.0),
                'reason': 'MACD bullish crossover',
                'details': 'MACD crossed above signal line'
            }
        # Bearish crossover
        elif prev_macd >= prev_signal and current_macd < current_signal:
            diff = abs(current_macd - current_signal)
            return {
                'action': 'SHORT',
                'strength': min(diff * 10, 1.0),
                'reason': 'MACD bearish crossover',
                'details': 'MACD crossed below signal line'
            }

        return None

    def analyze_pair(self, timeframe):
        """Analyze trading pair and generate signals"""
        logger.info(f"=== Analyzing {timeframe} timeframe ===")

        # Determine data parameters based on timeframe
        if timeframe == "15-Minutes":
            period = "5d"
            interval = "15m"
        elif timeframe == "Hourly":
            period = "1mo"
            interval = "1h"
        else:  # Daily
            period = "3mo"
            interval = "1d"

        # Fetch data
        stock1 = self.config['trading']['stock1']
        stock2 = self.config['trading']['stock2']

        df1 = self.fetch_stock_data(stock1, period=period, interval=interval)
        df2 = self.fetch_stock_data(stock2, period=period, interval=interval)

        if df1 is None or df2 is None:
            logger.error("Failed to fetch stock data")
            return None

        # Calculate spread
        df = self.calculate_spread(df1, df2, self.config['trading']['ratio'])

        if len(df) < 30:
            logger.warning(f"Insufficient data: {len(df)} rows")
            return None

        # Generate signals based on strategy
        strategy_name = self.config['strategy']['name']
        params = self.config['strategy']['params']

        signals = []

        if strategy_name == 'z_score':
            signal = self.generate_z_score_signal(df, params)
            if signal:
                signals.append(signal)

        elif strategy_name == 'bollinger':
            signal = self.generate_bollinger_signal(df, params)
            if signal:
                signals.append(signal)

        elif strategy_name == 'macd':
            signal = self.generate_macd_signal(df, params)
            if signal:
                signals.append(signal)

        elif strategy_name == 'combined':
            # Run all strategies
            z_signal = self.generate_z_score_signal(df, params)
            b_signal = self.generate_bollinger_signal(df, params)
            m_signal = self.generate_macd_signal(df, params)

            # Collect agreeing signals
            all_signals = [s for s in [
                z_signal, b_signal, m_signal] if s is not None]

            if len(all_signals) >= params['min_strategies']:
                # Check if they agree on direction
                actions = [s['action'] for s in all_signals]
                if len(set(actions)) == 1:  # All agree
                    avg_strength = sum(s['strength']
                                       for s in all_signals) / len(all_signals)
                    signals.append({
                        'action': actions[0],
                        'strength': avg_strength,
                        'reason': f"{len(all_signals)} strategies agree",
                        'details': ' | '.join([s['reason'] for s in all_signals])
                    })

        if not signals:
            logger.info("No signals generated")
            return None

        # Use the strongest signal
        signal = max(signals, key=lambda x: x['strength'])

        # Check minimum signal strength
        min_strength = self.config['alerts']['min_signal_strength']
        if signal['strength'] < min_strength:
            logger.info(
                f"Signal strength {signal['strength']:.2f} below threshold {min_strength}")
            return None

        logger.info(
            f"✓ Signal: {signal['action']} (strength: {signal['strength']:.2f})")
        return signal

    def should_send_alert(self, timeframe):
        """Check if enough time has passed since last alert"""
        cooldown = self.config['alerts']['cooldown_minutes'].get(timeframe, 60)

        if timeframe not in self.last_alert_times:
            return True

        elapsed = (datetime.now() -
                   self.last_alert_times[timeframe]).total_seconds() / 60

        if elapsed >= cooldown:
            return True

        logger.info(
            f"Cooldown active: {cooldown - elapsed:.1f} minutes remaining")
        return False

    def send_alert(self, signal, timeframe):
        """Send trading alert via Telegram"""
        if not self.should_send_alert(timeframe):
            return

        name1 = self.config['trading']['name1']
        name2 = self.config['trading']['name2']

        # Create message
        action_emoji = "🔴" if signal['action'] == "SHORT" else "🟢"
        strength_bars = "█" * int(signal['strength'] * 10)

        message = f"""
{action_emoji} <b>TRADING SIGNAL</b> {action_emoji}

<b>Pair:</b> {name1}/{name2}
<b>Action:</b> {signal['action']}
<b>Timeframe:</b> {timeframe}
<b>Strength:</b> {strength_bars} ({signal['strength']:.1%})

<b>Reason:</b> {signal['reason']}
<b>Details:</b> {signal['details']}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""

        if self.send_telegram_message(message):
            self.last_alert_times[timeframe] = datetime.now()
            logger.info(f"✓ Alert sent for {timeframe}")
        else:
            logger.error(f"✗ Failed to send alert for {timeframe}")

    def check_and_alert(self, timeframe):
        """Main function to check signals and send alerts"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Checking {timeframe} signals...")
            logger.info(f"{'='*60}")

            signal = self.analyze_pair(timeframe)

            if signal:
                self.send_alert(signal, timeframe)
            else:
                logger.info("No actionable signals at this time")

        except Exception as e:
            logger.error(f"Error in check_and_alert: {e}", exc_info=True)


def load_config():
    """Load configuration from YAML file"""
    if not CONFIG_FILE.exists():
        logger.error(f"Config file not found: {CONFIG_FILE}")
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded successfully")
    return config


def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("Trading Alert System Starting...")
    logger.info("="*60)

    try:
        # Load configuration
        config = load_config()

        # Initialize system
        alert_system = TradingAlertSystem(config)

        # Send startup message if enabled
        if config['alerts'].get('send_test_message', True):
            startup_msg = f"""
🚀 <b>Alert System Started</b>

<b>Pair:</b> {config['trading']['name1']}/{config['trading']['name2']}
<b>Strategy:</b> {config['strategy']['name']}

System is now monitoring for trading signals.

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
            alert_system.send_telegram_message(startup_msg)

        # Schedule checks based on config
        schedule_config = config['schedule']

        if schedule_config.get('check_15min', True):
            schedule.every(15).minutes.do(
                lambda: alert_system.check_and_alert("15-Minutes")
            )
            logger.info("✓ Scheduled: 15-minute checks")

        if schedule_config.get('check_hourly', True):
            schedule.every().hour.at(":00").do(
                lambda: alert_system.check_and_alert("Hourly")
            )
            logger.info("✓ Scheduled: Hourly checks")

        if schedule_config.get('check_daily', True):
            daily_hour = schedule_config.get('daily_check_hour', 17)
            schedule.every().day.at(f"{daily_hour:02d}:00").do(
                lambda: alert_system.check_and_alert("Daily")
            )
            logger.info(f"✓ Scheduled: Daily checks at {daily_hour:02d}:00")

        logger.info("="*60)
        logger.info("Alert system is running. Press Ctrl+C to stop.")
        logger.info("="*60)

        # Run first check immediately
        alert_system.check_and_alert("Daily")

        # Main loop
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds

    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("Alert system stopped by user")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
