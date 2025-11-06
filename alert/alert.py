import os
try:
    import yaml
except ImportError:
    # Friendly message with install instructions
    raise ModuleNotFoundError(
        "Missing required dependency 'PyYAML' (imported as 'yaml').\n"
        "Install it with: pip install PyYAML\n"
        "Or install all project requirements: pip install -r requirements.txt"
    )
import sys

# Ensure console output uses UTF-8 where possible to avoid UnicodeEncodeError on Windows
try:
    # Python 3.7+ supports reconfigure
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # Fallback: set PYTHONUTF8 environment variable for downstream subprocesses
    os.environ.setdefault('PYTHONUTF8', '1')
from datetime import datetime, timedelta
try:
    import telegram
    from telegram.error import TelegramError
except ImportError:
    raise ModuleNotFoundError(
        "Missing required dependency 'python-telegram-bot' (imported as 'telegram').\n"
        "Install it with: pip install python-telegram-bot\n"
        "Or install all project requirements: pip install -r requirements.txt"
    )
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class InteractiveSpreadAnalyzer:
    def __init__(self, stock1, stock2, ratio, name1, name2, z_score_window=20):
        self.stock1 = stock1
        self.stock2 = stock2
        self.ratio = ratio
        self.name1 = name1
        self.name2 = name2
        self.z_score_window = z_score_window

    def is_trading_hour(self, timestamp):
        """Check if the given timestamp is during trading hours (9:00-17:30 CET/CEST)"""
        if timestamp.weekday() >= 5:  # Weekend
            return False

        hour = timestamp.hour
        minute = timestamp.minute
        trading_start = 9 * 60
        trading_end = 17 * 60 + 30
        current_time = hour * 60 + minute

        return trading_start <= current_time <= trading_end

    def filter_non_trading_periods(self, df, interval):
        """Filter out non-trading periods based on interval"""
        if df is None or df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if interval == '1d':
            return df[df.index.dayofweek < 5]
        else:
            return df[df.index.map(self.is_trading_hour) & (df.index.dayofweek < 5)]

    async def fetch_data(self, ticker, interval, period, start=None, end=None):
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(ticker)
            if start and end:
                data = stock.history(interval=interval, start=start, end=end)
            else:
                data = stock.history(interval=interval, period=period)

            if data.empty:
                raise ValueError(f"No data retrieved for {ticker}")

            data = self.filter_non_trading_periods(data, interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_z_score(self, spread):
        """Calculate z-score for the spread series"""
        mean = spread.rolling(window=self.z_score_window).mean()
        std = spread.rolling(window=self.z_score_window).std()
        z_score = (spread - mean) / std
        return z_score

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators including z-score"""
        if df is None or df.empty:
            return df

        # Z-score
        df['Z_Score'] = self.calculate_z_score(df['Close'])

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        # Bollinger Bands
        df['MiddleBB'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['UpperBB'] = df['MiddleBB'] + (bb_std * 2)
        df['LowerBB'] = df['MiddleBB'] - (bb_std * 2)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        return df

    async def calculate_spread_with_indicators(self, interval, period, start=None, end=None):
        """Calculate spread and technical indicators"""
        stock1_data = await self.fetch_data(self.stock1, interval, period, start, end)
        stock2_data = await self.fetch_data(self.stock2, interval, period, start, end)

        if stock1_data is None or stock2_data is None:
            return None

        common_idx = stock1_data.index.intersection(stock2_data.index)
        stock1_data = stock1_data.loc[common_idx]
        stock2_data = stock2_data.loc[common_idx]

        if stock1_data.empty or stock2_data.empty:
            print(f"Data for {self.stock1} or {self.stock2} is empty.")
            return None

        spread_df = pd.DataFrame(index=common_idx)
        spread_df["Close"] = stock1_data["Close"] - \
            (stock2_data["Close"] * self.ratio)
        spread_df["Open"] = stock1_data["Open"] - \
            (stock2_data["Open"] * self.ratio)
        spread_df["High"] = stock1_data["High"] - \
            (stock2_data["Low"] * self.ratio)
        spread_df["Low"] = stock1_data["Low"] - \
            (stock2_data["High"] * self.ratio)
        spread_df["Volume"] = (stock1_data["Volume"] +
                               stock2_data["Volume"]) / 2

        spread_df = self.calculate_technical_indicators(spread_df)

        return spread_df


class TradingStrategy:
    """Base class for trading strategies - use as placeholder for custom strategies"""

    def __init__(self, name="Base Strategy"):
        self.name = name

    def generate_signal(self, spread_df):
        """
        Generate trading signal based on strategy
        Returns: (signal, signal_strength, details)
            signal: 1 (long), -1 (short), 0 (no signal)
            signal_strength: float 0-1 indicating confidence
            details: dict with additional info
        """
        return 0, 0.0, {}


class ZScoreStrategy(TradingStrategy):
    """Mean reversion strategy based on Z-Score"""

    def __init__(self, z_threshold=2.0, z_exit=0.5):
        super().__init__("Z-Score Mean Reversion")
        self.z_threshold = z_threshold
        self.z_exit = z_exit

    def generate_signal(self, spread_df):
        if spread_df is None or spread_df.empty:
            return 0, 0.0, {}

        latest = spread_df.iloc[-1]
        z_score = latest['Z_Score']

        signal = 0
        strength = 0.0
        details = {
            'z_score': z_score,
            'threshold': self.z_threshold,
            'strategy': self.name
        }

        # Long signal: Z-score below negative threshold
        if z_score < -self.z_threshold:
            signal = 1
            strength = min(abs(z_score) / (self.z_threshold * 2), 1.0)
            details['reason'] = f"Z-Score {z_score:.2f} below -{self.z_threshold}"

        # Short signal: Z-score above positive threshold
        elif z_score > self.z_threshold:
            signal = -1
            strength = min(abs(z_score) / (self.z_threshold * 2), 1.0)
            details['reason'] = f"Z-Score {z_score:.2f} above {self.z_threshold}"

        return signal, strength, details


class BollingerBandsStrategy(TradingStrategy):
    """Strategy based on Bollinger Bands breakout"""

    def __init__(self, rsi_oversold=30, rsi_overbought=70):
        super().__init__("Bollinger Bands + RSI")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, spread_df):
        if spread_df is None or spread_df.empty:
            return 0, 0.0, {}

        latest = spread_df.iloc[-1]
        close = latest['Close']
        upper_bb = latest['UpperBB']
        lower_bb = latest['LowerBB']
        rsi = latest['RSI']

        signal = 0
        strength = 0.0
        details = {
            'close': close,
            'upper_bb': upper_bb,
            'lower_bb': lower_bb,
            'rsi': rsi,
            'strategy': self.name
        }

        # Long signal: Price at lower band + oversold RSI
        if close <= lower_bb and rsi < self.rsi_oversold:
            signal = 1
            strength = (self.rsi_oversold - rsi) / self.rsi_oversold
            details['reason'] = f"Price at lower BB ({close:.2f}) with RSI {rsi:.1f}"

        # Short signal: Price at upper band + overbought RSI
        elif close >= upper_bb and rsi > self.rsi_overbought:
            signal = -1
            strength = (rsi - self.rsi_overbought) / \
                (100 - self.rsi_overbought)
            details['reason'] = f"Price at upper BB ({close:.2f}) with RSI {rsi:.1f}"

        return signal, strength, details


class MACDStrategy(TradingStrategy):
    """Strategy based on MACD crossover"""

    def __init__(self):
        super().__init__("MACD Crossover")

    def generate_signal(self, spread_df):
        if spread_df is None or len(spread_df) < 2:
            return 0, 0.0, {}

        latest = spread_df.iloc[-1]
        previous = spread_df.iloc[-2]

        macd = latest['MACD']
        signal_line = latest['Signal_Line']
        prev_macd = previous['MACD']
        prev_signal = previous['Signal_Line']

        signal = 0
        strength = 0.0
        details = {
            'macd': macd,
            'signal_line': signal_line,
            'histogram': latest['MACD_Hist'],
            'strategy': self.name
        }

        # Bullish crossover
        if prev_macd <= prev_signal and macd > signal_line:
            signal = 1
            strength = min(abs(macd - signal_line) /
                           abs(signal_line + 0.0001), 1.0)
            details['reason'] = f"MACD bullish crossover ({macd:.3f} > {signal_line:.3f})"

        # Bearish crossover
        elif prev_macd >= prev_signal and macd < signal_line:
            signal = -1
            strength = min(abs(macd - signal_line) /
                           abs(signal_line + 0.0001), 1.0)
            details['reason'] = f"MACD bearish crossover ({macd:.3f} < {signal_line:.3f})"

        return signal, strength, details


class CombinedStrategy(TradingStrategy):
    """Combined strategy using multiple indicators"""

    def __init__(self, z_threshold=2.0, min_strategies=2):
        super().__init__("Combined Multi-Strategy")
        self.strategies = [
            ZScoreStrategy(z_threshold),
            BollingerBandsStrategy(),
            MACDStrategy()
        ]
        self.min_strategies = min_strategies

    def generate_signal(self, spread_df):
        if spread_df is None or spread_df.empty:
            return 0, 0.0, {}

        signals = []
        strengths = []
        all_details = []

        # Get signals from all strategies
        for strategy in self.strategies:
            sig, strength, details = strategy.generate_signal(spread_df)
            if sig != 0:
                signals.append(sig)
                strengths.append(strength)
                all_details.append(details)

        # Require minimum number of strategies to agree
        if len(signals) < self.min_strategies:
            return 0, 0.0, {'reason': 'Insufficient strategy agreement'}

        # Check if signals agree
        if len(set(signals)) > 1:
            return 0, 0.0, {'reason': 'Conflicting signals from strategies'}

        # All signals agree
        final_signal = signals[0]
        final_strength = np.mean(strengths)

        details = {
            'strategy': self.name,
            'agreeing_strategies': len(signals),
            'average_strength': final_strength,
            'individual_details': all_details
        }

        return final_signal, final_strength, details


class SignalAlertSystem:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.analyzer = InteractiveSpreadAnalyzer(
            self.config['trading']['stock1'],
            self.config['trading']['stock2'],
            self.config['trading']['ratio'],
            self.config['trading']['name1'],
            self.config['trading']['name2'],
            self.config['trading'].get('z_score_window', 20)
        )

        # Initialize strategy based on config
        self.strategy = self.get_strategy(self.config['strategy'])

        self.last_alerts = {}
        self.bot = None
        self.running = False

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        # Allow override via environment variable
        env_path = os.getenv('ALERT_CONFIG')
        if env_path:
            if os.path.exists(env_path):
                print(f"Using config from ALERT_CONFIG: {env_path}")
                with open(env_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(
                    f"ALERT_CONFIG is set to {env_path} but file was not found.")

        # If the provided path exists, use it
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)

        # Fallback: look in the same directory as this script (alert/config.yaml)
        script_dir = os.path.dirname(__file__)
        alt_path = os.path.join(script_dir, 'config.yaml')
        if os.path.exists(alt_path):
            print(f"Using fallback config at {alt_path}")
            with open(alt_path, 'r') as file:
                return yaml.safe_load(file)

        # Another fallback: parent directory (repo root)
        parent_path = os.path.join(script_dir, '..', 'config.yaml')
        parent_path = os.path.normpath(parent_path)
        if os.path.exists(parent_path):
            print(f"Using parent config at {parent_path}")
            with open(parent_path, 'r') as file:
                return yaml.safe_load(file)

        # Not found anywhere: create default at the requested config_path
        self.create_default_config(config_path)
        print(
            f"Created default config at {config_path}. Please edit with your credentials.")
        raise FileNotFoundError(
            f"Please edit {config_path} with your credentials")

    def create_default_config(self, config_path):
        """Create a default configuration file"""
        default_config = {
            'trading': {
                'stock1': 'AGN.AS',
                'stock2': 'ASML.AS',
                'ratio': 0.0072,
                'name1': 'AGN',
                'name2': 'ASML',
                'z_score_window': 20
            },
            'telegram': {
                'token': 'YOUR_BOT_TOKEN',
                'chat_id': 'YOUR_CHAT_ID'
            },
            'strategy': {
                'name': 'z_score',  # Options: z_score, bollinger, macd, combined
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

        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)

    def get_strategy(self, strategy_config):
        """Initialize the appropriate strategy based on config"""
        strategy_name = strategy_config.get('name', 'z_score').lower()
        params = strategy_config.get('params', {})

        if strategy_name == 'z_score':
            return ZScoreStrategy(
                z_threshold=params.get('z_threshold', 2.0),
                z_exit=params.get('z_exit', 0.5)
            )
        elif strategy_name == 'bollinger':
            return BollingerBandsStrategy(
                rsi_oversold=params.get('rsi_oversold', 30),
                rsi_overbought=params.get('rsi_overbought', 70)
            )
        elif strategy_name == 'macd':
            return MACDStrategy()
        elif strategy_name == 'combined':
            return CombinedStrategy(
                z_threshold=params.get('z_threshold', 2.0),
                min_strategies=params.get('min_strategies', 2)
            )
        else:
            print(
                f"Unknown strategy: {strategy_name}. Using Z-Score strategy.")
            return ZScoreStrategy()

    async def setup_telegram(self):
        """Setup Telegram bot connection"""
        try:
            self.bot = telegram.Bot(token=self.config['telegram']['token'])
            await self.bot.get_me()
            print("✅ Telegram bot successfully configured")

            if self.config['alerts'].get('send_test_message', True):
                await self.send_test_message()
        except TelegramError as e:
            # Don't crash the whole service if Telegram is misconfigured.
            # Log the error and continue without a bot; alerts will be skipped.
            try:
                print(f"Failed to setup Telegram bot: {str(e)}")
            except Exception:
                # avoid any encoding issues when printing
                print("Failed to setup Telegram bot (see logs)")
            self.bot = None
            return False

    async def send_test_message(self):
        """Send initial test message"""
        if not self.bot:
            return

        try:
            test_message = (
                "🔔 <b>Alert System Activated</b> 🔔\n\n"
                f"✅ Connection successful\n"
                f"📊 Monitoring: {self.analyzer.name1}/{self.analyzer.name2}\n"
                f"🎯 Strategy: {self.strategy.name}\n"
                f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "System is now monitoring for trading signals..."
            )
            await self.bot.send_message(
                chat_id=self.config['telegram']['chat_id'],
                text=test_message,
                parse_mode='HTML'
            )
            print("✅ Test message sent successfully")
        except TelegramError as e:
            print(f"❌ Failed to send test message: {str(e)}")

    async def send_telegram_alert(self, message):
        """Send alert via Telegram"""
        if not self.bot:
            return

        try:
            await self.bot.send_message(
                chat_id=self.config['telegram']['chat_id'],
                text=message,
                parse_mode='HTML'
            )
            print("✅ Telegram alert sent")
        except TelegramError as e:
            print(f"❌ Failed to send Telegram alert: {str(e)}")

    def check_cooldown(self, timeframe):
        """Check if cooldown period has passed"""
        if timeframe not in self.last_alerts:
            return True

        cooldown_minutes = self.config['alerts']['cooldown_minutes'].get(
            timeframe, 60)
        time_since_last = datetime.now() - self.last_alerts[timeframe]
        return time_since_last.total_seconds() / 60 > cooldown_minutes

    async def check_signals(self, interval, period, timeframe):
        """Check for trading signals"""
        if not self.check_cooldown(timeframe):
            print(f"⏳ {timeframe}: Cooldown active, skipping check")
            return

        try:
            print(f"🔍 Checking {timeframe} signals...")
            spread_df = await self.analyzer.calculate_spread_with_indicators(interval, period)

            if spread_df is None or spread_df.empty:
                print(f"⚠️ No data available for {timeframe}")
                return

            # Get signal from strategy
            signal, strength, details = self.strategy.generate_signal(
                spread_df)

            # Check if signal meets minimum strength threshold
            min_strength = self.config['alerts'].get(
                'min_signal_strength', 0.5)

            if signal != 0 and strength >= min_strength:
                latest = spread_df.iloc[-1]
                signal_type = "🟢 LONG" if signal == 1 else "🔴 SHORT"

                message = self.format_alert_message(
                    signal_type, timeframe, latest, strength, details
                )

                await self.send_telegram_alert(message)
                self.last_alerts[timeframe] = datetime.now()
                print(f"🚨 {timeframe} {signal_type} signal sent!")
            else:
                print(f"✓ {timeframe}: No signals (strength: {strength:.2f})")

        except Exception as e:
            error_msg = f"Error checking {timeframe}: {str(e)}"
            print(f"❌ {error_msg}")
            await self.send_telegram_alert(f"⚠️ <b>Error:</b> {error_msg}")

    def format_alert_message(self, signal_type, timeframe, latest, strength, details):
        """Format the alert message"""
        message = (
            f"🚨 <b>{timeframe} Trading Signal</b> 🚨\n\n"
            f"<b>Signal:</b> {signal_type}\n"
            f"<b>Strategy:</b> {self.strategy.name}\n"
            f"<b>Confidence:</b> {strength:.1%}\n"
            f"<b>Timeframe:</b> {timeframe}\n\n"
            f"📊 <b>Current Metrics:</b>\n"
            f"• Spread: {latest['Close']:.4f}\n"
        )

        if 'Z_Score' in latest.index:
            message += f"• Z-Score: {latest['Z_Score']:.2f}\n"
        if 'RSI' in latest.index:
            message += f"• RSI: {latest['RSI']:.1f}\n"
        if 'MACD' in latest.index:
            message += f"• MACD: {latest['MACD']:.3f}\n"

        if 'reason' in details:
            message += f"\n💡 <b>Reason:</b> {details['reason']}\n"

        message += (
            f"\n<b>Pair:</b> {self.analyzer.name1} - {self.analyzer.name2}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return message

    async def schedule_checker(self):
        """Main scheduling loop"""
        self.running = True
        print("🚀 Alert system started. Monitoring for signals...\n")

        while self.running:
            now = datetime.now()

            # Check 15-minute signals
            if self.config['schedule'].get('check_15min', True):
                if now.minute % 15 == 0 and now.second < 10:
                    await self.check_signals("15m", "5d", "15-Minutes")

            # Check hourly signals
            if self.config['schedule'].get('check_hourly', True):
                if now.minute == 0 and now.second < 10:
                    await self.check_signals("1h", "1mo", "Hourly")

            # Check daily signals
            if self.config['schedule'].get('check_daily', True):
                daily_hour = self.config['schedule'].get(
                    'daily_check_hour', 17)
                if now.hour == daily_hour and now.minute == 0 and now.second < 10:
                    await self.check_signals("1d", "1y", "Daily")

            # Sleep for a short interval
            await asyncio.sleep(10)

    def stop(self):
        """Stop the alert system"""
        self.running = False
        print("\n⏹️ Stopping alert system...")

    async def run(self):
        """Run the alert system"""
        await self.setup_telegram()
        await self.schedule_checker()


async def main():
    """Main entry point"""
    try:
        alert_system = SignalAlertSystem('config.yaml')
        await alert_system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Keyboard interrupt received")
        alert_system.stop()
        if alert_system.bot:
            await alert_system.send_telegram_alert(
                "⚠️ <b>Alert System Stopped</b>\n"
                "Monitoring has been terminated by user."
            )
    except FileNotFoundError as e:
        print(f"\n❌ {str(e)}")
    except Exception as e:
        print(f"\n❌ Error running alert system: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
