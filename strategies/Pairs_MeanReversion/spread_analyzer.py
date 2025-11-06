import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class InteractiveSpreadAnalyzer:
    def __init__(self, stock1, stock2, ratio, name1=None, name2=None, z_score_window=20):
        """
        Initialize InteractiveSpreadAnalyzer with two stocks and their ratio
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.ratio = ratio
        self.name1 = name1 or stock1
        self.name2 = name2 or stock2
        self.z_score_window = z_score_window

    def is_trading_hour(self, timestamp):
        """
        Check if the given timestamp is during trading hours (9:00-17:30 CET/CEST)
        """
        if timestamp.weekday() >= 5:  # Weekend
            return False

        hour = timestamp.hour
        minute = timestamp.minute

        trading_start = 9 * 60  # 9:00 in minutes
        trading_end = 17 * 60 + 30  # 17:30 in minutes
        current_time = hour * 60 + minute

        return trading_start <= current_time <= trading_end

    def filter_non_trading_periods(self, df, interval):
        """
        Filter out non-trading periods based on interval
        """
        if df is None or df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if interval == '1d':
            return df[df.index.dayofweek < 5]
        else:
            return df[df.index.map(self.is_trading_hour) & (df.index.dayofweek < 5)]

    def fetch_data(self, ticker, interval, period, start=None, end=None):
        """
        Fetch stock data with error handling and trading hours filtering
        """
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
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_z_score(self, spread):
        """
        Calculate z-score for the spread series
        """
        mean = spread.rolling(window=self.z_score_window).mean()
        std = spread.rolling(window=self.z_score_window).std()
        z_score = (spread - mean) / std
        return z_score

    def generate_signals(self, z_score, threshold=2):
        """
        Generate trading signals based on z-score
        """
        signals = pd.Series(index=z_score.index, data=0)
        signals[z_score > threshold] = -1  # Short signal
        signals[z_score < -threshold] = 1   # Long signal
        return signals

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators including z-score
        """
        if df is None or df.empty:
            return df

        # Z-score and signals
        df['Z_Score'] = self.calculate_z_score(df['Close'])
        df['Signals'] = self.generate_signals(df['Z_Score'])

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

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

    def calculate_spread_with_indicators(self, interval, period, start=None, end=None):
        """
        Calculate spread and technical indicators with trading hours handling
        """
        stock1_data = self.fetch_data(
            self.stock1, interval, period, start, end)
        stock2_data = self.fetch_data(
            self.stock2, interval, period, start, end)

        if stock1_data is None or stock2_data is None:
            return None

        common_idx = stock1_data.index.intersection(stock2_data.index)
        stock1_data = stock1_data.loc[common_idx]
        stock2_data = stock2_data.loc[common_idx]

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

    def create_interactive_plot(self, spread_df, title, interval):
        """
        Create interactive plot using Plotly with proper time axis handling
        """
        if spread_df is None or spread_df.empty:
            return None

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=(
                'Spread Price & Bollinger Bands with Signals',
                'Z-Score',
                'Volume & MACD',
                'RSI',
                'ATR'
            )
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=spread_df.index,
                open=spread_df['Open'],
                high=spread_df['High'],
                low=spread_df['Low'],
                close=spread_df['Close'],
                name='Spread OHLC'
            ),
            row=1, col=1
        )

        # Add Bollinger Bands
        for band, color in [('UpperBB', 'rgba(173, 204, 255, 0.7)'),
                            ('MiddleBB', 'rgba(89, 89, 89, 0.7)'),
                            ('LowerBB', 'rgba(173, 204, 255, 0.7)')]:
            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df[band],
                    name=band,
                    line=dict(color=color)
                ),
                row=1, col=1
            )

        # Add trading signals
        long_signals = spread_df[spread_df['Signals'] == 1].index
        short_signals = spread_df[spread_df['Signals'] == -1].index

        fig.add_trace(
            go.Scatter(
                x=long_signals,
                y=spread_df.loc[long_signals, 'Low'],
                name='Long Signal',
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green'
                )
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=short_signals,
                y=spread_df.loc[short_signals, 'High'],
                name='Short Signal',
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red'
                )
            ),
            row=1, col=1
        )

        # Add Z-score
        fig.add_trace(
            go.Scatter(
                x=spread_df.index,
                y=spread_df['Z_Score'],
                name='Z-Score',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # Add Z-score threshold lines
        fig.add_hline(y=2, line_width=1, line_dash="dash",
                      line_color="red", row=2)
        fig.add_hline(y=-2, line_width=1, line_dash="dash",
                      line_color="green", row=2)

        # Add Volume with colors
        colors = ['red' if row['Open'] - row['Close'] >= 0
                  else 'green' for index, row in spread_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=spread_df.index,
                y=spread_df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=3, col=1
        )

        # Add MACD
        for indicator, color in [('MACD', 'blue'), ('Signal', 'orange')]:
            fig.add_trace(
                go.Scatter(
                    x=spread_df.index,
                    y=spread_df[indicator],
                    name=indicator,
                    line=dict(color=color)
                ),
                row=3, col=1
            )

        # Add MACD histogram
        colors = ['red' if val <
                  0 else 'green' for val in spread_df['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=spread_df.index,
                y=spread_df['MACD_Hist'],
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.5
            ),
            row=3, col=1
        )

        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=spread_df.index,
                y=spread_df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=4, col=1
        )

        # Add RSI reference lines
        fig.add_hline(y=70, line_width=1, line_dash="dash",
                      line_color="red", row=4)
        fig.add_hline(y=30, line_width=1, line_dash="dash",
                      line_color="green", row=4)

        # Add ATR
        fig.add_trace(
            go.Scatter(
                x=spread_df.index,
                y=spread_df['ATR'],
                name='ATR',
                line=dict(color='orange')
            ),
            row=5, col=1
        )

        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title=f"Spread Price ({self.name1} - {self.name2}×{self.ratio})",
            yaxis2_title="Z-Score",
            yaxis3_title="Volume & MACD",
            yaxis4_title="RSI",
            yaxis5_title="ATR",
            xaxis5_title="Date",
            showlegend=True,
            height=1400,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )

        # Configure x-axis based on interval
        if interval in ['15m', '1h']:
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[17.5, 9], pattern="hour"),
                ]
            )
        else:
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                ]
            )

        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig.update_yaxes(title_text="Volume & MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=4, col=1)
        fig.update_yaxes(title_text="ATR", row=5, col=1)

        return fig


def main():
    st.set_page_config(page_title="Spread Analyzer", layout="wide")

    st.title("📊 Interactive Spread Trading Analyzer")
    st.markdown(
        "Analyze the spread between two stocks with technical indicators and trading signals")

    # Sidebar for inputs
    st.sidebar.header("Configuration")

    # Stock pair presets
    preset = st.sidebar.selectbox(
        "Select Preset Pair",
        ["Custom", "AGN/ASML", "WKL/REN"]
    )

    if preset == "AGN/ASML":
        default_stock1 = "AGN.AS"
        default_stock2 = "ASML.AS"
        default_ratio = 0.0072
        default_name1 = "AGN"
        default_name2 = "ASML"
    elif preset == "WKL/REN":
        default_stock1 = "WKL.AS"
        default_stock2 = "REN.AS"
        default_ratio = 3.6
        default_name1 = "WKL"
        default_name2 = "REN"
    else:
        default_stock1 = "AGN.AS"
        default_stock2 = "ASML.AS"
        default_ratio = 0.0072
        default_name1 = "AGN"
        default_name2 = "ASML"

    # Stock inputs
    st.sidebar.subheader("Stock Pair")
    stock1 = st.sidebar.text_input("Stock 1 Ticker", value=default_stock1)
    name1 = st.sidebar.text_input("Stock 1 Name", value=default_name1)
    stock2 = st.sidebar.text_input("Stock 2 Ticker", value=default_stock2)
    name2 = st.sidebar.text_input("Stock 2 Name", value=default_name2)
    ratio = st.sidebar.number_input(
        "Ratio (Stock2 multiplier)", value=default_ratio, format="%.4f")

    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    z_score_window = st.sidebar.slider("Z-Score Window", 10, 50, 20)

    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["Daily (1 Year)", "Hourly (1 Month)",
         "15-Minutes (5 Days)", "All Timeframes"]
    )

    # Map timeframe to interval and period
    timeframe_map = {
        "Daily (1 Year)": ("1d", "1y", "Daily"),
        "Hourly (1 Month)": ("1h", "1mo", "Hourly"),
        "15-Minutes (5 Days)": ("15m", "5d", "15-Minutes")
    }

    # Analyze button
    if st.sidebar.button("🚀 Analyze Spread", type="primary"):
        with st.spinner("Fetching data and calculating indicators..."):
            # Initialize analyzer
            analyzer = InteractiveSpreadAnalyzer(
                stock1, stock2, ratio, name1, name2, z_score_window
            )

            if timeframe == "All Timeframes":
                intervals = [
                    ("1d", "1y", "Daily"),
                    ("1h", "1mo", "Hourly"),
                    ("15m", "5d", "15-Minutes")
                ]
            else:
                intervals = [timeframe_map[timeframe]]

            for interval, period, tf_name in intervals:
                try:
                    # Calculate spread and indicators
                    spread_df = analyzer.calculate_spread_with_indicators(
                        interval, period)

                    if spread_df is not None and not spread_df.empty:
                        # Get current values
                        current_spread = spread_df['Close'].iloc[-1]
                        current_z_score = spread_df['Z_Score'].iloc[-1]

                        # Display metrics
                        st.subheader(f"📈 {tf_name} Analysis")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Current Spread",
                                      f"{current_spread:.2f}")
                        with col2:
                            st.metric("Z-Score", f"{current_z_score:.2f}")
                        with col3:
                            signal = "LONG" if current_z_score < - \
                                2 else "SHORT" if current_z_score > 2 else "NEUTRAL"
                            st.metric("Signal", signal)
                        with col4:
                            st.metric(
                                "RSI", f"{spread_df['RSI'].iloc[-1]:.2f}")

                        # Create title
                        title = (f"Spread Analysis: {name1} - {name2}×{ratio}\n"
                                 f"Timeframe: {tf_name} | Current Spread: {current_spread:.2f} | Z-Score: {current_z_score:.2f}")

                        # Create and display plot
                        fig = analyzer.create_interactive_plot(
                            spread_df, title, interval)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        # Display recent signals
                        recent_signals = spread_df[spread_df['Signals'] != 0].tail(
                            10)
                        if not recent_signals.empty:
                            st.subheader("Recent Trading Signals")
                            signal_df = pd.DataFrame({
                                'Date': recent_signals.index,
                                'Signal': recent_signals['Signals'].map({1: 'LONG', -1: 'SHORT'}),
                                'Spread': recent_signals['Close'].round(2),
                                'Z-Score': recent_signals['Z_Score'].round(2)
                            })
                            st.dataframe(signal_df, use_container_width=True)
                    else:
                        st.error(f"No data available for {tf_name} timeframe")

                except Exception as e:
                    st.error(f"Error processing {tf_name} timeframe: {str(e)}")

    else:
        st.info(
            "👈 Configure your analysis parameters and click 'Analyze Spread' to start")

        # Display instructions
        st.markdown("""
        ### How to use this analyzer:
        
        1. **Select a preset pair** or enter custom stock tickers
        2. **Configure the ratio** - the multiplier for Stock 2 in the spread calculation
        3. **Choose a timeframe** - Daily, Hourly, 15-Minutes, or analyze all at once
        4. **Adjust Z-Score window** - the rolling window for mean reversion signals
        5. Click **Analyze Spread** to generate the analysis
        
        ### Understanding the signals:
        
        - **Long Signal** (Green ▲): Z-Score < -2 (spread is below mean, expect reversion up)
        - **Short Signal** (Red ▼): Z-Score > 2 (spread is above mean, expect reversion down)
        - **RSI**: Shows momentum (>70 overbought, <30 oversold)
        - **MACD**: Trend following indicator
        - **Bollinger Bands**: Volatility bands around the spread
        """)


if __name__ == "__main__":
    main()
