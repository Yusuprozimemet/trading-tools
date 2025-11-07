#!/usr/bin/env python3
"""
Terminal-based Algo Trading System
Run algo trading directly in terminal with real-time updates
"""

import os
import sys
import time
import yaml
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Back, Style, init

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

# Initialize colorama for cross-platform color support
init(autoreset=True)

# Load environment variables
load_dotenv()
ALPACA_ENDPOINT = os.getenv("ALPACA_ENDPOINT")
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = Fore.GREEN
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    RESET = Style.RESET_ALL
    BRIGHT = Style.BRIGHT


class TerminalTrading:
    def __init__(self):
        """Initialize trading system"""
        self.trading_client = TradingClient(
            ALPACA_KEY, 
            ALPACA_SECRET, 
            paper=(ALPACA_ENDPOINT == "https://paper-api.alpaca.markets")
        )
        self.data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        self.algo_running = False
        self.config = self.load_config()
        self.strategy = None
        self.update_interval = 30  # seconds

    def load_config(self):
        """Load main config"""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_strategy(self, strategy_name):
        """Load strategy config"""
        path = os.path.join(
            os.path.dirname(__file__), "strategies", f"{strategy_name}.yaml"
        )
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BRIGHT}{Colors.CYAN}{title.center(80)}{Colors.RESET}")
        print(f"{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

    def print_timestamp(self):
        """Print current timestamp"""
        print(f"{Colors.YELLOW}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]{Colors.RESET}")

    def format_currency(self, value):
        """Format value as currency"""
        try:
            return f"${float(value):,.2f}"
        except:
            return value

    def format_percentage(self, value):
        """Format value as percentage"""
        try:
            return f"{float(value):.2f}%"
        except:
            return value

    def get_color_for_value(self, value):
        """Return color based on value sign"""
        try:
            if float(value) >= 0:
                return Colors.GREEN
            else:
                return Colors.RED
        except:
            return Colors.WHITE

    def display_account_overview(self):
        """Display account overview"""
        try:
            account = self.trading_client.get_account()
            self.print_header("ACCOUNT OVERVIEW")
            
            account_data = [
                ["Portfolio Value", self.format_currency(account.portfolio_value)],
                ["Cash", self.format_currency(account.cash)],
                ["Buying Power", self.format_currency(account.buying_power)],
                ["Long Market Value", self.format_currency(account.long_market_value)],
                ["Short Market Value", self.format_currency(account.short_market_value)],
                ["Status", account.status],
                ["Pattern Day Trader", account.pattern_day_trader],
                ["Daytrade Count", account.daytrade_count],
            ]
            
            print(tabulate(account_data, headers=["Metric", "Value"], tablefmt="grid"))
            
            # Daily P/L
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            pl = equity - last_equity
            pl_pct = (pl / last_equity * 100) if last_equity > 0 else 0
            
            color = self.get_color_for_value(pl)
            print(f"\n{color}Today's P/L: {self.format_currency(pl)} ({self.format_percentage(pl_pct)}){Colors.RESET}")
            
        except Exception as e:
            print(f"{Colors.RED}Error fetching account info: {e}{Colors.RESET}")

    def display_positions(self):
        """Display current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            self.print_header("CURRENT POSITIONS")
            
            if not positions:
                print(f"{Colors.YELLOW}No open positions{Colors.RESET}\n")
                return
            
            positions_data = []
            for p in positions:
                side = "LONG" if float(p.qty) > 0 else "SHORT"
                pl = float(p.unrealized_pl)
                pl_color = self.get_color_for_value(pl)
                
                positions_data.append([
                    p.symbol,
                    f"{float(p.qty):.0f}",
                    side,
                    self.format_currency(p.avg_entry_price),
                    self.format_currency(p.current_price),
                    self.format_currency(p.market_value),
                    f"{pl_color}{self.format_currency(pl)}{Colors.RESET}",
                    f"{pl_color}{self.format_percentage(float(p.unrealized_plpc) * 100)}{Colors.RESET}"
                ])
            
            print(tabulate(
                positions_data,
                headers=["Symbol", "Qty", "Side", "Avg Entry", "Current", "Market Value", "P/L", "P/L %"],
                tablefmt="grid"
            ))
            print()
            
        except Exception as e:
            print(f"{Colors.RED}Error fetching positions: {e}{Colors.RESET}")

    def display_orders(self, limit=10):
        """Display recent orders"""
        try:
            orders = self.trading_client.get_orders(GetOrdersRequest(status="all", limit=limit))
            self.print_header(f"RECENT ORDERS (Latest {limit})")
            
            if not orders:
                print(f"{Colors.YELLOW}No recent orders{Colors.RESET}\n")
                return
            
            orders_data = []
            for o in orders:
                submitted = pd.to_datetime(o.submitted_at).strftime("%H:%M:%S")
                orders_data.append([
                    o.symbol,
                    o.side.upper(),
                    f"{o.qty}",
                    f"{o.filled_qty}",
                    o.filled_avg_price if o.filled_avg_price else "—",
                    o.status.upper(),
                    submitted,
                    str(o.id)[:8] + "..."
                ])
            
            print(tabulate(
                orders_data,
                headers=["Symbol", "Side", "Qty", "Filled", "Fill Price", "Status", "Time", "Order ID"],
                tablefmt="grid"
            ))
            print()
            
        except Exception as e:
            print(f"{Colors.RED}Error fetching orders: {e}{Colors.RESET}")

    def display_dashboard(self):
        """Display main dashboard"""
        self.clear_screen()
        self.print_header("ALGO TRADING DASHBOARD")
        self.print_timestamp()
        
        # Status Bar
        mode = "PAPER" if ALPACA_ENDPOINT == "https://paper-api.alpaca.markets" else "LIVE"
        mode_color = Colors.YELLOW if mode == "PAPER" else Colors.RED
        
        status_line = f"Mode: {mode_color}{mode}{Colors.RESET}"
        if self.strategy:
            status_line += f" | Strategy: {Colors.CYAN}{self.strategy}{Colors.RESET}"
        
        algo_status = "RUNNING" if self.algo_running else "STOPPED"
        algo_color = Colors.GREEN if self.algo_running else Colors.RED
        status_line += f" | Algo: {algo_color}{algo_status}{Colors.RESET}"
        
        print(status_line)
        print(f"{'─'*80}\n")
        
        self.display_account_overview()
        print()
        self.display_positions()
        print()
        self.display_orders()

    def place_order(self, symbol, qty, side):
        """Place market order"""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol.upper(),
                qty=int(qty),
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            order = self.trading_client.submit_order(order_data)
            
            color = Colors.GREEN if side.lower() == "buy" else Colors.RED
            print(f"\n{color}✓ ORDER PLACED: {side.upper()} {qty} {symbol} (ID: {str(order.id)[:8]}...){Colors.RESET}")
            return True
        except Exception as e:
            print(f"\n{Colors.RED}✗ ORDER FAILED: {str(e)}{Colors.RESET}")
            return False

    def cancel_order(self, order_id):
        """Cancel existing order"""
        try:
            self.trading_client.cancel_order(order_id)
            print(f"\n{Colors.YELLOW}✓ ORDER CANCELLED: {order_id[:8]}...{Colors.RESET}\n")
            return True
        except Exception as e:
            print(f"\n{Colors.RED}✗ ERROR: {str(e)}{Colors.RESET}\n")
            return False

    def get_latest_price(self, symbol):
        """Get latest price for symbol"""
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol.upper())
            trade = self.data_client.get_stock_latest_trade(req)
            if isinstance(trade, dict):
                return float(list(trade.values())[0].price)
            return float(trade.trade.price)
        except Exception as e:
            print(f"{Colors.RED}Cannot fetch price for {symbol}: {e}{Colors.RESET}")
            return None

    def execute_algo_logic(self, strategy_name, strategy_config):
        """Execute the actual trading logic based on strategy rules"""
        rules = strategy_config.get("rules", {})
        
        for symbol, rule in rules.items():
            try:
                # Get latest price
                latest_price = self.get_latest_price(symbol)
                if not latest_price:
                    continue
                
                buy_below = rule.get("buy_below")
                sell_above = rule.get("sell_above")
                qty = rule.get("qty", 1)
                
                # Display current status
                status = f"{symbol}: {self.format_currency(latest_price)}"
                if buy_below:
                    status += f" | Buy Below: {self.format_currency(buy_below)}"
                if sell_above:
                    status += f" | Sell Above: {self.format_currency(sell_above)}"
                print(status)
                
                # Buy logic
                if buy_below and latest_price <= buy_below:
                    print(f"{Colors.GREEN}► BUY SIGNAL: {symbol} at {self.format_currency(latest_price)}{Colors.RESET}")
                    if self.place_order(symbol, qty, "buy"):
                        print(f"{Colors.GREEN}✓ Buy order executed for {symbol}{Colors.RESET}")
                
                # Sell logic
                elif sell_above and latest_price >= sell_above:
                    print(f"{Colors.RED}► SELL SIGNAL: {symbol} at {self.format_currency(latest_price)}{Colors.RESET}")
                    if self.place_order(symbol, qty, "sell"):
                        print(f"{Colors.RED}✓ Sell order executed for {symbol}{Colors.RESET}")
                
            except Exception as e:
                print(f"{Colors.RED}Error processing {symbol}: {e}{Colors.RESET}")

    def run_algo(self, strategy_name):
        """Run algo trading"""
        try:
            self.strategy = strategy_name
            strategy_config = self.load_strategy(strategy_name)
            self.algo_running = True
            
            print(f"\n{Colors.GREEN}{'='*80}{Colors.RESET}")
            print(f"{Colors.GREEN}STARTING ALGO: {strategy_name.upper()}{Colors.RESET}")
            print(f"{Colors.GREEN}{'='*80}{Colors.RESET}\n")
            
            # Display strategy info
            print(f"{Colors.CYAN}Strategy: {strategy_name}{Colors.RESET}")
            print(f"{Colors.CYAN}Update Interval: {self.update_interval}s{Colors.RESET}")
            print(f"{Colors.CYAN}Symbols: {', '.join(strategy_config.get('rules', {}).keys())}{Colors.RESET}\n")
            
            iteration = 0
            while self.algo_running:
                iteration += 1
                self.clear_screen()
                self.display_dashboard()
                
                print(f"\n{Colors.BRIGHT}{Colors.CYAN}ALGO TRADING STATUS{Colors.RESET}")
                print(f"{'─'*80}")
                self.print_timestamp()
                print(f"Iteration: #{iteration}")
                print(f"Strategy: {Colors.CYAN}{strategy_name}{Colors.RESET}")
                print(f"{'─'*80}\n")
                
                # Execute strategy logic
                self.execute_algo_logic(strategy_name, strategy_config)
                
                print(f"\n{Colors.YELLOW}⏱  Next check in {self.update_interval} seconds... (Press Ctrl+C to stop){Colors.RESET}")
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Algo Trading stopped by user{Colors.RESET}")
            self.algo_running = False
        except Exception as e:
            print(f"\n{Colors.RED}Algo Error: {e}{Colors.RESET}")
            self.algo_running = False

    def start_algo_background(self, strategy_name):
        """Start algo in background thread (deprecated - now runs in foreground)"""
        # This method is kept for compatibility but algo now runs in foreground
        # for better control and visibility
        pass

    def show_menu(self):
        """Show main menu"""
        try:
            while True:
                self.display_dashboard()

                print(f"\n{Colors.BRIGHT}{Colors.CYAN}╔{'═'*78}╗{Colors.RESET}")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{' '*30}COMMANDS{' '*38}║{Colors.RESET}")
                print(f"{Colors.BRIGHT}{Colors.CYAN}╠{'═'*78}╣{Colors.RESET}")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.WHITE}[1]{Colors.RESET} Refresh Dashboard{' '*53}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.GREEN}[2]{Colors.RESET} Place Buy Order{' '*55}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.RED}[3]{Colors.RESET} Place Sell Order{' '*54}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.YELLOW}[4]{Colors.RESET} Cancel Order{' '*57}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.BLUE}[5]{Colors.RESET} View Strategy Config{' '*49}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.GREEN}[6]{Colors.RESET} Start Algo Trading{' '*51}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.RED}[7]{Colors.RESET} Stop Algo Trading{' '*52}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}║{Colors.RESET}  {Colors.WHITE}[8]{Colors.RESET} Exit{' '*65}║")
                print(f"{Colors.BRIGHT}{Colors.CYAN}╚{'═'*78}╝{Colors.RESET}")
                print()

                choice = input(f"{Colors.CYAN}► Enter command (1-8): {Colors.RESET}").strip()

                if choice == "1":
                    continue

                elif choice == "2":
                    symbol = input(f"{Colors.CYAN}► Symbol (e.g., AAPL): {Colors.RESET}").strip().upper()
                    qty = input(f"{Colors.CYAN}► Quantity: {Colors.RESET}").strip()
                    if symbol and qty and qty.isdigit():
                        self.place_order(symbol, qty, "buy")
                    else:
                        print(f"{Colors.RED}Invalid input{Colors.RESET}")
                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")

                elif choice == "3":
                    symbol = input(f"{Colors.CYAN}► Symbol (e.g., AAPL): {Colors.RESET}").strip().upper()
                    qty = input(f"{Colors.CYAN}► Quantity: {Colors.RESET}").strip()
                    if symbol and qty and qty.isdigit():
                        self.place_order(symbol, qty, "sell")
                    else:
                        print(f"{Colors.RED}Invalid input{Colors.RESET}")
                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")

                elif choice == "4":
                    order_id = input(f"{Colors.CYAN}► Order ID: {Colors.RESET}").strip()
                    if order_id:
                        self.cancel_order(order_id)
                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")

                elif choice == "5":
                    try:
                        active_strategies = self.config.get("strategies", {}).get("active", [])
                        print(f"\n{Colors.BRIGHT}{Colors.CYAN}Available Strategies:{Colors.RESET}")
                        for i, strat in enumerate(active_strategies, 1):
                            print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} {strat}")

                        strat_choice = input(f"\n{Colors.CYAN}► Select strategy (1-{len(active_strategies)}): {Colors.RESET}").strip()

                        if strat_choice.isdigit() and 1 <= int(strat_choice) <= len(active_strategies):
                            strat_name = active_strategies[int(strat_choice) - 1]
                            strat_config = self.load_strategy(strat_name)
                            print(f"\n{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}")
                            print(f"{Colors.BRIGHT}{Colors.YELLOW}{strat_name.upper()} Configuration{Colors.RESET}")
                            print(f"{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
                            print(yaml.dump(strat_config, default_flow_style=False))
                        else:
                            print(f"{Colors.RED}Invalid selection{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")

                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")

                elif choice == "6":
                    if self.algo_running:
                        print(f"\n{Colors.YELLOW}Algo is already running!{Colors.RESET}")
                        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                        continue
                        
                    try:
                        active_strategies = self.config.get("strategies", {}).get("active", [])
                        print(f"\n{Colors.BRIGHT}{Colors.CYAN}Available Strategies:{Colors.RESET}")
                        for i, strat in enumerate(active_strategies, 1):
                            print(f"  {Colors.YELLOW}[{i}]{Colors.RESET} {strat}")

                        strat_choice = input(f"\n{Colors.CYAN}► Select strategy (1-{len(active_strategies)}): {Colors.RESET}").strip()

                        if strat_choice.isdigit() and 1 <= int(strat_choice) <= len(active_strategies):
                            strat_name = active_strategies[int(strat_choice) - 1]
                            print(f"\n{Colors.GREEN}Starting {strat_name}...{Colors.RESET}")
                            print(f"{Colors.YELLOW}Press Ctrl+C in the algo screen to stop{Colors.RESET}")
                            input(f"\n{Colors.CYAN}Press Enter to start...{Colors.RESET}")
                            self.run_algo(strat_name)
                        else:
                            print(f"{Colors.RED}Invalid selection{Colors.RESET}")
                            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")

                elif choice == "7":
                    self.algo_running = False
                    self.strategy = None
                    print(f"\n{Colors.YELLOW}✓ Algo Trading Stopped{Colors.RESET}")
                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")

                elif choice == "8":
                    print(f"\n{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}")
                    print(f"{Colors.GREEN}Thank you for using the Algo Trading System!{Colors.RESET}".center(80))
                    print(f"{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
                    sys.exit(0)

                else:
                    print(f"\n{Colors.RED}Invalid command. Please enter 1-8.{Colors.RESET}")
                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.RESET}")
                    
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully without a traceback
            print(f"\n\n{Colors.YELLOW}Interrupted by user — exiting...{Colors.RESET}\n")
            sys.exit(0)


def main():
    """Main entry point"""
    print(f"\n{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BRIGHT}{Colors.GREEN}TERMINAL ALGO TRADING SYSTEM{Colors.RESET}".center(80))
    print(f"{Colors.BRIGHT}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    # Validate environment
    if not all([ALPACA_KEY, ALPACA_SECRET, ALPACA_ENDPOINT]):
        print(f"{Colors.RED}✗ Error: Missing Alpaca credentials in .env file{Colors.RESET}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Environment loaded successfully{Colors.RESET}")
    print(f"  Mode: {ALPACA_ENDPOINT}\n")
    
    # Initialize trading system
    try:
        trading = TerminalTrading()
        print(f"{Colors.GREEN}✓ Connected to Alpaca{Colors.RESET}\n")
        
        # Start menu
        input(f"{Colors.CYAN}Press Enter to start trading dashboard...{Colors.RESET}")
        trading.show_menu()
        
    except Exception as e:
        print(f"{Colors.RED}✗ Fatal Error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()