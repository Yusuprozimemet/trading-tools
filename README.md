# Trading Tools

Interactive Python toolkit for trading strategy prototyping, backtesting, and visualization using Streamlit.

## Features
- Pluggable strategy modules
- Real-time and historical data analysis
- Interactive charts and metrics
- Sidebar controls for tickers, parameters, and strategies

## Requirements
- Python 3.11+
- Streamlit
- yfinance
- pandas, numpy, plotly, matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
````

## Quickstart

1. Clone the repository:

```bash
git clone https://github.com/Yusuprozimemet/trading-tools.git
cd trading-tools
```

2. Run the Streamlit app:

```bash
streamlit run run.py
```

3. Use the sidebar to select tickers, strategies, and parameters. The app will generate charts and backtest results interactively.

## Project Structure

```
trading-tools/
├─ run.py             # Main Streamlit entry point
├─ strategies/        # Trading strategy modules
├─ finance/           # Data and computation utilities
├─ chart/             # Plotting functions
├─ alert/             # Alerts and notifications
├─ docs/              # Documentation and examples
├─ requirements.txt
└─ LICENSE
```

## Contributing

* Add new strategies in the `strategies/` folder
* Update visualizations in `chart/`
* Ensure new features are compatible with Streamlit UI
* Submit pull requests with clear descriptions

### How to contribute (recommended)

1. Fork the repository on GitHub: https://github.com/Yusuprozimemet/trading-tools
2. Create a feature branch: `git checkout -b feat/my-page`
3. Add your page/script under the project (for example `alert/` or `strategies/`).
4. Optionally add a display name mapping in `streamlit_pages.json` at the repo root to control
	how the page appears in the runner dropdown.
5. Commit and push your branch, then open a Pull Request describing your changes.

If you prefer, open an Issue first to discuss larger changes before submitting a PR.

## License

Apache-2.0 License

