"""Portfolio tracker Streamlit app converted from
`TradingBasics/portofolio_tracker.ipynb`.

Run with:
    streamlit run portofolio/portofolio.py

The app shows sample holdings, transactions and dividends, computes
summary statistics and allows downloading an Excel export with multiple
sheets.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Tuple

import pandas as pd
import plotly.express as px

try:
    import streamlit as st
except Exception:
    st = None


def _sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings_data = {
        'Ticker': ['AAPL', 'MSFT', 'SPY', 'BTC-USD'],
        'Asset_Name': ['Apple Inc.', 'Microsoft Corp.', 'SPDR S&P 500 ETF', 'Bitcoin'],
        'Asset_Type': ['Stock', 'Stock', 'ETF', 'Crypto'],
        'Shares_Owned': [100, 50, 200, 0.5],
        'Purchase_Date': ['2023-01-15', '2023-06-20', '2022-12-01', '2024-03-10'],
        'Purchase_Price': [150.00, 300.00, 400.00, 40000.00],
        'Current_Price': [220.00, 350.00, 450.00, 60000.00],
    }
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df['Total_Cost_Basis'] = holdings_df['Shares_Owned'] * \
        holdings_df['Purchase_Price']
    holdings_df['Current_Value'] = holdings_df['Shares_Owned'] * \
        holdings_df['Current_Price']
    holdings_df['Unrealized_Gain_Loss'] = holdings_df['Current_Value'] - \
        holdings_df['Total_Cost_Basis']
    holdings_df['Unrealized_Gain_Loss_Percent'] = (
        holdings_df['Unrealized_Gain_Loss'] / holdings_df['Total_Cost_Basis']) * 100
    holdings_df['Dividends_Received'] = [100.00, 75.00, 600.00, 0.00]
    holdings_df['Sector'] = ['Technology',
                             'Technology', 'Broad Market', 'Cryptocurrency']
    holdings_df['Brokerage'] = ['Fidelity', 'Schwab', 'Vanguard', 'Coinbase']
    holdings_df['Currency'] = ['USD', 'USD', 'USD', 'USD']

    transactions_data = {
        'Date': ['2023-01-15', '2023-06-20', '2022-12-01', '2024-03-10'],
        'Ticker': ['AAPL', 'MSFT', 'SPY', 'BTC-USD'],
        'Transaction_Type': ['Buy', 'Buy', 'Buy', 'Buy'],
        'Quantity': [100, 50, 200, 0.5],
        'Price_per_Share': [150.00, 300.00, 400.00, 40000.00],
        'Total_Amount': [15000.00, 15000.00, 80000.00, 20000.00],
        'Fees': [0.00, 5.00, 0.00, 50.00],
        'Notes': ['Initial purchase', '', 'Rebalance', '']
    }
    transactions_df = pd.DataFrame(transactions_data)

    dividends_data = {
        'Date': ['2023-12-15', '2024-01-10'],
        'Ticker': ['AAPL', 'SPY'],
        'Amount': [100.00, 600.00],
        'Reinvested': ['No', 'Yes'],
        'Tax_Withheld': [10.00, 60.00]
    }
    dividends_df = pd.DataFrame(dividends_data)

    return holdings_df, transactions_df, dividends_df


def build_summary(holdings_df: pd.DataFrame, dividends_df: pd.DataFrame) -> pd.DataFrame:
    summary_data = {
        'Metric': [
            'Total Portfolio Value',
            'Total Cost Basis',
            'Total Unrealized Gain/Loss',
            'Total Dividends Received',
            'Last Updated'
        ],
        'Value': [
            holdings_df['Current_Value'].sum(),
            holdings_df['Total_Cost_Basis'].sum(),
            holdings_df['Unrealized_Gain_Loss'].sum(),
            dividends_df['Amount'].sum() if not dividends_df.empty else 0.0,
            datetime.now().strftime('%Y-%m-%d')
        ]
    }
    return pd.DataFrame(summary_data)


def to_excel_bytes(holdings: pd.DataFrame, transactions: pd.DataFrame, dividends: pd.DataFrame, summary: pd.DataFrame) -> Tuple[bytes, str, str]:
    """Return a bytes object containing an Excel workbook (or CSV fallback),
    plus the appropriate MIME type and filename.
    """
    buffer = io.BytesIO()

    # Pick an available engine: prefer xlsxwriter, then openpyxl. If none
    # available, fall back to CSV bytes.
    engine = None
    try:
        import xlsxwriter  # type: ignore
        engine = 'xlsxwriter'
    except Exception:
        try:
            import openpyxl  # type: ignore
            engine = 'openpyxl'
        except Exception:
            engine = None

    if engine is None:
        # Fallback: return holdings as CSV
        csv_bytes = holdings.to_csv(index=False).encode('utf-8')
        return csv_bytes, 'text/csv', 'Portfolio_Tracker.csv'

    try:
        with pd.ExcelWriter(buffer, engine=engine) as writer:
            holdings.to_excel(writer, sheet_name='Holdings', index=False)
            transactions.to_excel(
                writer, sheet_name='Transactions', index=False)
            dividends.to_excel(writer, sheet_name='Dividends', index=False)
            summary.to_excel(writer, sheet_name='Summary', index=False)

            # If using xlsxwriter, add a simple header format for Holdings
            if engine == 'xlsxwriter':
                try:
                    workbook = writer.book
                    worksheet = writer.sheets['Holdings']
                    header_format = workbook.add_format(
                        {'bold': True, 'bg_color': '#4F81BD', 'font_color': '#FFFFFF'})
                    for col_num, _ in enumerate(holdings.columns.values):
                        worksheet.write(
                            0, col_num, holdings.columns.values[col_num], header_format)
                except Exception:
                    # non-fatal: formatting failed
                    pass

        return buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'Portfolio_Tracker.xlsx'
    except Exception:
        # Final fallback to CSV
        csv_bytes = holdings.to_csv(index=False).encode('utf-8')
        return csv_bytes, 'text/csv', 'Portfolio_Tracker.csv'


def plot_allocation_pie(holdings: pd.DataFrame):
    """Return a Plotly pie chart figure for allocation by asset type.

    The figure is sized to a reasonable default height to avoid layout issues
    when embedded in Streamlit.
    """
    allocation = holdings.groupby('Asset_Type')[
        'Current_Value'].sum().reset_index()
    fig = px.pie(allocation, names='Asset_Type', values='Current_Value',
                 title='Allocation by Asset Type', hole=0.0)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    if st is None:
        raise RuntimeError(
            'Streamlit is required to run this app. Install with `pip install streamlit`.')

    st.set_page_config(page_title='Portfolio Tracker', layout='wide')
    st.title('Portfolio Tracker — quick UI')

    # Sidebar controls
    with st.sidebar:
        st.header('Data')
        use_sample = st.checkbox('Use sample data', value=True)
        run = st.button('Run')

    if not run:
        st.write('Use the sidebar and press Run to prepare the portfolio view.')
        return

    if use_sample:
        holdings, transactions, dividends = _sample_data()
    else:
        st.error('Only sample data is supported in this lightweight app.')
        return

    summary = build_summary(holdings, dividends)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader('Holdings')
        st.dataframe(holdings)

    with col2:
        st.subheader('Summary')
        st.table(summary)
        st.markdown('---')
        # Allocation plot intentionally shown after the main tables to
        # avoid disrupting the page layout. It will appear below.

    st.subheader('Transactions')
    st.dataframe(transactions)

    st.subheader('Dividends')
    st.dataframe(dividends)

    # Allocation plot (showed after tables to preserve layout)
    st.subheader('Allocation')
    fig = plot_allocation_pie(holdings)
    st.plotly_chart(fig, use_container_width=True)

    # Excel download
    excel_bytes, mime, filename = to_excel_bytes(
        holdings, transactions, dividends, summary)
    # If we returned CSV bytes (fallback), Streamlit will use the provided mime and filename
    st.download_button('Download portfolio Excel',
                       data=excel_bytes, file_name=filename, mime=mime)


if __name__ == '__main__':
    if st is None:
        print(
            'Streamlit is not installed. Install via `pip install streamlit` to run the UI.')
    else:
        main()
