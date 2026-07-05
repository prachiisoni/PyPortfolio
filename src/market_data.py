"""
data_loader.py

Download historical stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd


def download_stock_data(tickers, start_date, end_date):
    """
    Download adjusted closing prices.

    Parameters
    ----------
    tickers : list
    start_date : str
    end_date : str

    Returns
    -------
    pandas.DataFrame
    """

    prices = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    # Remove stocks that completely failed to download
    prices = prices.dropna(axis=1, how="all")

    # Remove dates with missing values
    prices = prices.dropna()

    return prices