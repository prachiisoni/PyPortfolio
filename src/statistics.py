"""
statistics.py

Statistical calculations for portfolio optimization.
"""

import pandas as pd
import numpy as np
from src.config import TRADING_DAYS


def calculate_daily_returns(prices):
    """
    Calculate percentage daily returns.
    """
    returns = prices.pct_change()

    returns = returns.dropna()

    return returns


def calculate_annual_returns(daily_returns):
    """
    Calculate annualized returns.

    Formula:
    mean_daily_return × TRADING_DAYS
    """

    return daily_returns.mean() * TRADING_DAYS


def calculate_covariance_matrix(daily_returns):
    """
    Annualized covariance matrix.
    """

    return daily_returns.cov() * TRADING_DAYS


def calculate_annual_volatility(daily_returns):
    """
    Annualized standard deviation.

    Formula:
    daily standard deviation × sqrt(TRADING_DAYS)
    """

    return daily_returns.std() * np.sqrt(TRADING_DAYS)