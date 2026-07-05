"""
optimizer.py

Portfolio optimization functions.
"""

import numpy as np
from src.config import RISK_FREE_RATE


def portfolio_return(weights, annual_returns):
    """
    Calculate expected portfolio return.
    """
    return np.sum(weights * annual_returns)


def portfolio_volatility(weights, covariance_matrix):
    """
    Calculate portfolio volatility.
    """
    return np.sqrt(
        np.dot(
            weights.T,
            np.dot(covariance_matrix, weights)
        )
    )


def sharpe_ratio(portfolio_return, portfolio_volatility):
    """
    Calculate Sharpe Ratio.

    Sharpe = (Return - Risk Free Rate) / Volatility
    """

    return (portfolio_return - RISK_FREE_RATE) / portfolio_volatility