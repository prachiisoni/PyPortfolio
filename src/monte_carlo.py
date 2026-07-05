"""
monte_carlo.py

Monte Carlo simulation for portfolio optimization.
"""

import numpy as np

from src.random_portfolio import generate_random_weights
from src.optimizer import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)


def simulate_portfolios(
    annual_returns,
    covariance_matrix,
    stock_names,
    num_portfolios=10000,
):
    """
    Simulate thousands of random portfolios.

    Returns
    -------
    list
        List of portfolio dictionaries.
    """

    portfolios = []

    for _ in range(num_portfolios):

        weights = generate_random_weights(len(stock_names))

        expected_return = portfolio_return(
            weights,
            annual_returns,
        )

        risk = portfolio_volatility(
            weights,
            covariance_matrix,
        )

        sharpe = sharpe_ratio(
            expected_return,
            risk,
        )

        portfolios.append(
            {
                "Return": expected_return,
                "Risk": risk,
                "Sharpe": sharpe,
                "Weights": weights,
            }
        )

    return portfolios