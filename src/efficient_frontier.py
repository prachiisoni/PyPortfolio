"""
efficient_frontier.py

Optimization functions using SciPy.
"""

import numpy as np
from scipy.optimize import minimize

from src.optimizer import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)


def negative_sharpe_ratio(
    weights,
    annual_returns,
    covariance_matrix,
):
    """
    Objective function.

    We minimize negative Sharpe
    because scipy only minimizes.
    """

    expected_return = portfolio_return(
        weights,
        annual_returns,
    )

    risk = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    return -sharpe_ratio(
        expected_return,
        risk,
    )


def optimize_portfolio(
    annual_returns,
    covariance_matrix,
):
    """
    Find the portfolio with the maximum Sharpe Ratio.
    """

    number_of_assets = len(annual_returns)

    initial_weights = np.ones(number_of_assets) / number_of_assets

    bounds = tuple(
        (0, 1)
        for _ in range(number_of_assets)
    )

    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1,
        },
    )

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(
            annual_returns,
            covariance_matrix,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal_weights = result.x

    expected_return = portfolio_return(
        optimal_weights,
        annual_returns,
    )

    risk = portfolio_volatility(
        optimal_weights,
        covariance_matrix,
    )

    sharpe = sharpe_ratio(
        expected_return,
        risk,
    )

    return {
        "Weights": optimal_weights,
        "Return": expected_return,
        "Risk": risk,
        "Sharpe": sharpe,
        "Success": result.success,
        "Message": result.message,
    }