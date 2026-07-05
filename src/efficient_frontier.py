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
    Objective function for maximizing Sharpe Ratio.
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


def portfolio_variance(
    weights,
    covariance_matrix,
):
    """
    Portfolio variance.
    """

    return np.dot(
        weights.T,
        np.dot(covariance_matrix, weights),
    )


def optimize_portfolio(
    annual_returns,
    covariance_matrix,
):
    """
    Find portfolio with maximum Sharpe Ratio.
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
            "fun": lambda w: np.sum(w) - 1,
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

    weights = result.x

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

    return {
        "Weights": weights,
        "Return": expected_return,
        "Risk": risk,
        "Sharpe": sharpe,
        "Success": result.success,
        "Message": result.message,
    }


def minimum_variance_portfolio(
    annual_returns,
    covariance_matrix,
):
    """
    Find the Global Minimum Variance Portfolio.
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
            "fun": lambda w: np.sum(w) - 1,
        },
    )

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x

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

    return {
        "Weights": weights,
        "Return": expected_return,
        "Risk": risk,
        "Sharpe": sharpe,
        "Success": result.success,
        "Message": result.message,
    }