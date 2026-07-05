"""
Generate Efficient Frontier.
"""

import numpy as np
from scipy.optimize import minimize

from src.optimizer import (
    portfolio_return,
    portfolio_volatility,
)


def minimum_volatility_for_return(
    target_return,
    annual_returns,
    covariance_matrix,
):
    """
    Find the minimum-volatility portfolio
    for a given target return.
    """

    number_of_assets = len(annual_returns)

    initial_weights = np.ones(number_of_assets) / number_of_assets

    bounds = tuple((0, 1) for _ in range(number_of_assets))

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        },
        {
            "type": "eq",
            "fun": lambda w: portfolio_return(
                w,
                annual_returns,
            ) - target_return,
        },
    )

    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def generate_efficient_frontier(
    annual_returns,
    covariance_matrix,
    points=50,
):
    """
    Generate Efficient Frontier.
    """

    target_returns = np.linspace(
        annual_returns.min(),
        annual_returns.max(),
        points,
    )

    frontier_risk = []
    frontier_return = []

    for target in target_returns:

        result = minimum_volatility_for_return(
            target,
            annual_returns,
            covariance_matrix,
        )

        if result.success:

            risk = portfolio_volatility(
                result.x,
                covariance_matrix,
            )

            frontier_risk.append(risk)
            frontier_return.append(target)

    return frontier_risk, frontier_return