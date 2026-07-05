"""
portfolio_analysis.py

Functions for analyzing simulated portfolios.
"""


def get_max_sharpe_portfolio(portfolios):
    """
    Return the portfolio with the highest Sharpe Ratio.
    """
    return max(portfolios, key=lambda portfolio: portfolio["Sharpe"])


def get_min_risk_portfolio(portfolios):
    """
    Return the portfolio with the lowest volatility.
    """
    return min(portfolios, key=lambda portfolio: portfolio["Risk"])