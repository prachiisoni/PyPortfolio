"""
Main entry point for Portfolio Optimization Project.
"""

from src.market_data import download_stock_data
from src.statistics import (
    calculate_daily_returns,
    calculate_annual_returns,
    calculate_covariance_matrix,
    calculate_annual_volatility,
)
from src.random_portfolio import generate_random_weights
from src.optimizer import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
)
from src.monte_carlo import simulate_portfolios
from src.portfolio_analysis import (
    get_max_sharpe_portfolio,
    get_min_risk_portfolio,
)
from src.visualization import (
    plot_monte_carlo,
    plot_portfolio_weights,
)
from src.efficient_frontier import (
    optimize_portfolio,
    minimum_variance_portfolio,
)
from src.frontier import generate_efficient_frontier

STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"


def main():

    prices = download_stock_data(
        STOCKS,
        START_DATE,
        END_DATE,
    )

    # Keep stock order consistent
    prices = prices[STOCKS]

    daily_returns = calculate_daily_returns(prices)

    annual_returns = calculate_annual_returns(
        daily_returns,
    )

    covariance_matrix = calculate_covariance_matrix(
        daily_returns,
    )

    annual_volatility = calculate_annual_volatility(
        daily_returns,
    )

    # -------------------------------------------------
    # Random Portfolio
    # -------------------------------------------------

    weights = generate_random_weights(len(STOCKS))

    expected_return = portfolio_return(
        weights,
        annual_returns,
    )

    portfolio_risk = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    portfolio_sharpe = sharpe_ratio(
        expected_return,
        portfolio_risk,
    )

    print("=" * 60)
    print("RANDOM PORTFOLIO")
    print("=" * 60)

    for stock, weight in zip(STOCKS, weights):
        print(f"{stock:<8}: {weight:.2%}")

    print("-" * 60)
    print(f"Weight Sum        : {weights.sum():.2f}")
    print(f"Expected Return   : {expected_return:.2%}")
    print(f"Portfolio Risk    : {portfolio_risk:.2%}")
    print(f"Sharpe Ratio      : {portfolio_sharpe:.3f}")
    print()

    # -------------------------------------------------
    # Monte Carlo Simulation
    # -------------------------------------------------

    portfolios = simulate_portfolios(
        annual_returns,
        covariance_matrix,
        STOCKS,
    )

    best_portfolio = get_max_sharpe_portfolio(
        portfolios,
    )

    lowest_risk_portfolio = get_min_risk_portfolio(
        portfolios,
    )

    # -------------------------------------------------
    # SciPy Optimization
    # -------------------------------------------------

    optimal_portfolio = optimize_portfolio(
        annual_returns,
        covariance_matrix,
    )

    min_variance = minimum_variance_portfolio(
        annual_returns,
        covariance_matrix,
    )

    frontier_risk, frontier_return = generate_efficient_frontier(
        annual_returns,
       covariance_matrix,
    )

    # -------------------------------------------------
    # Monte Carlo Best Sharpe
    # -------------------------------------------------

    print("=" * 60)
    print("BEST SHARPE PORTFOLIO (MONTE CARLO)")
    print("=" * 60)

    print(f"Expected Return : {best_portfolio['Return']:.2%}")
    print(f"Risk            : {best_portfolio['Risk']:.2%}")
    print(f"Sharpe Ratio    : {best_portfolio['Sharpe']:.3f}")

    print("\nWeights")

    for stock, weight in zip(STOCKS, best_portfolio["Weights"]):
        print(f"{stock:<8}: {weight:.2%}")

    print()

    # -------------------------------------------------
    # Monte Carlo Minimum Risk
    # -------------------------------------------------

    print("=" * 60)
    print("MINIMUM RISK PORTFOLIO (MONTE CARLO)")
    print("=" * 60)

    print(f"Expected Return : {lowest_risk_portfolio['Return']:.2%}")
    print(f"Risk            : {lowest_risk_portfolio['Risk']:.2%}")
    print(f"Sharpe Ratio    : {lowest_risk_portfolio['Sharpe']:.3f}")

    print("\nWeights")

    for stock, weight in zip(STOCKS, lowest_risk_portfolio["Weights"]):
        print(f"{stock:<8}: {weight:.2%}")

    print()

    # -------------------------------------------------
    # SciPy Maximum Sharpe
    # -------------------------------------------------

    print("=" * 60)
    print("SCIPY OPTIMAL PORTFOLIO")
    print("=" * 60)

    print(f"Optimization Success : {optimal_portfolio['Success']}")
    print(f"Message              : {optimal_portfolio['Message']}")
    print()

    print(f"Expected Return : {optimal_portfolio['Return']:.2%}")
    print(f"Risk            : {optimal_portfolio['Risk']:.2%}")
    print(f"Sharpe Ratio    : {optimal_portfolio['Sharpe']:.3f}")

    print("\nWeights")

    for stock, weight in zip(STOCKS, optimal_portfolio["Weights"]):
        print(f"{stock:<8}: {weight:.2%}")

    print()

    # -------------------------------------------------
    # SciPy Minimum Variance
    # -------------------------------------------------

    print("=" * 60)
    print("SCIPY MINIMUM VARIANCE PORTFOLIO")
    print("=" * 60)

    print(f"Optimization Success : {min_variance['Success']}")
    print(f"Message              : {min_variance['Message']}")
    print()

    print(f"Expected Return : {min_variance['Return']:.2%}")
    print(f"Risk            : {min_variance['Risk']:.2%}")
    print(f"Sharpe Ratio    : {min_variance['Sharpe']:.3f}")

    print("\nWeights")

    for stock, weight in zip(STOCKS, min_variance["Weights"]):
        print(f"{stock:<8}: {weight:.2%}")

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    plot_monte_carlo(
        portfolios,
        best_portfolio,
        lowest_risk_portfolio,
        frontier_risk,
        frontier_return,
    )

    plot_portfolio_weights(
        best_portfolio["Weights"],
        STOCKS,
        "Monte Carlo - Best Sharpe Portfolio",
    )

    plot_portfolio_weights(
        lowest_risk_portfolio["Weights"],
        STOCKS,
        "Monte Carlo - Minimum Risk Portfolio",
    )

    plot_portfolio_weights(
        optimal_portfolio["Weights"],
        STOCKS,
        "SciPy - Maximum Sharpe Portfolio",
    )

    plot_portfolio_weights(
        min_variance["Weights"],
        STOCKS,
        "SciPy - Minimum Variance Portfolio",
    )

if __name__ == "__main__":
    main()