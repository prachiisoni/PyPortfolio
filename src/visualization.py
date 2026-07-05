"""
visualization.py

Visualization functions.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_monte_carlo(
    portfolios,
    best_portfolio,
    lowest_risk_portfolio,
    frontier_risk=None,
    frontier_return=None,
):
    """
    Interactive Monte Carlo visualization.
    """

    df = pd.DataFrame(
        {
            "Risk": [p["Risk"] for p in portfolios],
            "Return": [p["Return"] for p in portfolios],
            "Sharpe": [p["Sharpe"] for p in portfolios],
        }
    )

    fig = px.scatter(
        df,
        x="Risk",
        y="Return",
        color="Sharpe",
        color_continuous_scale="Viridis",
        title="Monte Carlo Portfolio Simulation",
        labels={
            "Risk": "Annual Risk (Volatility)",
            "Return": "Annual Expected Return",
            "Sharpe": "Sharpe Ratio",
        },
        hover_data={
            "Risk": ":.2%",
            "Return": ":.2%",
            "Sharpe": ":.3f",
        },
    )

    # Best Sharpe Portfolio
    fig.add_trace(
        go.Scatter(
            x=[best_portfolio["Risk"]],
            y=[best_portfolio["Return"]],
            mode="markers",
            marker=dict(
                size=18,
                color="gold",
                symbol="star",
                line=dict(color="black", width=2),
            ),
            name="Best Sharpe",
        )
    )

    # Minimum Risk Portfolio
    fig.add_trace(
        go.Scatter(
            x=[lowest_risk_portfolio["Risk"]],
            y=[lowest_risk_portfolio["Return"]],
            mode="markers",
            marker=dict(
                size=15,
                color="red",
                symbol="diamond",
                line=dict(color="black", width=2),
            ),
            name="Minimum Risk",
        )
    )

    # Efficient Frontier
    if frontier_risk is not None:

        fig.add_trace(
            go.Scatter(
                x=frontier_risk,
                y=frontier_return,
                mode="lines",
                line=dict(
                    color="red",
                    width=4,
                ),
                name="Efficient Frontier",
            )
        )

    fig.update_layout(
        width=1100,
        height=700,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    fig.show()

def plot_portfolio_weights(
    weights,
    stocks,
    title,
):
    """
    Plot portfolio allocation as a pie chart.
    """

    fig = px.pie(
        names=stocks,
        values=weights,
        title=title,
        hole=0.4,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        width=700,
        height=600,
        template="plotly_white",
    )

    fig.show()