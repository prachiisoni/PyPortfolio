from src.data_loader import download_stock_data

stocks = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL"
]

prices = download_stock_data(
    stocks,
    "2020-01-01",
    "2025-01-01"
)

print(prices.head())