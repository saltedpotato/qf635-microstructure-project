from Data.BinancePriceFetcher import *

class SimpleStrategy:
    def __init__(self, tickers, weights, data):
        """
        Initialize the portfolio strategy.

        Parameters:
        - tickers (list): List of stock tickers (e.g., ['AAPL', 'MSFT']).
        - weights (np array): Portfolio weights (e.g., [0.5, 0.5] for 50% AAPL, 50% MSFT).
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        """
        self.tickers = tickers
        self.weights = np.array(weights)
        self.data = data[tickers]

        if len(tickers) != len(weights):
            raise Exception("Size of tickers and weights do not match.")

        if weights.sum() != 1:
            raise Exception("Weights do not sum to 1.")

    def momentum_strategy(self, lookback=20, hold_period=5):
        """
        Simple momentum strategy:
        - Rank stocks by past `lookback`-day returns.
        - Invest in top-performing stocks for `hold_period` days.

        Returns:
        - pd.Series: Portfolio returns.
        """
        # Calculate returns and momentum
        returns = self.data.pct_change().dropna()
        momentum = self.data.pct_change(lookback).dropna()

        # Align the indices
        valid_dates = returns.index.intersection(momentum.index)
        returns = returns.loc[valid_dates]
        momentum = momentum.loc[valid_dates]

        portfolio_returns = pd.Series(index=returns.index, dtype=float)

        # Ensure we have enough data
        if len(returns) <= lookback:
            raise ValueError("Not enough data points for the given lookback period")

        for i in range(lookback, len(returns) - hold_period + 1, hold_period):
            try:
                # Rank stocks by momentum and select top half
                current_momentum = momentum.iloc[i]
                top_stocks = current_momentum.nlargest(len(self.tickers) // 2).index.tolist()

                # Create weights: 1/N for top stocks, 0 otherwise
                weights = np.array([
                    1 / len(top_stocks) if ticker in top_stocks else 0
                    for ticker in self.tickers
                ])

                # Apply weights to the next `hold_period` days
                period_returns = returns.iloc[i:i + hold_period]
                portfolio_returns.iloc[i:i + hold_period] = (period_returns * weights).sum(axis=1)

            except IndexError:
                # Handle cases where we reach the end of the data
                break
        portfolio_returns = pd.Series(portfolio_returns, name='Simple Momentum')
        portfolio_returns = pd.DataFrame(portfolio_returns)
        return portfolio_returns.dropna()


# Example Usage
if __name__ == "__main__":
    # Define portfolio
    symbol_manager = BinanceSymbolManager()

    # Add symbols
    print(symbol_manager.add_symbol("BTCUSDT"))  # Success
    print(symbol_manager.add_symbol("ETHUSDT"))  # Success
    print(symbol_manager.add_symbol("INVALID"))  # Will add but fail in API

    tickers = symbol_manager.get_symbols()
    print(tickers)

    price_fetcher = BinancePriceFetcher(tickers)
    btc_portfolio_daily = price_fetcher.get_grp_historical_ohlcv(
        interval="1d",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    weights = np.array([0.5, 0.5])  # Equal-weighted


    # Initialize strategy
    strategy = SimpleStrategy(
        tickers=tickers,
        weights=weights,
        data=btc_portfolio_daily
    )
    print(btc_portfolio_daily.shape)
    # Get momentum strategy returns
    momentum_returns = strategy.momentum_strategy(lookback=90, hold_period=30)
    print("\nMomentum Strategy Returns (Head):\n", momentum_returns.head())