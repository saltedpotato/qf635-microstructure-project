from Utils.import_files import *

class MomentumStrat:
    def __init__(self, data):
        """
        Initialize the portfolio strategy.

        Parameters:
        - tickers (list): List of stock tickers (e.g., ['AAPL', 'MSFT']).
        - weights (np array): Portfolio weights (e.g., [0.5, 0.5] for 50% AAPL, 50% MSFT).
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        """
        self.tickers = data.columns.tolist()[1:]
        self.data = data
        self.df_train = data[self.tickers]

    def momentum_strategy(self, lookback=20, hold_period=5):
        """
        Simple momentum strategy:
        - Rank stocks by past `lookback`-day returns.
        - Invest in top-performing stocks for `hold_period` days.

        Returns:
        - tuple: (strategy_df, weights_array) where:
            strategy_df: DataFrame containing price, signal, and exit_signal for each ticker
            weights_array: numpy array of weights for each period
        """
        # Calculate returns and momentum
        returns = self.df_train.pct_change().dropna()
        momentum = self.df_train.pct_change(lookback).dropna()

        # Align the indices
        valid_dates = returns.index.intersection(momentum.index)
        returns = returns.loc[valid_dates]
        momentum = momentum.loc[valid_dates]

        # Initialize output DataFrames and arrays
        signal_df = pd.DataFrame(index=returns.index, columns=self.tickers)
        exit_signal_df = pd.DataFrame(index=returns.index, columns=self.tickers)
        
        # Ensure we have enough data
        if len(returns) <= lookback:
            raise ValueError("Not enough data points for the given lookback period")

        for i in range(lookback, len(returns) - hold_period + 1, hold_period):
            try:
                # Rank stocks by momentum and select top half
                current_momentum = momentum.iloc[i]
                top_stocks = current_momentum.nlargest(len(self.tickers) // 2).index.tolist()

                # Generate signals
                for j in range(hold_period):
                    current_idx = i + j
                    if current_idx >= len(returns):
                        break
                    
                    # Entry signal (1 for buy at start of period)
                    if j == 0:
                        signal_df.iloc[current_idx] = [1 if ticker in top_stocks else 0 for ticker in self.tickers]
                    else:
                        signal_df.iloc[current_idx] = 0
                    
                    # Exit signal (1 for sell at end of period)
                    if j == (hold_period - 1):
                        exit_signal_df.iloc[current_idx] = [1 if ticker in top_stocks else 0 for ticker in self.tickers]
                    else:
                        exit_signal_df.iloc[current_idx] = 0

            except IndexError:
                # Handle cases where we reach the end of the data
                break
        
        # Combine all data into output DataFrame
        strategy_df = self.data[["timestamp"] + self.tickers]
        for ticker in self.tickers:
            strategy_df[ticker+'_signal'] = signal_df[ticker].fillna(0)
            strategy_df[ticker+'_exit_signal'] = exit_signal_df[ticker].fillna(0)    
        
        return strategy_df