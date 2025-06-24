from Utils.import_files import *

class SimpleStrat:
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
        self.df = data
        self.df_train = data[self.tickers]

    def simple_strategy(self, lookback=100, hold_period=30):
        """
        Simple momentum strategy:
        - Rank stocks by past `lookback`-period returns.
        - Invest in top-performing stocks for `hold_period` periods.

        Returns:
        - tuple: (strategy_df, weights_array) where:
            strategy_df: DataFrame containing price, signal, and exit_signal for each ticker
            weights_array: numpy array of weights for each period
        """
        # Calculate returns and momentum
        returns = self.df_train.pct_change().dropna()
        signals = [np.array([0 for i in self.tickers])] * lookback
        exit_signals = [np.array([0 for i in self.tickers])] * (lookback+hold_period-1)
        for i in range(lookback, len(returns) + 1, hold_period):
            lookback_df = returns[i-lookback:i].dropna()

            # Rank stocks by sharpe and select top half
            if len(self.tickers) > 1:
                topn_tickers = (lookback_df.mean() / lookback_df.std()).nlargest(len(self.tickers) // 2).index.tolist()
                bottomn_tickers = (lookback_df.mean() / lookback_df.std()).nlargest(len(self.tickers) // 2, keep='last').index.tolist()

            signal_temp = []
            signal_exit_temp = []
            for t in self.tickers:
                if t in topn_tickers:
                    signal_temp += [1]
                elif t in bottomn_tickers:
                    signal_temp += [-1]
                else:
                    signal_temp += [0]
                
                if (i-lookback) % hold_period == 0:
                    signal_exit_temp += [1]
                         
            signals += [np.array(signal_temp)] + [np.array([0 for i in self.tickers])] * (hold_period - 1)
            exit_signals += [np.array(signal_exit_temp)] + [np.array([0 for i in self.tickers])] * (hold_period - 1)
        
        signals_df = pd.DataFrame(signals, columns=[f"{t}_signal" for t in self.tickers])[:len(self.df)]
        signals_df['timestamp'] = self.df['timestamp']
        self.df = pd.merge(self.df, signals_df, on=['timestamp'])

        signals_exit_df = pd.DataFrame(exit_signals, columns=[f"{t}_exit_signal" for t in self.tickers])[:len(self.df)]
        signals_exit_df['timestamp'] = self.df['timestamp']
        self.df = pd.merge(self.df, signals_exit_df, on=['timestamp'])

        return self.df