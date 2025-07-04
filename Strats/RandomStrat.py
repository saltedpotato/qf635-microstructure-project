from Utils.import_files import *

class RandomStrat:
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

    def random_strategy(self, lookback=100, hold_period=30, seed=42):
        """
        Random strategy:
        - Randomly assigns long (1), short (-1), or flat (0) positions every `hold_period`.
        - Signals are held constant for `hold_period` periods.

        Returns:
        - DataFrame: Same format as simple_strategy() with price, signal, and exit_signal columns
        """
        np.random.seed(seed)
        n = len(self.df)
        signals = [np.array([0 for _ in self.tickers])] * lookback
        exit_signals = [np.array([0 for _ in self.tickers])] * (lookback + hold_period - 1)

        for i in range(lookback, n, hold_period):
            # Generate random signals (-1, 0, 1)
            random_signals = np.random.choice([-1, 0, 1], size=len(self.tickers))

            # Create signals for hold_period
            signals += [random_signals] + [np.array([0 for _ in self.tickers])] * (hold_period - 1)
            
            # Exit signal at start of new period
            exit_signal_temp = np.array([1 for _ in self.tickers])
            exit_signals += [exit_signal_temp] + [np.array([0 for _ in self.tickers])] * (hold_period - 1)

        # Format signal DataFrame
        signals_df = pd.DataFrame(signals[:len(self.df)], columns=[f"{t}_signal" for t in self.tickers])
        signals_df['timestamp'] = self.df['timestamp']
        self.df = pd.merge(self.df, signals_df, on='timestamp')

        # Format exit signal DataFrame
        exit_signals_df = pd.DataFrame(exit_signals[:len(self.df)], columns=[f"{t}_exit_signal" for t in self.tickers])
        exit_signals_df['timestamp'] = self.df['timestamp']
        self.df = pd.merge(self.df, exit_signals_df, on='timestamp')

        return self.df
