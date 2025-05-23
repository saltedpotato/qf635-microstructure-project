from Utils.import_files import *
from scipy.stats import skew, kurtosis

from Strats.SimpleStrat import *


class PortfolioMetrics:
    def __init__(self, returns):
        """
        Initialize the metrics calculator with portfolio returns.

        Parameters:
        - returns: pd.DataFrame or pd.Series containing return data
                   (if DataFrame, each column represents a different asset/strategy)
        """
        if isinstance(returns, pd.Series):
            self.returns = returns.to_frame(name='Returns')
        else:
            self.returns = returns.copy()

        # Convert returns to float if not already
        self.returns = self.returns.astype(float)

    def annualized_return(self, periods_per_year=252):
        """
        Calculate annualized return.

        Parameters:
        - periods_per_year: trading days/year (252 for daily, 12 for monthly)
        """
        compounded_growth = (1 + self.returns).prod()
        n_periods = len(self.returns)
        return compounded_growth ** (periods_per_year / n_periods) - 1

    def annualized_volatility(self, periods_per_year=252):
        """
        Calculate annualized volatility (standard deviation).
        """
        return self.returns.std() * np.sqrt(periods_per_year)

    def sharpe_ratio(self, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculate Sharpe ratio.
        """
        excess_returns = self.returns - risk_free_rate / periods_per_year
        return (excess_returns.mean() * periods_per_year) / self.annualized_volatility(periods_per_year)

    def max_drawdown(self):
        """
        Calculate maximum drawdown.
        """
        wealth_index = (1 + self.returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    def calmar_ratio(self, periods_per_year=252):
        """
        Calculate Calmar ratio (return vs max drawdown).
        """
        return self.annualized_return(periods_per_year) / abs(self.max_drawdown())

    def skewness(self):
        """
        Calculate return skewness.
        """
        return skew(self.returns)

    def kurtosis(self):
        """
        Calculate return kurtosis.
        """
        return kurtosis(self.returns)

    def value_at_risk(self, level=5):
        """
        Calculate Value at Risk (VaR) at specified confidence level.
        """
        return np.percentile(self.returns, level)

    def conditional_var(self, level=5):
        """
        Calculate Conditional Value at Risk (CVaR).
        """
        var = self.value_at_risk(level)
        return self.returns[self.returns <= var].mean()

    def tail_ratio(self, level=5):
        """
        Calculate Tail Ratio (ratio of right tail to left tail).
        """
        right_tail = abs(np.percentile(self.returns, 100 - level))
        left_tail = abs(np.percentile(self.returns, level))
        return right_tail / left_tail

    def omega_ratio(self, threshold=0.0):
        """
        Calculate Omega ratio.
        """
        excess = self.returns - threshold
        return excess[excess > 0].sum() / abs(excess[excess < 0].sum())

    def summary(self, periods_per_year=252, risk_free_rate=0.0):
        """
        Generate comprehensive performance summary.
        """
        metrics = {
            'Annualized Return': self.annualized_return(periods_per_year),
            'Annualized Volatility': self.annualized_volatility(periods_per_year),
            'Sharpe Ratio': self.sharpe_ratio(risk_free_rate, periods_per_year),
            'Max Drawdown': self.max_drawdown(),
            'Calmar Ratio': self.calmar_ratio(periods_per_year),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            f'VaR ({5}%)': self.value_at_risk(5),
            f'CVaR ({5}%)': self.conditional_var(5),
            'Tail Ratio': self.tail_ratio(5),
            'Omega Ratio': self.omega_ratio(0.0)
        }

        return pd.DataFrame(metrics, index=self.returns.columns)

if __name__ == "__main__":
    # Define portfolio
    symbol_manager = BinanceSymbolManager()
    price_fetcher = BinancePriceFetcher(symbol_manager)

    # Add symbols
    print(symbol_manager.add_symbol("BTCUSDT"))  # Success
    print(symbol_manager.add_symbol("ETHUSDT"))  # Success

    btc_portfolio_daily = price_fetcher.get_grp_historical_ohlcv(
        interval="1d",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    weights = np.array([0.5, 0.5])  # Equal-weighted

    # Initialize strategy
    strategy = SimpleStrategy(
        tickers=symbol_manager.get_symbols(),
        weights=weights,
        data=btc_portfolio_daily
    )

    returns = strategy.momentum_strategy(lookback=90, hold_period=30)

    # Initialize metrics calculator
    metrics = PortfolioMetrics(returns)

    # Get individual metrics
    print("Annualized Returns:\n", metrics.annualized_return())
    print("\nSharpe Ratios:\n", metrics.sharpe_ratio())

    # Get full summary
    print("\nPerformance Summary:")
    summary = metrics.summary(risk_free_rate=0.02)  # 2% risk-free rate
    print(summary)