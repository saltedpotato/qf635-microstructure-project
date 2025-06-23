from scipy.stats import skew, kurtosis

from Strats.SimpleStrat import *


class PortfolioMetrics:
    def __init__(self, returns, periods_per_year=252):
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
        self.periods_per_year = periods_per_year

    def annualized_return(self):
        """
        Calculate annualized return.

        Parameters:
        - periods_per_year: trading days/year (252 for daily, 12 for monthly)
        """
        try:
            compounded_growth = float((1 + self.returns).prod())
        except:
            compounded_growth = (1 + self.returns).prod()
        n_periods = len(self.returns)
        return compounded_growth ** (self.periods_per_year / n_periods) - 1

    def annualized_volatility(self):
        """
        Calculate annualized volatility (standard deviation).
        """
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio.
        """
        excess_returns = self.returns - risk_free_rate / self.periods_per_year
        return (excess_returns.mean() * self.periods_per_year) / self.annualized_volatility()

    def max_drawdown(self):
        """
        Calculate maximum drawdown.
        """
        wealth_index = (1 + self.returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    def max_drawdown_duration(self):
        """
        Calculate maximum drawdown duration.
        """
        wealth_index = (1 + self.returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        durations = []
        current_start = None

        for i, val in enumerate(drawdowns.values.flatten()):
            if val == 0:
                if current_start is not None:
                    durations.append(i - current_start)
                    current_start = None
            else:
                if current_start is None:
                    current_start = i
        if current_start is not None:
            durations.append(len(drawdowns) - current_start)
        return max(durations, default=0)

    def calmar_ratio(self):
        """
        Calculate Calmar ratio (return vs max drawdown).
        """
        return self.annualized_return() / abs(self.max_drawdown())

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

    def summary(self, risk_free_rate=0.0, filter=False):
        """
        Generate comprehensive performance summary.
        """
        all_metrics = {
            'Annualized Return': self.annualized_return(),
            'Annualized Volatility': self.annualized_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(risk_free_rate),
            'Max Drawdown Duration': self.max_drawdown_duration(),
            'Max Drawdown': self.max_drawdown(),
            'Calmar Ratio': self.calmar_ratio(),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            f'VaR ({5}%)': self.value_at_risk(5),
            f'CVaR ({5}%)': self.conditional_var(5),
            'Tail Ratio': self.tail_ratio(5),
            'Omega Ratio': self.omega_ratio(0.0)
        }
        summary_df = pd.DataFrame(all_metrics, index=self.returns.columns)

        if filter:
            return summary_df[filter]

        return summary_df
