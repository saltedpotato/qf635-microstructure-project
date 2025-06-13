from statsmodels.tsa.stattools import adfuller
from Utils.BinancePriceFetcher import *
import matplotlib.pyplot as plt
import statsmodels.api as sm

import sys
import threading

try:
    import thread
except ImportError:
    import _thread as thread

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    # print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt
    # thread.error()


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

@exit_after(1)
def get_pair_prices(ticker, interval = '1d', start_date="2023-01-01", end_date="2023-12-31"):
    price_fetcher = BinancePriceFetcher([ticker]) #Initialise price fetcher
    prices = price_fetcher.get_historical_ohlcv(
        symbol=ticker,
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )
    return prices


def get_coint_pairs(tickers, interval = '1d', start_date="2023-01-01", end_date="2023-12-31"):
    coint_pairs = []
    for ticker in tqdm(tickers):
        try:
            prices = get_pair_prices(ticker, interval = interval, start_date=start_date, end_date=end_date)
            if len(prices) == 0:
                continue
            result = adfuller(prices["close"])
            if result[1] < 0.05:
                coint_pairs.append(ticker)

        except KeyboardInterrupt as e:
            continue
        except KeyError as e:
            continue

    # with open('Strats/pairs/saved_pairs.pkl', 'wb') as f:
    #     pickle.dump(coint_pairs, f)
    return coint_pairs

class pair_trading:
    def __init__(self,x_data,y_data,lookback=None,ZSCORE_ENTRY=None, initial_capital=100000):
        self.x_data = x_data
        self.y_data = y_data
        self.lookback = lookback
        self.ZSCORE_ENTRY = ZSCORE_ENTRY
        self.initial_capital = initial_capital
        self.spread = None
        self.z_scores = None
        self.positions = None
        self.beta = None # beta

    def compute_spread(self,lookback=None):
        x = self.x_data.iloc[:lookback]
        y = self.y_data.iloc[:lookback]

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        self.beta = model.params[1]
        intercept = model.params[0]
        self.spread = y - (intercept + self.beta*x)

        return self.spread

    def compute_z_score(self, rolling_window=20):
        """
        Compute z-scores using only past data at each point
        """
        z_scores = pd.Series(index=self.spread.index, dtype=float)

        for i in range(self.lookback + rolling_window, len(self.spread)):
            window = self.spread.iloc[i-rolling_window:i]
            mean = window.mean()
            std = window.std()

            if std > 0:  # Avoid division by zero
                z_scores.iloc[i] = (self.spread.iloc[i] - mean) / std
            else:
                z_scores.iloc[i] = 0

        self.z_scores = z_scores
        return z_scores

    def generate_signals(self):
        """
        Generate trading signals without lookahead
        """
        df = pd.DataFrame({
            "z_scores": self.z_scores,
            "beta": self.beta
        })

        # Initialize signal columns
        df["z_signal"] = 0
        df["positions"] = 0
        df["BUY_or_SELL"] = 0

        # Generate signals without lookahead
        for i in range(1, len(df)):
            if df.z_scores.iloc[i-1] < -self.ZSCORE_ENTRY:
                df.z_signal.iloc[i] = 1
            elif df.z_scores.iloc[i-1] > self.ZSCORE_ENTRY:
                df.z_signal.iloc[i] = -1

            # Position management
            if (df.z_scores.iloc[i-1] * df.z_scores.iloc[i] < 0) and (df.positions.iloc[i-1] != 0):
                df.positions.iloc[i] = 0
            else:
                df.positions.iloc[i] = df.z_signal.iloc[i]

            # Determine trade direction
            if df.positions.iloc[i] != df.positions.iloc[i-1]:
                df.BUY_or_SELL.iloc[i] = df.positions.iloc[i] - df.positions.iloc[i-1]

        return df

    def plot_signals(self, df):
        fig, axs = plt.subplots(2,1, figsize=(15,9), sharex=True)
        axs[0].plot(df.z_scores, label="Z-score")
        axs[0].axhline(0,color='red')
        axs[0].set_ylabel("Z Scores")
        axs[0].set_xlabel("Date")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].plot(df.loc[df["BUY_or_SELL"] == -1].index, df.z_scores[df["BUY_or_SELL"] == -1], color='r', marker="v", linestyle='')
        axs[0].plot(df.loc[df["BUY_or_SELL"] == +1].index, df.z_scores[df["BUY_or_SELL"] == +1], color='g', marker="^", linestyle='')

        axs[1].plot(self.x_data, label="Asset x")
        axs[1].plot(self.y_data, label="Asset y")
        axs[1].legend()
        axs[1].grid(True)
        plt.show()

    def computePnL(self,df): #have not account for capital
        df["x_pnl"] = self.x_data*df["BUY_or_SELL"]*self.beta
        df["y_pnl"] = self.y_data*-df["BUY_or_SELL"]
        df['daily_pnl'] = df['x_pnl'] + df['y_pnl']
        df['cumulative_pnl'] = (df['daily_pnl']).cumsum()

        portfolio_value = self.initial_capital + df['cumulative_pnl']
        # Daily return = (Today's PnL) / (Yesterday's Portfolio Value)
        df['daily_return'] = df['cumulative_pnl'].diff() / portfolio_value.shift(1)
        df['daily_return'].fillna(0, inplace=True)  # First day has no prior value

        return df