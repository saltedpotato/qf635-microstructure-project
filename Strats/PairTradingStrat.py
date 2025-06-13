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
    def __init__(self, df):
        self.df = df

    def generate_signals(self, lookback, threshold):
        # Calculate the rolling mean with a window size of lookback
        self.df['rolling_mean'] = self.df.iloc[:, 1].rolling(window=lookback).mean()
        self.df['rolling_std'] = self.df.iloc[:, 1].rolling(window=lookback).std()
        self.df['z_scores'] = (self.df.iloc[:, 1] - self.df['rolling_mean']) / self.df['rolling_std']

        self.df.loc[self.df['z_scores'] > threshold, 'signal'] = -1
        self.df.loc[self.df['z_scores'] < -threshold, 'signal'] = 1
        self.df["signal"] = self.df["signal"].fillna(0)
        self.df["signal"] = self.df["signal"].shift(1)


    def plot_signals(self):
        fig, axs = plt.subplots(2,1, figsize=(15,9), sharex=True)
        axs[0].plot(self.df.z_scores, label="Z-score")
        axs[0].axhline(0,color='red')
        axs[0].set_ylabel("Z Scores")
        axs[0].set_xlabel("Date")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].plot(self.df.loc[self.df["signal"] == -1].index, self.df.z_scores[self.df["signal"] == -1], color='r', marker="v", linestyle='')
        axs[0].plot(self.df.loc[self.df["signal"] == +1].index, self.df.z_scores[self.df["signal"] == +1], color='g', marker="^", linestyle='')

        axs[1].plot(self.df.iloc[:, 1], label="Asset y")
        axs[1].legend()
        axs[1].grid(True)
        plt.show()

    def computePnL(self, test_start_date): #have not account for capital
        trades = self.df[self.df['timestamp'] >= test_start_date].copy().reset_index(drop=True)
        # PnL variables - one set per security
        position = 0
        pnlUnrealized = 0
        pnlRealized = 0

        avg_short_price = 0
        short_pos = 0
        avg_long_price = 0
        long_pos = 0
        closed_pos = 0

        pnl_df =pd.DataFrame()
        positions = []
        pnlUnrealized_list = []
        pnlRealized_list = []
        daily_pnl = []

        # for each trade
        for i in range(0, len(trades)):
            qty = trades['signal'][i]
            price = trades.iloc[:, 1][i]

            if qty < 0:
                avg_short_price = (avg_short_price * short_pos + price * qty) / (short_pos + qty)
                short_pos += qty
            elif qty > 0:
                avg_long_price = (avg_long_price * long_pos + price * qty) / (long_pos + qty)
                long_pos += qty

            if i > 0:
                prev_qty = trades['signal'][i - 1]
                if (qty * position) < 0:
                    closed_pos = min(abs(qty), abs(position))
                else:
                    closed_pos = 0
                short_pos += closed_pos
                long_pos -= closed_pos

                if (position+qty) < 0:
                    pnlUnrealized = (avg_short_price - price) * -(position+qty)
                else:
                    pnlUnrealized = (avg_long_price - price) * (position+qty)
                # print(closed_pos)

            position += qty
            pnlRealized += (avg_short_price - avg_long_price) * closed_pos
            daily_pnl += [(avg_short_price - avg_long_price) * closed_pos + pnlUnrealized]

            positions += [position]
            pnlUnrealized_list += [pnlUnrealized]
            pnlRealized_list += [pnlRealized]

            if short_pos == 0:
                avg_short_price = 0
            if long_pos == 0:
                avg_long_price = 0


        pnl_df["Date"] = trades["timestamp"]
        pnl_df["Price"] = trades.iloc[:, 1]
        pnl_df["Signal"] = trades["signal"]
        pnl_df["Positions"] = positions
        pnl_df["Realized_PnL"] = pnlRealized_list
        pnl_df["Unrealized_PnL"] = pnlUnrealized_list
        pnl_df["Daily_PnL"] = daily_pnl
        pnl_df["PnL_Total"] = pnl_df["Realized_PnL"] + pnl_df["Unrealized_PnL"]

        return pnl_df