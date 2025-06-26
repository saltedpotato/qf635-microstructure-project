from statsmodels.tsa.stattools import adfuller
from Data.BinancePriceFetcher import *
import matplotlib.pyplot as plt
from PnL_Metrics.PortfolioMetrics import *
from Utils.Hurst import *
from Utils.config import *

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

@exit_after(5)
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

class MeanReversionStrat:
    def __init__(self, df, tickers):
        self.df = df
        self.tickers = tickers


    def generate_signals(self, lookback, execute_threshold, close_threshold):
        # Calculate the rolling mean with a window size of lookback
        for t in self.tickers:
            self.df[t+'_rolling_mean'] = self.df[t].rolling(window=lookback).mean()
            self.df[t+'_rolling_std'] = self.df[t].rolling(window=lookback).std()
            self.df[t+'_z_scores'] = (self.df[t] - self.df[t+'_rolling_mean']) / self.df[t+'_rolling_std']
            is_mean_revert = []
            counter = 0
            for i in tqdm(range(lookback, len(self.df))):
                # if hurst_exponent(self.df[t][i-lookback:i].values) < Hurst_Type.mean_revert[-1]:
                #     is_mean_revert += [1]
                if counter == lookback:
                    if adfuller(self.df[t][i-lookback:i].values)[1] < MeanReversionStrat_PARAMS.stationarity_cutoff:
                        is_mean_revert += [1] * lookback
                    else:
                        is_mean_revert += [0] * lookback
                    counter = 0
                counter += 1
                    
            self.df[t+'_is_mean_revert'] = [0] * (len(self.df) - len(is_mean_revert)) + is_mean_revert
            
            self.df[t+"_is_mean_revert"] = self.df[t+"_is_mean_revert"].fillna(0)

            self.df[t+'_signal'] = np.where(self.df[t+'_z_scores'] < -execute_threshold, 1, 0)
            self.df[t+'_signal'] = np.where(self.df[t+'_z_scores'] > execute_threshold, -1, self.df[t+'_signal'])
            self.df[t+'_signal'] = np.where(self.df[t+'_is_mean_revert'] != 1, 0, self.df[t+'_signal'])
        

            self.df[t+'_exit_signal'] = np.where(((self.df[t+'_z_scores'] > -close_threshold) & (self.df[t+'_z_scores'] < close_threshold)), 1, 0)

        return self.df
    
    def generate_single_signal(self, t, prices, execute_threshold, close_threshold):
        signal_df = pd.DataFrame()
        signal_df['Tickers'] = [t]
        rolling_mean = self.df[t].mean()
        rolling_std = self.df[t].std()
        bid, mid, ask = prices[0], prices[1], prices[2]
        price = mid
        z_score = (mid - rolling_mean) / rolling_std

        is_mean_revert = False
        if adfuller(self.df[t].values)[1] < 0.05:
            is_mean_revert = True

        signal = 0
        exit_signal = 0
        if z_score < -execute_threshold:
            signal = 1
            price = ask
        elif z_score > execute_threshold:
            signal = -1
            price = bid

        if not is_mean_revert:
            signal = 0
            price = mid
            
        if ((z_score > -close_threshold) & (z_score < close_threshold)):
            exit_signal = 1
        
        signal_df['signals'] = [signal]
        signal_df['exit_signals'] = [exit_signal]
        signal_df['Price'] = price
        return signal_df