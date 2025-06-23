from statsmodels.tsa.stattools import adfuller
from Data.BinancePriceFetcher import *
import matplotlib.pyplot as plt
from PnL_Metrics.PortfolioMetrics import *

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

def hurst_exponent(ts: pd.Series, max_lags: int = 50) -> float:
    """
    Calculate the Hurst Exponent of a time series.
        
    Args:
        time_series (pd.Series): Input time series (e.g., stock prices, spreads).
        max_lags (int): Maximum number of lags to compute rescaled range (R/S).
        
    Returns:
        float: Hurst Exponent value.
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


class pair_trading:
    def __init__(self, df, weights):
        self.df = df
        self.tickers = df.columns.tolist()[1:]
        self.weights = weights


    def generate_signals(self, lookback, execute_threshold, close_threshold):
        # Calculate the rolling mean with a window size of lookback
        for t in self.tickers:
            self.df[t+'_rolling_mean'] = self.df[t].rolling(window=lookback).mean()
            self.df[t+'_rolling_std'] = self.df[t].rolling(window=lookback).std()
            self.df[t+'_z_scores'] = (self.df[t] - self.df[t+'_rolling_mean']) / self.df[t+'_rolling_std']
            is_mean_revert = []
            for i in range(lookback, len(self.df)):
                if hurst_exponent(self.df[t][i-lookback:i].values) < 0.5:
                    is_mean_revert += [1]
                else:
                    is_mean_revert += [0]
                
            self.df[t+'_is_mean_revert'] = [0] * (len(self.df) - len(is_mean_revert)) + is_mean_revert
            self.df[t+"_is_mean_revert"] = self.df[t+"_is_mean_revert"].fillna(0)

            self.df[t+'_signal'] = np.where((self.df[t+'_z_scores'] > execute_threshold) | (self.df[t+'_z_scores'] < -execute_threshold), 1, 0)
            self.df[t+'_signal'] = np.where(self.df[t+'_is_mean_revert'] != 1, 0, self.df[t+'_signal'])
        

            self.df[t+'_exit_signal'] = np.where((self.df[t+'_z_scores'] > -close_threshold) & (self.df[t+'_z_scores'] < close_threshold), 1, 0)
            self.df[t+'_exit_signal'] = np.where(self.df[t+'_exit_signal'] != 1, 1, self.df[t+'_exit_signal'])

        return self.df


# Example Usage
if __name__ == "__main__":
    # Get all available tickers
    response = requests.get(f"{BASE_URL}/ticker/price")
    data = response.json()
    BTC_pairs = [i["symbol"] for i in data if "BTC" in i["symbol"]]
    coint_pairs = get_coint_pairs(BTC_pairs, interval='1d', start_date="2023-01-01", end_date="2023-12-31")

    test_pair = coint_pairs[10]

    symbol_manager = BinanceSymbolManager()
    # Add symbols
    print(symbol_manager.add_symbol(test_pair))  # Success

    price_fetcher = BinancePriceFetcher(symbol_manager.get_symbols())
    # Fetch pair historical price
    pair_portfolio = price_fetcher.get_grp_historical_ohlcv(
        interval="1d",
        start_date="2023-01-01",
        end_date="2024-12-31"
    )


    model = pair_trading(pair_portfolio.copy())
    spread = model.generate_signals(lookback=10, threshold=1.5)
    pnl_df = model.computePnL(test_start_date="2024-01-01")

    returns = pnl_df[("Daily_PnL")]

    port_metrics = PortfolioMetrics(returns.dropna())
    summary = port_metrics.summary(risk_free_rate=0.02)  # 2% risk-free rate
    print(summary)