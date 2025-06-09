from statsmodels.tsa.stattools import coint
from itertools import combinations
import pickle
from Utils.BinancePriceFetcher import *

import sys
import threading
from time import sleep

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
def get_pair_prices(pair, interval = '1d', start_date="2023-01-01", end_date="2023-12-31"):
    price_fetcher = BinancePriceFetcher(pair) #Initialise price fetcher
    prices = price_fetcher.get_grp_historical_ohlcv(
            interval=interval,
            start_date=start_date,
            end_date=end_date
    )
    return prices

def coint_test(p1, p2):
    score, p_value, _ = coint(p1, p2)
    if p_value < 0.05:
        return True
    else:
        return False


def get_coint_pairs(tickers, r=2, max_pairs = 100, interval = '1d', start_date="2023-01-01", end_date="2023-12-31"):
    all_pairs = list(combinations(tickers, r))
    coint_pairs = []
    corrupt_tickers = []
    for pair in tqdm(all_pairs):
        try:
            if (pair[0] in corrupt_tickers) or (pair[1] in corrupt_tickers):
                continue

            prices = get_pair_prices(pair, interval = interval, start_date=start_date, end_date=end_date)

            # To prevent survivorship bias
            prices = prices.dropna()

            if coint_test(prices[pair[0]], prices[pair[1]]):
                coint_pairs += [pair]

            if len(coint_pairs) > max_pairs:
                break;
        except KeyboardInterrupt as e:
            corrupt_tickers += [pair[-1]]
            continue
        except KeyError as e:
            continue
    with open('Strats/pairs/saved_pairs.pkl', 'wb') as f:
        pickle.dump(coint_pairs, f)
    return coint_pairs