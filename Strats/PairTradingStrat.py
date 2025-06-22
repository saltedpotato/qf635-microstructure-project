from statsmodels.tsa.stattools import adfuller
from Data.BinancePriceFetcher import *
import matplotlib.pyplot as plt
from Strats.PortfolioMetrics import *

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

            self.df[t+'_signal'] = np.where((self.df[t+'_z_scores'] > execute_threshold) | (self.df[t+'_z_scores'] < -execute_threshold), 1, 0)
            # self.df.loc[self.df[t+'_z_scores'] > execute_threshold, t+'_signal'] = -1
            # self.df.loc[self.df[t+'_z_scores'] < -execute_threshold, t+'_signal'] = 1
            self.df[t+"_signal"] = self.df[t+"_signal"].fillna(0)

            self.df[t+'_exit_signal'] = np.where((self.df[t+'_z_scores'] > -close_threshold) & (self.df[t+'_z_scores'] < close_threshold), 1, 0)
            self.df[t+"_exit_signal"] = self.df[t+"_exit_signal"].fillna(0)

        return self.df


    def plot_signals(self, ticker):
        fig, axs = plt.subplots(2,1, figsize=(15,9), sharex=True)
        axs[0].plot(self.df[ticker+"_z_scores"], label="Z-score")
        axs[0].axhline(0,color='red')
        axs[0].set_ylabel("Z Scores")
        axs[0].set_xlabel("Date")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].plot(self.df.loc[self.df[ticker+"_signal"] == -1].index, self.df[ticker+"_z_scores"][self.df[ticker+"_signal"] == -1], color='r', marker="v", linestyle='')
        axs[0].plot(self.df.loc[self.df[ticker+"_signal"] == +1].index, self.df[ticker+"_z_scores"][self.df[ticker+"_signal"] == +1], color='g', marker="^", linestyle='')

        axs[1].plot(self.df[ticker], label="Asset y")
        axs[1].legend()
        axs[1].grid(True)
        plt.show()

    def computePnL(self, ticker, test_start_date): #have not account for capital
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

        positions = []
        pnlUnrealized_list = []
        pnlRealized_list = []
        daily_pnl = []
        daily_pnl_pct = []

        # for each trade
        for i in range(0, len(trades)):
            qty = trades[ticker+'_signal'][i]
            price = trades[ticker][i]
            exit_signal = trades[ticker+'_exit_signal'][i]

            if exit_signal == 1:
                if position != 0:
                    qty = -position #close position

            if qty < 0:
                avg_short_price = (avg_short_price * short_pos + price * qty) / (short_pos + qty)
                short_pos += qty
            elif qty > 0:
                avg_long_price = (avg_long_price * long_pos + price * qty) / (long_pos + qty)
                long_pos += qty

            if i > 0:
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
            try:
                daily_pnl_pct += [((pnlRealized+pnlUnrealized) - (pnlUnrealized_list[-2] + pnlRealized_list[-2]))/(pnlUnrealized_list[-2] + pnlRealized_list[-2])]
            except:
                daily_pnl_pct += [np.nan]

            if short_pos == 0:
                avg_short_price = 0
            if long_pos == 0:
                avg_long_price = 0

        pnl_df = trades[["timestamp", ticker, ticker+"_signal"]].copy()
        pnl_df.columns = ["Date", "Price", "Signal"]
        pnl_df["Positions"] = positions
        pnl_df["Realized_PnL"] = pnlRealized_list
        pnl_df["Unrealized_PnL"] = pnlUnrealized_list
        pnl_df["Daily_PnL"] = daily_pnl
        pnl_df["Daily_PnL_Pct"] = daily_pnl_pct
        pnl_df["PnL_Total"] = pnl_df["Realized_PnL"] + pnl_df["Unrealized_PnL"]

        return pnl_df

    def computePortfolioPnL(self, test_start_date):
        portfolioPnL = pd.DataFrame()
        for ind, t in enumerate(self.tickers):
            pnl_df = self.computePnL(t, test_start_date)
            portfolioPnL[t+"_daily_pnl"] = pnl_df["Daily_PnL"]
            portfolioPnL[t+"_daily_pnl_pct"] = pnl_df["Daily_PnL_Pct"]
            if ind > 0:
                portfolioPnL["total_daily_pnl"] = portfolioPnL["total_daily_pnl"] + portfolioPnL[t+"_daily_pnl"] * self.weights[ind]
                portfolioPnL["total_daily_pnl_pct"] = portfolioPnL["total_daily_pnl_pct"] + portfolioPnL[t+"_daily_pnl_pct"] * self.weights[ind]
                portfolioPnL["total_pnl"] = portfolioPnL["total_pnl"] + pnl_df["PnL_Total"] * self.weights[ind]
            else:
                portfolioPnL["total_daily_pnl"] = portfolioPnL[t+"_daily_pnl"] * self.weights[ind]
                portfolioPnL["total_daily_pnl_pct"] = portfolioPnL[t+"_daily_pnl_pct"] * self.weights[ind]
                portfolioPnL["total_pnl"] = pnl_df["PnL_Total"] * self.weights[ind]

        portfolioPnL["timestamp"] = pnl_df["Date"]
        portfolioPnL = portfolioPnL[["timestamp", "total_daily_pnl", "total_daily_pnl_pct", "total_pnl"]]

        return portfolioPnL



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