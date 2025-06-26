from Data.BinancePriceFetcher import *
from Data.GetOrderBook import *
from Utils.config import *
from Strats.MeanReversionStrat import *
from Strats.RegressionStrat import *


print("Trading parameters (edit in config.py file):")
print(f"Tickers: {tickers}")
print(f"Stop loss: {stoploss}")
print(f"Max drawdown duration: {drawdown_duration}")
print(f"Rolling window: {rolling}")
print(f"Portfolio Allocation method: {weight_method.__name__}")
print(f"Allow shorting in weight allocation: {short}")
print(f"Train start date: {start_date}")
print(f"Trading Frequency: {interval}")


now = datetime.today()

symbol_manager = BinanceSymbolManager()
# Add symbols
for t in tickers:
    symbol_manager.add_symbol(t) # Success

passed_tickers = symbol_manager.get_symbols()
price_fetcher = BinancePriceFetcher(passed_tickers)

# Fetch pair historical price up till latest point in time
portfolio_prices = price_fetcher.get_grp_historical_ohlcv(
        interval=interval,
        start_date=start_date,
        end_date=now.strftime('%Y-%m-%d')
    )

def get_signal(prices, portfolio_prices):
    signals = pd.DataFrame()
    mr_model = MeanReversionStrat(portfolio_prices[-MeanReversionStrat_PARAMS.lookback:].copy(), passed_tickers)
    lr_model = RegressionStrat(portfolio_prices[-RegressionStrat_PARAMS.lookback_window-10:].copy(), passed_tickers, 
                                    lookback_window=RegressionStrat_PARAMS.lookback_window, 
                                    regression_type=RegressionStrat_PARAMS.regression_type)
    
    for ind, t in enumerate(passed_tickers):
        huber_result = hurst_exponent(portfolio_prices[t].values[-rolling:])
        if huber_result < Hurst_Type.mean_revert[-1]:
            # mean-revert
            signal = mr_model.generate_single_signal(
                t, prices[ind], 
                execute_threshold=MeanReversionStrat_PARAMS.execute_threshold, 
                close_threshold=MeanReversionStrat_PARAMS.close_threshold)
            
        elif huber_result > Hurst_Type.trend[0]:
            # Regression or smth else
            signal = lr_model.generate_single_signal(t, prices[ind], 
                                                     pca_components=RegressionStrat_PARAMS.pca_components, 
                                                     execute_threshold=RegressionStrat_PARAMS.execute_threshold, 
                                                     r2_exit=RegressionStrat_PARAMS.r2_exit)
        else:
            signal = pd.DataFrame([[t, 0, 0, prices[ind][1]]], columns = ['Tickers', 'signals', 'exit_signals', 'Price'])
        
        signals = pd.concat([signals, signal])
    signals['timestamp'] = datetime.now()
    
    signals = signals[['timestamp', 'Tickers', 'signals', 'exit_signals', "Price"]]
    return signals

order_book_manager = OrderBookManager(passed_tickers)
# Main loop
while True:
    try:
        start_time = time.time()
        prices = []
        for t in tickers:
            # Print current state
            order_book = order_book_manager.fetch_order_book(t, limit = 5)
            prices += [[order_book.best_bid(), order_book.get_mid_price(), order_book.best_ask()]]
        signals = get_signal(prices, portfolio_prices)
        print(signals)
        
        #portfolio price scrape time start
        portfolio_prices_start = time.perf_counter()
        portfolio_prices = price_fetcher.get_grp_historical_ohlcv(
                interval=interval,
                start_date=start_date,
                end_date=now.strftime('%Y-%m-%d')
            )
        portfolio_prices_end = time.perf_counter()

        # Sleep to maintain ~frequency interval 
        time.sleep(interval_seconds - (portfolio_prices_end-portfolio_prices_start))

    except KeyboardInterrupt:
        logging.info("Stopping order book monitor...")
        break
    except Exception as e:
        logging.error(f"Error: {e}")
        time.sleep(5)  # Wait before retrying after error


