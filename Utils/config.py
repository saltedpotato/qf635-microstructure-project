from Strats.WeightAllocation import *

tickers = ['BTCUSDT', 'ETHUSDT']
stoploss = 0.05
drawdown_duration = 2880
rolling=1000
weight_method=hierarchical_risk_parity_weighting
short = False
start_date="2024-07-01"
interval = '5m'
interval_seconds = 5*60

class Hurst_Type:
    mean_revert = [0, 0.5] # true if hurst result < mean_revert
    gbm = [0.45, 0.55] # true if hurst result around 0.5
    trend = [0.85, 1] # true if hurst result > 0.85


class MeanReversionStrat_PARAMS:
    execute_threshold = 1.5
    close_threshold = 1
    lookback = 8640
    stationarity_cutoff = 0.1

class RegressionStrat_PARAMS:
    lookback_window=8000
    regression_type='huber'
    pca_components=3
    execute_threshold=0