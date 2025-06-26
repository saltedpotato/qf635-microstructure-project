from Strats.WeightAllocation import *

tickers = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
stoploss = 10e6 # don't activate for now
drawdown_duration = 10e6 # don't activate for now
rolling=1000
weight_method=hierarchical_risk_parity_weighting
short = False
start_date="2024-07-01"
interval = '5m'
interval_seconds = 5*60

class Hurst_Type:
    mean_revert = [0, 0.5] # true if hurst result < mean_revert
    momentum = [0.5, 1] # true if hurst result > 0.5


class MeanReversionStrat_PARAMS:
    execute_threshold = 1.5
    close_threshold = 1
    lookback = 8640
    stationarity_cutoff = 0.1

class TrendStrat_PARAMS:
    stationarity_cutoff = 0.1


class RegressionStrat_PARAMS:
    lookback_window=8000
    regression_type='theilsen'
    pca_components=3
    execute_threshold=0
    stationarity_cutoff = 0.05
    r2_exit=0.7


class RSI_PARAMS:
    rsi_period=14
    stoch_period=14
    k_smooth=3
    d_smooth=3