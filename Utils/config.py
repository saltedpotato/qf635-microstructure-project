from Strats.WeightAllocation import *
from Utils.import_files import *


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
    r2_exit=0.7


class RSI_PARAMS:
    rsi_period=14
    stoch_period=14
    k_smooth=3
    d_smooth=3


now = datetime.today()

tickers = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
stoploss = 10e6 # don't activate for now
drawdown_duration = 10e6 # don't activate for now
rolling=1000
weight_method=hierarchical_risk_parity_weighting
short = False

interval = '5m'
interval_mins = 5
interval_seconds = interval_mins*60

end_date = now
start_date=(end_date - timedelta(days=max(rolling, MeanReversionStrat_PARAMS.lookback, RegressionStrat_PARAMS.lookback_window)//(interval_mins * 24) + 2)).strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')