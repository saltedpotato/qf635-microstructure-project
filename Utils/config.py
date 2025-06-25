from Strats.WeightAllocation import *

tickers = ['BTCUSDT', 'ETHUSDT']
stoploss = 0.05
drawdown_duration = 2880
rolling=1000
weight_method=hierarchical_risk_parity_weighting
short = False
start_date="2024-07-01"
interval = '5m'