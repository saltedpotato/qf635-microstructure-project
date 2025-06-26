from Data.BinancePriceFetcher import *
from PnL_Metrics.PortfolioMetrics import *
from Utils.Hurst import *
from Utils.config import *
from PnL_Metrics.Backtest import *

class TrendStrat:
    def __init__(self, df, tickers):
        self.df = df
        self.tickers = tickers


    def generate_signals(self, slow_ma_period, fast_ma_period, roc_period, execute_threshold, exit_threshold):
        # Calculate the rolling mean with a window size of lookback
        for t in self.tickers:
            self.df[t+'_slow_ma'] = self.df[t].rolling(window=slow_ma_period).mean()
            self.df[t+'_fast_ma'] = self.df[t].rolling(window=fast_ma_period).mean()
            self.df[t+'_momentum'] = self.df[t].pct_change(roc_period)

            # Generate signals
            self.df[t+'_signal'] = 0  # Default: no position
            self.df[t+'_exit_signal'] = 0  # Default: no exit

            # Long: Fast MA > Slow MA + Positive Momentum
            long_condition = (
                (self.df[t+'_fast_ma'] > self.df[t+'_slow_ma']) & 
                (self.df[t+'_momentum'] > execute_threshold)
            )
            
            # Short: Fast MA < Slow MA + Negative Momentum
            short_condition = (
                (self.df[t+'_fast_ma'] < self.df[t+'_slow_ma']) & 
                (self.df[t+'_momentum'] < -execute_threshold)
            )
            

            exit_condition = (
                (abs(self.df[t+'_momentum']) < exit_threshold)
            )
            
            self.df.loc[long_condition, t+'_signal'] = 1
            self.df.loc[short_condition, t+'_signal'] = -1
            self.df.loc[exit_condition, t+'_exit_signal'] = 1

        return self.df
    
    
    def generate_single_signal(self, t, prices, slow_ma_period, fast_ma_period, roc_period, execute_threshold, exit_threshold):
        signals = self.generate_signals(slow_ma_period, fast_ma_period, roc_period, execute_threshold, exit_threshold)
        signal, exit_signal = signals.tail(1)[t+"_signal"].item(), signals.tail(1)[t+"_exit_signal"].item()
        bid, mid, ask = prices[0], prices[1], prices[2]

        signal_df = pd.DataFrame()
        signal_df['Tickers'] = [t]

        if signal == -1:
            price = bid
        elif signal == 1:
            price = ask
        else:
            price = mid
        
        backtest = Backtest(signals.copy(), tickers = self.tickers, test_start_date=start_date, test_end_date=end_date, stoploss=stoploss, drawdown_duration=drawdown_duration)
        weights = backtest.get_weights(self, rolling, weight_method, short)
        index = self.tickers.index(t)

        # Store results
        signal_df['signals'] = [signal]
        signal_df['weights'] = [weights[index]]
        signal_df['exit_signals'] = [exit_signal]
        signal_df['Price'] = price
        return signal_df