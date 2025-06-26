from Data.BinancePriceFetcher import *
from PnL_Metrics.PortfolioMetrics import *
from Utils.Hurst import *
from Utils.config import *


class TrendStrat:
    def __init__(self, df, tickers):
        self.df = df
        self.tickers = tickers


    def generate_signals(self, slow_ma_period, fast_ma_period, roc_period, execute_threshold):
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
                (abs(self.df[t+'_momentum']) < -execute_threshold)
            )
            
            self.df.loc[long_condition, t+'_signal'] = 1
            self.df.loc[short_condition, t+'_signal'] = -1
            self.df.loc[exit_condition, t+'_exit_signal'] = 1

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