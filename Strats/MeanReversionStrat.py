from Data.BinancePriceFetcher import *
from PnL_Metrics.PortfolioMetrics import *
from Utils.Hurst import *
from Utils.config import *

class MeanReversionStrat:
    def __init__(self, df, tickers):
        self.df = df
        self.tickers = tickers


    def generate_signals(self, lookback, execute_threshold, close_threshold):
        # Calculate the rolling mean with a window size of lookback
        for t in self.tickers:
            self.df[t+'_rolling_mean'] = self.df[t].rolling(window=lookback).mean()
            self.df[t+'_rolling_std'] = self.df[t].rolling(window=lookback).std()
            self.df[t+'_z_scores'] = (self.df[t] - self.df[t+'_rolling_mean']) / self.df[t+'_rolling_std']
            is_mean_revert = []
            counter = 0
            for i in tqdm(range(lookback, len(self.df))):
                # if hurst_exponent(self.df[t][i-lookback:i].values) < Hurst_Type.mean_revert[-1]:
                #     is_mean_revert += [1]
                if counter == lookback:
                    if adfuller(self.df[t][i-lookback:i].values)[1] < MeanReversionStrat_PARAMS.stationarity_cutoff:
                        is_mean_revert += [1] * lookback
                    else:
                        is_mean_revert += [0] * lookback
                    counter = 0
                counter += 1
                    
            self.df[t+'_is_mean_revert'] = [0] * (len(self.df) - len(is_mean_revert)) + is_mean_revert
            
            self.df[t+"_is_mean_revert"] = self.df[t+"_is_mean_revert"].fillna(0)

            self.df[t+'_signal'] = np.where(self.df[t+'_z_scores'] < -execute_threshold, 1, 0)
            self.df[t+'_signal'] = np.where(self.df[t+'_z_scores'] > execute_threshold, -1, self.df[t+'_signal'])
            self.df[t+'_signal'] = np.where(self.df[t+'_is_mean_revert'] != 1, 0, self.df[t+'_signal'])
        

            self.df[t+'_exit_signal'] = np.where(((self.df[t+'_z_scores'] > -close_threshold) & (self.df[t+'_z_scores'] < close_threshold)), 1, 0)

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