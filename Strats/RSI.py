from Utils.config import *
from PnL_Metrics.Backtest import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class RSI:
    def __init__(self, df, tickers):
        self.df = df
        self.tickers = tickers

    # --- RSI using Wilder's EMA ---
    def __compute_rsi_wilder(self, t, period=14):
        delta = self.df[t].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # --- Stochastic RSI with EMA smoothing ---
    def __compute_stoch_rsi_ema(self, t, rsi_period, stoch_period, k_smooth, d_smooth):
        rsi = self.__compute_rsi_wilder(t, rsi_period)
        min_rsi = rsi.rolling(window=stoch_period).min()
        max_rsi = rsi.rolling(window=stoch_period).max()
        stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
        k = stoch_rsi.ewm(span=k_smooth, adjust=False).mean() * 100
        d = k.ewm(span=d_smooth, adjust=False).mean()
        return rsi, stoch_rsi, k, d
    
    def generate_signals(self, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
        for t in self.tickers:
            # --- Apply indicator calculations ---
            self.df[t+'_RSI'], self.df[t+'_StochRSI'], self.df[t+'_%K'], self.df[t+'_%D'] = self.__compute_stoch_rsi_ema(t, rsi_period, stoch_period, k_smooth, d_smooth)

            bullish_crossover = (self.df[t+'_%K'].shift(1) < self.df[t+'_%D'].shift(1)) & (self.df[t+'_%K'] > self.df[t+'_%D'])
            bearish_crossover = (self.df[t+'_%K'].shift(1) > self.df[t+'_%D'].shift(1)) & (self.df[t+'_%K'] < self.df[t+'_%D'])

            oversold_k = self.df[t+'_%K'] < 10
            overbought_k = self.df[t+'_%K'] > 80

            rsi_buy_condition = self.df[t+'_RSI'] < 10
            rsi_sell_condition = self.df[t+'_RSI'] >= 90

            buy_signal = bullish_crossover | oversold_k | rsi_buy_condition
            sell_signal = bearish_crossover | overbought_k | rsi_sell_condition

             # --- Signal Generation ---
            self.df[t+'_pseudo_signal'] = 0
            self.df.loc[buy_signal, t+'_pseudo_signal'] = 1
            self.df.loc[sell_signal, t+'_pseudo_signal'] = -1
            self.df.loc[buy_signal & sell_signal, t+'_pseudo_signal'] = 0

            self.df[t+'_signal'] = 0
            self.df.loc[self.df[t+'_pseudo_signal'] == 1, t+'_signal'] = 1
            self.df.loc[self.df[t+'_pseudo_signal'] == -1, t+'_signal'] = -1
            self.df[t+'_signal'] = self.df[t+'_signal'].replace(to_replace=0, method='ffill')
            self.df[t+'_pseudo_signal'] = self.df[t+'_pseudo_signal'].shift(1)

            self.df[t+'_exit_signal'] = 0
            
        return self.df
    
    def generate_single_signal(self, t, prices, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):

        signals = self.generate_signals(rsi_period, stoch_period, k_smooth, d_smooth)
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
        weights = backtest.get_weights(rolling, weight_method, short).tail(1)

        # Store results
        signal_df['signals'] = [signal]
        signal_df['weights'] = [weights[t].item()]
        signal_df['exit_signals'] = [exit_signal]
        signal_df['Price'] = price
        return signal_df