from Data.BinancePriceFetcher import *  # Module for fetching Binance price data
from PnL_Metrics.PortfolioMetrics import *  # Module for portfolio performance metrics
from Utils.Hurst import *  # Module for Hurst exponent calculations
from Utils.config import *  # Configuration parameters
from PnL_Metrics.Backtest import *

class MeanReversionStrat:
    def __init__(self, df, tickers):
        """
        Initialize the mean reversion strategy.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            tickers (list): List of ticker symbols to trade
        """
        self.df = df  # Store price data
        self.tickers = tickers  # Store ticker symbols

    def generate_signals(self, lookback, execute_threshold, close_threshold):
        """
        Generate trading signals for all tickers based on mean reversion logic.
        
        Args:
            lookback (int): Window size for rolling calculations
            execute_threshold (float): Z-score threshold for entering trades
            close_threshold (float): Z-score threshold for exiting trades
            
        Returns:
            pd.DataFrame: DataFrame with generated signals
        """
        # Calculate rolling statistics and z-scores for each ticker
        for t in self.tickers:
            # Calculate rolling mean and standard deviation
            self.df[t+'_rolling_mean'] = self.df[t].rolling(window=lookback).mean()
            self.df[t+'_rolling_std'] = self.df[t].rolling(window=lookback).std()
            self.df[t+'_z_scores'] = (self.df[t] - self.df[t+'_rolling_mean']) / self.df[t+'_rolling_std']
            
            # Check for mean reversion property using Augmented Dickey-Fuller test
            is_mean_revert = []
            counter = 0
            for i in range(lookback, len(self.df)):
                if counter == lookback:
                    # Perform ADF test on lookback window
                    if adfuller(self.df[t][i-lookback:i].values)[1] < MeanReversionStrat_PARAMS.stationarity_cutoff:
                        is_mean_revert += [1] * lookback  # Series is mean-reverting
                    else:
                        is_mean_revert += [0] * lookback  # Series is not mean-reverting
                    counter = 0
                counter += 1
                    
            # Store mean reversion flag in DataFrame
            self.df[t+'_is_mean_revert'] = [0] * (len(self.df) - len(is_mean_revert)) + is_mean_revert
            self.df[t+"_is_mean_revert"] = self.df[t+"_is_mean_revert"].fillna(0)

            # Generate entry signals based on z-scores and mean reversion property
            self.df[t+'_signal'] = np.where(
                (self.df[t+'_z_scores'] < -execute_threshold) & (self.df[t+'_is_mean_revert'] == 1), 
                1,  # Long signal
                0
            )
            self.df[t+'_signal'] = np.where(
                (self.df[t+'_z_scores'] > execute_threshold) & (self.df[t+'_is_mean_revert'] == 1),
                -1,  # Short signal
                self.df[t+'_signal']
            )

            # Generate exit signals when z-score returns near mean
            self.df[t+'_exit_signal'] = np.where(
                ((self.df[t+'_z_scores'] > -close_threshold) & 
                 (self.df[t+'_z_scores'] < close_threshold)),
                1,  # Exit signal
                0
            )

        return self.df
    
    def generate_single_signal(self, t, prices, lookback, execute_threshold, close_threshold):
        """
        Generate trading signal for a single ticker at current prices.
        
        Args:
            t (str): Ticker symbol
            prices (list): [bid, mid, ask] prices
            execute_threshold (float): Z-score threshold for entering trades
            close_threshold (float): Z-score threshold for exiting trades
            
        Returns:
            pd.DataFrame: DataFrame with signal, exit signal, and execution price
        """
        signals = self.generate_signals(lookback, execute_threshold, close_threshold)
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
    