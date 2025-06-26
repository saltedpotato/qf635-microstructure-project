from Data.BinancePriceFetcher import *  # Module for fetching Binance price data
from PnL_Metrics.PortfolioMetrics import *  # Module for portfolio performance metrics
from Utils.Hurst import *  # Module for Hurst exponent calculations
from Utils.config import *  # Configuration parameters

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
            for i in tqdm(range(lookback, len(self.df))):
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
    
    def generate_single_signal(self, t, prices, execute_threshold, close_threshold):
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
        signal_df = pd.DataFrame()
        signal_df['Tickers'] = [t]
        
        # Calculate current statistics
        rolling_mean = self.df[t].mean()
        rolling_std = self.df[t].std()
        bid, mid, ask = prices[0], prices[1], prices[2]
        price = mid  # Default to mid price if no trade
        
        # Calculate current z-score
        z_score = (mid - rolling_mean) / rolling_std

        # Check if series is mean-reverting using ADF test
        is_mean_revert = False
        if adfuller(self.df[t].values)[1] < MeanReversionStrat_PARAMS.stationarity_cutoff:  # 95% confidence level
            is_mean_revert = True

        # Determine trade signal
        signal = 0  # Default: no signal
        exit_signal = 0  # Default: don't exit
        
        # Long signal condition
        if z_score < -execute_threshold and is_mean_revert:
            signal = 1
            price = ask  # Buy at ask price
            
        # Short signal condition    
        elif z_score > execute_threshold and is_mean_revert:
            signal = -1
            price = bid  # Sell at bid price
            
        # Exit signal condition
        if ((z_score > -close_threshold) & (z_score < close_threshold)):
            exit_signal = 1
        
        # Store results
        signal_df['signals'] = [signal]
        signal_df['exit_signals'] = [exit_signal]
        signal_df['Price'] = price
        return signal_df