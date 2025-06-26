from Utils.import_files import *
import matplotlib.pyplot as plt
from Strats.WeightAllocation import *

class Backtest:
    def __init__(self, df, tickers, test_start_date, test_end_date, stoploss=0.1, drawdown_duration=100):
        '''
        Initialize the backtesting class with necessary parameters.
        
        Args:
            df (DataFrame): DataFrame containing price and signal data: columns required - timestamp, ticker (contains price), ticker_signal, ticker_exit_signal
            tickers (list): List of tickers/symbols to backtest
            weights (list): Portfolio weights for each ticker
            test_start_date (str/datetime): Start date for backtesting period
            stoploss (float): Stop-loss percentage (default: 0.1 for 10%)
            drawdown_duration (int): Number of days to consider for drawdown (default: 100)
        '''
        self.df = df
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.tickers = tickers
        self.weights = np.array([1/len(tickers) for i in tickers]) #equal weighting first
        self.stoploss = stoploss
        self.drawdown_duration = drawdown_duration
        

    def backtest(self, ticker):
        '''
        Backtest a single ticker based on trading signals.
        
        Args:
            ticker (str): The ticker symbol to backtest
            
        Returns:
            DataFrame: Contains PnL metrics and position information
        '''
        # Filter trades from the start date and reset index
        trades = self.df[(self.df['timestamp'] <= self.test_end_date) & (self.df['timestamp'] >= self.test_start_date)].copy().reset_index(drop=True)
        
        # Initialize PnL tracking variables
        position = 0  # Current position (positive for long, negative for short)
        pnlUnrealized = 0  # Unrealized profit/loss
        pnlRealized = 0  # Realized profit/loss
        
        # Price averaging for short and long positions
        avg_short_price = 0  # Average price of short positions
        short_pos = 0  # Size of short position
        avg_long_price = 0  # Average price of long positions
        long_pos = 0  # Size of long position
        closed_pos = 0  # Size of position being closed
        
        # Lists to track metrics over time
        positions = []  # Track position size at each step
        pnlUnrealized_list = []  # Track unrealized PnL
        pnlRealized_list = []  # Track realized PnL
        daily_pnl = []  # Daily PnL (realized + unrealized)
        daily_pnl_pct = []  # Daily PnL percentage change

        peak = 0
        peak_ind = len(trades)

        # Process each trade in sequence
        for i in range(0, len(trades)):
            # Get current trade information
            qty = trades[ticker+'_signal'][i]  # Signal quantity (positive for buy, negative for sell)
            price = trades[ticker][i]  # Current price
            exit_signal = trades[ticker+'_exit_signal'][i]  # Exit signal (1 = exit)

            if (pnlUnrealized + pnlRealized) > peak:
                peak = pnlUnrealized + pnlRealized
                peak_ind = i
                
            # Check exit conditions (three possible exit triggers)
            if exit_signal == 1:  # Exit signal triggered
                if position != 0:
                    qty = -position  # Close entire position
            elif (len(daily_pnl_pct) > 0) and (daily_pnl_pct[-1] < -self.stoploss):  # Stop-loss hit
                if position != 0:
                    qty = -position  # Close entire position
            elif (i - peak_ind) >= self.drawdown_duration:  # Drawdown duration exceeded
                if position != 0:
                    qty = -position  # Close entire position
                    peak_ind = len(trades)
            
            # Update position averages based on trade direction
            if qty < 0:  # Short position
                avg_short_price = (avg_short_price * short_pos + price * qty) / (short_pos + qty)
                short_pos += qty
            elif qty > 0:  # Long position
                avg_long_price = (avg_long_price * long_pos + price * qty) / (long_pos + qty)
                long_pos += qty
            
            # Handle position closing and PnL calculation
            if i > 0:  # Skip first trade as we need previous data
                if (qty * position) < 0:  # Check if we're closing part of the position
                    closed_pos = min(abs(qty), abs(position))
                else:
                    closed_pos = 0
                
                # Update position sizes
                short_pos += closed_pos
                long_pos -= closed_pos
                
                # Calculate unrealized PnL based on position direction
                if (position+qty) < 0:  # Net short position
                    pnlUnrealized = (avg_short_price - price) * -(position+qty)
                else:  # Net long position
                    pnlUnrealized = (avg_long_price - price) * (position+qty)
            
            # Update position and PnL metrics
            position += qty
            pnlRealized += (avg_short_price - avg_long_price) * closed_pos
            daily_pnl += [(avg_short_price - avg_long_price) * closed_pos + pnlUnrealized]
            
            # Append current metrics to tracking lists
            positions += [position]
            pnlUnrealized_list += [pnlUnrealized]
            pnlRealized_list += [pnlRealized]
            
            # Calculate daily PnL percentage change (handle first day case)
            try:
                daily_pnl_pct += [((pnlRealized+pnlUnrealized) - (pnlUnrealized_list[-2] + pnlRealized_list[-2])) / 
                                 abs(pnlUnrealized_list[-2] + pnlRealized_list[-2] + 1e-10)]  # Small value to avoid division by zero
            except:
                daily_pnl_pct += [np.nan]  # First day has no previous PnL
            
            # Reset average prices if positions are closed
            if short_pos == 0:
                avg_short_price = 0
            if long_pos == 0:
                avg_long_price = 0
        
        # Create final results DataFrame
        pnl_df = trades[["timestamp", ticker, ticker+"_signal"]].copy()
        pnl_df.columns = ["timestamp", "Price", "Signal"]
        pnl_df["Positions"] = positions
        pnl_df["Realized_PnL"] = pnlRealized_list
        pnl_df["Unrealized_PnL"] = pnlUnrealized_list
        pnl_df["Daily_PnL"] = daily_pnl
        pnl_df["PnL_Total"] = pnl_df["Realized_PnL"] + pnl_df["Unrealized_PnL"]
        
        return pnl_df
    
    def get_weights(self, rolling=5000, weight_method = inverse_volatility_weighting, allow_short=True):
        weights = [np.array([1/len(self.tickers) for i in self.tickers])] * rolling
        for i in range(0, len(self.df)-rolling, rolling):
            window = self.df[:i+rolling]
            pair_backtest_temp = Backtest(window.copy(), tickers = self.tickers, test_start_date=window['timestamp'].head(1).item(), test_end_date=window['timestamp'].tail(1).item(), stoploss=0.1, drawdown_duration=100)
            returns_temp = pair_backtest_temp.get_ticker_returns(notional = 10e6)
            weights_temp = weight_method(returns_temp, allow_short=allow_short)
            if np.isnan(weights_temp[0]):
                if len(weights) == 0:
                    weights_temp = np.array([1/len(self.tickers) for i in self.tickers])
                else:
                    weights_temp = weights[-1]
            weights += [weights_temp] * len(self.df[i+rolling:i+rolling*2])
        weights_df = pd.DataFrame(weights, columns=self.tickers)[:len(self.df)]
        weights_df['timestamp'] = self.df['timestamp']
        return weights_df
    
    def computePortfolioPnL(self, rolling=5000, weight_method = inverse_volatility_weighting, allow_short=True):
        '''
        Computes the portfolio-level PnL by aggregating individual asset PnLs using their weights.
        
        Args:
            test_start_date (str/datetime): Start date for the backtesting period
            stoploss (float): Stop-loss percentage to apply to each asset (default: 0.1 for 10%)
            
        Returns:
            DataFrame: Contains timestamp and portfolio-level PnL metrics including:
                - total_daily_pnl: Weighted sum of daily PnL across all assets
                - total_pnl: Weighted sum of cumulative PnL
        '''
        # Initialize empty DataFrame to store portfolio results
        portfolioPnL = pd.DataFrame()
        weights_df = self.get_weights(rolling, weight_method, allow_short)
        weights_df = weights_df[(weights_df['timestamp'] <= self.test_end_date) & (weights_df['timestamp'] >= self.test_start_date)].copy().reset_index(drop=True)
        # Loop through each ticker in the portfolio
        for ind, t in enumerate(self.tickers):
            pnl_df = self.backtest(t)
        
            # Store individual asset metrics
            portfolioPnL[t+"_daily_pnl"] = pnl_df["Daily_PnL"]
            
            # Calculate weighted portfolio metrics
            if ind > 0:
                # For subsequent assets, add weighted contribution to existing totals
                portfolioPnL["total_daily_pnl"] = portfolioPnL["total_daily_pnl"] + portfolioPnL[t+"_daily_pnl"] * weights_df[t]
                portfolioPnL["total_pnl"] = portfolioPnL["total_pnl"] + pnl_df["PnL_Total"] * weights_df[t]
            else:
                # For first asset, initialize totals with weighted values
                portfolioPnL["total_daily_pnl"] = portfolioPnL[t+"_daily_pnl"] * weights_df[t]
                portfolioPnL["total_pnl"] = pnl_df["PnL_Total"] * weights_df[t]
        
        # Add timestamp from the last backtested asset (all should have same dates)
        portfolioPnL["timestamp"] = pnl_df["timestamp"]
        
        # Select and return only the portfolio-level metrics
        portfolioPnL = portfolioPnL[["timestamp", "total_daily_pnl", "total_pnl"]]
        
        return portfolioPnL

    def plot_pnl(self, total_pnl_df):
        '''
        Visualizes the PnL performance of individual assets and the overall portfolio.
        
        Args:
            test_start_date (str/datetime): Start date for the backtesting period
            
        Returns:
            None (displays matplotlib plot)
        '''
        # Create subplots: one for each asset + one for portfolio
        fig, axs = plt.subplots(len(self.tickers)+1, 1, figsize=(15,9), sharex=True)
        
        # Plot each asset's PnL in separate subplot
        for i, t in enumerate(self.tickers):
            # Backtest the asset
            pnl_df = self.backtest(t)
            
            # Plot cumulative PnL
            axs[i].plot(pnl_df["timestamp"], pnl_df["PnL_Total"], label=f"{t}_PnL_Total")
            axs[i].set_ylabel("Accumulated PnL")
            axs[i].set_xlabel("timestamp")
            axs[i].legend()
            axs[i].grid(True)
        
        # Calculate and plot portfolio PnL in bottom subplot
        axs[len(self.tickers)].plot(total_pnl_df["timestamp"], total_pnl_df["total_pnl"], label="Portfolio")
        axs[len(self.tickers)].set_ylabel("Accumulated PnL")
        axs[len(self.tickers)].set_xlabel("timestamp")
        axs[len(self.tickers)].legend()
        axs[len(self.tickers)].grid(True)
        
        # Display the complete figure
        plt.show()

    def get_ticker_returns(self, notional = 10e6):
        grp_returns = pd.DataFrame()
        for t in self.tickers:
            pnl = self.backtest(t)['PnL_Total']
            pnl_diff = np.diff(pnl)
            grp_returns[t+'_returns'] = pnl_diff / notional

        return grp_returns
    
    def get_returns(self, pnl_df, notional = 10e6):
        pnl_filtered = pnl_df[pnl_df['total_pnl'] != 0]['total_pnl']
        pnl_diff = np.diff(pnl_filtered)
        returns = pnl_diff / notional
        return returns