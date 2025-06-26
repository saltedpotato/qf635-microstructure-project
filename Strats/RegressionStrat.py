import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, 
    TheilSenRegressor, 
    HuberRegressor, 
    RANSACRegressor
)
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from Utils.Hurst import *
from Utils.config import *
from PnL_Metrics.Backtest import *

class RegressionStrat:
    """
    A trading strategy that uses different robust regression techniques to generate long/short signals.
    
    Parameters:
    -----------
    lookback_window : int
        Number of periods to use for training the regression models
    regression_type : str
        Type of regression to use ('linear', 'theilsen', 'huber', 'ransac')
    z_score_threshold : float
        Threshold for z-score of prediction error to trigger trades
    volatility_scaling : bool
        Whether to scale position sizes by recent volatility
    """
    
    def __init__(self, df, tickers, lookback_window=63, regression_type='theilsen'):
        self.df = df
        self.tickers = tickers
        self.lookback_window = lookback_window
        self.regression_type = regression_type.lower()
        
        # Initialize models
        self.models = {
            'linear': LinearRegression(),
            'theilsen': TheilSenRegressor(random_state=42),
            'huber': HuberRegressor(),
            'ransac': RANSACRegressor(random_state=42)
        }
        
        # State variables
        self.current_position = 0
        self.holding_counter = 0
        self.prediction_error_history = []
        self.scaler = StandardScaler()
        
    def _get_features_targets(self, t):
        """Create feature matrix (lags) and target vector (future returns)"""
        returns = self.df[t].pct_change()

        # Create lagged features
        df_preprocessed = pd.DataFrame()
        for lag in range(1, self.lookback_window, 1000):
            df_preprocessed[f'lag_{lag}'] = returns.shift(lag)
        
        df_preprocessed['rolling_vol'] = returns.rolling(window=self.lookback_window).std().shift(1)
        
        df_preprocessed[t] = returns
        # Align and drop NA
        df_preprocessed = df_preprocessed.dropna().reset_index(drop=True)
        X, y = df_preprocessed.drop(columns=[t]), df_preprocessed[t]
        
        return X, y
    
    def _get_test_features(self, t):
        """Create feature matrix (lags) and target vector (future returns)"""
        
        returns = self.df[t].pct_change()

        # Create lagged features
        df_preprocessed = pd.DataFrame()
        for lag in range(0, self.lookback_window, 1000):
            df_preprocessed[f'lag_{lag+1}'] = returns.shift(lag)
        
        df_preprocessed['rolling_vol'] = returns.rolling(window=self.lookback_window).std()
        
        df_preprocessed[t] = returns
        # Align and drop NA
        df_preprocessed = df_preprocessed.dropna().reset_index(drop=True)
        X, y = df_preprocessed.drop(columns=[t]), df_preprocessed[t]
        
        return X
    
    
    def _get_model(self):
        """Get the appropriate regression model based on type"""
        if self.regression_type not in self.models:
            raise ValueError(f"Unknown regression type: {self.regression_type}. "
                           f"Available options: {list(self.models.keys())}")
        return self.models[self.regression_type]
    
    def _get_position_size(self, prediction_error_z, volatility):
        """Determine position size based on prediction error and volatility"""
        # Base position size based on z-score
        position_size = np.clip(prediction_error_z / self.z_score_threshold, -1, 1)
        
        # Scale by volatility if enabled
        if self.volatility_scaling and volatility > 0:
            position_size /= volatility
            
        return position_size
    
    def generate_signals(self, pca_components=3, threshold=1, r2_exit=0.7):
        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('PCA', PCA(n_components=min(pca_components, self.lookback_window))),
                             ('model', self._get_model())])
        
        for t in self.tickers:
            X, y = self._get_features_targets(t)
            preds = [0] * self.lookback_window
            exit_signal = [0] * self.lookback_window
            for i in range(self.lookback_window, len(self.df)-self.lookback_window, self.lookback_window):
                X_temp, y_temp = X[i-self.lookback_window: i], y[i-self.lookback_window: i]
                try:
                    pipeline.fit(X_temp, y_temp)
                    r2 = r2_score(y_temp, pipeline.predict(X_temp))
                except:
                    pred_price = pipeline.predict(X[i:i+self.lookback_window])
                    
                pred_price = pipeline.predict(X[i:i+self.lookback_window])
                preds += list(pred_price)

                if r2 < r2_exit:
                    exit_signal += [1] * self.lookback_window
                else:
                    exit_signal += [0] * self.lookback_window
            
            if len(preds) < len(self.df):
                pred_price = pipeline.predict(X[-(len(self.df)-len(preds)):])
                preds += list(pred_price)

                if r2 < r2_exit:
                    exit_signal += [1] * (len(self.df)-len(exit_signal))
                else:
                    exit_signal += [0] * (len(self.df)-len(exit_signal))

            preds = np.array(preds)[:len(self.df)]
            self.df[t+'_pred'] = preds
            self.df[t+'_is_trend'] = exit_signal
            self.df[t+'_signal'] = np.where(self.df[t+'_pred'] < -threshold, -1, 0)
            self.df[t+'_signal'] = np.where(self.df[t+'_pred'] > threshold, 1, self.df[t+'_signal'])
            self.df[t+'_signal'] = np.where(self.df[t+'_is_trend'] != 1, 0, self.df[t+'_signal'])
            self.df[t+'_exit_signal'] = exit_signal

        return self.df

    def generate_single_signal(self, t, prices, pca_components=3, execute_threshold=1, r2_exit=0.7):
        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('PCA', PCA(n_components=min(pca_components, self.lookback_window))),
                             ('model', self._get_model())])
        
        signal_df = pd.DataFrame()
        signal_df['Tickers'] = [t]

        bid, mid, ask = prices[0], prices[1], prices[2]
        price = mid

        X, y = self._get_features_targets(t)
        X_test = self._get_test_features(t)
        pipeline.fit(X, y)
        r2 = r2_score(y, pipeline.predict(X))
        pred = pipeline.predict(X_test)[-1]

        is_trend = False
        if r2 < r2_exit:
            is_trend = True

        signal = 0
        exit_signal = 0
        if pred < -execute_threshold:
            signal = -1
            price = bid
        elif pred > execute_threshold:
            signal = 1
            price = ask

        if not is_trend:
            exit_signal = 1

        signals = self.generate_signals(pca_components, execute_threshold, r2_exit)
        backtest = Backtest(signals.copy(), tickers = self.tickers, test_start_date=start_date, test_end_date=end_date, stoploss=stoploss, drawdown_duration=drawdown_duration)
        weights = backtest.get_weights(rolling, weight_method, short).tail(1)

        # Store results
        signal_df['signals'] = [signal]
        signal_df['weights'] = [weights[t].item()]
        signal_df['exit_signals'] = [exit_signal]
        signal_df['Price'] = price
        return signal_df