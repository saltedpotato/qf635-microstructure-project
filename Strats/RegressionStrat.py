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
        for lag in range(1, self.lookback_window):
            df_preprocessed[f'lag_{lag}'] = returns.shift(lag)
        
        df_preprocessed['rolling_vol'] = returns.rolling(window=self.lookback_window).std().shift(1)
        
        df_preprocessed[t] = returns
        # Align and drop NA
        df_preprocessed = df_preprocessed.dropna().reset_index(drop=True)
        X, y = df_preprocessed.drop(columns=[t]), df_preprocessed[t]
        
        return X, y
    
    
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
    
    def generate_signals(self, pca_components=3, threshold=1):
        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('PCA', PCA(n_components=min(pca_components, self.lookback_window))),
                             ('model', self._get_model())])
        
        for t in self.tickers:
            X, y = self._get_features_targets(t)
            signal = [0] * self.lookback_window
            is_trend = [0] * self.lookback_window
            for i in range(self.lookback_window, len(self.df)):
                X_temp, y_temp = X[i-self.lookback_window: i], y[i-self.lookback_window: i]
                pipeline.fit(X_temp, y_temp)

                pred_price = pipeline.predict(X[:i+1])[-1]
                
                if pred_price > threshold:
                    signal += [1]
                elif pred_price < -threshold:
                    signal += [-1]
                else:
                    signal += [0]

            for i in range(self.lookback_window, len(self.df)):
                if hurst_exponent(self.df[t][i-self.lookback_window:i].values) > 0.7:
                    is_trend += [1]
                else:
                    is_trend += [0]
            
            self.df[t+'_signal'] = signal[:len(self.df)]
            self.df[t+'_exit_signal'] = is_trend[:len(self.df)]

        return self.df

    