import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def sharpe_weighting(returns, allow_short=True):
    """
    Allocate weights proportional to asset sharpe
    Allows for short positions when allow_short=True
    """
    sharpe = returns.mean() / returns.std()
    weights = sharpe / sharpe.sum() * np.sqrt(252*12*24)

    if allow_short:
        # Allow weights to range between -1 and 1
        weights = weights * 2 - weights.sum()/len(weights)
        weights = weights / np.abs(weights).sum()  # Ensure sum of absolute weights = 1
    
    if len(weights[np.isnan(weights)]) == len(weights):
        return equal_weighting(returns)
    
    weights[np.isnan(weights)] = 0
    
    return weights.values

def inverse_volatility_weighting(returns, allow_short=True):
    """
    Allocate weights inversely proportional to asset volatility
    Allows for short positions when allow_short=True
    """
    volatilities = returns.std()
    inverse_vol = 1 / (volatilities+1e-10)
    weights = inverse_vol / inverse_vol.sum()

    if allow_short:
        # Allow weights to range between -1 and 1
        weights = weights * 2 - weights.sum()/len(weights)
        weights = weights / np.abs(weights).sum()  # Ensure sum of absolute weights = 1

    return weights.values

def mean_variance_optimization(returns, target_return=None, allow_short=True):
    """
    Markowitz portfolio optimization with shorting allowed
    """
    cov_matrix = returns.cov()
    expected_returns = returns.mean()
    n_assets = len(expected_returns)
    
    if target_return is None:
        target_return = expected_returns.mean()  # More reasonable default
    
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        {'type': 'eq', 'fun': lambda x: x @ expected_returns - target_return}  # target return
    ]
    
    if allow_short:
        bounds = [(-1, 1) for _ in range(n_assets)]  # Allow short positions (-100% to 100%)
    else:
        bounds = [(0, 1) for _ in range(n_assets)]  # long-only
    
    result = minimize(
        portfolio_volatility,
        x0=np.ones(n_assets)/n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def equal_weighting(returns, allow_short=True):
    n_assets = returns.shape[1]

    return np.ones(n_assets) / n_assets

class PortfolioAllocator:
    def __init__(self, returns):
        self.returns = returns
        self.cov_matrix = returns.cov()
        self.expected_returns = returns.mean()
        
    def allocate(self, method='inverse_vol', allow_short=True, **kwargs):
        if method == 'inverse_vol':
            return inverse_volatility_weighting(self.returns, allow_short=allow_short)
        elif method == 'mean_var':
            return mean_variance_optimization(self.returns, allow_short=allow_short, **kwargs)
        elif method == 'equal':
            return equal_weighting(self.returns, allow_short=allow_short)
        else:
            raise ValueError(f"Unknown method: {method}")
        
