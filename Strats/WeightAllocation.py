import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

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


def hierarchical_risk_parity_weighting(returns, allow_short=True):
    """
    Allocate weights using Hierarchical Risk Parity (HRP),
    with optional short selling.
    
    """
    cov = returns.cov()
    corr = returns.corr()

    try:
        dist = np.sqrt(0.5 * (1 - corr))
        dist_array = squareform(dist.values, checks=False)
        linkage_matrix = linkage(dist_array, method='ward')
        ordered_linkage_matrix = optimal_leaf_ordering(linkage_matrix, dist_array)
        sorted_indices = leaves_list(ordered_linkage_matrix)
        sorted_assets = corr.index[sorted_indices].tolist()
    except ValueError as e:
        return equal_weighting(returns, allow_short)

    cov_sorted = cov.loc[sorted_assets, sorted_assets]

    def get_ivp(cov_):
        ivp = 1. / np.diag(cov_.values)
        ivp /= ivp.sum()
        return pd.Series(ivp, index=cov_.index)

    def get_cluster_var(cov_, cluster_items):
        cov_sub = cov_.loc[cluster_items, cluster_items]
        w_ = get_ivp(cov_sub)
        return np.dot(w_, np.dot(cov_sub, w_))

    def apply_constraints(alpha, weights, left, right, min_weights, max_weights):
        alpha = min(
            np.sum(max_weights[left]) / weights[left[0]],
            max(np.sum(min_weights[left]) / weights[left[0]], alpha),
        )
        alpha = 1 - min(
            np.sum(max_weights[right]) / weights[right[0]],
            max(np.sum(min_weights[right]) / weights[right[0]], 1 - alpha),
        )
        return alpha

    def recursive_bisection(cov_, assets, min_weights, max_weights):
        n = len(assets)
        weights = pd.Series(1.0, index=assets)
        stack = [assets]

        while stack:
            cluster = stack.pop()
            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            left_var = get_cluster_var(cov_, left)
            right_var = get_cluster_var(cov_, right)
            alpha = right_var / (left_var + right_var)

            # Apply constraint logic
            left_idx = [cov_.columns.get_loc(i) for i in left]
            right_idx = [cov_.columns.get_loc(i) for i in right]
            all_idx = [cov_.columns.get_loc(i) for i in cluster]

            alpha = apply_constraints(alpha, weights.values, left_idx, right_idx, min_weights, max_weights)

            weights[left] *= (1 - alpha)
            weights[right] *= alpha
            stack.extend([left, right])

        return weights

    n_assets = len(sorted_assets)
    if allow_short:
        min_weights = np.full(n_assets, -np.inf)
        max_weights = np.full(n_assets, np.inf)
    else:
        min_weights = np.zeros(n_assets)
        max_weights = np.ones(n_assets)

    weights = recursive_bisection(cov_sorted, sorted_assets, min_weights, max_weights)
    weights = weights.reindex(returns.columns).fillna(0)

    # Normalize so weights sum to 1 (or sum of abs to 1 if shorts allowed)
    if allow_short:
        weights /= np.abs(weights).sum()
    else:
        weights /= weights.sum()

    return weights.values
        
