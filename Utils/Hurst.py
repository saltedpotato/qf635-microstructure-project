from Utils.import_files import *
def hurst_exponent(ts: pd.Series, max_lags: int = 50) -> float:
    """
    Calculate the Hurst Exponent of a time series.
        
    Args:
        time_series (pd.Series): Input time series (e.g., stock prices, spreads).
        max_lags (int): Maximum number of lags to compute rescaled range (R/S).
        
    Returns:
        float: Hurst Exponent value.
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

