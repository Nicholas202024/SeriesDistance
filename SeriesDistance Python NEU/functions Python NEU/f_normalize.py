import numpy as np

def f_normalize(k):
    """
    Function normalizes the range of the input vector k to the range [0, 1].
    """
    k = (k - np.nanmin(k)) / (np.nanmax(k) - np.nanmin(k))
    
    # Return zeros if all are NaN (e.g., relevant for e_sd_t if time series are shifted horizontally)
    if np.all(np.isnan(k)):
        k = np.zeros(len(k))
    
    return k