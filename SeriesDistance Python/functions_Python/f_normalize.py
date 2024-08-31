import numpy as np

def f_normalize(k):
    """
    Normalizes the range of the input vector k to the range [0, 1].

    Parameters:
    k (np.ndarray): Input vector to be normalized.

    Returns:
    np.ndarray: Normalized vector.
    """
    # Normalize the range of the input vector k to the range [0, 1]
    k = (k - np.nanmin(k)) / (np.nanmax(k) - np.nanmin(k))
    
    # Return zeros if all values are NaN (e.g., relevant for e_sd_t if time series are shifted horizontally)
    if np.all(np.isnan(k)):
        k = np.zeros(len(k))
    
    return k