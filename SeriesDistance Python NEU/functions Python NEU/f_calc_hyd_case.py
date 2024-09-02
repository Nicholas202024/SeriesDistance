import numpy as np

def f_calc_hyd_case(vals):
    """
    Returns the hydrological case for each timestep of a hydrological timeseries
    Uwe Ehret, 15.Nov.2013

    INPUT
        vals: (n,1) array with values
    OUTPUT
        hydcase: (n,1) array with hydrological case: -2=valley -1=drop, 1=rise 2=peak
    METHOD
        for each point, calculates the gradient to the previous and the next value
        drop-rise: valley   drop-drop: drop   rise-rise: rise   rise-drop : peak  
    """

    len_vals = len(vals)
    hydcase = np.full(len_vals, np.nan)  # initialize result array

    # Find hydrological case for first and last timestep (special case due to incomplete neighborhood)
    if (vals[1] - vals[0]) < 0:
        hydcase[0] = -1
    else:
        hydcase[0] = 1

    if (vals[-1] - vals[-2]) < 0:
        hydcase[-1] = -1
    else:
        hydcase[-1] = 1

    # Find hydrological case for each timestep
    for z in range(1, len_vals - 1):  # loop over all values except the first and last
        if (vals[z] - vals[z - 1]) < 0 and (vals[z + 1] - vals[z]) > 0:
            hydcase[z] = -2  # drop-rise: valley
        elif (vals[z] - vals[z - 1]) < 0 and (vals[z + 1] - vals[z]) < 0:
            hydcase[z] = -1  # drop-drop: drop
        elif (vals[z] - vals[z - 1]) > 0 and (vals[z + 1] - vals[z]) > 0:
            hydcase[z] = 1   # rise-rise: rise
        elif (vals[z] - vals[z - 1]) > 0 and (vals[z + 1] - vals[z]) < 0:
            hydcase[z] = 2   # rise-drop: peak

    return hydcase