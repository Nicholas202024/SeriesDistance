import numpy as np

def f_ReplaceEqualNeighbours(vals):
    """
    Replaces equal neighbouring values in a series
    Uwe Ehret, 15. Nov. 2013

    INPUT
        vals: (n,1) numpy array of values
    OUTPUT
        vals: (n,1) numpy array, same as input, but equal neighbours replaced
        count: number of found equal neighbours
    METHOD
        If a series of equal neighbouring values is found, they are successively increased by 1/1000
        --> in a sequence of equal values, the last one will be the largest
    """

    count = 0

    for z in range(len(vals) - 1):  # Loop over all values
        if vals[z] == vals[z + 1]:  # Equal neighbouring values?
            count = 1

            while (z + count) < len(vals) and vals[z + count] == vals[z]:  # Loop over all neighbouring equal values
                vals[z + count] = vals[z + count - 1] + 0.0001  # Increase the value of the previous point by 0.0001
                count += 1

    return vals