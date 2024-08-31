import numpy as np

def f_ReplaceEqualNeighbours(vals):

    """
    Replaces equal neighbouring values in a series.
    Uwe Ehret, 15.Nov.2013

    Parameters:
    vals (numpy array): (n,1) array with values

    Returns:
    vals (numpy array): (n,1) array, same as input, but equal neighbors replaced
    count (int): number of equal neighbors found

    Method:
    If a series of equal neigbouring values is found, they are successively raised by 1/1000 
    --> in a sequence of equal values, the last will become the largest
    """

    count = 0
    vals = np.array(vals)  # Ensure vals is a numpy array

    for z in range(len(vals) - 1):  # loop over all values
        if vals[z] == vals[z + 1]:  # Equal neighbouring values?
            count = 1

            while (z + count) < len(vals) and vals[z + count] == vals[z]:  # Loop over all neighboring equal values
                vals[z + count] = vals[z + count - 1] + 0.0001  # raise the value of the previous point by 0.0001
                count += 1

    return vals, count