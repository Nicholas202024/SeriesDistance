import numpy as np
from f_CalcHydCase import f_CalcHydCase

def f_DefineSegments(x, y):
    """
    Defines all segments in an event (starttime, endtime, relative duration, relative magnitude change (dQ fraction), importance)
    Uwe Ehret, 15.Nov.2013, modified Simon Seibert, March 2014

    Parameters:
    x (numpy array): (n,1) array with time position (x-position) of values
    y (numpy array): (n,1) array with values

    Returns:
    segs (list of dicts): List of segments found in the entire event
    Note: A segment always includes its first and last point (start, valley, peak or end) --> peaks and valleys are used twice!
    """

    # initialize structure
    segs = []

    # find all segments
    segs.append({
        'starttime_local': 0,
        'endtime_local': None,
        'starttime_global': x[0],
        'endtime_global': None,
        'length': None,
        'rel_length': None,
        'sum_dQ': None,
        'rel_dQ': None,
        'relevance': None
    })

    for z in range(1, len(y) - 1):  # loop over all values except the first and last
        hydcase = f_CalcHydCase(y)
        if hydcase[z] == 2 or hydcase[z] == -2:  # peak or valley
            segs[-1]['endtime_global'] = x[z]
            segs[-1]['endtime_local'] = z
            segs.append({
                'starttime_local': z,
                'endtime_local': None,
                'starttime_global': x[z],
                'endtime_global': None,
                'length': None,
                'rel_length': None,
                'sum_dQ': None,
                'rel_dQ': None,
                'relevance': None
            })

    segs[-1]['endtime_global'] = x[-1]
    segs[-1]['endtime_local'] = len(y) - 1

    # compute segment properties
    for seg in segs:
        # length of the segment
        seg['length'] = seg['endtime_local'] - seg['starttime_local']
        # relative length to the entire time series [0,1]
        seg['rel_length'] = (seg['endtime_local'] - seg['starttime_local']) / (len(y) - 1)
        # sum of segment slopes
        seg['sum_dQ'] = np.sum(np.diff(y[seg['starttime_local']:seg['endtime_local'] + 1]))
        # sum of slopes relative to entire event [0,1]
        seg['rel_dQ'] = (np.sum(np.abs(np.diff(y[seg['starttime_local']:seg['endtime_local'] + 1])))) / np.sum(np.abs(np.diff(y)))
        # relative importance of the segment
        seg['relevance'] = np.sqrt(seg['rel_length']**2 + seg['rel_dQ']**2)  # relevance as the euclidean distance of rel_length and rel_dQ

    # normalize the segment relevance with overall relevance of the entire event [0,1]
    Relevance = np.sum([seg['relevance'] for seg in segs])
    for seg in segs:
        seg['relevance'] = seg['relevance'] / Relevance

    return segs