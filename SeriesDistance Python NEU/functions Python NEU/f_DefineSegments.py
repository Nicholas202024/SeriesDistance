import numpy as np
from f_calc_hyd_case import f_calc_hyd_case

def f_define_segments(x, y):
    """
    Defines all segments in an event (starttime, endtime, relative duration, relative magnitude change (dQ fraction), importance)
    Uwe Ehret, 15.Nov.2013, modified Simon Seibert, March 2014

    INPUT
        x: (n,1) array with time position (x-position) of values
        y: (n,1) array with values
    OUTPUT
        segs: list of dictionaries, where each dictionary represents a segment found in the entire event
        Note: A segment always includes its first and last point (start, valley, peak or end) --> peaks and valleys are used twice!
    """

    # Initialize structure
    segs = []

    # Find all segments
    segs.append({
        'starttime_global': x[0],
        'starttime_local': 0
    })

    hydcase = f_calc_hyd_case(y)
    for z in range(1, len(y) - 1):  # loop over all values except the first and last
        if hydcase[z] == 2 or hydcase[z] == -2:  # peak or valley
            segs[-1]['endtime_global'] = x[z]
            segs[-1]['endtime_local'] = z
            segs.append({
                'starttime_global': x[z],
                'starttime_local': z
            })

    segs[-1]['endtime_global'] = x[-1]
    segs[-1]['endtime_local'] = len(y) - 1

    # Compute segment properties
    for seg in segs:
        # Length of the segment
        seg['length'] = seg['endtime_local'] - seg['starttime_local']
        # Relative length to the entire time series [0,1]
        seg['rel_length'] = (seg['endtime_local'] - seg['starttime_local']) / (len(y) - 1)
        # Sum of segment slopes
        seg['sum_dQ'] = np.sum(np.diff(y[seg['starttime_local']:seg['endtime_local'] + 1]))
        # Sum of slopes relative to entire event [0,1]
        seg['rel_dQ'] = np.sum(np.abs(np.diff(y[seg['starttime_local']:seg['endtime_local'] + 1]))) / np.sum(np.abs(np.diff(y)))
        # Relative importance of the segment
        seg['relevance'] = np.sqrt(seg['rel_length']**2 + seg['rel_dQ']**2)  # relevance as the Euclidean distance of rel_length and rel_dQ

    # Normalize the segment relevance with overall relevance of the entire event [0,1]
    total_relevance = np.sum([seg['relevance'] for seg in segs])
    for seg in segs:
        seg['relevance'] = seg['relevance'] / total_relevance

    return segs