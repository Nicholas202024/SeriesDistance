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
        segs: (1,x) list of dicts, where x is the number of segments found in the entire event
        Note: A segment always includes its first and last point (start, valley, peak or end) --> peaks and valleys are used twice!
    """

    # initialize structure
    segs = []

    # find all segments
    segs.append({
        'starttime_global': x[0],
        'starttime_local': 0
    })

    for z in range(1, len(y) - 1):  # loop over all values except the first and last
        # print('Input for f_calc_hyd_case:')
        # print('type y: ', type(y))
        # print('dtype y: ', y.dtype)
        # print('shape y: ', y.shape)
        # print('\n')

        hydcase = f_calc_hyd_case(y)

        # print('Output of f_calc_hyd_case:')
        # print('type hydcase: ', type(hydcase))
        # print('dtype hydcase: ', hydcase.dtype)
        # print('shape hydcase: ', hydcase.shape)
        # print('\n')
        
        if hydcase[z] == 2 or hydcase[z] == -2:  # peak or valley
            segs[-1]['endtime_global'] = x[z]
            segs[-1]['endtime_local'] = z
            segs.append({
                'starttime_global': x[z],  # (end+1) is on purpose! It creates a new segment template in the structure
                'starttime_local': z       # (end) is on purpose! As a new segment has been created in the previous line, the entry is now made in the new segment
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
        seg['rel_dQ'] = (np.sum(np.abs(np.diff(y[seg['starttime_local']:seg['endtime_local'] + 1])))) / np.sum(np.abs(np.diff(y)))  # discharge changes relative to entire event [0,1]
        # relative importance of the segment
        seg['relevance'] = np.sqrt(seg['rel_length']**2 + seg['rel_dQ']**2)  # relevance as the euclidean distance of rel_length and rel_dQ

    # normalize the segment relevance with overall relevance of the entire event [0,1]
    total_relevance = np.sum([seg['relevance'] for seg in segs])
    for seg in segs:
        seg['relevance'] = seg['relevance'] / total_relevance

    return segs