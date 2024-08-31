import numpy as np

def f_AggregateSegment(segs, hydcase, y, seg2erase=None):
    """
    Erases a specified or the least relevant segment of an event. The segment is then merged with its neighbors.
    Uwe Ehret, 15.Nov.2013, modified: Simon Seibert March 3rd 2014

    Parameters:
    segs (list): List of segment dictionaries, where each dictionary contains segment properties.
    hydcase (np.ndarray): (n,1) matrix with hydrological case: -2=valley, -1=drop, 1=rise, 2=peak.
    y (np.ndarray): Array of discharge values.
    seg2erase (int, optional): Number of the segment to erase.

    Returns:
    tuple: Updated segments list and hydrological cases.
    """
    
    # Identify the least relevant segment if the segment to erase is not specified
    if seg2erase is None:  # the segment to erase is not specified
        if len(segs) > 2:  # if more than the first and last segment are left ...
            dum = [seg['relevance'] for seg in segs]
            dum = dum[1:-1]  # exclude the first and last from deletion
            seg2erase = dum.index(min(dum)) + 1  # find the least relevant segment
        elif len(segs) == 2:  # only the first and last of the segments are left
            dum = [seg['relevance'] for seg in segs]
            seg2erase = dum.index(min(dum))
        else:  # less than 2 segments left
            raise ValueError('f_AggregateSegment: less than 2 segments left!')
    else:  # the segment to erase is specified
        if len(segs) > 2:  # if more than the first and last segment are left ...
            if seg2erase == 1 or seg2erase == len(segs):  # if more than the first and last segment are left ...
                raise ValueError('f_AggregateSegment: more than 2 segments left: cannot erase the first or last segment!')
        elif len(segs) == 2:  # only the first and last of the segments are left
            # no problem
            pass
        else:  # less than 2 segments left
            raise ValueError('f_AggregateSegment: less than 2 segments left!')

    # Join and modify segment properties

    # Delete the first segment and join it with the second
    if seg2erase == 1:
        # Adjust the hydrological cases (take over the value of the following segment)
        hydcase[segs[seg2erase]['starttime_local']:segs[seg2erase]['endtime_local']] = hydcase[segs[seg2erase]['starttime_local'] + 1]

        # Join 2 segments: the segment to erase, and the following. The following takes it all
        segs[seg2erase + 1]['starttime_local'] = segs[seg2erase]['starttime_local']
        segs[seg2erase + 1]['starttime_global'] = segs[seg2erase]['starttime_global']
        segs[seg2erase + 1]['length'] = segs[seg2erase + 1]['endtime_local'] - segs[seg2erase]['starttime_local']
        segs[seg2erase + 1]['rel_length'] = (segs[seg2erase + 1]['endtime_local'] - segs[seg2erase]['starttime_local']) / (len(y) - 1)
        segs[seg2erase + 1]['sum_dQ'] = sum(np.diff(y[segs[seg2erase]['starttime_local']:segs[seg2erase + 1]['endtime_local']]))
        segs[seg2erase + 1]['rel_dQ'] = sum(abs(np.diff(y[segs[seg2erase]['starttime_local']:segs[seg2erase + 1]['endtime_local']]))) / sum(abs(np.diff(y)))
        segs[seg2erase + 1]['relevance'] += segs[seg2erase]['relevance']

        # Erase the first segment
        del segs[seg2erase]

    # Delete the last segment and join it with the second last
    elif seg2erase == len(segs):
        # Adjust the hydrological cases (take over the value of the previous segment)
        hydcase[segs[seg2erase]['starttime_local']:segs[seg2erase]['endtime_local']] = hydcase[segs[seg2erase]['starttime_local'] - 1]

        # Join 2 segments: the segment to erase, and the previous. The previous takes it all
        segs[seg2erase - 1]['endtime_local'] = segs[seg2erase]['endtime_local']
        segs[seg2erase - 1]['endtime_global'] = segs[seg2erase]['endtime_global']
        segs[seg2erase - 1]['length'] = segs[seg2erase]['endtime_local'] - segs[seg2erase - 1]['starttime_local']
        segs[seg2erase - 1]['rel_length'] = (segs[seg2erase]['endtime_local'] - segs[seg2erase - 1]['starttime_local']) / (len(y) - 1)
        segs[seg2erase - 1]['sum_dQ'] = sum(np.diff(y[segs[seg2erase - 1]['starttime_local']:segs[seg2erase]['endtime_local']]))
        segs[seg2erase - 1]['rel_dQ'] = sum(abs(np.diff(y[segs[seg2erase - 1]['starttime_local']:segs[seg2erase]['endtime_local']]))) / sum(abs(np.diff(y)))
        segs[seg2erase - 1]['relevance'] += segs[seg2erase]['relevance']

        # Erase the last segment
        del segs[seg2erase]

    # 3 or more segments are left, join all three of them and update segment properties
    else:
        # Adjust the hydrological cases (take over the value of the previous segment)
        hydcase[segs[seg2erase]['starttime_local']:segs[seg2erase]['endtime_local']] = hydcase[segs[seg2erase]['starttime_local'] - 1]

        # Join 3 segments: the segment to erase, the previous and the following. The previous takes it all
        segs[seg2erase - 1]['endtime_local'] = segs[seg2erase + 1]['endtime_local']
        segs[seg2erase - 1]['endtime_global'] = segs[seg2erase + 1]['endtime_global']
        segs[seg2erase - 1]['length'] = segs[seg2erase + 1]['endtime_local'] - segs[seg2erase - 1]['starttime_local']
        segs[seg2erase - 1]['rel_length'] = (segs[seg2erase + 1]['endtime_local'] - segs[seg2erase - 1]['starttime_local']) / (len(y) - 1)
        segs[seg2erase - 1]['sum_dQ'] = sum(np.diff(y[segs[seg2erase - 1]['starttime_local']:segs[seg2erase + 1]['endtime_local']]))
        segs[seg2erase - 1]['rel_dQ'] = sum(abs(np.diff(y[segs[seg2erase - 1]['starttime_local']:segs[seg2erase + 1]['endtime_local']]))) / sum(abs(np.diff(y)))
        segs[seg2erase - 1]['relevance'] += segs[seg2erase]['relevance'] + segs[seg2erase + 1]['relevance']

        # Erase the segment to erase and the following
        # NOTE: here the order of deletion is important! Work from the end backwards, otherwise enumeration gets screwed up!
        del segs[seg2erase + 1]
        del segs[seg2erase]

    return segs, hydcase