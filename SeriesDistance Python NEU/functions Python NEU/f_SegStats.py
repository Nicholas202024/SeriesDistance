def f_SegStats(segs):
    """
    Calculate segment statistics.
    
    Parameters:
    segs (list): List of dictionaries with keys 'sum_dQ', 'length', 'endtime_global', and 'starttime_global'.
    
    Returns:
    list: A list containing the number of peaks, number of troughs, total rise duration, total fall duration, and total duration.
    """
    
    # Calculate the number of peaks (segments with positive sum_dQ)
    peaks = len([seg for seg in segs if seg['sum_dQ'] > 0])
    
    # Calculate the number of troughs (segments with negative sum_dQ)
    troughs = len([seg for seg in segs if seg['sum_dQ'] < 0])
    
    # Calculate the total rise duration (sum of lengths of segments with positive sum_dQ)
    rise_dur = sum(seg['length'] for seg in segs if seg['sum_dQ'] > 0)
    
    # Calculate the total fall duration (sum of lengths of segments with negative sum_dQ)
    fall_dur = sum(seg['length'] for seg in segs if seg['sum_dQ'] < 0)
    
    # Calculate the total duration (difference between the end time of the last segment and the start time of the first segment)
    tot_dur = segs[-1]['endtime_global'] - segs[0]['starttime_global']
    
    # Summarize in a single list
    segment_stats = [peaks, troughs, rise_dur, fall_dur, tot_dur]
    
    return segment_stats