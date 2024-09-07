def f_ComputeContingencyTable(obs_events, sim_events, obs_sim_events_mapped):
    """
    Computes the contingency table for observed and simulated events.
    """
    # Initialize the contingency table with NaN values
    contingency_table = {'hits': float('nan'), 'misses': float('nan'), 'false_alarms': float('nan')}

    # Calculate hits, misses, and false alarms
    contingency_table['hits'] = len(obs_sim_events_mapped)
    contingency_table['misses'] = len(obs_events) - contingency_table['hits']
    contingency_table['false_alarms'] = len(sim_events) - contingency_table['hits']

    return contingency_table