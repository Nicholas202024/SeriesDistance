import numpy as np
from f_DefineSegments import f_DefineSegments

def f_FindSplitPoints(obs, sim, split_frequency):
    """
    Splits the time series at times where both obs and sim are in low flow.

    Parameters:
    obs (numpy array): Observed values
    sim (numpy array): Simulated values
    split_frequency (int): Frequency at which to consider splits

    Returns:
    timeseries_splits (list): List of split points
    """
    # method
    # - overall, we want to split the time series at times where both obs and sim are in low flow
    # - at each candidate split point, sample obs-sim pairs within a searchrange of
    #   'perc' percent of the split frequency (forward and backward in time)
    # - from each sample set, determine the ranks of the obs and sim values after sorting by size
    # - the time step that has min(rank(obs) + rank(sim)) is the best split canditate
    # - compare the obs and sim values of that time step with the cdf of the entire time series: 
    #   In order to make a good split, the values should be also globally of low rank, expressed by paramater 'max_quantile' (probability of unexceedance)
    #   If the obs and sim values are BOTH below the Pu limit of their time series, a split is done. If not, not. UNLESS ...
    #   there are more than 'max_num_segs' segments in the obs or sim timeseries since the last split. This will make the iterative SD
    #   algorithm very slow. Therefore, in this case, keep the candidate split point nevertheless

    if split_frequency is None or np.isnan(split_frequency):
        raise ValueError('split_frequency IS EITHER MISSING OR NaN')

    max_quantile = 0.50  # candidate split points are only kept if both the obs and sim value are low enough to be below this probability of unexceedance of the entire time series
    max_num_segs = 15    # a candidate split point is kept if since the last split point, more segments are contained in the obs or sim series
    perc = 15            # the region around a split point used for searching the optimal split point (percent of the split frequency)

    searchrange = round((perc / 100) * split_frequency)

    # determine t
    obs_max_global = np.quantile(obs, max_quantile)
    sim_max_global = np.quantile(sim, max_quantile)

    # build the list of splits

    # add the start of the time series (mandatory)
    timeseries_splits = [0]

    for i in range(split_frequency, len(obs) - split_frequency, split_frequency):
        obs_sample = obs[i - searchrange : i + searchrange + 1]
        sim_sample = sim[i - searchrange : i + searchrange + 1]
        obs_ranks = np.argsort(np.argsort(obs_sample))
        sim_ranks = np.argsort(np.argsort(sim_sample))
        best_split_time = np.argmin(obs_ranks + sim_ranks)
        best_split_time = i - searchrange + best_split_time

        if (obs[best_split_time] <= obs_max_global) and (sim[best_split_time] <= sim_max_global):
            # if at the candidate split point, both obs and sim are small enough, add the point in time to the list
            timeseries_splits.append(best_split_time)
        else:
            # if there are many segments in obs or sim since the last split point, keep the split point anyways
            times = np.arange(timeseries_splits[-1], best_split_time + 1)
            vals_obs = obs[times]
            vals_sim = sim[times]
            segs_obs = f_DefineSegments(times, vals_obs)
            segs_sim = f_DefineSegments(times, vals_sim)

            if (len(segs_obs) >= max_num_segs) or (len(segs_sim) >= max_num_segs):
                timeseries_splits.append(best_split_time)
            # original code:
            # if (segs_obs.shape[1] >= max_num_segs) or (segs_sim.shape[1] >= max_num_segs):
            #     timeseries_splits.append(best_split_time)

    # add the end of the time series (mandatory)
    timeseries_splits.append(len(obs) - 1)

    return timeseries_splits