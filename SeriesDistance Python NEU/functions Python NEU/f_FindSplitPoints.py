import numpy as np
from f_define_segments import f_define_segments

def f_FindSplitPoints(obs, sim, split_frequency):
    """
    Method
    - overall, we want to split the time series at times where both obs and sim are in low flow
    - at each candidate split point, sample obs-sim pairs within a searchrange of
      'perc' percent of the split frequency (forward and backward in time)
    - from each sample set, determine the ranks of the obs and sim values after sorting by size
    - the time step that has min(rank(obs) + rank(sim)) is the best split candidate
    - compare the obs and sim values of that time step with the cdf of the entire time series: 
      In order to make a good split, the values should be also globally of low rank, expressed by parameter 'max_quantile' (probability of unexceedance)
      If the obs and sim values are BOTH below the Pu limit of their time series, a split is done. If not, not. UNLESS ...
      there are more than 'max_num_segs' segments in the obs or sim timeseries since the last split. This will make the iterative SD
      algorithm very slow. Therefore, in this case, keep the candidate split point nevertheless
    """

    if split_frequency is None or np.isnan(split_frequency):
        print('Warning: split_frequency IS EITHER MISSING OR NaN')
        return []

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

            # print('Input for f_define_segments:')
            # print('type times: ', type(times))
            # print('dtype times: ', times.dtype)
            # print('shape times: ', times.shape)
            # print('\n')
            # print('type vals_obs: ', type(vals_obs))
            # print('dtype vals_obs: ', vals_obs.dtype)
            # print('shape vals_obs: ', vals_obs.shape)
            # print('\n')
            # print('type vals_sim: ', type(vals_sim))
            # print('dtype vals_sim: ', vals_sim.dtype)
            # print('shape vals_sim: ', vals_sim.shape)
            # print('\n')

            segs_obs = f_define_segments(times, vals_obs)
            segs_sim = f_define_segments(times, vals_sim)

            # print('Output of f_define_segments:')
            # print('type segs_obs: ', type(segs_obs))
            # print('\n')
            # print('type segs_sim: ', type(segs_sim))
            # print('\n')

            if (len(segs_obs) >= max_num_segs) or (len(segs_sim) >= max_num_segs):
                timeseries_splits.append(best_split_time)

    # add the end of the time series (mandatory)
    timeseries_splits.append(len(obs) - 1)

    return timeseries_splits