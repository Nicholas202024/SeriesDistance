import numpy as np

def f_SD_1dNoEventError(obs, sim, obs_events, sim_events, obs_sim_events_mapped, error_model):
    """
    Determines the 1-d error distribution between obs and sim for all time steps that are neither part of an obs nor a sim event
    """
    
    bad_times = []  # list of all times that shall NOT be used to determine the NoEventError, because they are part of either an obs or sim event
    cons = {'x_match_obs_global': [], 'y_match_obs': [], 'x_match_sim_global': [], 'y_match_sim': []}

    # loop over all items in the matching events list
    for i in range(obs_sim_events_mapped.shape[0]):
        start_obs = obs_sim_events_mapped[i, 0]
        start_sim = obs_sim_events_mapped[i, 1]
        end_obs = obs_events[np.where(obs_events[:, 0] == start_obs), 1][0]
        end_sim = sim_events[np.where(sim_events[:, 0] == start_sim), 1][0]

        bad_start = min(start_obs, start_sim)  # find the earlier start of the related obs and sim event
        bad_end = max(end_obs, end_sim)        # find the later end of the related obs and sim event

        bad_start = int(bad_start)
        bad_end = int(bad_end)

        # print('bad_start: ', bad_start)
        # print('bad_start type: ', type(bad_start))
        # print('bad_end: ', bad_end)
        # print('bad_end type: ', type(bad_end))
        # print('\n')

        bad_times.extend(range(bad_start, bad_end + 1))  # exclude these time steps from the 1d error calculation of low-flows

    # the list of times that shall be used for the NoEventError are ALL times minus the times that shall NOT be used
    good_times = np.setdiff1d(np.arange(1, obs.shape[0] + 1), bad_times)

    # determine the errors
    if error_model == 'standard':  # compute the simple difference
        e_q_1d = obs[good_times - 1] - sim[good_times - 1]  # > 0 means obs is larger than sim
    elif error_model == 'relative':  # compute a scaled difference
        e_q_1d = (obs[good_times - 1] - sim[good_times - 1]) / ((obs[good_times - 1] + sim[good_times - 1]) * 0.5)  # > 0 means obs is larger than sim
    else:
        raise ValueError('distance function not properly specified')

    # save matching points (for plotting)
    cons['x_match_obs_global'] = good_times
    cons['y_match_obs'] = obs[good_times - 1]
    cons['x_match_sim_global'] = good_times
    cons['y_match_sim'] = sim[good_times - 1]

    return e_q_1d, cons