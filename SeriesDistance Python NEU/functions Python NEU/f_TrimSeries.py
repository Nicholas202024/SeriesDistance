import numpy as np
from f_calc_hyd_case import f_calc_hyd_case

def f_trim_series(obs, obs_eventindex, sim, sim_eventindex):
    """
    Trims if necessary the obs and sim series until
    they both start with either a 'rise' or a 'fall'
    they both end with either a 'rise' or a 'fall'

    INPUT
        obs: (n,1) array with observed discharge. n = number of time steps
        obs_eventindex: (n,1) array with indices of observed events
        sim: (n,1) array with simulated discharge. n = number of time steps
        sim_eventindex: (n,1) array with indices of simulated events

    OUTPUT
        obs: trimmed observed series
        x_obs: global time position of the trimmed observed series
        sim: trimmed simulated series
        x_sim: global time position of the trimmed simulated series
    """

    # Determine the hydrological case for each timestep in the original time series
    hydcase_obs = f_calc_hyd_case(obs)  # -2=valley -1=drop 0=no feature 1=rise 2=peak
    hydcase_sim = f_calc_hyd_case(sim)

    # find the best starting points
    pos_obs_rise = np.where(hydcase_obs == 1)[0][0]
    pos_sim_rise = np.where(hydcase_sim == 1)[0][0]
    sum_pos_rise = pos_obs_rise + pos_sim_rise

    pos_obs_fall = np.where(hydcase_obs == -1)[0][0]
    pos_sim_fall = np.where(hydcase_sim == -1)[0][0]
    sum_pos_fall = pos_obs_fall + pos_sim_fall

    # choose the starting point pair where the least trimming is required
    if sum_pos_rise <= sum_pos_fall:
        start_obs = pos_obs_rise
        start_sim = pos_sim_rise
    else:
        start_obs = pos_obs_fall
        start_sim = pos_sim_fall

    # find the best end points
    pos_obs_rise = np.where(hydcase_obs == 1)[0][-1]
    pos_sim_rise = np.where(hydcase_sim == 1)[0][-1]
    sum_pos_rise = pos_obs_rise + pos_sim_rise

    pos_obs_fall = np.where(hydcase_obs == -1)[0][-1]
    pos_sim_fall = np.where(hydcase_sim == -1)[0][-1]
    sum_pos_fall = pos_obs_fall + pos_sim_fall

    # choose the end point pair where the least trimming is required
    if sum_pos_rise > sum_pos_fall:
        end_obs = pos_obs_rise
        end_sim = pos_sim_rise
    else:
        end_obs = pos_obs_fall
        end_sim = pos_sim_fall

    # memorize temporal offsets of trimmed time series
    offset_obs = obs_eventindex[0] + start_obs - 1  # offset of trimmed obs series
    offset_sim = sim_eventindex[0] + start_sim - 1  # offset of trimmed sim series

    # trim the time series
    obs = obs[start_obs:end_obs + 1]  # obs trimmed
    x_obs = np.arange(offset_obs, offset_obs + len(obs))  # the global time position of obs

    sim = sim[start_sim:end_sim + 1]  # sim trimmed
    x_sim = np.arange(offset_sim, offset_sim + len(sim))  # the global time position of sim

    # update flow condition index/ hydrological cases
    hydcase_obs = f_calc_hyd_case(obs)
    hydcase_sim = f_calc_hyd_case(sim)

    return obs, x_obs, sim, x_sim