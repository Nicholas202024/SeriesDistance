import numpy as np
from f_CalcHydCase import f_CalcHydCase

def f_TrimSeries(obs, obs_eventindex, sim, sim_eventindex):
    """
    Trims if necessary the obs and sim series until
    they both start with either a 'rise' or a 'fall'
    they both end with either a 'rise' or a 'fall'

    Parameters:
    obs (np.ndarray): (n,1) matrix with observed discharge. n = number of time steps
    obs_eventindex (np.ndarray): indices of the observed events
    sim (np.ndarray): (n,1) matrix with simulated discharge. n = number of time steps
    sim_eventindex (np.ndarray): indices of the simulated events

    Returns:
    tuple: trimmed observed and simulated series along with their global time positions
    """
    
    # Determine the hydrological case for each timestep in the original time series
    hydcase_obs = f_CalcHydCase(obs)  # -2=valley -1=drop 0=no feature 1=rise 2=peak
    hydcase_sim = f_CalcHydCase(sim)

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
    hydcase_obs = f_CalcHydCase(obs)
    hydcase_sim = f_CalcHydCase(sim)

    return obs, x_obs, sim, x_sim