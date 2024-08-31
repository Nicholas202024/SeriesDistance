import numpy as np
from scipy.interpolate import interp1d

def f_smooth_DP(obs, sim, nse_smooth_limit):

    # calculate statistics to compare original and smoothed time series
    def calculate_extremes(series):
        # number of extremes
        local_mins = len(np.where(np.diff(np.sign(np.diff(series))) == 2)[0]) + 1 # number of positive sign changes in the temporal derivative of obs (local maxima)
        local_maxs = len(np.where(np.diff(np.sign(np.diff(series))) == -2)[0]) + 1 # number of negative sign changes in the temporal derivative of obs (local minima)
        return local_mins + local_maxs

    obs_tot_extremes = calculate_extremes(obs)
    sim_tot_extremes = calculate_extremes(sim)

    SumAbsSIM = np.sum(np.abs(np.diff(sim)))
    SumAbsOBS = np.sum(np.abs(np.diff(obs)))

    print(f'original obs: var: {np.var(obs)}, # extremes: {obs_tot_extremes}, diff(obs)={SumAbsOBS}')
    print(f'original sim: var: {np.var(sim)}, # extremes: {sim_tot_extremes}, diff(sim)={SumAbsSIM}')

    obs_old = obs.copy()  # keep the original time series
    xes = np.arange(1, len(obs) + 1)  # create x data (points in time)
    xy = np.column_stack((xes, obs_old))  # prepare input for f_dp1d

    # Placeholder for f_dp1d function
    def f_dp1d(xy, param1, param2):
        # This is a placeholder. Replace with the actual implementation.
        return xy

    # simplify obs up to a point number criterion
    xy_dp = f_dp1d(xy, -999, sim_tot_extremes)

    # sample the simplified line at the original x-locations (points in time)
    interp_func = interp1d(xy_dp[:, 0], xy_dp[:, 1], kind='linear')
    obs = interp_func(xes)

    # Uncomment and repeat for sim if needed
    # sim_old = sim.copy()  # keep the original time series
    # xes = np.arange(1, len(sim) + 1)  # create x data (points in time)
    # xy = np.column_stack((xes, sim_old))  # prepare input for f_dp1d
    # num_points = xy_dp.shape[0]  # the number of points of the simplified 'sim' time series
    # xy_dp = f_dp1d(xy, -999, num_points)
    # interp_func = interp1d(xy_dp[:, 0], xy_dp[:, 1], kind='linear')
    # sim = interp_func(xes)

    # number of extremes smoothed time series
    obs_tot_extremes = calculate_extremes(obs)
    sim_tot_extremes = calculate_extremes(sim)

    SumAbsSIM = np.sum(np.abs(np.diff(sim)))
    SumAbsOBS = np.sum(np.abs(np.diff(obs)))

    # plot statistics on the command window
    print(f'smoothed obs: var: {np.var(obs)}, # extremes: {obs_tot_extremes}, diff(obs)={SumAbsOBS}')
    print(f'smoothed sim: var: {np.var(sim)}, # extremes: {sim_tot_extremes}, diff(sim)={SumAbsSIM}')

    return obs, sim