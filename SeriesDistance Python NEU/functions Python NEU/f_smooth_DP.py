import numpy as np
from scipy.interpolate import interp1d
from f_dp1d import f_dp1d

def f_smooth_DP(obs, sim, nse_smooth_limit):
    # Berechne Statistiken, um die ursprüngliche und geglättete Zeitreihe zu vergleichen
    # Anzahl der Extremwerte
    local_mins = len(np.where(np.diff(np.sign(np.diff(obs))) == 2)[0]) + 1  # Anzahl der positiven Vorzeichenwechsel in der zeitlichen Ableitung von obs (lokale Maxima)
    local_maxs = len(np.where(np.diff(np.sign(np.diff(obs))) == -2)[0]) + 1  # Anzahl der negativen Vorzeichenwechsel in der zeitlichen Ableitung von obs (lokale Minima)
    obs_tot_extremes = local_mins + local_maxs

    local_mins = len(np.where(np.diff(np.sign(np.diff(sim))) == 2)[0]) + 1  # dasselbe für sim
    local_maxs = len(np.where(np.diff(np.sign(np.diff(sim))) == -2)[0]) + 1  # dasselbe für sim
    sim_tot_extremes = local_mins + local_maxs

    SumAbsSIM = np.sum(np.abs(np.diff(sim)))
    SumAbsOBS = np.sum(np.abs(np.diff(obs)))

    print(f'orginal obs: var: {np.var(obs)}, # extremes: {obs_tot_extremes}, diff(obs)={SumAbsOBS}')
    print(f'orginal sim: var: {np.var(sim)}, # extremes: {sim_tot_extremes}, diff(sim)={SumAbsSIM}')

    obs_old = obs.copy()  # behalte die ursprüngliche Zeitreihe
    xes = np.arange(1, len(obs) + 1)  # erstelle x-Daten (Zeitpunkte)
    xy = np.column_stack((xes, obs_old))  # bereite Eingabe für f_dp1d vor

    # print('\n')
    # print('input data for f_dp1d:')
    # # print("xy: ", xy)
    # print('type xy: ', type(xy))
    # print('dtype xy: ', xy.dtype)
    # print('shape xy: ', xy.shape)
    # print('\n')

    # füge hinzu: vereinfache obs bis zu einem Punktzahlkriterium
    xy_dp = f_dp1d(xy, -999, sim_tot_extremes)
    # frühere Version:
    # vereinfache obs bis zu einem NSE-Übereinstimmungsniveau, das durch 'nse_limit_obs' angegeben wird
    # xy_dp = f_dp1d(xy, -999, -999, nse_smooth_limit)

    # print('output data for f_dp1d:')
    # # print("xy_dp: ", xy_dp)
    # print('type xy_dp: ', type(xy_dp))
    # print('dtype xy_dp: ', xy_dp.dtype)
    # print('shape xy_dp: ', xy_dp.shape)
    # print('\n')

    # sample die vereinfachte Linie an den ursprünglichen x-Positionen (Zeitpunkten)
    interp_func = interp1d(xy_dp[:, 0], xy_dp[:, 1], kind='linear')
    obs = interp_func(xes)

    # Anzahl der Extremwerte der geglätteten Zeitreihe
    local_mins = len(np.where(np.diff(np.sign(np.diff(obs))) == 2)[0]) + 1  # Anzahl der positiven Vorzeichenwechsel (lokale Maxima)
    local_maxs = len(np.where(np.diff(np.sign(np.diff(obs))) == -2)[0]) + 1  # Anzahl der negativen Vorzeichenwechsel (lokale Minima)
    obs_tot_extremes = local_mins + local_maxs

    local_mins = len(np.where(np.diff(np.sign(np.diff(sim))) == 2)[0]) + 1  # Anzahl der positiven Vorzeichenwechsel (lokale Maxima)
    local_maxs = len(np.where(np.diff(np.sign(np.diff(sim))) == -2)[0]) + 1  # Anzahl der negativen Vorzeichenwechsel (lokale Minima)
    sim_tot_extremes = local_mins + local_maxs

    SumAbsSIM = np.sum(np.abs(np.diff(sim)))
    SumAbsOBS = np.sum(np.abs(np.diff(obs)))

    # Statistiken im Befehlsfenster ausgeben
    print(f'smoothed obs: var: {np.var(obs)}, # extremes: {obs_tot_extremes}, diff(obs)={SumAbsOBS}')
    print(f'smoothed sim: var: {np.var(sim)}, # extremes: {sim_tot_extremes}, diff(sim)={SumAbsSIM}')

    return obs, sim