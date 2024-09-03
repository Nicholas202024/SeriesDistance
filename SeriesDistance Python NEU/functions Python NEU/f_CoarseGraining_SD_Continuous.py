import numpy as np
from f_TrimSeries import f_trim_series
from f_calc_hyd_case import f_calc_hyd_case
from f_DefineSegments import f_define_segments
from f_AggregateSegment import f_aggregate_segment
from f_SD import f_sd

def f_coarse_graining_continuous(obs, sim, timeseries_splits, weight_nfc, weight_rds, weight_sdt, weight_sdv, error_model):
    """
    Coarse-graining function for continuous series distance calculation.
    """

    # initialize arrays
    cons_all = []
    segs_obs_opt_all = []
    segs_sim_opt_all = []
    e_sd_rise_all = []
    e_sd_fall_all = []
    e_sd_t_all = []
    e_sd_q_all = []
    obs_org = obs.copy()
    sim_org = sim.copy()

    # pre-processing
    for i in range(len(timeseries_splits) - 1):
        # display progress information
        txt = f'time series split {i} of {len(timeseries_splits) - 1}'
        print(txt)

        # create subset/ split the time series
        obs_split = obs_org[timeseries_splits[i]:timeseries_splits[i + 1]]
        sim_split = sim_org[timeseries_splits[i]:timeseries_splits[i + 1]]
        
        print('\n')
        print('Input for f_trim_series')
        print('obs_split shape', obs_split.shape)
        print('obs_split type', type(obs_split))
        print('sim_split shape', sim_split.shape)
        print('sim_split type', type(sim_split))
        print('\n')

        # Trim sim and obs to ensure that both start and end with either rise or fall (if necessary)
        obs, x_obs, sim, x_sim = f_trim_series(obs_split, np.arange(timeseries_splits[i], timeseries_splits[i + 1]), sim_split, np.arange(timeseries_splits[i], timeseries_splits[i + 1]))

        print('Output from f_trim_series/Input for f_define_segments')
        print('obs type', type(obs))
        print('obs shape', obs.shape)
        print('sim type', type(sim))
        print('sim shape', sim.shape)
        print('\n')
        print('x_obs type', type(x_obs))
        print('x_obs shape', x_obs.shape)
        print('x_sim type', type(x_sim))
        print('x_sim shape', x_sim.shape)
        print('\n')

        # Determine the hydrological case for each timestep in the original time series
        hydcase_obs_orig = f_calc_hyd_case(obs)
        hydcase_sim_orig = f_calc_hyd_case(sim)

        hydcase_obs = hydcase_obs_orig.copy()
        hydcase_sim = hydcase_sim_orig.copy()

        # define segments in the two time series
        segs_obs = f_define_segments(x_obs, obs)
        segs_sim = f_define_segments(x_sim, sim)

        print('Output from f_define_segments')
        # print('segs_obs', segs_obs)
        print('segs_obs type', type(segs_obs))
        print('segs_obs dtype', type(segs_obs[0]))
        print('segs_obs len', len(segs_obs))
        # print('segs_sim', segs_sim)
        print('segs_sim type', type(segs_sim))
        print('segs_sim dtype', type(segs_sim[0]))
        print('segs_sim len', len(segs_sim))
        print('\n')

        # check for differences in the number of segments in obs and sim
        seg_diff = len(segs_obs) - len(segs_sim)

        # error checking: events must have either both even or both odd # of segments
        if seg_diff % 2 != 0:
            raise ValueError('f_SeriesDistance: events must have either both even or both odd # of segments!')

        print('Input for f_aggregate_segment')
        # print('hydcase_obs', hydcase_obs)
        print('hydcase_obs type', type(hydcase_obs))
        print('hydcase shape', hydcase_obs.shape)
        print('\n')
        # print('hydcase_sim', hydcase_sim)
        print('hydcase_sim type', type(hydcase_sim))
        print('hydcase_sim shape', hydcase_sim.shape)
        print('\n')

        # equalize the # of segments starting with the least relevant segment in the time series which has more segments
        while seg_diff != 0:
            if seg_diff > 0:  # more obs than sim segments
                segs_obs, hydcase_obs = f_aggregate_segment(segs_obs, hydcase_obs, obs)
            else:  # more sim than obs segments
                segs_sim, hydcase_sim = f_aggregate_segment(segs_sim, hydcase_sim, sim)
            seg_diff = len(segs_obs) - len(segs_sim)

        print('Output from f_aggregate_segment')
        print('segs_obs type', type(segs_obs))
        print('segs_obs dtype', type(segs_obs[0]))
        print('segs_obs len', len(segs_obs))
        print('\n')
        # print('hydcase_obs', hydcase_obs)
        print('hydcase_obs type', type(hydcase_obs))
        print('hydcase_obs shape', hydcase_obs.shape)
        print('\n')
    
        print('segs_sim type', type(segs_sim))
        print('segs_sim dtype', type(segs_sim[0]))
        print('segs_sim len', len(segs_sim))
        print('\n')
        # print('hydcase_sim', hydcase_sim)
        print('hydcase_sim type', type(hydcase_sim))
        print('hydcase_sim shape', hydcase_sim.shape)
        print('\n')

        # cleanup
        del seg_diff

        # iterative reduction of segments and calculation of the selected statistics of agreement

        # determine number of reduction steps
        num_red = (len(segs_obs) // 2) - 1

        # initialize arrays
        percfalsecase = np.full(num_red + 1, np.nan)
        mafdist_t = np.full(num_red + 1, np.nan)
        mafdist_v = np.full(num_red + 1, np.nan)
        segment_data = [None] * (num_red + 1)
        connector_data = [None] * (num_red + 1)
        e_sd_rise = [None] * (num_red + 1)
        e_sd_fall = [None] * (num_red + 1)

        # Apply SD and calculate all three statistics for the initial conditions (no reduction, only equalized # of segments)
        fdist_q, fdist_t, _, e_q_rise, e_t_rise, _, e_q_fall, e_t_fall, _, cons = f_sd(obs, segs_obs, sim, segs_sim, error_model)
        raise Exception('Froced stop')

        # store segments and connectors for initial conditions
        segment_data[0] = (segs_obs, segs_sim)
        connector_data[0] = (cons, 0)
        e_sd_rise[0] = (e_t_rise, e_q_rise)
        e_sd_fall[0] = (e_t_fall, e_q_fall)

        # calculate objective function inputs for initial conditions
        percfalsecase[0] = (np.sum(hydcase_obs_orig != hydcase_obs) / len(obs)) + (np.sum(hydcase_sim_orig != hydcase_sim) / len(sim))
        mafdist_t[0] = np.mean(np.abs(fdist_t))
        mafdist_v[0] = np.mean(np.abs(fdist_q))

        # apply coarse-graining to all time series splits (big for-loop): Jointly reduce obs/sim segments, one by one, until only one obs and one sim segment are left
        for z in range(num_red):
            if hydcase_obs[0] != hydcase_sim[0] or len(segs_obs) != len(segs_sim):
                raise ValueError('error in big loop')

            # initialize temporary variables (temporary for one reduction step)
            tmp_percfalsecase = np.full((len(segs_obs), len(segs_obs)), np.nan)
            tmp_rel_del_seg = np.full((len(segs_obs), len(segs_obs)), np.nan)
            tmp_mafdist_t = np.full((len(segs_obs), len(segs_obs)), np.nan)
            tmp_mafdist_v = np.full((len(segs_obs), len(segs_obs)), np.nan)

            # loop over all possible segment reduction combinations
            for z_obs in range(1, len(segs_obs) - 1):
                tmp_segs_obs = segs_obs.copy()
                tmp_hydcase_obs = hydcase_obs.copy()

                tmp_rel_obs = tmp_segs_obs[z_obs]['relevance']
                tmp_segs_obs, tmp_hydcase_obs = f_aggregate_segment(tmp_segs_obs, tmp_hydcase_obs, obs, z_obs)

                for z_sim in range(1, len(segs_sim) - 1):
                    tmp_segs_sim = segs_sim.copy()
                    tmp_hydcase_sim = hydcase_sim.copy()

                    tmp_rel_sim = tmp_segs_sim[z_sim]['relevance']
                    tmp_segs_sim, tmp_hydcase_sim = f_aggregate_segment(tmp_segs_sim, tmp_hydcase_sim, sim, z_sim)

                    # compute the objective function
                    tmp_percfalsecase[z_obs, z_sim] = (np.sum(hydcase_obs_orig != tmp_hydcase_obs) / len(obs)) + (np.sum(hydcase_sim_orig != tmp_hydcase_sim) / len(sim))
                    tmp_rel_del_seg[z_obs, z_sim] = tmp_rel_obs + tmp_rel_sim

                    fdist_q, fdist_t, _, _, _, _, _, _, _, cons, _, _ = f_sd(obs, tmp_segs_obs, sim, tmp_segs_sim, error_model, 'true')
                    tmp_mafdist_t[z_obs, z_sim] = np.mean(np.abs(fdist_t))
                    tmp_mafdist_v[z_obs, z_sim] = np.mean(np.abs(fdist_q))

            # find the best erase-combination for the given reduction step
            norm_tmp_percfalsecase = weight_nfc * f_normalize(tmp_percfalsecase)
            norm_tmp_rel_del_seg = weight_rds * f_normalize(tmp_rel_del_seg)
            norm_tmp_mafdist_t = weight_sdt * f_normalize(tmp_mafdist_t)
            norm_tmp_mafdist_v = weight_sdv * f_normalize(tmp_mafdist_v)

            tmp_opt = np.sqrt(norm_tmp_percfalsecase**2 + norm_tmp_rel_del_seg**2 + norm_tmp_mafdist_t**2 + norm_tmp_mafdist_v**2)

            pos_obs, pos_sim = np.unravel_index(np.argmin(tmp_opt), tmp_opt.shape)
            pos_obs = pos_obs[0]
            pos_sim = pos_sim[0]

            # execute the change on the real events
            segs_obs, hydcase_obs = f_aggregate_segment(segs_obs, hydcase_obs, obs, pos_obs)
            segs_sim, hydcase_sim = f_aggregate_segment(segs_sim, hydcase_sim, sim, pos_sim)

            # Calculate Series Distance on the optimized level of aggregated segments return SD errors
            fdist_q, fdist_t, _, e_q_rise, e_t_rise, _, e_q_fall, e_t_fall, _, cons = f_sd(obs, segs_obs, sim, segs_sim, error_model)

            # add segment data, connectors and time/ magnitude errors of the best solution for this time series split to that of the entire time series
            segment_data[z + 1] = (segs_obs, segs_sim)
            connector_data[z + 1] = (cons, z)
            e_sd_rise[z + 1] = (e_t_rise, e_q_rise)
            e_sd_fall[z + 1] = (e_t_fall, e_q_fall)

            # compute objective function inputs
            percfalsecase[z + 1] = (np.sum(hydcase_obs_orig != hydcase_obs) / len(obs)) + (np.sum(hydcase_sim_orig != hydcase_sim) / len(sim))
            mafdist_t[z + 1] = np.mean(np.abs(fdist_t))
            mafdist_v[z + 1] = np.mean(np.abs(fdist_q))

            # progress info
            txt = f'reduction step {z} of {num_red}'
            print(txt)

        # Calculate objective function and find the optimal coarse graining step
        ObFuncVal = np.sqrt(weight_nfc * f_normalize(percfalsecase)**2 + weight_sdt * f_normalize(mafdist_t)**2 + weight_sdv * f_normalize(mafdist_v)**2)
        opt_step = np.argmin(ObFuncVal)
        opt_step = opt_step if opt_step == 0 else opt_step[0]

        if len(ObFuncVal) > 1:
            if opt_step == 0:
                print('selected step # initial conditions')
            else:
                print(f'selected step # {opt_step}')

        # select and return coarse-grained segments and connectors for optimal level of generalization
        segs_obs_opt = segment_data[opt_step][0]
        segs_sim_opt = segment_data[opt_step][1]
        cons = connector_data[opt_step][0]
        e_sd_rise_opt = e_sd_rise[opt_step]
        e_sd_fall_opt = e_sd_fall[opt_step]

        # add segment data and connectors of the splitted subset to that of the entire time series
        if not cons_all:
            cons_all.append({
                'x_match_obs_global': cons[0]['x_match_obs_global'],
                'y_match_obs': cons[0]['y_match_obs'],
                'x_match_sim_global': cons[0]['x_match_sim_global'],
                'y_match_sim': cons[0]['y_match_sim']
            })
        else:
            cons_all[0]['x_match_obs_global'] += cons[0]['x_match_obs_global']
            cons_all[0]['y_match_obs'] += cons[0]['y_match_obs']
            cons_all[0]['x_match_sim_global'] += cons[0]['x_match_sim_global']
            cons_all[0]['y_match_sim'] += cons[0]['y_match_sim']

        segs_obs_opt_all.extend(segs_obs_opt)
        segs_sim_opt_all.extend(segs_sim_opt)
        e_sd_rise_all.extend(e_sd_rise_opt)
        e_sd_fall_all.extend(e_sd_fall_opt)
        e_sd_t_all.extend([e_sd_rise_opt[0], e_sd_fall_opt[0]])
        e_sd_q_all.extend([e_sd_rise_opt[1], e_sd_fall_opt[1]])

    return segs_obs_opt_all, segs_sim_opt_all, cons_all, e_sd_t_all, e_sd_q_all