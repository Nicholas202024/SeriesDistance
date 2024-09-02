import numpy as np
from f_CalcHydCase import f_CalcHydCase
from f_TrimSeries import f_TrimSeries
from f_DefineSegments import f_DefineSegments
from f_AggregateSegment import f_AggregateSegment
from f_normalize import f_normalize
from f_SD import f_SD

def f_CoarseGraining_SD_Continuous(obs, sim, timeseries_splits, weight_nfc, weight_rds, weight_sdt, weight_sdv, error_model):
    """
    Coarse-grains the time series and calculates the selected statistics of agreement.
    """

    # Initialize arrays
    cons_all = {'x_match_obs_global': [], 'y_match_obs': [], 'x_match_sim_global': [], 'y_match_sim': []}
    segs_obs_opt_all = []  # coarse-grained segments of entire 'obs' time series
    segs_sim_opt_all = []  # coarse-grained segments of entire 'sim' time series
    e_sd_rise_all = []     # SD errors rising segments (final output)
    e_sd_fall_all = []     # SD errors falling segments (final output)
    e_sd_t_all = []        # SD timing errors (final output)
    e_sd_q_all = []        # SD magnitude errors (final output)
    obs_org = obs[0].copy()   # backup original time series
    sim_org = sim[0].copy()   # backup original time series

    # Pre-processing
    for i in range(len(timeseries_splits) - 1):  # loop over all items in timeseries_splits list
        # Display progress information
        txt = f'time series split {i} of {len(timeseries_splits) - 1}'
        print(txt)

        # Create subset/ split the time series
        obs_split = obs_org[timeseries_splits[i]:timeseries_splits[i + 1]]  # extract the obs values within the time series subset
        sim_split = sim_org[timeseries_splits[i]:timeseries_splits[i + 1]]  # extract the sim values within the time series subset

        # Trim sim and obs to ensure that both start and end with either rise or fall (if necessary)
        obs, x_obs, sim, x_sim = f_TrimSeries(obs_split, list(range(timeseries_splits[i], timeseries_splits[i + 1])), sim_split, list(range(timeseries_splits[i], timeseries_splits[i + 1])))

        # Determine the hydrological case for each timestep in the original time series
        hydcase_obs_orig = f_CalcHydCase(obs)
        hydcase_sim_orig = f_CalcHydCase(sim)

        hydcase_obs = hydcase_obs_orig.copy()  # will change with increasing segment merging
        hydcase_sim = hydcase_sim_orig.copy()  # will change with increasing segment merging

        # Define segments in the two time series
        segs_obs = f_DefineSegments(x_obs, obs)
        segs_sim = f_DefineSegments(x_sim, sim)

        # Check for differences in the number of segments in obs and sim
        seg_diff = len(segs_obs) - len(segs_sim)

        # Error checking: events must have either both even or both odd # of segments
        if seg_diff % 2 != 0:
            raise ValueError('f_SeriesDistance: events must have either both even or both odd # of segments!')

        # Equalize the # of segments starting with the least relevant segment in the time series which has more segments
        while seg_diff != 0:
            if seg_diff > 0:  # more obs than sim segments
                segs_obs, hydcase_obs = f_AggregateSegment(segs_obs, hydcase_obs, obs)  # erase the least relevant segment
            else:  # more sim than obs segments
                segs_sim, hydcase_sim = f_AggregateSegment(segs_sim, hydcase_sim, sim)  # erase the least relevant segment
            seg_diff = len(segs_obs) - len(segs_sim)  # number of segments still unequal?

        # Cleanup
        del seg_diff

        # Iterative reduction of segments and calculation of the selected statistics of agreement

        # Determine number of reduction steps
        num_red = (len(segs_obs) // 2) - 1  # reduce to 2 (if starting with an even number of segments)
                                            # reduce to 3 (if starting with an odd number of segments)

        # Initialize arrays
        percfalsecase = np.full(num_red + 1, np.nan)  # number of wrong hydcase assignments
        mafdist_t = np.full(num_red + 1, np.nan)      # Timing Error of SD
        mafdist_v = np.full(num_red + 1, np.nan)      # Magnitude Error of SD
        segment_data = [None] * (num_red + 1)         # (num_red,3) cell array which contains the best obs and sim segments (col 1 and 2) found for each reduction step and joint SD properties (col 3)
        connector_data = [None] * (num_red + 1)       # cell array which contains the connectors of the different coarse graining steps
        e_sd_rise = [None] * (num_red + 1)            # timing & magnitude errors of each coarse-graining step (rising)
        e_sd_fall = [None] * (num_red + 1)            # timing & magnitude errors of each coarse-graining step (falling)
        # NOTE: first entries contain the initial state before coarse graining

        # Apply SD and calculate all three statistics for the initial conditions (no reduction, only equalized # of segments)
        # Apply SD
        fdist_q, fdist_t, _, e_q_rise, e_t_rise, _, e_q_fall, e_t_fall, _, cons, e_rise_MD, e_fall_MD = f_SD(obs, segs_obs, sim, segs_sim, error_model)
        
        # Store segments and connectors for initial conditions
        segment_data[0] = (segs_obs, segs_sim)
        connector_data[0] = (cons, 0)  # field stores coarse graining step; 0=initial conditions
        e_sd_rise[0] = (e_t_rise, e_q_rise)  # for timing errors of rising segments
        e_sd_fall[0] = (e_t_fall, e_q_fall)  # for magnitude errors of falling segments

        # Calculate objective function inputs for initial conditions
        percfalsecase[0] = (len(np.where(hydcase_obs_orig != hydcase_obs)[0]) / len(obs)) + (len(np.where(hydcase_sim_orig != hydcase_sim)[0]) / len(sim))
        mafdist_t[0] = np.mean(np.abs(fdist_t))
        mafdist_v[0] = np.mean(np.abs(fdist_q))

        # Apply coarse-graining to all time series splits (big for-loop): Jointly reduce obs/sim segments, one by one, until only one obs and one sim segment are left
        for z in range(num_red):  # reduce until only 2 or 3 segments are left (2: when started with even # of segments, 3: when started with odd # of segments)
            # Some error checking
            if hydcase_obs[0] != hydcase_sim[0] or len(segs_obs) != len(segs_sim):
                raise ValueError('error in big loop')

            # Initialize temporary variables (temporary for one reduction step)
            tmp_percfalsecase = np.full((len(segs_obs), len(segs_sim)), np.nan)  # (m,m) matrix with percentage of false hydcases (obs + sim) for all possible reduction combinations
            tmp_rel_del_seg = np.full((len(segs_obs), len(segs_sim)), np.nan)    # (m,m) matrix with relevance of deleted segments (obs + sim)
            tmp_mafdist_t = np.full((len(segs_obs), len(segs_sim)), np.nan)      # (m,m) matrix with timing error
            tmp_mafdist_v = np.full((len(segs_obs), len(segs_sim)), np.nan)      # (m,m) matrix with value error

            # Loop over all possible segment reduction combinations
            for z_obs in range(1, len(segs_obs) - 1):  # loop over all observed segments, except the first and last
                # For each new loop, start with the best selection of the previous z-round
                tmp_segs_obs = segs_obs.copy()
                tmp_hydcase_obs = hydcase_obs.copy()

                tmp_rel_obs = tmp_segs_obs[z_obs]['relevance']  # save the relevance of the segment before it is deleted
                # original code
                # tmp_segs_obs, tmp_hydcase_obs = f_AggregateSegment(tmp_segs_obs, tmp_hydcase_obs, obs, z_obs)  # erase the specified segment

                try:
                    tmp_segs_obs, tmp_hydcase_obs = f_AggregateSegment(tmp_segs_obs, tmp_hydcase_obs, obs, z_obs)  # erase the specified segment
                except ValueError:
                    continue

                for z_sim in range(1, len(segs_sim) - 1):  # loop over all simulated segments, except the first and last
                    # For each new loop, start with the best selection of the previous z-round
                    tmp_segs_sim = segs_sim.copy()
                    tmp_hydcase_sim = hydcase_sim.copy()

                    tmp_rel_sim = tmp_segs_sim[z_sim]['relevance']  # save the relevance of the segment before it is deleted
                    # original code
                    # tmp_segs_sim, tmp_hydcase_sim = f_AggregateSegment(tmp_segs_sim, tmp_hydcase_sim, sim, z_sim)  # erase the specified segment

                    try:
                        tmp_segs_sim, tmp_hydcase_sim = f_AggregateSegment(tmp_segs_sim, tmp_hydcase_sim, sim, z_sim)  # erase the specified segment
                    except ValueError:
                        continue
                            
                    # Compute the objective function
                    tmp_percfalsecase[z_obs, z_sim] = (len(np.where(hydcase_obs_orig != tmp_hydcase_obs)[0]) / len(obs)) + (len(np.where(hydcase_sim_orig != tmp_hydcase_sim)[0]) / len(sim))
                    tmp_rel_del_seg[z_obs, z_sim] = tmp_rel_obs + tmp_rel_sim

                    fdist_q, fdist_t, _, _, _, _, _, _, _, cons, _, _ = f_SD(obs, tmp_segs_obs, sim, tmp_segs_sim, error_model, 'true')  # UE: used to be 'standard'
                    tmp_mafdist_t[z_obs, z_sim] = np.mean(np.abs(fdist_t))
                    tmp_mafdist_v[z_obs, z_sim] = np.mean(np.abs(fdist_q))

            # Find the best erase-combination for the given reduction step
            # Normalize and weight the criteria. NOTE: For all criteria: the smaller = the better 0=best, 1=worst
            norm_tmp_percfalsecase = weight_nfc * f_normalize(tmp_percfalsecase)
            norm_tmp_rel_del_seg = weight_rds * f_normalize(tmp_rel_del_seg)
            norm_tmp_mafdist_t = weight_sdt * f_normalize(tmp_mafdist_t)
            norm_tmp_mafdist_v = weight_sdv * f_normalize(tmp_mafdist_v)

            # Join the criteria to calculate the objective function (euclidean distance)
            tmp_opt = np.full((len(segs_obs), len(segs_sim)), np.nan)
            for yyy in range(len(tmp_opt)):
                for zzz in range(len(tmp_opt)):
                    tmp_opt[yyy, zzz] = np.sqrt(norm_tmp_percfalsecase[yyy, zzz]**2 + norm_tmp_rel_del_seg[yyy, zzz]**2 + norm_tmp_mafdist_t[yyy, zzz]**2 + norm_tmp_mafdist_v[yyy, zzz]**2)
            
            # Find the minimum (=best) value
            pos_obs, pos_sim = np.unravel_index(np.argmin(tmp_opt), tmp_opt.shape)
            pos_obs = pos_obs  # reduce to size 1 in case several equally small values were found
            pos_sim = pos_sim  # reduce to size 1 in case several equally small values were found

            # Execute the change on the real events
            segs_obs, hydcase_obs = f_AggregateSegment(segs_obs, hydcase_obs, obs, pos_obs)  # erase the specified segment
            segs_sim, hydcase_sim = f_AggregateSegment(segs_sim, hydcase_sim, sim, pos_sim)  # erase the specified segment

            # Calculate Series Distance on the optimized level of aggregated segments return SD errors
            fdist_q, fdist_t, _, e_q_rise, e_t_rise, _, e_q_fall, e_t_fall, _, cons, e_rise_MD, e_fall_MD = f_SD(obs, segs_obs, sim, segs_sim, error_model)

            # Add segment data, connectors and time/ magnitude errors of the best solution for this time series split to that of the entire time series
            segment_data[z + 1] = (segs_obs, segs_sim)
            connector_data[z + 1] = (cons, z)  # add coarse graining step
            e_sd_rise[z + 1] = (e_t_rise, e_q_rise)  # for timing errors of rising segments
            e_sd_fall[z + 1] = (e_t_fall, e_q_fall)  # for magnitude errors of falling segments

            # Compute objective function inputs
            percfalsecase[z + 1] = (len(np.where(hydcase_obs_orig != hydcase_obs)[0]) / len(obs)) + (len(np.where(hydcase_sim_orig != hydcase_sim)[0]) / len(sim))
            mafdist_t[z + 1] = np.mean(np.abs(fdist_t))
            mafdist_v[z + 1] = np.mean(np.abs(fdist_q))

            # Progress info
            txt = f'reduction step {z} of {num_red}'
            print(txt)

    # Calculate objective function and find the optimal coarse graining step
    ObFuncVal = np.sqrt(weight_nfc * f_normalize(percfalsecase)**2 + weight_sdt * f_normalize(mafdist_t)**2 + weight_sdv * f_normalize(mafdist_v)**2)

    opt_step = np.argmin(ObFuncVal)
    if len(ObFuncVal) > 1:  # display best coarse graining step
        if opt_step == 0:
            print('selected step # initial conditions')
        else:
            print(f'selected step # {opt_step}')

    # Select and return coarse-grained segments and connectors for optimal level of generalization
    segs_obs_opt = segment_data[opt_step][0]  # segment data obs
    segs_sim_opt = segment_data[opt_step][1]  # segment data sim
    cons = connector_data[opt_step][0]        # connectors
    e_sd_rise_opt = e_sd_rise[opt_step]       # timing & magnitude errors of rising segments
    e_sd_fall_opt = e_sd_fall[opt_step]       # timing & magnitude errors of falling segments

    # Add segment data and connectors of the splitted subset to that of the entire time series
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

    # segment data
    segs_obs_opt_all += segs_obs_opt
    segs_sim_opt_all += segs_sim_opt

    # time-magnitude errors for all rising/ falling sections
    e_sd_rise_all = np.concatenate((e_sd_rise_all, e_sd_rise_opt), axis=0)
    e_sd_fall_all = np.concatenate((e_sd_fall_all, e_sd_fall_opt), axis=0)

    # time and magnitude errors for entire hydrographs
    e_sd_t_all = np.concatenate((e_sd_t_all, np.concatenate((e_sd_rise_opt[:, 0], e_sd_fall_opt[:, 0]))), axis=0)
    e_sd_q_all = np.concatenate((e_sd_q_all, np.concatenate((e_sd_rise_opt[:, 2], e_sd_fall_opt[:, 2]))), axis=0)

    return segs_obs_opt_all, segs_sim_opt_all, cons_all, e_sd_t_all, e_sd_q_all