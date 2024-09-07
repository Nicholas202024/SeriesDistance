import numpy as np
from f_TrimSeries import f_trim_series
from f_calc_hyd_case import f_calc_hyd_case
from f_DefineSegments import f_define_segments
from f_AggregateSegment import f_aggregate_segment
from f_normalize import f_normalize
from f_SD import f_sd
from f_SegStats import f_SegStats
from f_PlotCoarseGrainIntSteps import f_PlotCoarseGrainIntSteps


def f_CoarseGraining_Event(obs, obs_eventindex, sim, sim_eventindex, weight_nfc, weight_rds, weight_sdt, weight_sdv, error_model, plot_intermedSteps):
    """
    Apply coarse-graining and SD method to each event.

    Parameters:
    obs: (n,1) matrix with observed discharge
    obs_eventindex: index of observed events
    sim: (n,1) matrix with simulated discharge
    sim_eventindex: index of simulated events
    weight_nfc: (1) weighting factor for the number of false hydrological cases, used in the objective function.
                The higher the more relevant. Recommended value: 1
    weight_rds: (1) weighting factor for the relevance of the deleted segments, used in the objective function.
                The higher the more relevant. Recommended value: 1 
    weight_sdt: (1) weighting factor for the series distance time error, used in the objective function.
                The higher the more relevant. Recommended value: 5
    weight_sdv: (1) weighting factor for the series distance value (or magnitude) error, used in the objective function.
                The higher the more relevant. Recommended value: 0         
    error_model: (1) sets the way the magnitude distance among obs and sim is computed
                but only in the last step, (section %%'compute and return the final, optimized series distances'
                with the optimized set of segments. During optimization, a simple distance (obs - sim) is used
                if 'true': dist_v = (obs - sim) / ((obs + sim)*0.5)
                if 'false': dist_v = (obs - sim)
                Recommended: 'true'    
    plot_intermedSteps: Plots intermediate coarse graining steps

    Returns:
    segs_obs_opt, segs_sim_opt, cons, connector_data, ObFuncVal, opt_step, CoarseGrain_segs, seg_raw_statistics
    """

    # Pre-processing
    obs = obs[obs_eventindex]
    sim = sim[sim_eventindex]

    # Trim sim and obs to ensure that both start and end with either rise or fall (if necessary)
    obs, x_obs, sim, x_sim = f_trim_series(obs, obs_eventindex, sim, sim_eventindex)
        
    # Determine the hydrological case for each timestep in the original time series
    hydcase_obs_orig = f_calc_hyd_case(obs)
    hydcase_sim_orig = f_calc_hyd_case(sim)
    
    hydcase_obs = hydcase_obs_orig  # will change with increasing segment merging
    hydcase_sim = hydcase_sim_orig  # will change with increasing segment merging

    # Define segments in the two time series
    segs_obs = f_define_segments(x_obs, obs)
    segs_sim = f_define_segments(x_sim, sim)

    # print('\n')
    # print('Input f_SegStats:')
    # print('segs_obs type:', type(segs_obs))
    # print('segs_obs[0] type:', type(segs_obs[0]))
    # print('segs_obs[0]:', segs_obs[0])
    # print('segs_sim type:', type(segs_sim))
    # print('segs_sim[0] type:', type(segs_sim[0]))
    # print('segs_sim[0]:', segs_sim[0])
    # print('\n')

    # print('Output f_SegStats:')
    # print('Segment_Stats_obs:', f_SegStats(segs_obs))
    # print('Segment_Stats_sim:', f_SegStats(segs_sim))
    # print('\n')

    # Determine statistics of initial segment definition
    # Original code der Übersetzung
    # seg_raw_statistics = np.concatenate((f_SegStats(segs_obs), f_SegStats(segs_sim)), axis=1)
    seg_raw_statistics = [f_SegStats(segs_obs), f_SegStats(segs_sim)]

    # print('Output f_SegStats zusammengefasst:')
    # print('seg_raw_statistics:', seg_raw_statistics)
    # print('seg_raw_statistics type:', type(seg_raw_statistics))
    # print('\n')

    # Check for differences in the number of segments of obs and sim
    seg_diff = len(segs_obs) - len(segs_sim)
      
    # Error checking: events must have either both even or both odd # of segments
    if seg_diff % 2 != 0:
        raise ValueError('f_SeriesDistance: events must have either both even or both odd # of segments!')
    
    # Equalize the # of segments starting with the least relevant segment in the event which has more segments
    while seg_diff != 0:  # only required if the number of segments differs 
        if seg_diff > 0:  # more obs than sim segments
            segs_obs, hydcase_obs = f_aggregate_segment(segs_obs, hydcase_obs, obs)  # erase the least relevant segment
        else:  # more sim than obs segments
            segs_sim, hydcase_sim = f_aggregate_segment(segs_sim, hydcase_sim, sim)  # erase the least relevant segment
        seg_diff = len(segs_obs) - len(segs_sim)  # number of segments still unequal?

    # Cleanup
    del seg_diff, x_sim, x_obs

    # print('check 1')
    # print('segs_obs:', segs_obs)
    # print('segs_obs len:', len(segs_obs))
    # print('\n')
    # Reduction of segments (coarse graining) and calculation of statistics of agreement

    # Determine number of reduction steps
    num_red = (len(segs_obs) // 2) - 1  # reduce to 2 (if starting with an even number of segments)
                                        # reduce to 3 (if starting with an odd number of segments)    
    # Initialize arrays for the objective function
    percfalsecase = np.full((num_red + 1, 1), np.nan)  # number of wrong hydcase assignments
    mafdist_t = np.full((num_red + 1, 1), np.nan)  # Mean Absolute Time Error of SD [h]
    mafdist_v = np.full((num_red + 1, 1), np.nan)  # Mean Absolute Value Error of SD [m3/s]
    segment_data = [None] * (num_red + 1)  # (num_red,3) list which contains the best obs and sim segments (col 1 and 2) found for each reduction step and joint SD properties (col 3)
    connector_data = [None] * (num_red + 1)  # list which contains the connectors of the different coarse graining steps
    # NOTE: first entries contain the initial state before coarse graining

    # Apply SD and calculate all three statistics for initial conditions (no reduction, only equalized # of segments)
    # Apply SD 
    fdist_q, fdist_t, _, _, _, _, _, _, _, cons, _, _ = f_sd(obs, segs_obs, sim, segs_sim, error_model, 'false')

    # Plot initial conditions
    if plot_intermedSteps:
        f_PlotCoarseGrainIntSteps(obs, segs_obs, sim, segs_sim, cons, 'initial conditions')
    
    # print('check 2')
    # print('num_red:', num_red)
    # print('segment_data:', segment_data)
    # print('\n')
    
    # Store segments and connectors for initial conditions
    segment_data[0] = (segs_obs, segs_sim)
    connector_data[0] = (cons, 0)  # field stores coarse graining step; 0=initial conditions
    
    # print('segment_data[0][0]:', segment_data[0][0])
    # print('\n')
    
    # Calculate objective function inputs for initial conditions
    percfalsecase[0] = (len(np.where(hydcase_obs_orig != hydcase_obs)[0]) / len(obs)) + \
                       (len(np.where(hydcase_sim_orig != hydcase_sim)[0]) / len(sim))
    mafdist_t[0] = np.mean(np.abs(fdist_t))  # mean SD time error
    mafdist_v[0] = np.mean(np.abs(fdist_q))  # mean SD value error    
    
    # Store segment combinations for initial conditions
    CoarseGrain_segs = []
    current_segs = np.column_stack((np.ones(len(segs_obs)), [seg['starttime_global'] for seg in segs_obs], [seg['endtime_global'] for seg in segs_obs], [seg['starttime_global'] for seg in segs_sim], [seg['endtime_global'] for seg in segs_sim]))
    CoarseGrain_segs.append(current_segs)
    
    # print('Check store segment')
    # print('current_segs:', current_segs)
    # print('current_segs type:', type(current_segs))
    # print('\n')
    # print('CoarseGrain_segs:', CoarseGrain_segs)
    # print('CoarseGrain_segs type:', type(CoarseGrain_segs))
    # print('\n')

    # print('check 3')
    # print('num_red:', num_red)
    # print('\n')

    # Iterative coarse-graining: Jointly aggregate segments in obs and sim, one by one, until the event is represented by two obs and two sim segments
    for z in range(num_red):  # reduce until only 2 or 3 segments are left (2: when started with even # of segments, 3: when started with odd # of segments)
        # Error checking
        if hydcase_obs[0] != hydcase_sim[0] or len(segs_obs) != len(segs_sim):
            raise ValueError('error in big loop')

        # Initialize temporary variables (temporary for one reduction step)
        tmp_percfalsecase = np.full((len(segs_obs), len(segs_obs)), np.nan)  # (m,m) matrix with percentage of false hydcases (obs + sim) for all possible reduction combinations
        tmp_rel_del_seg = np.full((len(segs_obs), len(segs_obs)), np.nan)  # (m,m) matrix with relevance of deleted segments (obs + sim)
        tmp_mafdist_t = np.full((len(segs_obs), len(segs_obs)), np.nan)  # (m,m) matrix with timing error
        tmp_mafdist_v = np.full((len(segs_obs), len(segs_obs)), np.nan)  # (m,m) matrix with value error
    
        # Loop over all possible segment reduction combinations
        for z_obs in range(1, len(segs_obs) - 1):  # loop over all observed segments, except the first and last
            # For each new loop, start with the best selection of the previous z-round
            tmp_segs_obs = segs_obs.copy()
            tmp_hydcase_obs = hydcase_obs.copy()

            tmp_rel_obs = tmp_segs_obs[z_obs]['relevance']  # save the relevance of the segment before it is deleted
            tmp_segs_obs, tmp_hydcase_obs = f_aggregate_segment(tmp_segs_obs, tmp_hydcase_obs, obs, z_obs)  # erase the specified segment
        
            for z_sim in range(1, len(segs_sim) - 1):  # loop over all simulated segments, except the first and last
                # For each new loop, start with the best selection of the previous z-round
                tmp_segs_sim = segs_sim.copy()
                tmp_hydcase_sim = hydcase_sim.copy()
            
                tmp_rel_sim = tmp_segs_sim[z_sim]['relevance']  # save the relevance of the segment before it is deleted
                tmp_segs_sim, tmp_hydcase_sim = f_aggregate_segment(tmp_segs_sim, tmp_hydcase_sim, sim, z_sim)  # erase the specified segment
           
                # Compute percentage of false hyd. cases and relative importance
                tmp_percfalsecase[z_obs, z_sim] = (len(np.where(hydcase_obs_orig != tmp_hydcase_obs)[0]) / len(obs)) + \
                                                  (len(np.where(hydcase_sim_orig != tmp_hydcase_sim)[0]) / len(sim))
                tmp_rel_del_seg[z_obs, z_sim] = tmp_rel_obs + tmp_rel_sim
            
                # Apply SD
                fdist_q, fdist_t, _, _, _, _, _, _, _, cons, _, _ = f_sd(obs, tmp_segs_obs, sim, tmp_segs_sim, error_model, 'true')
                tmp_mafdist_t[z_obs, z_sim] = np.mean(np.abs(fdist_t))
                tmp_mafdist_v[z_obs, z_sim] = np.mean(np.abs(fdist_q))
        
        # Find the best erase-combination for the given step using an objective function
        # Normalize and weight the criteria. NOTE: For all criteria: the smaller = the better 0=best, 1=worst
        norm_tmp_percfalsecase = f_normalize(tmp_percfalsecase)
        norm_tmp_rel_del_seg = f_normalize(tmp_rel_del_seg)
        norm_tmp_mafdist_t = f_normalize(tmp_mafdist_t)
        norm_tmp_mafdist_v = f_normalize(tmp_mafdist_v)

        # Join the criteria to calculate the objective function (euclidean distance)   
        tmp_opt_step = np.full((len(segs_obs), len(segs_obs)), np.nan)        
        for yyy in range(len(tmp_opt_step)):
            for zzz in range(len(tmp_opt_step)):
                tmp_opt_step[yyy, zzz] = np.sqrt(weight_nfc * norm_tmp_percfalsecase[yyy, zzz]**2 + \
                                                 weight_rds * norm_tmp_rel_del_seg[yyy, zzz]**2 + \
                                                 weight_sdt * norm_tmp_mafdist_t[yyy, zzz]**2 + \
                                                 weight_sdv * norm_tmp_mafdist_v[yyy, zzz]**2)

        # Find the minimum (=best) value
        pos_obs, pos_sim = np.unravel_index(np.argmin(tmp_opt_step), tmp_opt_step.shape)
        pos_obs = pos_obs  # reduce to size 1 in case several equally small values were found
        pos_sim = pos_sim  # reduce to size 1 in case several equally small values were found  

        # Execute the change on the real events

        # Original code der Übersetzung
        # segs_obs, hydcase_obs = f_aggregate_segment(segs_obs, hydcase_obs, obs, pos_obs)  # erase the specified segment
        # segs_sim, hydcase_sim = f_aggregate_segment(segs_sim, hydcase_sim, sim, pos_sim)  # erase the specified segment

        try:
            segs_obs, hydcase_obs = f_aggregate_segment(segs_obs, hydcase_obs, obs, pos_obs)  # erase the specified segment
            segs_sim, hydcase_sim = f_aggregate_segment(segs_sim, hydcase_sim, sim, pos_sim)  # erase the specified segment
        except:
            # segs_obs, hydcase_obs = f_aggregate_segment(segs_obs, hydcase_obs, obs)  # erase the specified segment
            # segs_sim, hydcase_sim = f_aggregate_segment(segs_sim, hydcase_sim, sim)  # erase the specified segment
            pass

        # calculate Series Distance
        fdist_q, fdist_t, _, _, _, _, _, _, _, cons, _, _ = f_sd(obs, segs_obs, sim, segs_sim, error_model)  # UE: used to be 'standard'

        # store all segment data (needed for final SD calculation and plotting of the overall winner)
        segment_data[z + 1] = [segs_obs, segs_sim]
        connector_data[z + 1] = [cons, z]  # add coarse graining step

        # compute objective function for initial conditions (before coarse graining)
        percfalsecase[z + 1] = (len(np.where(hydcase_obs_orig != hydcase_obs)[0]) / len(obs)) + \
                            (len(np.where(hydcase_sim_orig != hydcase_sim)[0]) / len(sim))
        mafdist_t[z + 1] = np.mean(np.abs(fdist_t))
        mafdist_v[z + 1] = np.mean(np.abs(fdist_q))

        # Store segment combinations and corresponding coarse graining steps

        # print('check type')
        # print('segs_obs:', segs_obs)
        # print('type(segs_obs):', type(segs_obs))
        # print('type(segs_obs[0]):', type(segs_obs[0]))
        # print('len(segs_obs):', len(segs_obs))
        # print('\n')

        # Original code der Übersetzung
        # rowindex = (z + 1) * np.ones(len(segs_obs['starttime_global']))
        # current_segs = np.column_stack((rowindex, segs_obs['starttime_global'], segs_obs['endtime_global'], segs_sim['starttime_global'], segs_sim['endtime_global']))
        # CoarseGrain_segs = np.vstack((CoarseGrain_segs, current_segs))

        rowindex = (z + 2) * np.ones(len(segs_obs))
        current_segs = np.column_stack((rowindex, [seg['starttime_global'] for seg in segs_obs], [seg['endtime_global'] for seg in segs_obs], [seg['starttime_global'] for seg in segs_sim], [seg['endtime_global'] for seg in segs_sim]))
        
        # print('rowindex:', rowindex)
        # print('rowindex type:', type(rowindex))
        # print('rowindex shape:', rowindex.shape)
        # print('\n')
        # print('current_segs:', current_segs)
        # print('current_segs type:', type(current_segs))
        # print('current_segs shape:', current_segs.shape)
        # print('\n')

        CoarseGrain_segs.append(current_segs)

        # print('CoarseGrain_segs:', CoarseGrain_segs)
        # print('CoarseGrain_segs type:', type(CoarseGrain_segs))
        # print('CoarseGrain_segs shape:', len(CoarseGrain_segs))
        # print('\n')
        
        # raise Exception('STOP COARSE GRAINING')
        # plot intermediate coarse graining steps
        if plot_intermedSteps:
            f_PlotCoarseGrainIntSteps(obs, segs_obs, sim, segs_sim, cons, f'coarse graining step: {z}')

        # display progress info
        print(f'coarse graining step {z} of {num_red}')

        # Calculate objective function and find the optimal coarse graining step
        ObFuncVal = np.sqrt(weight_nfc * f_normalize(percfalsecase) ** 2 +
                            weight_sdt * f_normalize(mafdist_t) ** 2 +
                            weight_sdv * f_normalize(mafdist_v) ** 2)

        opt_step = np.argmin(ObFuncVal)
        if len(ObFuncVal) > 1:  # display best coarse graining step
            if opt_step == 0:
                print('selected step # initial conditions')
            else:
                print(f'selected step # {opt_step}')

        # select and return coarse-grained segments and connectors for optimal level of generalization
        segs_obs_opt = np.array(segment_data[opt_step][0])
        segs_sim_opt = np.array(segment_data[opt_step][1])
        cons = np.array(connector_data[opt_step][0])

    # Code zu Übersetzung hinzugefügt
    if num_red == 0:
        segs_obs_opt = np.array(segment_data[0][0])
        segs_sim_opt = np.array(segment_data[0][1])

        ObFuncVal = np.empty(0)

        opt_step = None

    # print('check 4')
    # print('num_red:', num_red)
    # print('segs_obs_opt:', segs_obs_opt)
    # print('segs_sim_opt:', segs_sim_opt)
    # print('\n')

    return segs_obs_opt, segs_sim_opt, cons, connector_data, ObFuncVal, opt_step, CoarseGrain_segs, seg_raw_statistics