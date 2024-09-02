import numpy as np

def f_SD(y_obs, segs_obs, y_sim, segs_sim, error_model, printflag=False):
    """
    Calculates the distance vectors in time and value between two matching events (obs/sim).

    Parameters:
    y_obs (np.ndarray): Observed values.
    segs_obs (list): List of observed segments.
    y_sim (np.ndarray): Simulated values.
    segs_sim (list): List of simulated segments.
    error_model (str): Error model to use ('standard' or 'relative').
    printflag (bool, optional): Flag to print debug information. Default is False.

    Returns:
    tuple: Various error metrics and connectors.
    """
    
    # Initialize connectors
    cons = {
        'x_match_obs_global': [],
        'y_match_obs': [],
        'x_match_sim_global': [],
        'y_match_sim': []
    }

    # Specify connectors
    totnumcons = (len(y_obs) + len(y_sim)) * 0.5
    num_segs = len(segs_obs)
    segs_cons = np.full(num_segs, np.nan)
    sum_rels = sum([seg['relevance'] for seg in segs_obs]) + sum([seg['relevance'] for seg in segs_sim])
    segs_cons = np.round((np.array([seg['relevance'] for seg in segs_obs]) + np.array([seg['relevance'] for seg in segs_sim])) * totnumcons / sum_rels).astype(int)

    # Initialize output variables
    e_q_rise = []      # magnitude errors in rising limbs
    e_t_rise = []      # time errors in rising limbs
    e_q_fall = []      # magnitude errors in falling limbs
    e_t_fall = []      # time errors in falling limbs
    e_ysim_rise = []   # simulated discharge for rising limbs, corresponding to each error (needed to subdivide errors in discharge classes)
    e_ysim_fall = []   # simulated discharge for falling limbs, corresponding to each error (needed to subdivide errors in discharge classes)
    e_rise_MD = []     # 1D magnitude errors in corresponding rising limb sections 
    e_fall_MD = []     # 1D magnitude errors in corresponding falling limb sections 

    # Loop over all segments
    for z in range(num_segs):
        # Determine the GLOBAL x-location (time) of the connectors in the current segment  
        num = segs_cons[z]
        con_x_obs_global_seg = np.linspace(segs_obs[z]['starttime_global'], segs_obs[z]['endtime_global'], num)
        con_x_sim_global_seg = np.linspace(segs_sim[z]['starttime_global'], segs_sim[z]['endtime_global'], num)

        # Determine the LOCAL x-location (time) of the connectors in the current segment     
        con_x_obs_local_seg = np.linspace(segs_obs[z]['starttime_local'], segs_obs[z]['endtime_local'], num)
        con_x_sim_local_seg = np.linspace(segs_sim[z]['starttime_local'], segs_sim[z]['endtime_local'], num)

        # Determine the local x-locations of the segment
        x_obs_local_seg = np.arange(segs_obs[z]['starttime_local'], segs_obs[z]['endtime_local'] + 1)
        x_sim_local_seg = np.arange(segs_sim[z]['starttime_local'], segs_sim[z]['endtime_local'] + 1)

        xobs = np.arange(segs_obs[z]['starttime_local'], segs_obs[z]['endtime_local'] + 1)
        xsim = np.arange(segs_sim[z]['starttime_local'], segs_sim[z]['endtime_local'] + 1)
        xint = np.intersect1d(xobs, xsim)

        # Show vertically compared segments 
        if printflag:
            import matplotlib.pyplot as plt
            plt.plot(xint, y_obs[xint], 'k', linewidth=2)
            plt.plot(np.arange(len(y_obs)), y_obs, 'r')
            plt.plot(np.arange(len(y_sim)), y_sim, 'b')
            plt.show()

        # Determine the local y-values of the segment (based on the local x-locations)
        y_obs_seg = y_obs[x_obs_local_seg]
        y_sim_seg = y_sim[x_sim_local_seg]

        # Determine the y-values of the connectors in the current segment (based on the local x-locations) with linear interpolation
        try:
            con_y_obs_seg = np.interp(con_x_obs_global_seg, x_obs_local_seg, y_obs_seg)
            con_y_sim_seg = np.interp(con_x_sim_local_seg, x_sim_local_seg, y_sim_seg)
        except ValueError:
            continue

        # Find out whether the current segment is 'rise' or 'fall'
        if segs_obs[z]['sum_dQ'] > 0:  # rise
            # Calculate the length of the connectors (distance between connector points on obs and sim) in the current segment    
            # Time (x) distances 
            e_t_rise_seg = con_x_obs_global_seg - con_x_sim_global_seg  # > 0 means obs is later than sim

            # Magnitude distances
            if error_model == 'standard':  # compute the simple difference 
                e_q_rise_seg = con_y_obs_seg - con_y_sim_seg  # > 0 means obs is larger than sim
            elif error_model == 'relative':  # compute a scaled difference
                e_q_rise_seg = (con_y_obs_seg - con_y_sim_seg) / ((con_y_obs_seg + con_y_sim_seg) * 0.5)  # > 0 means obs is larger than sim
            else:
                raise ValueError('distance function not properly specified')

            # Add the errors of the segment to the overall errors of the event
            e_q_rise.extend(e_q_rise_seg)
            e_t_rise.extend(e_t_rise_seg)
            e_ysim_rise.extend(con_y_sim_seg)

            # Add vertical 1D error to output array
            e_rise_MD.extend((y_obs[xint] - y_sim[xint]) / ((y_obs[xint] + y_sim[xint]) * 0.5))
        else:  # fall
            # Calculate the length of the connectors (distance between connector points on obs and sim) in the current segment    
            # Time (x) distances
            e_t_fall_seg = con_x_obs_global_seg - con_x_sim_global_seg  # > 0 means obs is later than sim

            # Magnitude distances
            if error_model == 'standard':  # compute the simple difference
                e_q_fall_seg = con_y_obs_seg - con_y_sim_seg  # > 0 means obs is larger than sim
            elif error_model == 'relative':  # compute a scaled difference
                e_q_fall_seg = (con_y_obs_seg - con_y_sim_seg) / ((con_y_obs_seg + con_y_sim_seg) * 0.5)  # > 0 means obs is larger than sim
            else:
                raise ValueError('distance function not properly specified')

            # Add the errors of the segment to the overall errors of the event
            e_q_fall.extend(e_q_fall_seg)
            e_t_fall.extend(e_t_fall_seg)
            e_ysim_fall.extend(con_y_sim_seg)

            # Add vertical 1D error to output array
            e_fall_MD.extend((y_obs[xint] - y_sim[xint]) / ((y_obs[xint] + y_sim[xint]) * 0.5))

        # Add matching points(x,y) of the segment to overall matching points(for plotting)
        cons['x_match_obs_global'].extend(con_x_obs_global_seg)
        cons['y_match_obs'].extend(con_y_obs_seg)
        cons['x_match_sim_global'].extend(con_x_sim_global_seg)
        cons['y_match_sim'].extend(con_y_sim_seg)

    # Combine all case-specific error distributions to one for magnitude and one for time
    e_q = np.concatenate([e_q_rise, e_q_fall])
    e_t = np.concatenate([e_t_rise, e_t_fall])
    e_ysim = np.concatenate([e_ysim_rise, e_ysim_fall])

    return e_q, e_t, e_ysim, e_q_rise, e_t_rise, e_ysim_rise, e_q_fall, e_t_fall, e_ysim_fall, cons, e_rise_MD, e_fall_MD