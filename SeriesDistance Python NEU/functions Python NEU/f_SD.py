import numpy as np

def f_sd(y_obs, segs_obs, y_sim, segs_sim, error_model, printflag=False):
    """
    Calculates the distance vectors in time and value between two matching events (obs/sim)

    Modification history
    - 2015/08/03: Uwe Ehret, Simon Seibert: First version
    - 2015/09/16: Uwe Ehret: Included possibility for separate error distributions for discharge ranges

    OUTPUT
        e_q           # magnitude errors for both rise and fall, e_q=[e_q_rise; e_q_fall]
        e_t           # timing errors for both rise and fall,    e_t=[e_t_rise; e_t_fall]
        e_ysim        # 
        e_q_rise      # magnitude errors in rising limbs
        e_t_rise      # time errors in rising limbs
        e_ysim_rise
        e_q_fall      # magnitude errors in falling limbs
        e_t_fall      # time errors in falling limbs
        e_ysim_fall
        cons          # SD connectors
        e_rise_MD     # 1D magnitude errors of corresponding rising limb sections
        e_fall_MD     # 1D magnitude errors of corresponding falling limbs sections

    INPUT
        y_obs: (n,1) array with observed values
        segs_obs: list of dictionaries with observed segments, where each dictionary represents a segment
        y_sim: (m,1) array with simulated values
        segs_sim: list of dictionaries with simulated segments, where each dictionary represents a segment
        error_model: string, sets the way the magnitude distance among obs and sim is computed
                     if 'relative': dist_v = (obs - sim) / ((obs + sim)*0.5)
                     if 'standard': dist_v = (obs - sim)
                     Default: 'standard'
        printflag: boolean, if true, the connector points of series distance will be stored in the global variables *_match_*

    METHOD
        Note: 
        - The # of obs and sim segments needs to be equal
        - The order of the segment types of obs and sim has to be equal: either both start with a 'rise' or a 'fall'
        the total number of connectors for the event equals mean(length(obs_event),length(sim_event))
        the number of connectors per segment is determined by the mean importance of the segment (mean of obs and sim relevance)
    """

    cons = {
        'x_match_obs_global': [],
        'y_match_obs': [],
        'x_match_sim_global': [],
        'y_match_sim': []
    }

    # specify connectors

    # determine the total number of connectors (average of total length of obs and sim event times the user-specified percentage)
    totnumcons = (len(y_obs) + len(y_sim)) * 0.5

    # determine the share of connectors for each segment
    num_segs = len(segs_obs)  # # of segments (could also be len(segs_sim))
    segs_cons = np.full(num_segs, np.nan)  # variable for the number of connectors assigned to each segment
    sum_rels = sum([seg['relevance'] for seg in segs_obs]) + sum([seg['relevance'] for seg in segs_sim])  # the overall sum of relevance (as relevance is already normalized, should be 1 + 1 = 2)

    # raise Exception('f_SD STOP')
    # the share of connectors for each segment is proportional to its relative relevance
    segs_cons = np.round(([seg['relevance'] for seg in segs_obs] + [seg['relevance'] for seg in segs_sim]) * int(np.round(totnumcons)) / sum_rels)

    # initialize output variables   

    e_q_rise = []      # magnitude errors in rising limbs
    e_t_rise = []      # time errors in rising limbs
    e_q_fall = []      # magnitude errors in falling limbs
    e_t_fall = []      # time errors in falling limbs
    e_ysim_rise = []   # simulated discharge for rising limbs, corresponding to each error (needed to subdivide errors in discharge classes)
    e_ysim_fall = []   # simulated discharge for falling limbs, corresponding to each error (needed to subdivide errors in discharge classes)
    e_rise_MD = []     # 1D magnitude errors in corresponding rising limb sections 
    e_fall_MD = []     # 1D magnitude errors in corresponding falling limb sections 

    # loop over all segments
    for z in range(num_segs):

        # determine the GLOBAL x-location (time) of the connectors in the current segment  
        num = int(segs_cons[z])      
        con_x_obs_global_seg = np.linspace(segs_obs[z]['starttime_global'], segs_obs[z]['endtime_global'], num)
        con_x_sim_global_seg = np.linspace(segs_sim[z]['starttime_global'], segs_sim[z]['endtime_global'], num)  

        # determine the LOCAL x-location (time) of the connectors in the current segment     
        con_x_obs_local_seg = np.linspace(segs_obs[z]['starttime_local'], segs_obs[z]['endtime_local'], num)
        con_x_sim_local_seg = np.linspace(segs_sim[z]['starttime_local'], segs_sim[z]['endtime_local'], num)      

        # determine the local x-locations of the segment
        x_obs_local_seg = np.arange(segs_obs[z]['starttime_local'], segs_obs[z]['endtime_local'] + 1)
        x_sim_local_seg = np.arange(segs_sim[z]['starttime_local'], segs_sim[z]['endtime_local'] + 1)    

        xobs = np.arange(segs_obs[z]['starttime_local'], segs_obs[z]['endtime_local'] + 1)
        xsim = np.arange(segs_sim[z]['starttime_local'], segs_sim[z]['endtime_local'] + 1)
        xint = np.intersect1d(xobs, xsim)

        # show vertically compared segments 
        if False:
            import matplotlib.pyplot as plt
            plt.plot(xint, y_obs[xint], 'k', linewidth=2)
            plt.plot(xint, y_sim[xint], 'k', linewidth=2)
            plt.plot(np.arange(len(y_obs)), y_obs, 'r')
            plt.plot(np.arange(len(y_sim)), y_sim, 'b')
            plt.show()

        # determine the local y-values of the segment (based on the local x-locations)
        y_obs_seg = y_obs[x_obs_local_seg]
        y_sim_seg = y_sim[x_sim_local_seg]  

        # determine the y-values of the connectors in the current segment (based on the local x-locations) with linear interpolation
        con_y_obs_seg = np.interp(con_x_obs_local_seg, x_obs_local_seg, y_obs_seg) 
        con_y_sim_seg = np.interp(con_x_sim_local_seg, x_sim_local_seg, y_sim_seg)     

        # find out whether the current segment is 'rise' or 'fall'
        if segs_obs[z]['sum_dQ'] > 0:   # rise
            # calculate the length of the connectors (distance between connector points on obs and sim) in the current segment    
            # time (x) distances 
            e_t_rise_seg = (con_x_obs_global_seg - con_x_sim_global_seg)  # > 0 means obs is later than sim

            # magnitude distances
            if error_model == 'standard':  # compute the simple difference 
                e_q_rise_seg = (con_y_obs_seg - con_y_sim_seg)  # > 0 means obs is larger than sim
            elif error_model == 'relative':  # compute a scaled difference
                e_q_rise_seg = (con_y_obs_seg - con_y_sim_seg) / ((con_y_obs_seg + con_y_sim_seg) * 0.5)  # > 0 means obs is larger than sim
            else:
                raise ValueError('distance function not properly specified')

            # add the errors of the segment to the overall errors of the event
            e_q_rise.extend(e_q_rise_seg)
            e_t_rise.extend(e_t_rise_seg)  
            e_ysim_rise.extend(con_y_sim_seg)

            # add vertical 1D error to output array
            e_rise_MD.extend((y_obs[xint] - y_sim[xint]) / ((y_obs[xint] + y_sim[xint]) * 0.5))
        else:  # fall
            # calculate the length of the connectors (distance between connector points on obs and sim) in the current segment    
            # time (x) distances
            e_t_fall_seg = (con_x_obs_global_seg - con_x_sim_global_seg)  # > 0 means obs is later than sim

            # magnitude distances
            if error_model == 'standard':  # compute the simple difference
                e_q_fall_seg = (con_y_obs_seg - con_y_sim_seg)  # > 0 means obs is larger than sim
            elif error_model == 'relative':  # compute a scaled difference
                e_q_fall_seg = (con_y_obs_seg - con_y_sim_seg) / ((con_y_obs_seg + con_y_sim_seg) * 0.5)  # > 0 means obs is larger than sim
            else:
                raise ValueError('distance function not properly specified')

            # add the errors of the segment to the overall errors of the event
            e_q_fall.extend(e_q_fall_seg)
            e_t_fall.extend(e_t_fall_seg) 
            e_ysim_fall.extend(con_y_sim_seg)

            # add vertical 1D error to output array
            e_fall_MD.extend((y_obs[xint] - y_sim[xint]) / ((y_obs[xint] + y_sim[xint]) * 0.5))

        # add matching points(x,y) of the segment to overall matching points(for plotting)
        cons['x_match_obs_global'].extend(con_x_obs_global_seg)
        cons['y_match_obs'].extend(con_y_obs_seg)
        cons['x_match_sim_global'].extend(con_x_sim_global_seg)
        cons['y_match_sim'].extend(con_y_sim_seg)

    # combine all case-specific error distributions to one for magnitude and one for time
    e_q = np.concatenate([e_q_rise, e_q_fall])
    e_t = np.concatenate([e_t_rise, e_t_fall])
    e_ysim = np.concatenate([e_ysim_rise, e_ysim_fall])

    return e_q, e_t, e_ysim, e_q_rise, e_t_rise, e_ysim_rise, e_q_fall, e_t_fall, e_ysim_fall, cons, e_rise_MD, e_fall_MD