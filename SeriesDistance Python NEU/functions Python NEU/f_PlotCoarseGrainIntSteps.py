import matplotlib.pyplot as plt
import numpy as np

def f_PlotCoarseGrainIntSteps(obs, segs_obs, sim, segs_sim, connectors, titlestring):
    """
    Plots two related time series (obs and sim) and their series-distance connectors.
    
    Parameters:
    obs: array-like, observed time series
    segs_obs: list of dictionaries, segments of the observed time series
    sim: array-like, simulated time series
    segs_sim: list of dictionaries, segments of the simulated time series
    connectors: dictionary, containing matching points between obs and sim
    titlestring: string, title of the plot
    
    Returns:
    None
    """
    
    show_connectors = True

    print('connectors:', connectors)
    print('\n')
    # Distance vectors between matching points obs/sim
    if show_connectors:
        # Original code der übersetzung
        # u = connectors['x_match_sim_global'] - connectors['x_match_obs_global']
        # v = connectors['y_match_sim'] - connectors['y_match_obs']
        u = [a - b for a, b in zip (connectors['x_match_sim_global'], connectors['x_match_obs_global'])]
        v = [a - b for a, b in zip (connectors['y_match_sim'], connectors['y_match_obs'])]

    plt.figure()

    # Plot the time series
    plt.plot(range(segs_obs[0]['starttime_global'], segs_obs[-1]['endtime_global'] + 1), obs, '-b')
    plt.plot(range(segs_sim[0]['starttime_global'], segs_sim[-1]['endtime_global'] + 1), sim, '--r')

    # Plot Feature Distance lines
    # Distance vectors between matching points obs/sim
    if show_connectors:
        plt.quiver(connectors['x_match_obs_global'], connectors['y_match_obs'], u, v, angles='xy', scale_units='xy', scale=1, color=[0.6, 0.6, 0.6], headwidth=0, headlength=0)

    # Plot the connected segments in unique color
    num_segs = len(segs_obs)
    cmap = np.array([[1, 0, 0], [0, 0.4, 1], [0, 1, 0], [0.8, 0, 1], [1, 0.8, 0.5]])  # colormap

    cmap_count = 0
    for z in range(num_segs):
        xes_global = range(segs_obs[z]['starttime_global'], segs_obs[z]['endtime_global'] + 1)
        plt.plot(xes_global, obs[segs_obs[z]['starttime_local']:segs_obs[z]['endtime_local'] + 1], '-', color=cmap[cmap_count], linewidth=2)
        
        xes_global = range(segs_sim[z]['starttime_global'], segs_sim[z]['endtime_global'] + 1)
        plt.plot(xes_global, sim[segs_sim[z]['starttime_local']:segs_sim[z]['endtime_local'] + 1], '--', color=cmap[cmap_count], linewidth=2)
        
        cmap_count += 1
        if cmap_count >= len(cmap):
            cmap_count = 0

    # Formatting
    if not show_connectors:
        plt.legend(['observation', 'simulation'], loc='northeast')
    else:
        plt.legend(['observation', 'simulation', 'connectors'], loc='upper right')
    plt.gca().legend().set_visible(False)
    
    ax1 = plt.gca()
    ax1.tick_params(axis='both', which='major', labelsize=11, width=2)
    plt.title(titlestring, fontsize=14, fontweight='bold')
    plt.box(True)
    plt.show()