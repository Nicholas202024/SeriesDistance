import matplotlib.pyplot as plt
import numpy as np

def f_PlotConnectedSeries(obs, segs_obs, sim, segs_sim, connectors, showEventIndex=False):
    """
    Plots two related time series (obs and sim) and their series-distance connectors.
    19.Nov.2013 Uwe Ehret
    """

    show_connectors = True

    # Distance vectors between matching points obs/sim
    if show_connectors:
        # Original code der Übersetzung
        # u = connectors[0]['x_match_sim_global'] - connectors[0]['x_match_obs_global']
        # v = connectors[0]['y_match_sim'] - connectors[0]['y_match_obs']
        
        # print('connectors:', connectors)
        # print('\n')
        
        try:
            u = [a - b for a, b in zip(connectors[0]['x_match_sim_global'], connectors[0]['x_match_obs_global'])]
            v = [a - b for a, b in zip(connectors[0]['y_match_sim'], connectors[0]['y_match_obs'])]
        except:
            u = [a - b for a, b in zip(connectors['x_match_sim_global'], connectors['x_match_obs_global'])]
            v = [a - b for a, b in zip(connectors['y_match_sim'], connectors['y_match_obs'])]

    fig, ax = plt.subplots()
    # ax.hold(True)

    # Plot the timeseries
    ax.plot(np.arange(1, len(obs) + 1), obs, color=[0.2, 0.2, 0.2], label='observation')
    ax.plot(np.arange(1, len(sim) + 1), sim, '--', color=[0.2, 0.2, 0.2], label='simulation')

    # Plot Feature Distance lines
    # Distance vectors between matching points obs/sim
    if show_connectors:
        try:
            ax.quiver(connectors[0]['x_match_obs_global'], connectors[0]['y_match_obs'], u, v, angles='xy', scale_units='xy', scale=1, color=[0.6, 0.6, 0.6], headlength=0, headwidth=0)
        except:
            ax.quiver(connectors['x_match_obs_global'], connectors['y_match_obs'], u, v, angles='xy', scale_units='xy', scale=1, color=[0.6, 0.6, 0.6], headlength=0, headwidth=0)

    # Plot the connected segments in unique color
    num_segs = len(segs_obs)
    cmap = np.array([[1, 0, 0], [0, 0.4, 1], [0, 1, 0], [0.8, 0, 1], [1, 0.8, 0.5]])  # colormap

    cmap_count = 0
    for z in range(num_segs):
        xes_global = np.arange(segs_obs[z]['starttime_global'], segs_obs[z]['endtime_global'] + 1)
        yes_global = obs[xes_global - 1]  # Adjust for 0-based indexing in Python
        ax.plot(xes_global, yes_global, '-', color=cmap[cmap_count], linewidth=2)

        xes_global = np.arange(segs_sim[z]['starttime_global'], segs_sim[z]['endtime_global'] + 1)
        yes_global = sim[xes_global - 1]  # Adjust for 0-based indexing in Python
        ax.plot(xes_global, yes_global, '--', color=cmap[cmap_count], linewidth=2)

        cmap_count += 1
        if cmap_count >= len(cmap):
            cmap_count = 0

    # Formatting
    if not show_connectors:
        ax.legend(['observation', 'simulation'], loc='upper right')
    else:
        ax.legend(['observation', 'simulation', 'connectors'], loc='upper right')
    ax.legend().set_visible(True)

    ax.tick_params(axis='both', which='major', labelsize=11, width=2)
    ax.grid(True)

    # Add event index to plot
    if showEventIndex:
        event_ind = [0] + list(np.where(np.diff([seg['eventID'] for seg in segs_obs]) == 1)[0] + 1)
        for kk in range(len(event_ind)):
            ax.text(segs_obs[event_ind[kk]]['starttime_global'], 0, f'# {kk + 1}')
    plt.box(True)
    # ax.hold(False)
    plt.show()