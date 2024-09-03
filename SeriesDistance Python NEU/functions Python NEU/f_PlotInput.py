import matplotlib.pyplot as plt
import numpy as np

def f_plot_input(obs_org, obs, obs_events, sim_org, sim, sim_events, obs_sim_events_mapped, timeseries_splits):
    """
    Plots time series, events, events mappings and time series splits
    """

    # initialize
    legendtext = []
    nextlegendentry = 0
    h = []

    plt.figure()

    # if specified, plot time series of obs
    if obs is not None and len(obs) > 0:
        xes = np.arange(1, len(obs) + 1)
        h.append(plt.plot(xes, obs, '-b')[0])
        legendtext.append('obs')
        nextlegendentry += 1

    # if specified, plot time series of obs_org
    if obs_org is not None and len(obs_org) > 0:
        xes = np.arange(1, len(obs_org) + 1)
        h.append(plt.plot(xes, obs_org, '--b')[0])
        legendtext.append('obs_org')
        nextlegendentry += 1

    # if specified, plot time series of sim
    if sim is not None and len(sim) > 0:
        xes = np.arange(1, len(sim) + 1)
        h.append(plt.plot(xes, sim, '-r')[0])
        legendtext.append('sim')
        nextlegendentry += 1

    # if specified, plot time series of sim_org
    if sim_org is not None and len(sim_org) > 0:
        xes = np.arange(1, len(sim_org) + 1)
        h.append(plt.plot(xes, sim_org, '--r')[0])
        legendtext.append('sim_org')
        nextlegendentry += 1

    # if specified, plot obs events
    if obs_events is not None and len(obs_events) > 0:
        for i in range(obs_events.shape[0]):
            xes_obs = np.arange(obs_events[i, 0], obs_events[i, 1] + 1)
            if i == 0:
                h.append(plt.plot(xes_obs, obs[xes_obs - 1], '-b', linewidth=4)[0])
                legendtext.append('obs events')
                nextlegendentry += 1
            else:
                plt.plot(xes_obs, obs[xes_obs - 1], '-b', linewidth=4)

    # if specified, plot sim events
    if sim_events is not None and len(sim_events) > 0:
        for i in range(sim_events.shape[0]):
            xes_sim = np.arange(sim_events[i, 0], sim_events[i, 1] + 1)
            if i == 0:
                h.append(plt.plot(xes_sim, sim[xes_sim - 1], '-r', linewidth=4)[0])
                legendtext.append('sim events')
                nextlegendentry += 1
            else:
                plt.plot(xes_sim, sim[xes_sim - 1], '-r', linewidth=4)

    # if specified, plot event connections
    if obs_sim_events_mapped is not None and len(obs_sim_events_mapped) > 0:
        for i in range(obs_sim_events_mapped.shape[0]):
            # event starts
            xes_connect = [obs_sim_events_mapped[i, 0], obs_sim_events_mapped[i, 1]]
            yes_connect = [obs[xes_connect[0] - 1], sim[xes_connect[1] - 1]]
            plt.plot(xes_connect, yes_connect, '-k', linewidth=1)

            # event ends
            x_obs_end = obs_events[np.where(obs_events[:, 0] == obs_sim_events_mapped[i, 0])[0][0], 1]
            x_sim_end = sim_events[np.where(sim_events[:, 0] == obs_sim_events_mapped[i, 1])[0][0], 1]
            xes_connect = [x_obs_end, x_sim_end]
            yes_connect = [obs[xes_connect[0] - 1], sim[xes_connect[1] - 1]]
            plt.plot(xes_connect, yes_connect, '-k', linewidth=1)

        h.append(plt.plot([1, 1], [1, 1], '-k', linewidth=1)[0])
        legendtext.append('connected events')
        nextlegendentry += 1

    # if specified, plot time series splits
    if timeseries_splits is not None and len(timeseries_splits) > 0:
        for i in range(len(timeseries_splits)):
            xes = [timeseries_splits[i], timeseries_splits[i]]
            yes = [min(np.concatenate((obs, sim))), max(np.concatenate((obs, sim)))]
            plt.plot(xes, yes, '-.', linewidth=1, color=[0.65, 0.65, 0.65])

        h.append(plt.plot([1, 1], [1, 1], '-.', linewidth=1, color=[0.65, 0.65, 0.65])[0])
        legendtext.append('timesplits')
        nextlegendentry += 1

    # plot legend
    plt.legend(h, legendtext)
    plt.show()