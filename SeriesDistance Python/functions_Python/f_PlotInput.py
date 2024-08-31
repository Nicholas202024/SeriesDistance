import matplotlib.pyplot as plt

def f_PlotInput(obs_org, obs, obs_events, sim_org, sim, sim_events, obs_sim_events_mapped, timeseries_splits):
    """
    Plots time series, events, events mappings and time series splits
    """

    # initialize
    legendtext = []
    nextlegendentry = 1
    h = []
    plt.figure()

    # if specified, plot time series of obs
    if obs is not None:
        xes = range(1, len(obs[0]) + 1)
        h.append(plt.plot(xes, obs[0], '-b')[0])
        legendtext.append('obs')
        nextlegendentry += 1

    # if specified, plot time series of obs_org
    if obs_org is not None:
        xes = range(1, len(obs_org) + 1)
        h.append(plt.plot(xes, obs_org, '--b')[0])
        legendtext.append('obs_org')
        nextlegendentry += 1

    # if specified, plot time series of sim
    if sim is not None:
        xes = range(1, len(sim[0]) + 1)
        h.append(plt.plot(xes, sim[0], '-r')[0])
        legendtext.append('sim')
        nextlegendentry += 1

    # if specified, plot time series of sim_org
    if sim_org is not None:
        xes = range(1, len(sim_org) + 1)
        h.append(plt.plot(xes, sim_org, '--r')[0])
        legendtext.append('sim_org')
        nextlegendentry += 1

    # if specified, plot obs events
    if obs_events is not None:
        for i in range(len(obs_events)):
            xes_obs = range(obs_events[i, 0], obs_events[i, 1] + 1)
            if i == 0:
                h.append(plt.plot(xes_obs, obs[obs_events[i, 0]:obs_events[i, 1] + 1], '-b', linewidth=4)[0])
                legendtext.append('obs events')
                nextlegendentry += 1
            else:
                plt.plot(xes_obs, obs[obs_events[i, 0]:obs_events[i, 1] + 1], '-b', linewidth=4)

    # if specified, plot sim events
    if sim_events is not None:
        for i in range(len(sim_events)):
            xes_sim = range(sim_events[i, 0], sim_events[i, 1] + 1)
            if i == 0:
                h.append(plt.plot(xes_sim, sim[sim_events[i, 0]:sim_events[i, 1] + 1], '-r', linewidth=4)[0])
                legendtext.append('sim events')
                nextlegendentry += 1
            else:
                plt.plot(xes_sim, sim[sim_events[i, 0]:sim_events[i, 1] + 1], '-r', linewidth=4)

    # if specified, plot event connections
    if obs_sim_events_mapped is not None:
        for i in range(len(obs_sim_events_mapped)):
            # event starts
            xes_connect = [obs_sim_events_mapped[i, 0], obs_sim_events_mapped[i, 1]]
            yes_connect = [obs[xes_connect[0]], sim[xes_connect[1]]]
            plt.plot(xes_connect, yes_connect, '-k', linewidth=1)

            # event ends
            x_obs_end = obs_events[np.where(obs_events[:, 0] == obs_sim_events_mapped[i, 0])[0][0], 1]
            x_sim_end = sim_events[np.where(sim_events[:, 0] == obs_sim_events_mapped[i, 1])[0][0], 1]
            xes_connect = [x_obs_end, x_sim_end]
            yes_connect = [obs[xes_connect[0]], sim[xes_connect[1]]]
            plt.plot(xes_connect, yes_connect, '-k', linewidth=1)

        h.append(plt.plot([1, 1], [1, 1], '-k', linewidth=1)[0])
        legendtext.append('connected events')
        nextlegendentry += 1

    # if specified, plot time series splits
    if timeseries_splits is not None:
        for i in range(len(timeseries_splits)):
            xes = [timeseries_splits[i], timeseries_splits[i]]
            yes = [min(min(obs[0]), min(sim[0])), max(max(obs[0]), max(sim[0]))]
            plt.plot(xes, yes, '-.', linewidth=1, color=[0.65, 0.65, 0.65])

        h.append(plt.plot([1, 1], [1, 1], '-k', linewidth=1)[0])
        legendtext.append('timesplits')
        nextlegendentry += 1

    # plot legend
    plt.legend(h, legendtext)
    plt.show()