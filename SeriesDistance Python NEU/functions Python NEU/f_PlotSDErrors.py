import matplotlib.pyplot as plt
import numpy as np

def f_PlotSDErrors(e_sd_q_rise, e_sd_t_rise, e_sd_q_fall, e_sd_t_fall, e_sd_q_rise_subset, e_sd_t_rise_subset, e_sd_q_fall_subset, e_sd_t_fall_subset):
    """
    Plots 2-d error distributions
    """
    
    ff = plt.figure(figsize=(16/2.54, 8/2.54))  # Convert cm to inches
    plt.gcf().set_size_inches(16, 8)
    
    # magnitude errors
    ax1 = plt.subplot(1, 2, 1)
    # ax1.hold(True)
    
    # plot the error distribution
    scatter1 = ax1.scatter(e_sd_t_rise, e_sd_q_rise, facecolors=[0.75, 0.75, 0.75], edgecolors=[0.75, 0.75, 0.75], marker='.')
    
    # set x- and ylim, but only if there is at least one non-NaN value in the plot
    if np.any(~np.isnan(e_sd_q_rise)):
        ylim = [np.nanmax([np.nanmax(e_sd_q_rise), abs(np.nanmin(e_sd_q_rise))]) * -1.1, 
                np.nanmax([np.nanmax(e_sd_q_rise), abs(np.nanmin(e_sd_q_rise))]) * 1.1]
        ax1.set_ylim(ylim)
        if np.all(e_sd_t_rise == 0):
            ax1.set_xlim([-0.1, 0.1])
        else:
            xlim = [np.nanmax([np.nanmax(e_sd_t_rise), abs(np.nanmin(e_sd_t_rise))]) * -1.1, 
                    np.nanmax([np.nanmax(e_sd_t_rise), abs(np.nanmin(e_sd_t_rise))]) * 1.1]
            ax1.set_xlim(xlim)
    
    # add vertical and horizontal lines
    ax1.axhline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    ax1.axvline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    
    # if specified, plot subset error distribution
    if e_sd_q_rise_subset and e_sd_t_rise_subset:
        scatter2 = ax1.scatter(e_sd_t_rise_subset, e_sd_q_rise_subset, facecolors=[0.3, 0.3, 0.3], edgecolors=[0.3, 0.3, 0.3], marker='.')
    
    # plot the mean (center) of the error distribution
    ax1.plot(np.nanmean(e_sd_t_rise), np.nanmean(e_sd_q_rise), marker='+', markersize=10, color='k', linestyle='none')
    
    # add labels
    ax1.set_xlabel('timing error (>0: obs later sim)')
    ax1.set_ylabel('magnitude error (>0: obs larger sim)')
    ax1.set_title('rising segments')
    
    # format the figure
    ax1.axis('square')
    # ax1.box(True)
    # ax1.hold(False)
    
    # add second plot on timing errors
    ax2 = plt.subplot(1, 2, 2)
    # ax2.hold(True)
    
    # plot the error distribution
    scatter1 = ax2.scatter(e_sd_t_fall, e_sd_q_fall, facecolors=[0.75, 0.75, 0.75], edgecolors=[0.75, 0.75, 0.75], marker='.')
    
    # set x- and ylim, but only if there is at least one non-NaN value in the plot
    if np.any(~np.isnan(e_sd_q_fall)):
        ylim = [np.nanmax([np.nanmax(e_sd_q_fall), abs(np.nanmin(e_sd_q_fall))]) * -1.1, 
                np.nanmax([np.nanmax(e_sd_q_fall), abs(np.nanmin(e_sd_q_fall))]) * 1.1]
        ax2.set_ylim(ylim)
        if np.all(e_sd_t_rise == 0):
            ax2.set_xlim([-0.1, 0.1])
        else:
            xlim = [np.nanmax([np.nanmax(e_sd_t_rise), abs(np.nanmin(e_sd_t_rise))]) * -1.1, 
                    np.nanmax([np.nanmax(e_sd_t_rise), abs(np.nanmin(e_sd_t_rise))]) * 1.1]
            ax2.set_xlim(xlim)
    
    # add vertical and horizontal lines
    ax2.axhline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    ax2.axvline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    
    # if specified, plot subset error distribution
    if e_sd_q_fall_subset and e_sd_t_fall_subset:
        scatter2 = ax2.scatter(e_sd_t_fall_subset, e_sd_q_fall_subset, facecolors=[0.3, 0.3, 0.3], edgecolors=[0.3, 0.3, 0.3], marker='.')
    
    # plot the mean (center) of the error distribution
    ax2.plot(np.nanmean(e_sd_t_fall), np.nanmean(e_sd_q_fall), marker='+', markersize=10, color='k', linestyle='none')
    
    # add labels
    ax2.set_xlabel('timing error (>0: obs later sim)')
    ax2.set_ylabel('magnitude error (>0: obs larger sim)')
    ax2.set_title('falling segments')
    
    # format the figure
    ax2.axis('square')
    # ax2.box(True)
    # ax2.hold(False)
    
    plt.show()