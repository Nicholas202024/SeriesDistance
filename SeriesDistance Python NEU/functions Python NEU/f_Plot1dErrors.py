import matplotlib.pyplot as plt
import numpy as np

def f_Plot1dErrors(e_sd_q_1d, e_sd_q_1d_subset, titlestring):
    """
    Plots 1-d error distributions
    """
    
    # Modification history
    # - 2015/08/01:  Uwe Ehret and Simon Seibert: First version
    # - 2015/09/16: Uwe Ehret: Included possibility for separate error distributions for discharge ranges

    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))  # Convert cm to inches
    fig.set_size_inches(8, 8)
    # ax.hold(True)
    
    # plot the error distribution
    cdf_e_sd_q_1d = np.linspace(0, 1, len(e_sd_q_1d))
    e_sd_q_1d_sorted = np.sort(e_sd_q_1d)
    ax.plot(e_sd_q_1d_sorted, cdf_e_sd_q_1d, marker='.', linestyle='none', 
            markerfacecolor=[0.75, 0.75, 0.75], markeredgecolor=[0.75, 0.75, 0.75])
    
    # plot the mean (center) of the error distribution
    e_mean = np.mean(e_sd_q_1d)
    pu_e_mean = cdf_e_sd_q_1d[np.where(e_sd_q_1d_sorted >= e_mean)[0][0]]
    ax.plot(e_mean, pu_e_mean, marker='+', markersize=10, linestyle='none', 
            markerfacecolor='none', markeredgecolor='k')
    
    # if specified, plot subset error distribution
    if e_sd_q_1d_subset is not None and len(e_sd_q_1d_subset) > 0:
        # find the pu-values for all points in the subset
        indx = np.where(np.isin(e_sd_q_1d_sorted, e_sd_q_1d_subset))[0]
        ax.plot(e_sd_q_1d_subset, cdf_e_sd_q_1d[indx], '.r')
    
    # set x-lim, but only if there is at least one non-NaN value in the plot
    if np.any(~np.isnan(e_sd_q_1d_sorted)):
        xlim = [max(np.max(e_sd_q_1d_sorted), abs(np.min(e_sd_q_1d_sorted))) * -1.1, 
                max(np.max(e_sd_q_1d_sorted), abs(np.min(e_sd_q_1d_sorted))) * 1.1]
        ax.set_xlim(xlim)
    
    # Add vertical and horizontal lines and format axes
    ax.axhline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    ax.axvline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    ax.set_xlabel('magnitude error (>0: obs larger sim)')
    ax.set_ylabel('cum. prob. of unexceedance')
    ax.set_title(titlestring)
    ax.axis('square')
    # ax.box(True)
    # ax.hold(False)
    
    plt.show()