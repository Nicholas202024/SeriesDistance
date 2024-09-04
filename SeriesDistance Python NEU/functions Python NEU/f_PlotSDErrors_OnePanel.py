import matplotlib.pyplot as plt
import numpy as np

def f_PlotSDErrors_OnePanel(t_errors, q_errors):
    # Plots 2-d error distributions
    
    fig, ax = plt.subplots(figsize=(8/2.54, 8/2.54))  # Convert cm to inches
    fig.set_size_inches(8/2.54, 8/2.54)  # Set paper size in inches
    
    # ax.hold(True)  # Hold on (deprecated in newer versions of matplotlib)
    
    # magnitude errors
    # plot the error distribution
    scatter1 = ax.scatter(t_errors, q_errors, facecolors=[0.75, 0.75, 0.75], edgecolors=[0.75, 0.75, 0.75], marker='.')
    
    # set x- and ylim, but only if there is at least one non-NaN value in the plot
    if np.any(~np.isnan(t_errors)):
        ax.set_ylim([np.nanmax([np.nanmax(q_errors), abs(np.nanmin(q_errors))])*-1.1, np.nanmax([np.nanmax(q_errors), abs(np.nanmin(q_errors))])*1.1])
        if np.all(q_errors == 0):
            ax.set_xlim([-0.1, 0.1])
        else:
            ax.set_xlim([np.nanmax([np.nanmax(t_errors), abs(np.nanmin(t_errors))])*-1.1, np.nanmax([np.nanmax(t_errors), abs(np.nanmin(t_errors))])*1.1])
    
    # add vertical and horizontal lines
    ax.axhline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    ax.axvline(0, linestyle='-.', linewidth=1, color=[0.65, 0.65, 0.65])
    
    # plot the mean (center) of the error distribution
    ax.plot(np.nanmean(t_errors), np.nanmean(q_errors), marker='+', markersize=10, markerfacecolor='none', markeredgecolor='k', linestyle='none')
    
    # add labels
    ax.set_xlabel('timing error (>0: obs later sim)')
    ax.set_ylabel('magnitude error (>0: obs larger sim)')
    # titlestr = '2d error distribution for rising segments, Q range: {} <= Qsim < {}'.format(error_lvls[i], error_lvls[i+1])
    ax.set_title('entire time series')
    
    # format the figure
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.show()