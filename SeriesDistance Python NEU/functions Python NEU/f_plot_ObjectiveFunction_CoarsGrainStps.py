import matplotlib.pyplot as plt
import numpy as np

def f_plot_ObjectiveFunction_CoarsGrainStps(CoarseGrainStps, opt_step, titlestring, savefigure=False):
    """
    Plot the results of the objective function for all aggregation steps.
    
    Parameters:
    CoarseGrainStps (array-like): The values of the objective function for each coarse graining step.
    opt_step (int): The index of the optimal step.
    titlestring (str): The title of the plot.
    savefigure (bool, optional): Whether to save the figure. Default is False.
    """
    
    # plot the results of the objective function for all aggregation steps
    hfig = plt.figure(figsize=(8/2.54, 8/2.54))  # Convert size from cm to inches
    plt.hold(True)

    x = np.arange(1, len(CoarseGrainStps) + 1)  # create x-values
    rr, = plt.plot(x, CoarseGrainStps, 'ko')
    rr.set_markerfacecolor([0.85, 0.85, 0.85])
    rr.set_markersize(6)
    plt.ylim([0, max(1, max(CoarseGrainStps) * 1.1)])
    plt.gca().tick_params(axis='both', which='major', labelsize=10)
    # lbl = num2cell((0:length(CoarseGrainStps)-1));
    plt.box(True)
    
    # highlight the best
    oo, = plt.plot(opt_step + 1, CoarseGrainStps[opt_step], 'ko')  # opt_step + 1 to match MATLAB 1-based indexing
    oo.set_markerfacecolor('r')
    oo.set_markersize(6)
    
    plt.xlabel('coarse graining step (1=initial cond.)')
    plt.ylabel('objective function value (-)')
    plt.title(titlestring)
    plt.hold(False)
    
    # save output file
    if savefigure:
        plt.savefig('./results/results_of_aggregation.emf')
    
    plt.show()