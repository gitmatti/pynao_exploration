# Copyright 2015 Mathias Schmerling
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================
"""
Some high-level plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.ndimage.filters import gaussian_filter1d
from pynao_exploration.utils.plotting.limits_reachability import limits

def plot_marker_detection(ax, marker_detected, width=500, **kwargs_plot):
    """Plots the frequency of detected markers.

    Args:
        ax (matplotlib.axes.AxesSubplot): the axis on which to plot.
        marker_detected (ndarray): the binary detection array
        width (float, optional): the smoothing window width. Gaussian
            window function.

    Keyword Args:
        label_fontsize (int): Default 10
        title_fontsize (int): Default 12
        tick_fontsize (int): Default 8
        linestyle ({'-', ...})
    """
    label_fontsize = kwargs_plot.pop('label_fontsize', 10)
    title_fontsize = kwargs_plot.pop('title_fontsize', 12)
    tick_fontsize = kwargs_plot.pop('tick_fontsize', 8)
    plot_specs = {'linestyle': '-'}
    plot_specs.update(kwargs_plot)
    ax.plot(gaussian_filter1d(marker_detected,sigma=width), **plot_specs)
    ax.set_xticks(np.arange(0,1001,200))
    ax.set_xticklabels(['0','10','20','30','40','50'])
    ax.set_ylim((-0.05,1.2))
    ax.set_xlim((0,1000))
    ax.set_ylabel('percentage (running average)', fontsize=label_fontsize)
    ax.set_xlabel('movements (x20 samples)', fontsize=label_fontsize)
    ax.set_title('Percentage of Marker Detection', fontsize=title_fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_fontsize) 
    ax.legend(loc='upper left', fontsize=11, frameon=False)


def scatter_plot(self, ax, *topics_dims, **kwargs):
    """TODO"""
    ms_limits = kwargs.pop('ms_limits', True)
    indices = kwargs.pop('indices',None)
    plot_specs = {'marker': 'o', 'linestyle': 'None'}
    plot_specs.update(kwargs)

    data, names = self.data_from_topics_dims(*topics_dims, indices=indices)

    for i in range(len(data)):
        if data[i].shape[1]==2:
            ax.plot(data[i][:,0],data[i][:,1], **plot_specs)
        elif data[i].shape[1]==3:
            ax.plot(data[i][:,0],data[i][:,1],data[i][:,2], **plot_specs)
        else:
            logger.warning("""Scatterplot only works for 2D and 3D data. 
                              Use NaoBabbling.scatter_matrix instead""")



def plot_learning_curve(eval_at, errors, fig=None, axes=None, **kwargs):
    """Plots a learning curve of the error over time.

    Args:
        eval_at (list[int]): evaluation times, plotted on x-axis
        errors (list[int]): list of errors of length len(eval_at)
        fig (matplotlib.figure.Figure, optional): a figure in which to plot.
            If 'None', a new figure is created.
        axes (list[matplotlib.axes.AxesSubplot], optional): possibility to 
            specify the list of axes in which to plot.

    Keyword Args:
        errorbar (boolean): If True (default), errorbars are plotted.
        capstyle ({'x','o','^', ...})
        show_misses (boolean): should the amount of visually missed
            testcases be displayed.
        shift (int): shifts the errorbars around to increase readability
        figsize (tuple[int]): figsize
        color
    """
    nans = np.isnan(errors)
    missed = np.sum(nans,axis=1)
    show_misses = kwargs.pop('show_misses', np.any(nans))
    figsize = kwargs.pop('figsize', (16,10))
    color = kwargs.pop('color', 'b')
    shift = kwargs.pop('shift', 0)
    capstyle = kwargs.pop('capstyle',None)
    errorbar = kwargs.pop('errorbar',True)

    if fig==None:
        if show_misses:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
            pos = axes[0].get_position()
            axes[0].set_position([pos.x0, 0.25, pos.width, 0.65 ])
            axes[1].set_position([pos.x0, 0.05, pos.width, 0.20 ])
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            axes = [axes]     

    avg_error = np.zeros(errors.shape[0])
    std_error = np.zeros(errors.shape[0])

    for i in range(errors.shape[0]):
        e = errors[i,np.invert(nans[i,:])]
        avg_error[i] = np.mean(e)
        std_error[i] = np.std(e)

    if errorbar:
        eb = axes[0].errorbar(np.array(eval_at)+shift, avg_error,
                              yerr=std_error,color=color, **kwargs)
        eb[-1][0].set_linestyle(kwargs.pop('linestyle','-'))
        for cap in eb[1]:
            cap.set_marker(capstyle)
    else:
        axes[0].plot(np.array(eval_at)+shift, avg_error, color=color, **kwargs)


    axis = axes[0].axis()
    axes[0].axis([-100, eval_at[-1] * 1.1, axis[2], axis[3]])
    if show_misses:
        axes[1].bar(np.array(eval_at)+shift, missed, 100, color=color)
        axes[1].axis([-100, eval_at[-1] * 1.1, 0, np.max(missed)+5])
        axes[0].xaxis.set_visible(False)
        axes[1].yaxis.set_ticks_position('right')
        axes[1].yaxis.set_label_position('right')
        axes[1].set_ylabel('trials where marker not visible', fontsize=14)
        axes[1].set_xlabel('Executed movements (x 20 samples)', fontsize=18)
    else:
        axes[0].set_xlabel('Executed movements (x 20 samples)', fontsize=18)

    axes[0].set_title('Test on {} sensory goals'.format(str(errors.shape[1])),
                      fontsize=22, y=1.03)
    axes[0].set_ylabel('Mean error, euclidian distance [m]', fontsize=14)

    return fig, axes


def show_development(data, intervals, **kwargs):
    """High-level hexbin plotting of the requested data over time.

    Args:
        data (ndarray): the data to plot
        intervals (ndarray, shape(k,2)): the k intervals that are to be
            plotted seperately. In the k-th row of the plot, all entries of
            topic[intervals[k,0]:intervals[k,1],dims] will be displayed.

    Keyword Args:
        color_offset (float): offsets the empty bins from the bins that
            contain at least one sample by some amount to increase
            readability.
        vmax (list[int]): list of length intervals.shape[0]. The k-th entry
            specifies the cap value for color in the k-th row of the plot.
        figsize (tuple[int]): figsize
        cmap (Colormap)
        gridsize (int): as in matplotlib.pyplot.hexbin.
        axes_names (list[str]): the labels for axes.
    """

    n_data, n_vars = data.shape
    n_interval = intervals.shape[0]
    n_plots = (n_vars**2 - n_vars)/2

    axes_limits = kwargs.pop('axes_limits',
                             np.array([[0.0,0.25],[0.0,0.3],[-0.1,0.3]]))
    topic = kwargs.pop('topic', 'dim')
    names = kwargs.pop('axes_names', ['{} {}'.format(topic, str(i))
                                      for i in range(n_vars)])
    #['X [m]','Y [m]','Z [m]'])
    figsize = kwargs.pop('figsize', None)
    vmax = kwargs.pop('vmax', None)
    masked = kwargs.pop('masked', False)
    color_offset = kwargs.pop('color_offset', -0.1)
    plot_specs = {'gridsize':20,'cmap':None}
    plot_specs.update(kwargs)

    fig, axes = plt.subplots(nrows=n_interval+1, ncols=n_plots, figsize=figsize)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    fig.subplots_adjust(right=0.9, bottom=0.1)
    #x = data[:,0]
    #y = data[:,1]
    #z = data[:,2]

    for i,interval in enumerate(intervals):
        for x in range(n_vars):
            for y in range(n_vars):
                if x<y:
                    ax = axes[i,x+y-1]
                    ax.set_xlabel(names[x], fontsize=10)
                    if not ax.is_first_col():
                        ax.set_ylabel(names[y], fontsize=10)
                    else:
                        ax.set_ylabel(names[y], fontsize=10)
                    ax.set_xticks([]) #ax.set_xticks(axes_limits[x])
                    ax.set_yticks([]) #ax.set_yticks(axes_limits[y])
                    #ax.set_xlim(axes_limits[x])
                    #ax.set_ylim(axes_limits[y])
                    try:
                        vm = vmax[i]
                    except TypeError:
                        vm = vmax
                    im = ax.hexbin(data[interval[0]:interval[1],x],
                                   data[interval[0]:interval[1],y],
                                   vmax=vm,
                                   extent=(axes_limits[x,0],
                                           axes_limits[x,1],
                                           axes_limits[y,0],
                                           axes_limits[y,1]),
                                   **plot_specs)
                    #if masked:
                    #    im.set_array(np.ma.array(im.get_array(), mask=im.get_array()==0))
                    im.set_array(np.where(im.get_array()==0,
                                          color_offset*vm,
                                          im.get_array()))
                    ax.plot(limits[x+y-1][:,0],
                            limits[x+y-1][:,1], lw=1, color='k')
                    if not ax.is_first_col() and not ax.is_last_col():
                        title = '{} {} to {}'.format(topic,
                                                     str(interval[0]),
                                                     str(interval[1]))
                        ax.set_title(title, fontsize=16, y=1.03)
                    if ax.is_last_col():
                        pos = ax.get_position()
                        cbar_ax = fig.add_axes([pos.x0+pos.width+0.02,
                                                pos.y0+0.01,
                                                0.01,
                                                pos.height-0.02])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        cb.set_ticks(np.round(np.linspace(0,vm,5)))

    for x in range(n_vars):
        for y in range(n_vars):
            if x<y:
                try:
                    vm = vmax[-1]
                except TypeError:
                    vm = vmax
                ax = axes[n_interval,x+y-1]
                ax.set_xlabel(names[x], fontsize=10)
                if not ax.is_first_col():
                    ax.set_ylabel(names[y], fontsize=10)
                else:
                    ax.set_ylabel(names[y], fontsize=10)
                ax.set_xticks([]) #ax.set_xticks(axes_limits[x])
                ax.set_yticks([]) #ax.set_yticks(axes_limits[y])
                
                im = ax.hexbin(data[:,x],data[:,y], 
                               vmax=vm,
                               extent=(axes_limits[x,0],
                                       axes_limits[x,1],
                                       axes_limits[y,0],
                                       axes_limits[y,1]),
                               **plot_specs)
                #if masked:
                #    im.set_array(np.ma.array(im.get_array(), mask=im.get_array()==0))
                im.set_array(np.where(im.get_array()==0,
                                      color_offset*vm,
                                      im.get_array()))
                ax.plot(limits[x+y-1][:,0],limits[x+y-1][:,1], lw=1, color='k')
                if not ax.is_first_col() and not ax.is_last_col():
                    title = ' overall'
                    ax.set_title(title, fontsize=16, y=1.03)
                if ax.is_last_col():
                    pos = ax.get_position()
                    cbar_ax = fig.add_axes([pos.x0+pos.width+0.02,
                                            pos.y0+0.01,
                                            0.01,
                                            pos.height-0.02])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    cb.set_ticks(np.linspace(0,vm,5))

    return fig, axes, im
