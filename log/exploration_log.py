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
This module provides an logging and plotting class for exploration experiments
with the framework.
"""
# generic imports
import numpy as np
import os
# relative imports
from pynao_exploration.utils.config import make_nao_configuration
from pynao_exploration.utils.plotting.plots import plot_learning_curve
from pynao_exploration.utils.plotting.plots import show_development
from pynao_exploration.utils.plotting.plots import plot_marker_detection
from pynao_exploration.utils.io import make_files, from_files


this_path = os.path.dirname(os.path.realpath(__file__))
default_knowledge_base = os.path.join(this_path, "..", "data", "knowledge",
                                      "knowledge_base_200samples_resampling.txt")

class NaoExplorationLog(object):
    """Logging class for sensorimotor exploration.

    Attributes:
        conf (NaoConfiguration): configuration object.
        expl_dims (list[int]): list of explored dimensions.
        inf_dims (list[int]): list of inference dimensions.
        call (dict): dictionary with the original experiment call.
        n_iterations (int): number of maximum iterations.
        t (int): number of current iterations.
        continued (boolean): does the log belong to a continued experiment?
        eval_at (list[int]): list of evaluation time points
        forward_error (list): list of forward prediction errors
        inverse_error (list): list of inverse prediction errors
        data (dict): the collected data of the experiment
        """
    def __init__(self, conf, expl_dims, inf_dims, call, **kwargs):
        """Initialize a logging class object.

        Args:
            conf (NaoConfiguration): a NaoConfiguration object.
            expl_dims (list[int]): explored dimensions.
            inf_dims (list[int]): inference dimensions.
            call (dict): a copy of the call to the respective logged experiment
        """
        self.conf = conf
        self.expl_dims = expl_dims
        self.inf_dims = inf_dims
        self.call = call
        self.n_iterations = self.call['n_iterations']
        self._subsampling = self.call['subsampling']
        try:
            self._samples_between_goals = self.call['samples_between_goals']
        except KeyError:
            self._samples_between_goals = self.call['samplesBetweenGoals']
        self.t = 0
        self.continued = False

        self.eval_at = []
        #self.evaluated_at = []
        self.forward_error = []
        self.inverse_error = []

        self.data = {}
        self.data['choices'] = np.zeros((self.n_iterations,
                                         len(self.expl_dims)))
        self.data['inferences'] = np.zeros((self.n_iterations,
                                            len(self.inf_dims)))
        self.data['movements'] = np.zeros((self.n_iterations,
                                           self.conf.m_ndims))
        self.data['sensors'] = np.zeros((self.n_iterations,
                                         self.conf.s_ndims))
        if self._subsampling:
            n_experience = self.n_iterations * self._samples_between_goals
        else:
            n_experience = self.n_iterations
        self.data['experience'] = np.zeros((n_experience,self.conf.ndims))
        self.data['marker_detected'] = np.zeros(n_experience)

    def evaluate_at(self, eval_at, append=False):
        """Sets the evaluation times of the experiment.

        Args:
            eval_at (list[int]): a list of evaluation time points.
        """
        if not append:
            self.eval_at = sorted(eval_at)
        else:
            self.eval_at = self.eval_at + eval_at

    def default_testcases(self, n_samples=-1):
        """Tells the experiment to use the default testcases."""
        self.make_testcases(default_knowledge_base)

    def make_testcases(self, fname, n_samples=-1, **load_kwargs):
        """Allows to explicitly specifiy which testcases to use.

        Args:
            fname (str): path to the testcases file
        """
        f = open(fname, 'r')
        header = f.readline()
        f.close()
        header = header.split(' ')[1:]
        m_dims = [header.index(item) for item 
                  in ['HeadPitch','HeadYaw','LShoulderRoll',
                      'LShoulderPitch','LElbowRoll','LElbowYaw']]
        s_dims = [header.index(item) for item 
                  in ['imageX','imageY','spaceX','spaceY','spaceZ']]
        testcases = np.loadtxt(fname, **load_kwargs)
        if n_samples > testcases.shape[0] or n_samples<=0:
            n_samples = testcases.shape[0]
        perm = np.random.permutation(testcases.shape[0])
        self.testcases = testcases[perm[:n_samples],:]
        self.testcases = self.testcases[:,m_dims+s_dims]

    # TODO
    def axes_limits(self, topic, dims):
        bounds = []
        for i, topic in zip(topic,dims):
            if topic == 'motor':
                bounds.extend(list(self.conf.m_bounds[:, dims].T.flatten()))
            elif topic == 'sensori':
                bounds.extend(list(self.conf.s_bounds[:, dims].T.flatten()))
            elif topic == 'choice':
                bounds.extend(list(self.conf.bounds[:, [self.expl_dims[d]
                                                        for d in dims]].T.flatten()))
            elif topic == 'inference':
                bounds.extend(list(self.conf.bounds[:, [self.inf_dims[d]
                                                        for d in dims]].T.flatten()))
            else:
                raise ValueError("Only valid for 'motor', 'sensori', 'choice' and 'inference' topics")
        return bounds

    def plot_marker_detection(self, ax, width=500, **kwargs_plot):
        """Plots the frequency of detected markers.

        Args:
            ax (matplotlib.axes.AxesSubplot): the axis on which to plot.
            width (float, optional): the smoothing window width. Gaussian
                window function.

        Keyword Args:
            label_fontsize (int): Default 10
            title_fontsize (int): Default 12
            tick_fontsize (int): Default 8
            linestyle ({'-', ...})
        """
        plot_marker_detection(ax, self.data['marker_detected'], width=width,
                              **kwargs_plot)

    def scatter_plot(self, ax, *topics_dims, **kwargs):
        """Not functional"""
        #ms_limits = kwargs.pop('ms_limits', True)
        #indices = kwargs.pop('indices',None)
        #plot_specs = {'marker': 'o', 'linestyle': 'None'}
        #plot_specs.update(kwargs)

        #data, names = self.data_from_topics_dims(*topics_dims, indices=indices)

        #for i in range(len(data)):
        #    if data[i].shape[1]==2:
        #        ax.plot(data[i][:,0],data[i][:,1], **plot_specs)
        #    elif data[i].shape[1]==3:
        #        ax.plot(data[i][:,0],data[i][:,1],data[i][:,2], **plot_specs)
        #    else:
        #        logger.warning("Scatterplot only works for 2D and 3D data.",
        #                       "Use NaoExplorationLog.scatter_matrix instead")
        pass

    def hexbin(self, ax, topic_dims, **kwargs):
        """Not functional"""
        #indices = kwargs.pop('indices',None)
        #plot_specs = {'gridsize':20,'cmap':None}
        #plot_specs.update(kwargs)

        #data, names = self.data_from_topics_dims(topic_dims, indices=indices)
        #if data[0].shape[1]==2:
        #    im = ax.hexbin(data[0][:,0],data[0][:,1],**plot_specs)
        #else:
        #    logger.warning('NaoBabbling.hexbin works only for 2D data')
        #return im
        pass

    def plot_learning_curve(self, fig=None, axes=None, mode='inverse',
                            **kwargs):
        """Plots a learning curve of the error over time.

        Args:
            fig (matplotlib.figure.Figure, optional): a figure in which to plot.
                If 'None', a new figure is created.
            axes (list[matplotlib.axes.AxesSubplot], optional): possibility to 
                specify the list of axes in which to plot.
            mode ({'forward','inverse'}, optional): specifies if the inverse or
                forward error should be plotted. Default 'inverse'.

        Keyword Args:
            errorbar (boolean): If True (default), errorbars are plotted.
            capstyle ({'x','o','^', ...})
            show_misses (boolean): should the amount of visually missed
                testcases be displayed.
            shift (int): shifts the errorbars around to increase readability
            figsize (tuple[int]): figsize
            color
        """
        if not hasattr(self, 'eval_at'):
            raise UserWarning('No evaluation available, '
                              'you need to specify the evaluate_at argument'
                              ' when constructing the experiment')

        if mode=='forward':
            error = np.array(self.forward_error)
            fig, axes = plot_learning_curve(self.eval_at, error,
                                            fig=fig, axes=axes, **kwargs)
        elif mode=='inverse':
            error = np.array(self.inverse_error)
            fig, axes = plot_learning_curve(self.eval_at, error,
                                            fig=fig, axes=axes, **kwargs)

        return fig, axes

    def scatter_matrix(self, *topics_dims, **kwargs):
        """Not functional"""
        #indices = kwargs.pop('indices',None)
        #plot_specs = {'marker': 'o', 'linestyle': 'None'}
        #plot_specs.update(kwargs)

        #data, names = self.data_from_topics_dims(*topics_dims, indices=indices )
        #plot_specs['names'] = names
        #scm = scatterplot_matrix(*data, **plot_specs)
        #return scm
        pass

    def hexbin_matrix(self, topic, dims, **kwargs):
        """Not functional"""
        #indices = kwargs.pop('indices',None)
        #plot_specs = {'gridsize':20,'cmap':None}
        #plot_specs.update(kwargs)

        #n = len(dims)
        #data = self.data[topic]
        #plot_specs['names'] = topic*n

        #hbm = hexbin_matrix(data, **plot_specs)
        #return hbm
        pass

    def show_development(self, topic, dims, intervals, **kwargs):
        """High-level hexbin plotting of the requested data over time.

        Args:
            topic (str): one of {'choices','inferences','movements',
                'sensory','experience'}
            dims (list[int]): the plottable dimensions of that topic.
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
        data = self.data[topic][:,dims]
        fig, axes, im = show_development(data, intervals, topic=topic, **kwargs)
        return fig, axes, im

    def to_file(self, fname):
        """Saves log to file.

        Args:
            fname (str): filename
        """
        idx = self.t * self._samples_between_goals
        config = dict(s_mins=list(self.conf.s_mins),
                      s_maxs=list(self.conf.s_maxs),
                      m_mins=list(self.conf.m_mins),
                      m_maxs=list(self.conf.m_maxs),
                      head_dims=list(self.conf.head_dims),
                      arm_dims=list(self.conf.arm_dims),
                      image_dims=list(self.conf.image_dims),
                      space_dims=list(self.conf.space_dims),
                      expl_dims=self.expl_dims,
                      inf_dims=self.inf_dims,
                      call=self.call)
        data = {'marker_detected':self.data['marker_detected'][:idx],
                'config':config,
                't':self.t,
                'choices':self.data['choices'][:self.t,:],
                'inferences':self.data['inferences'][:self.t,:],
                'experience':self.data['experience'][:idx,:],
                'movements':self.data['movements'][:self.t,:],
                'sensors':self.data['sensors'][:self.t,:],
                'forward_error':self.forward_error,
                'inverse_error':self.inverse_error,
                'evaluated_at':self.eval_at,
                'testcases':self.testcases}
        make_files(data, fname)

    @classmethod
    def from_file(cls, fname):
        """Classmethod to load log from file.

        Args:
            fname (str): filename to load from

        Returns:
            An instance of NaoExplorationLog."""
        # load the data saved by NaoBabbling.to_file
        data = from_files(fname)
        conf = make_nao_configuration(**data['config'])
        log = cls(conf, **data['config'])

        # load rest of the data into the new object
        log.data['marker_detected'] = data['marker_detected']
        log.data['choices'] = data['choices']
        log.data['inferences'] = data['inferences']
        log.data['sensors'] = data['sensors']
        log.data['experience'] = data['experience']
        log.data['movements'] = data['movements']
        log.forward_error = data['forward_error']
        log.inverse_error = data['inverse_error']
        log.eval_at = data['evaluated_at']
        log.testcases = data['testcases']
        log.t = data['choices'].shape[0]
        return log

        """
        # call NaoBabbling with the original call
        expe = NaoBabbling(**data['call'])
        expe.continued = True

        # load rest of the data into the new object
        expe.log.data['marker_detected'] = data['marker_detected']
        expe.log.data['choices'] = data['choices']
        expe.log.data['inferences'] = data['inferences']
        expe.log.data['sensors'] = data['sensors']
        expe.log.data['experience'] = data['experience']
        expe.log.data['movements'] = data['movements']
        try:
            # legacy issues
            expe.log.reaching_error = data['reaching_error']
            expe.log.focusing_error = data['focusing_error']
        except KeyError:
            expe.log.forward_error = data['forward_error']
            expe.log.inverse_error = data['inverse_error']
        expe.evaluated_at   = data['evaluated_at']
        expe.testcases      = data['testcases']
        expe.t              = data['choices'].shape[0]

        return expe
        """


if __name__=="__main__":
    pass

