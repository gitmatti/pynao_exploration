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
NaoBabbling implementation
"""
# generic imports
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import cPickle
import logging
logger = logging.getLogger(__name__)
# explauto import
from explauto.exceptions import ExplautoBootstrapError, ExplautoEnvironmentUpdateError
from explauto.utils.density_image import density_image
from explauto.utils import rand_bounds, bounds_min_max
from explauto.sensorimotor_model import sensorimotor_models
from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
from explauto.interest_model import interest_models
from explauto.interest_model.interest_model import InterestModel
# relative package imports
from pynao_exploration.sensorimotor_model.nao_sensorimotor_model import NaoSensorimotorModel
from pynao_exploration.sensorimotor_model.nao_sensorimotor_model import configurations as sm_configs
from pynao_exploration.interest_model.nao_interest_model import NaoInterestModel
from pynao_exploration.interest_model.nao_interest_model import configurations as im_configs
from pynao_exploration.environment.nao_environment import NaoEnvironment
from pynao_exploration.log.exploration_log import NaoExplorationLog


# add specific nao models to the existing explauto model lists
sensorimotor_models['nao_sensorimotor_model'] = (NaoSensorimotorModel, sm_configs)
interest_models['nao_interest_model'] = (NaoInterestModel, im_configs)




class NaoBabbling(object):
    """ Central class of the exploration experiments. Allows to set up and 
    run goal and motor babbling experiments on the Nao robot involving head
    and arm movements.

    Refer to our paper 
    >>> Schmerling M., Schillaci G. and Hafner V.V.,
    Goal-Directed Learning of Hand-Eye Coordination on a Humanoid Robot in 
    Proceedings of the IEEE 5th International Conference on Development and 
    Learning and on Epigenetic Robotics, pp. 168-175 (2015) <<<
    for more details on the scientific background.

    This class uses the Explauto framework: https://github.com/flowersteam/explauto

    Attributes:
        model_type
        expl_type
        nao_environment
        conf
        expl_dims
        inf_dims
        sensorimotor_model
        interest_model
        n_bootstrap_sm, n_bootstrap_im
        n_bootstrap
        m_home
        current_choice
        current_inference
        current_movement
        current_sensory
        log
        connected
        t
        n_iterations
    """
    def __init__(self, model_type, expl_type,
                       sm_name, sm_config, 
                       im_name, im_config, 
                       env_config,
                       n_iterations = 10000, sigma_m = 0.1,
                       subsampling=True, time_between_goals=2.0,
                       samples_between_goals=20, **kwargs):
        """Initialize NaoBabbling class

        Args:
            model_type: {'concurrent', 'combined'} 
                Determines if separate internal models should 
                control head and arm movements.

            expl_type: str
                should be in {'goal', 'motor'} for model_type='combined' or in 
                {'goal-goal', 'motor-motor', 'goal-motor','goal-motor'} for 
                model_type='concurrent'. In the latter case, the first part 
                determines head exploration type and the second part determines
                arm exploration type.

            sm_name: str
                should be 'nao_sensorimotor_model' for model_type='concurrent'.
                Otherwise, choose from the available sm_names of the explauto
                framework. Execute NaoBabbling.available_sm_models() to return
                a list of options.         

            sm_config: str 
                could be 'knn_default' or 'non_parametric_default' for 
                model_type='concurrent'. Otherwise, choose from the available 
                sm_configs of the explauto framework.

            im_name: str
                should be 'nao_interest_model' for model_type='concurrent'.
                Otherwise, choose from the available im_names of the explauto
                framework. Execute NaoBabbling.available_im_models() to return
                a list of options.           

            im_config: str 
                could be 'discretized_progress', 'random', 'progress-random'
                or 'random-progress' for model_type='concurrent'.
                Otherwise, choose from the available im_configs of the explauto
                framework.

            env_config: {'LArm4DoF_Head2DoF', 'LArmAndHeadCombined_6DoF'} 

            n_iterations: int, optional
                Default 10000 iterations

            sigma_m: float, optional
                Standard deviation of exploratory noise in joint angles. 
                Default 0.1 rad.

            subsampling: bool, optional
                If to collect samples during movement execution.
                Default true.

            time_between_goals: float, optional
                Execution time of movements. Default 2s.

            samples_between_goals: int, optional
                How many samples to take during execution. Default 20.

        """
        # models, configurations and dimensions
        self.model_type = model_type
        self.expl_type = expl_type

        # environment
        self.nao_environment = NaoEnvironment.from_configuration(env_config)
        self.conf = self.nao_environment.conf

        # set up model and exploration type and calculate the relevant 
        # dimensions. expl_dims are the dimensions that exploration is focused
        # on and inf_dims are the remaining dimensions. I.e. for goal babbling,
        # expl_dims would be the sensory/cartesian space. For motor babbling,
        # expl_dims would be the motor space
        assert model_type in ['combined', 'concurrent']

        # learn two separate models for head and arm kinematics
        if model_type=='concurrent':
            assert expl_type in ['goal-goal','motor-motor',
                                 'goal-motor','motor-goal']
            if expl_type=='goal-goal':
                self.expl_dims = self.conf.image_dims + self.conf.space_dims
            elif expl_type=='motor-motor':
                self.expl_dims = self.conf.head_dims + self.conf.arm_dims
            elif expl_type=='goal-motor':
                self.expl_dims = self.conf.image_dims + self.conf.arm_dims
            elif expl_type=='motor-goal':
                self.expl_dims = self.conf.head_dims + self.conf.space_dims

        # learn a combined model for head and arm kinematics
        if model_type=='combined':
            assert expl_type in ['goal','motor']
            if expl_type=='goal':
                self.expl_dims = self.conf.s_dims
            elif expl_type=='motor':
                self.expl_dims = self.conf.m_dims

        self.inf_dims = sorted(list(set(self.conf.dims) - set(self.expl_dims)))

        # sensorimotor model 
        self.sensorimotor_model = \
            SensorimotorModel.from_configuration(self.conf, sm_name, sm_config)
        self.sensorimotor_model.sigma_m = sigma_m
        self.sigma_m = sigma_m

        # interest model
        self.interest_model = \
            InterestModel.from_configuration(self.conf, self.expl_dims,
                                             im_name, im_config)

        # bootstrapping
        # how many samples should be collected before inferences start
        self.n_bootstrap_sm = 25
        self.n_bootstrap_im = 25
        self._n_bootstrap = 0

        # subsampling yes or no?
        self.subsampling = subsampling
        self.time_between_goals = time_between_goals
        if self.subsampling:
            self.samples_between_goals = samples_between_goals
        else:
            self.samples_between_goals = 1
        self.sample_time = time_between_goals/samples_between_goals
        if not self.nao_environment==None:
            self.nao_environment.subsampling = self.subsampling
            self.nao_environment.time_between_goals = self.time_between_goals
            self.nao_environment.samples_between_goals = self.samples_between_goals
            self.nao_environment.sample_time = self.sample_time
            #self.sensor_names = self.nao_environment.sensor_names
            #self.motor_names  = self.nao_environment.joint_names
            #self.all_names    = self.motor_names + self.sensor_names

        # parameters for running the experiment
        self.n_iterations = n_iterations
        self.continued = False
        self.t = 0
        self.p_going_home = 0.05
        self.m_home = [np.pi/32.,np.pi/8.,0,np.pi/8.,-np.pi/8.,0]
        #self._annealing_schedule()
        
        # data structures
        self.current_choice = np.zeros(len(self.expl_dims))
        self.current_inference = np.zeros(len(self.inf_dims))
        self.current_movement = np.zeros(self.conf.m_ndims)
        self.current_sensory = np.zeros(self.conf.s_ndims)
        self.current_detection = np.zeros(self.samples_between_goals)

        # save the init call
        self._call = {'model_type':model_type, 'expl_type':expl_type,
                      'sm_name':sm_name, 'sm_config':sm_config, 
                      'im_name':im_name, 'im_config':im_config, 
                      'env_config':env_config,
                      'n_iterations':n_iterations, 'sigma_m':sigma_m,
                      'subsampling':subsampling,
                      'time_between_goals':time_between_goals,
                      'samples_between_goals':samples_between_goals}


        self.log = NaoExplorationLog(self.conf, self.expl_dims,
                                     self.inf_dims, self._call)
        self.default_testcases()

        self.connected = False
        try:
            assert self.nao_environment.connected
            self.connected = True
        except AssertionError:
            print "Nao does not seem to be connected."
            print "You may proceed, but functionality will be limited"

        #self.choices = np.zeros((n_iterations,len(self.expl_dims)))
        #self.inferences = np.zeros((n_iterations,len(self.inf_dims)))
        #self.movements = np.zeros((n_iterations,self.conf.m_ndims))
        #self.sensors = np.zeros((n_iterations,self.conf.s_ndims))
        #if self.subsampling:
        #    n_experience = n_iterations * samples_between_goals
        #else:
        #    n_experience = n_iterations
        #self.experience = np.zeros((n_experience,self.conf.ndims))
        #self.markerDetected = np.zeros(n_experience)

    @classmethod
    def from_configuration(cls, config):
        """Classmethod to instaniate experiment."""
        return NaoBabbling(**config)

    @classmethod
    def available_sm_models(cls):
        """Classmethod to return list of available sensorimotor models."""
        return sensorimotor_models.keys()

    @classmethod
    def available_im_models(cls):
        """Classmethod to return list of available interest models."""
        return interest_models.keys()


    ############################################################################
    ########################       algorithm       #############################
    ############################################################################
    def choose(self):
        """Returns a point chosen by the interest model.
        Explauto-like syntax.

        Returns:
            An ndarray of length len(self.expl_dims) sampled from the interest
            model.
        """
        go_random = False
        # check the amount of samples
        if self._n_bootstrap >= self.n_bootstrap_im:
            # catch ExplautoBootstrapError
            try:
                choice = self.interest_model.sample()
            except ExplautoBootstrapError:
                go_random = True
                logger.warning('Interest model not bootstrapped yet')
        else:
            go_random = True
            logger.warning('Interest model still missing samples.')
        if go_random:
            choice = rand_bounds(self.conf.bounds[:, self.expl_dims]).flatten()
        return choice


    def infer(self, expl_dims, inf_dims, choice):
        """Employs the sensorimotor model to infer the result of an input. 
        Explauto-like syntax.

        Args:
            expl_dims, inf_dims (list[int]): Together determine if an inverse
                or forward prediction is made.
            choice (ndarray): input to inference engine.

        Returns:
            The sensorimotor prediction of choice given expl_dims and inf_dims.
        """
        go_random = False
        if self._n_bootstrap >= self.n_bootstrap_sm:
            try:
                inference = self.sensorimotor_model.infer(expl_dims,
                                                          inf_dims,
                                                          choice.flatten())
            except ExplautoBootstrapError:
                logger.warning('Sensorimotor model not bootstrapped yet')
                go_random = True
        else:
            go_random = True
            logger.warning('Sensorimotor model still missing samples.')
        if go_random:
            inference = rand_bounds(self.conf.bounds[:, inf_dims]).flatten()
        return inference    


    def produce(self):
        """Combines goal sampling and sensorimotor inference. Determines the
        the next movement to be made by setting self.current_movement,
        self.current_choice, self.current_inference and self.current_sensory
        """
        if np.random.rand() > self.p_going_home:
            self.current_choice = self.choose()
            self.current_inference = self.infer(self.expl_dims,
                                                self.inf_dims,
                                                self.current_choice)
            ms = np.zeros(self.conf.ndims)
            ms[self.expl_dims] = self.current_choice
            ms[self.inf_dims] = self.current_inference
            self.current_movement = ms[self.conf.m_dims]
            self.current_sensory = ms[self.conf.s_dims] 
        else:
            self.current_movement = self.m_home

        self.log.data['choices'][self.t,:] = self.current_choice
        self.log.data['inferences'][self.t,:] = self.current_inference
        self.log.data['movements'][self.t,:] = self.current_movement
        self.log.data['sensors'][self.t,:] = self.current_sensory


    def perceive(self, experience):
        """Updates the sensorimotor and interest model with the collected 
        experience.

        Args:
            experience (ndarray): should be an (m,s) pair of length
                self.conf.m_ndims + self.conf.s_ndims"""
        self.current_experience = experience

        if experience.ndim==1:
            # in the simple case when self.subsampling is not active,
            # experience will contain only one sample
            # TODO
            pass

        elif experience.ndim==2:
            # if self.subsampling is active,
            # experience will contain multiple samples

            # filter out nans
            self.current_detection = np.invert(
                                        np.isnan(
                                            np.sum(self.current_experience,
                                                   axis=1)))
            self._n_bootstrap += np.sum(self.current_detection)
            # update sensorimotor model on all subsamples
            for i in range(self.current_experience.shape[0]-1):
                if self.current_detection[i]:
                    ms = self.current_experience[i,:]
                    self.sensorimotor_model.update(ms[self.conf.m_dims],
                                                   ms[self.conf.s_dims])
                    # TBD: should interest model be updated with subsamples?
                    #self.interest_model.update(ms, ms)
            # update both sensorimotor model and interest model 
            # on the end point sample
            if self.current_detection[-1]:
                ms = self.current_experience[-1,:]
                self.sensorimotor_model.update(ms[self.conf.m_dims],
                                               ms[self.conf.s_dims])
                self.interest_model.update(np.hstack((self.current_movement,
                                                      self.current_sensory)),
                                           ms)
            else:
                # TODO: there might be a way to update on invisible marker
                # positions too
                #ms = experience[-1,:]
                #ms[self.conf.s_dims] = -1.0 * self.current_sensory
                #self.interest_model.update(np.hstack((self.current_movement, self.current_sensory)), ms)
                pass

        # logging
        start = self.t*self.samples_between_goals
        end = (self.t+1)*self.samples_between_goals
        self.log.data['experience'][start:end,:] = self.current_experience
        self.log.data['marker_detected'][start:end] = self.current_detection

    def _run_with_subsampling(self):
        """Runs experiment when self.subsampling=True"""
        # go home first
        self.nao_environment.go2position(self.m_home)
        time.sleep(2)
        # start iterations
        while self.t < self.n_iterations:
            # print progress
            if self.t%1000==0: 
                print self.t
            # evaluate if requested
            if self.t in self.log.eval_at:
                error = self.evaluate(topic=('arm','forward'))
                self.log.forward_error.append(error)
                error = self.evaluate(topic=('arm','inverse'))
                self.log.inverse_error.append(error)

            # actual iteration
            self.produce()
            try:
                env_state = self.nao_environment.update(self.current_movement)
            except ExplautoEnvironmentUpdateError:
                logger.warning('Environment update error at time %d with '
                               'motor command %s. '
                               'This iteration wont be used to update models',
                               self.t, self.current_movement)
            self.perceive(env_state)
            #self._anneal_exploratory_noise()
            self.t += 1
            self.log.t = self.t

    def _run(self):
        """Not yet implemented"""
        pass

    def run(self):
        """Run the experiment."""
        if self.connected:
            if self.subsampling:
                self._run_with_subsampling()
            else:
                self._run()
            self.nao_environment.cleanup()

    # unused
    def _annealing_schedule(self, onset=3000, tau=0.999):
        """Unused. Could potentially be used to anneal exploratory noise."""
        self.tau = tau
        self.onset = onset

    # unused
    def _anneal_exploratory_noise(self):
        """Unused. Could potentially be used to anneal exploratory noise."""
        if self.t>self.onset:
            self.sigma_m[0] *= self.tau
            self.sigma_m[1] *= self.tau
            self.sensorimotor_model.sigma_m = self.sigma_m


    ############################################################################
    ##################          evaluation facilities  #########################
    ############################################################################
    def evaluate_at(self, eval_at, **kwargs):
        """Sets the evaluation times of the experiment."""
        self.log.evaluate_at(eval_at, **kwargs)

    def default_testcases(self, n_samples=-1):
        """Tells the experiment to use the default testcases."""
        self.log.default_testcases()

    def make_testcases(self, fname, n_samples=-1, **load_kwargs):
        """Allows to explicitly specifiy which testcases to use."""
        self.log.make_testcases(fname, n_samples=n_samples, **load_kwargs)

    def evaluate(self, topic=('arm','forward')):
        """Evaluate model performances"""
        if (not topic[0] in ['arm','head']
            or not topic[1] in ['forward','inverse']):
            logger.warning('Evaluation not (yet) possible for this topic/model')
            return []  
        if self.log.testcases==None:
            logger.warning('No testcases have been specified, cannot evaluate.'
                           'Use NaoBabbling.make_testcases.')
            return []      
            
        n_testcases = self.log.testcases.shape[0]
        error = np.zeros(n_testcases)

        self.sensorimotor_model.sigma_m = [0.0,0.0]

        for i in range(n_testcases):
            ms_true = self.log.testcases[i,:]
            ms = np.zeros_like(ms_true)
            if topic[1]=='forward':    
                ms[self.conf.m_dims] = ms_true[self.conf.m_dims]     
                ms[self.conf.s_dims] = self.sensorimotor_model.infer(
                                          self.conf.m_dims,
                                          self.conf.s_dims,
                                          ms_true[self.conf.m_dims])
            elif topic[1]=='inverse':
                ms[self.conf.m_dims] = self.sensorimotor_model.infer(
                                          self.conf.s_dims,
                                          self.conf.m_dims,
                                          ms_true[self.conf.s_dims])
                ms[self.conf.s_dims] = self.nao_environment.compute_sensori_effect(
                                          ms[self.conf.m_dims])[self.conf.s_dims]
            if topic[0]=='arm':
                error[i] = np.linalg.norm(ms[self.conf.space_dims]
                                          - ms_true[self.conf.space_dims])
            if topic[1]=='head':
                error[i] = np.linalg.norm(ms[self.conf.image_dims]
                                          - ms_true[self.conf.image_dims])

        self.sensorimotor_model.sigma_m = self.sigma_m

        return error

    ############################################################################
    ##########################   low-level plotting   ##########################
    ############################################################################
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
        self.log.plot_marker_detection(ax, width=width, **kwargs_plot)

    def scatter_plot(self, ax, *topics_dims, **kwargs):
        """Does a scatterplot of the requested data"""
        self.log.scatter_plot(ax, *topics_dims, **kwargs)

    def hexbin(self, ax, topic_dims, **kwargs):
        """Does a hexbin plot of the requested data"""
        self.log.hexbin(ax, topic_dims, **kwargs)

    ############################################################################
    ##########################   high-level plotting   #########################
    ############################################################################
    def plot_learning_curve(self, fig=None, axes=None,
                            mode='forward', **kwargs):
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
        fig, axes = self.log.plot_learning_curve(fig=fig, axes=axes,
                                                 mode=mode, **kwargs)
        return fig, axes

    def scatter_matrix(self, *topics_dims, **kwargs):
        """Plots a scatter matrix of the requested data."""
        scm = self.log.scatter_matrix(*topics_dims, **kwargs)
        return scm

    def hexbin_matrix(self, topic, dims, indices=None, **kwargs):
        """Plots a hexbin matrix of the requested data."""
        hbm = self.log.hexbin_matrix(topic, dims, indices=indices, **kwargs)
        return hbm

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
        fig, axes, im = self.log.show_development(topic, dims, intervals,
                                                  **kwargs)
        return fig, axes, im

    ############################################################################
    ##########################   loading and saving   ##########################
    ############################################################################
    def to_file(self, fname):
        """Saves experiment to file.

        Args:
            fname (str): filename
        """
        self.log.to_file(fname)

    @classmethod
    def from_file(cls, fname):
        """Classmethod to load experiment from file.

        Args:
            fname (str): filename to load from

        Returns:
            An instance of NaoBabbling."""
        log = NaoExplorationLog.from_file(fname)
        expe = cls(**log.call)
        expe.log = log
        return expe
    
    @classmethod
    def continue_from_file(cls, new_iter, eval_at, fname,
                           re_evaluate=True, testcases=None):
        """Initializes experiment from file with the option to continue via 
        NaoBabbling.run().

        Args:
            new_iter (int): new iteration maximum
            eval_at (list[int]): new evaluation times
            fname (str): file to load from via NaoBabbling.from_file()
        """
        expe = NaoExplorationLog.from_file(fname)
        if not testcases==None:
            expe.log.make_testcases(testcases, n_samples=-1)
        expe.continued = True
        previous_iterations = expe.choices.shape[0]
        assert new_iter >= previous_iterations
        
        expe.evaluate_at(eval_at, re_evaluate=re_evaluate)
        new_iterations = new_iter - previous_iterations
        expe.n_iterations = new_iter

        expe._retrain_models(re_evaluate=re_evaluate)
        if new_iterations > 0:
            expe.log['choices'] = np.row_stack(
                                      (expe.log['choices'],
                                       np.zeros((new_iterations,
                                                 len(expe.expl_dims)))))
            expe.log['inferences'] = np.row_stack(
                                      (expe.log['inferences'],
                                       np.zeros((new_iterations,
                                                 len(expe.inf_dims)))))
            expe.log['movements'] = np.row_stack(
                                      (expe.log['movements'],
                                       np.zeros((new_iterations,
                                                 expe.conf.m_ndims))))
            expe.log['sensors'] = np.row_stack(
                                      (expe.log['sensors'],
                                       np.zeros((new_iterations,
                                                 expe.conf.s_ndims))))
            if expe.subsampling:
                n_experience = new_iterations * expe.samples_between_goals
            else:
                n_experience = new_iterations
            expe.log['experience'] = np.row_stack(
                                        (expe.log['experience'],
                                         np.zeros((n_experience,
                                                   expe.conf.ndims))))
            expe.log['marker_detected'] = np.hstack((expe.log['marker_detected'],
                                                    np.zeros(n_experience)))
        return expe

    def _retrain_models(self, re_evaluate=False):
        if not self.subsampling:
            # TODO
            pass
        elif self.subsampling:
            for t in range(self.log.data['choices'].shape[0]):
                # optionally re-evaluate
                if re_evaluate and t in self.log.eval_at:
                    error = self.evaluate(topic=('arm','forward'))
                    self.log.forward_error.append(error)
                    error = self.evaluate(topic=('arm','inverse'))
                    self.log.inverse_error.append(error)
                # first update models based on the subsamples
                for i in range(self.samples_between_goals-1):
                    idx = t * self.samples_between_goals + i
                    if self.log.data['marker_detected'][idx]:
                        ms = self.log.data['experience'][idx,:]
                        self.sensorimotor_model.update(ms[self.conf.m_dims],
                                                       ms[self.conf.s_dims])
                        self.interest_model.update(ms, ms)
                # then update on the major iteration point
                idx = (t+1) * self.samples_between_goals - 1
                ms = self.log.data['experience'][idx,:]
                if self.log.data['marker_detected'][idx]:
                    self.sensorimotor_model.update(ms[self.conf.m_dims],
                                                   ms[self.conf.s_dims])
                    self.interest_model.update(
                        np.hstack((self.log.data['movements'][t,:],
                                   self.log.data['sensors'][t,:])),
                        ms)


if __name__=="__main__":
    pass

