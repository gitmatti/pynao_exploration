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
This module provides an Explauto-compatible (https://github.com/flowersteam/explauto)
Nao environment by implementing the nvironment class' main abstract functions 
compute_sensori_effect() and compute_motor_command().

Example Use
-----------
from pynaoexploration.environment.nao import NaoEnvironment

nao = NaoEnvironment.from_configuration(config_name='LArm4DoF_Head2DoF')
nao.compute_sensori_effect([-0.25,0.45,0.15,0.1,0.0,0.0) # in radians
nao.goHome()
"""
import sys
import time
import numpy as np
# explauto import
from explauto.environment.environment import Environment
from explauto.utils import bounds_min_max
# naoqi import
from naoqi import ALProxy
# relative package import
from pynao_exploration.utils.config import make_nao_configuration
from . import configurations

class NaoEnvironment(Environment):
    """NaoEnvironment is a class that inherits from Explauto's (TODO) 
    Environment class and extends the functionality to Nao by using NaoQi 
    and allows subsampling of observations and concurrent head and arm control.
        
    Attributes:
        conf (NaoConfiguration): a configuration class object
        connected (boolean): is there an connection to NaoQi and its necessary
            modules, i.e. ALMotion and NaoDetection. Limited functionality
            if False.
        joint_names (list[str]): the list of controllable joints
        sensor_names (list[str]): the list of relevant sensory dimensions
        home_position (dict): specifying the joint home position
        subsampling (boolean): should sensors be polled in between movements
        time_between_goals (float): movement interpolation time
        samples_between_goals (float): TODO
        sample_time (float): TODO
    """
    use_process = False
    def __init__(self, joint_names, sensor_names, camera_params,
                 dims, m_mins, m_maxs, s_mins, s_maxs,
                 home_position={'HeadPitch':-0.25,
                                'HeadYaw':0.45,
                                'LShoulderRoll':0.0,
                                'LShoulderPitch':0.0,
                                'LElbowRoll':0.0},
                 IP="127.0.0.1", PORT=9559):
        """NaoEnvironment init method.

        Args:
            joint_names (list[str]): List of joint to be used
            sensor_names (list[str]): List of sensory dimensions to be used
            camera_params (dict): camera parameters, width and height
            dims (dict): dictionary specifying head_dims, arm_dims, space_dims
              and image dims
            m_mins, m_maxs, s_mins, s_maxs (ndarrays): as in Explauto
            home_position (dict, optional): dictionary specifying a home postion
              for joints
            IP (str, optional): to connect to Nao, default local ip "127.0.0.1"
            PORT (int, optional): to connect to Nao, default 9559
        """
        # metaclass init
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        # adapt configuration
        self.conf = make_nao_configuration(m_mins, m_maxs,
                                           s_mins, s_maxs, **dims)

        # camera setup
        self._image_width = camera_params['width']
        self._image_height = camera_params['height']
        self._image_center = np.array([self._image_width/2,self._image_height/2])

        # joint setup
        self.joint_names = joint_names
        self.sensor_names = sensor_names
        #self.joint_stiffness_lists = [1.0] * len(self.joint_names)
        #self.joint_time_lists = [1.0] * len(self.joint_names)
        self.home_position = home_position

        # proxy setup
        self.connected = False   
        try:
            self._motion_proxy = ALProxy("ALMotion", IP, PORT)
        except Exception, e:
            print "Could not create proxy to ALMotion"
            print "Error was: ", e

        try:
            self._detection_proxy = ALProxy("NaoDetection", IP, PORT)
            self._detection_proxy.init(self.joint_names, self.sensor_names,
                                       self._image_width, self._image_height)
            self._detection_proxy.post.run()
        except Exception, e:
            print "Could not create proxy to NaoDetection"
            print "Error was: ", e

        if hasattr(self, '_motion_proxy') and hasattr(self, '_detection_proxy'):
            self.connected = True

        self.data = None

        # subsampling
        self.subsampling = False
        self.time_between_goals = None
        self.samples_between_goals = None
        self.sample_time = None

    @classmethod
    def from_configuration(cls, config_name='LArm4DoF_Head2DoF'):
        """ As Explauto: Environment factory from configuration strings.

        Args:
            config_name (str): the configuration string

        """
        return NaoEnvironment(**configurations[config_name])

    def _stiffness_on(self, head=False):
        """Turns on stiffnesses for joints"""
        self._motion_proxy.setStiffnesses(self.joint_names, 1.0)
        if head:
            self._motion_proxy.setStiffnesses('Head', 1.0)
                
    def go2position(self, m_env, wait=True):
        """Move joints to position.

        Args:
            m_env (list, ndarray): position to go to
            wait (boolean, optional): wait for movement to finish? default true
        """
        self._stiffness_on()
        if wait==True:
            self._motion_proxy.post.angleInterpolation(self.joint_names,
                                                       list(m_env),
                                                       1.5, True) 
        else:
            self._motion_proxy.setAngles(self.joint_names,
                                         list(m_env), 0.5)

    def go_home(self, move_speed=0.5):
        """Sends joints to home position"""
        self._motion_proxy.setStiffnesses(self.home_position.keys(), 1.0)
        idx = self._motion_proxy.post.setAngles(self.home_position.keys(), 
                                                self.home_position.values(),
                                                move_speed)
        time.sleep(0.5) 

    def one_update(self, m_ag, log=True):
        m_env = self.compute_motor_command(m_ag)
        if not self.subsampling:
            s = self.compute_sensori_effect(m_env)
            m_env = np.array(self._motion_proxy.getAngles(self.joint_names,
                                                          True))

            if log:
                self.emit('motor', m_env)
                self.emit('sensori', s)
            return np.hstack((m_env, s))
        else:
            out = self.compute_sensori_effect(m_env,
                                              subsampling=self.subsampling)
            if log:
                self.emit('motor', m_env)
                self.emit('sensori', out[-1,self.conf.s_dims])
            return out

    def cleanup(self):
        self._detection_proxy.cleanup()

    def compute_motor_command(self, m_ag):
        """As Explauto: Compute the motor command by restricting it
        to the bounds."""
        m_env = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        return m_env

    def compute_sensori_effect(self, m_env, subsampling=False):
        """Execute a motor command and retrieve sensory information.

        Args:
            m_env (list, ndarray): position to go to
        """

        # this is the case if using the explauto library
        # (agent, experiment, etc.)
        if not subsampling:
            self.go2position(m_env, wait=True)
            self.data = self._detection_proxy.get_data()
            ms = np.array(self.data[1:])
            if self.data[0]==1 or len(ms)==self.conf.ndims: # marker detected?
                return ms
            else: # if not, only angles are provided
                out = np.zeros(self.conf.ndims)*np.nan
                out[self.conf.m_dims] = ms
                return out

        # this can be the case when using class NaoBabbling
        elif subsampling:
            out = np.zeros((self.samples_between_goals, self.conf.ndims))
            self._stiffness_on()
            self._motion_proxy.post.angleInterpolation(self.joint_names,
                                                       list(m_env),
                                                       self.time_between_goals,
                                                       True)
            for i in range(self.samples_between_goals):
                time.sleep(self.sample_time)
                self.data = self._detection_proxy.get_data()
                ms = np.array(self.data[1:])
                # marker detected?
                if self.data[0]==1 or len(ms)==self.conf.ndims:
                    out[i,:] = ms
                # if not only angles are provided
                else:
                    out[i,:] = np.nan
                    out[i,self.conf.m_dims] = ms
                # some dirty hacking...
                if i==self.samples_between_goals-2:
                    time.sleep(self.sample_time*2)
            return out


if __name__=="__main__":
    pass

