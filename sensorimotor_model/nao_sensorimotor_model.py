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
sensorimotor model specifically tailored to Aldebaran Nao. The class subsumes
two sensorimotor models for reaching and gazing movements.

Example Use
-----------
from pynaoexploration.environment.nao_environment import NaoEnvironment
from pynaoexploration.sensorimotor_model.nao_sensorimotor_model import NaoSensorimotorModel

nao = NaoEnvironment.from_configuration(config_name='LArm4DoF_Head2DoF')

nao_sm_model = NaoSensorimotorModel(nao.conf,
                                    model_type_head='WNN',
                                    config_str_head='default',
                                    model_type_arm='WNN',
                                    config_str_arm='default')

"""
# generic imports
import numpy as np
# explauto imports
from explauto.exceptions import ExplautoBootstrapError
from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
from explauto.utils.config import make_configuration


class NaoSensorimotorModel(SensorimotorModel):
    """NaoSensorimotorModel is a class that inherits from Explauto's (https://github.com/flowersteam/explauto) 
    SensorimotorModel class. It combines two sub-sensorimotor models for 
    reaching and gazing movements.
        
    Attributes:
        conf (NaoConfiguration): a configuration class object
        sm_model_head (SensorimotorModel): an Explauto sm model
        sm_model_arm (SensorimotorModel): an Explauto sm model
        mode ({'explore', 'exploit'}): with or without exploratory noise?
        sigma_m (list[float]): list of exploratory noise amplitudes for
            sub-models
    """
    def __init__(self,  conf,
                        model_type_head='WNN',  config_str_head='default',
                        model_type_arm= 'WNN',  config_str_arm ='default'):
        """NaoSensorimotorModel init method.

        Args:
            conf (NaoConfiguration): a configuration class object
            model_type_head (str, optional)
            config_str_head (str, optional)
            model_type_arm (str, optional)
            config_str_arm (str, optional)
        """
        SensorimotorModel.__init__(self,conf)

        conf_sm_model_head = make_configuration(
                                self.conf.mins[self.conf.head_dims],
                                self.conf.maxs[self.conf.head_dims],
                                self.conf.mins[self.conf.image_dims
                                               + self.conf.arm_dims],
                                self.conf.maxs[self.conf.image_dims
                                               + self.conf.arm_dims])

        conf_sm_model_arm  = make_configuration(
                                self.conf.mins[self.conf.arm_dims],
                                self.conf.maxs[self.conf.arm_dims],
                                self.conf.mins[self.conf.space_dims],
                                self.conf.maxs[self.conf.space_dims])
        self.sm_model_head = \
            SensorimotorModel.from_configuration(conf_sm_model_head,
                                                 model_type_head,
                                                 config_str_head)
        self.sm_model_arm = \
            SensorimotorModel.from_configuration(conf_sm_model_arm,
                                                 model_type_arm,
                                                 config_str_arm)
        self.t = 0
        self.mode = 'explore'
        self.sm_model_head.mode = 'explore'
        self.sm_model_arm.mode = 'explore'
        self.sigma_m = [0.1,0.1]

    @classmethod
    def from_configuration(cls, conf, **config):
        """NaoSensorimotorModel classmethod instantiation method.

        Args:
            conf (NaoConfiguration): a configuration class object
            config (str): a config string
        """
        return NaoSensorimotorModel(conf, **config)

    @classmethod
    def available_configurations(cls):
        """Classmethod display possible configurations."""
        return configurations.keys()


    def infer(self, in_dims, out_dims, x):
        """Inference of input x in sensorimotor space.

        Args:
            in_dims (list[int]): list of input dimensions.
                If in_dims==self.conf.m_dims its a forward prediction.
                If in_dims==self.conf.s_dims its an inverse prediction
            out_dims (list[int]): list of output dimensions.
                If out_dims==self.conf.m_dims its an inverse prediction.
                If out_dims==self.conf.s_dims its a forward prediction 
            x (ndarray): Input vector on which inference should be executed

        Returns:
            The prediction corresponding to the input vector x in sensorimotor
            space.
            """
        ms = np.zeros(self.conf.ndims)
        head_dims = self.conf.head_dims
        arm_dims = self.conf.arm_dims
        space_dims = self.conf.space_dims
        image_dims = self.conf.image_dims

        # if arm dims are input and space dims are output
        # -----> forward prediction for arm model
        if (set(arm_dims).issubset(set(in_dims))
        and set(space_dims).issubset(set(out_dims))):

            # if head dims are input and image dims are output
            # -----> forward prediction for head model
            if (set(head_dims).issubset(set(in_dims))
            and set(image_dims).issubset(set(out_dims))):

                # get the relevant x indices first
                x_arm_dims = [in_dims.index(arm_dim) for arm_dim in arm_dims]
                x_head_dims =[in_dims.index(head_dim) for head_dim in head_dims]
                # insert x into ms
                ms[arm_dims] = x[x_arm_dims]
                ms[head_dims] = x[x_head_dims]
                # inference
                # forward prediction for mArm really doesn't make sense here
                # TODO, got enough but not perfect
                sImage_mArmNonsensePrediction = \
                    self.sm_model_head.forward_prediction(ms[head_dims]) 
                ms[image_dims] = sImage_mArmNonsensePrediction[:2]
                ms[space_dims] = \
                    self.sm_model_arm.forward_prediction(ms[arm_dims])
                return ms[out_dims]

            # if image dims are input and head dims are output
            # -----> inverse prediction for head model
            elif (set(image_dims).issubset(set(in_dims))
            and set(head_dims).issubset(set(out_dims))):

                # get the relevant x indices first
                x_arm_dims = [in_dims.index(arm_dim) for arm_dim in arm_dims]
                x_image_dims = [in_dims.index(im_dim) for im_dim in image_dims]
                # insert x into ms
                ms[arm_dims]   =  x[x_arm_dims]
                ms[image_dims] =  x[x_image_dims]
                # inference
                ms[space_dims] = \
                    self.sm_model_arm.forward_prediction(ms[arm_dims])
                ms[head_dims] = \
                    self.sm_model_head.inverse_prediction(ms[image_dims
                                                             + arm_dims])
                # exploratory noise
                if self.sm_model_head.mode == 'explore':
                    ms[head_dims] = self._exploratory_noise(ms[head_dims],
                                                            self.sigma_m[0])
                return ms[out_dims]

        # if space dims are input and arm dims are output
        # -----> inverse prediction for arm model
        elif (set(space_dims).issubset(set(in_dims))
        and   set(arm_dims).issubset(set(out_dims))):

            # if head dims are input and image dims are output
            # -----> forward prediction for head model
            if (set(head_dims).issubset(set(in_dims))
            and set(image_dims).issubset(set(out_dims))):

                # get the relevant x indices first
                x_head_dims =[in_dims.index(head_dim) for head_dim in head_dims]
                x_space_dims = [in_dims.index(sp_dim) for sp_dim in space_dims]
                # insert x into ms
                ms[head_dims]  =  x[x_head_dims]
                ms[space_dims] =  x[x_space_dims]
                # inference and exploratory noise
                ms[arm_dims] = \
                    self.sm_model_arm.inverse_prediction(ms[space_dims])
                if self.sm_model_arm.mode == 'explore':
                    ms[arm_dims] = self._exploratory_noise(ms[arm_dims],
                                                           self.sigma_m[1])
                ms[image_dims] = \
                    self.sm_model_head.forward_prediction(ms[head_dims])[:2]
                return ms[out_dims]

            # if image dims are input and head dims are output
            # -----> inverse prediction for head model
            elif (set(image_dims).issubset(set(in_dims))
            and set(head_dims).issubset(set(out_dims))):

                # get the relevant x indices first
                x_image_dims =[in_dims.index(im_dim) for im_dim in image_dims]
                x_space_dims = [in_dims.index(sp_dim) for sp_dim in space_dims]
                # insert x into ms
                ms[image_dims] =  x[x_image_dims]
                ms[space_dims] =  x[x_space_dims]
                # arm inference and exploratory noise
                ms[arm_dims] = \
                    self.sm_model_arm.inverse_prediction(ms[space_dims])
                if self.sm_model_arm.mode == 'explore':
                    ms[arm_dims] = self._exploratory_noise(ms[arm_dims],
                                                           self.sigma_m[1])
                # head inference and exploratory noise
                ms[head_dims] = \
                    self.sm_model_head.inverse_prediction(ms[image_dims
                                                             + arm_dims])
                if self.sm_model_head.mode == 'explore':
                    ms[head_dims] = self._exploratory_noise(ms[head_dims],
                                                            self.sigma_m[0])
                return ms[out_dims]

    def update(self, m, s):
        """ Update both sensorimotor models given a new (m, s) pair,
        where m is a motor command and s is the corresponding observed
        sensory effect.

        Args:
            m (ndarray): motor part
            s (ndarray): sensory part
        """
        if not s==[] and not np.any(np.isnan(s)) and len(s)>0:
            ms = np.zeros(self.conf.ndims)
            ms[self.conf.m_dims] = m
            ms[self.conf.s_dims] = s
            # update arm model with arm and space input
            self.sm_model_arm.update(ms[self.conf.arm_dims],
                                     ms[self.conf.space_dims])
            # update head model with head and image+arm input
            self.sm_model_head.update(ms[self.conf.head_dims],
                                      ms[self.conf.image_dims
                                         + self.conf.arm_dims])
            self.t += 1

    def forward_prediction(self, m):
        """ This method overrides the parent method because things are a little 
        more complicated here.
        """
        # TODO
        pass

    def inverse_prediction(self, s_g):
        """ This method overrides the parent method because things are a little 
        more complicated here.
        """
        # TODO
        pass

    def _exploratory_noise(self, m, sigma_m):
        """Adds exploratory noise to m with std sigma_m"""
        if sigma_m > 0.0:
            m += np.random.normal(scale=sigma_m, size=m.shape[0])
        return m

    def set_mode(self,mode):
        """Sets the exploration mode"""
        self.mode = mode
        self.sm_model_head.mode = mode
        self.sm_model_arm.mode = mode


configurations = {'knn_default':
                     {'model_type_head':'WNN','config_str_head':'default',
                      'model_type_arm': 'WNN',  'config_str_arm' :'default'},
                  'non_parametric_default':
                     {'model_type_head':'non_parametric',
                      'config_str_head':'default',
                      'model_type_arm': 'non_parametric',
                      'config_str_arm' :'default'},
                  'gmm_default':
                     {'model_type_head':'ilo_gmm',
                      'config_str_head':'default',
                      'model_type_arm': 'ilo_gmm',
                      'config_str_arm' :'default'}
                  }
sensorimotor_models = {'nao_sensorimotor_model':
                       (NaoSensorimotorModel, configurations)}

if __name__=="__main__":
    pass
