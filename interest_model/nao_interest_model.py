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
interest model specifically tailored to Aldebaran Nao. The class subsumes two
interest models for reaching and gazing movements.

Example Use
-----------
from pynaoexploration.environment.nao import NaoEnvironment
from pynaoexploration.interest_model.nao_interest_model import NaoInterestModel

nao = NaoEnvironment.from_configuration(config_name='LArm4DoF_Head2DoF')

nao_in_model = NaoInterestModel(nao.conf, nao.conf.s_dims,
                                model_type_head='discretized_progress',
                                config_str_head='default',
                                model_type_arm ='discretized_progress',
                                config_str_arm ='default')

from explauto import InterestModel

nao_in_model = InterestModel.from_configuration(nao.conf, nao.conf.s_dims,
                                                'nao_interest_model',
                                                'discretized_progress')

nao_in_model.sample()
"""
# generic imports
import sys
import numpy as np
# explauto import
from explauto.utils import discrete_random_draw
from explauto.utils.config import make_configuration
from explauto.interest_model.interest_model import InterestModel


class NaoInterestModel(InterestModel):
    """NaoInterestModel is a class that inherits from Explauto's (TODO) 
    InterestModel class. It combines two sub-interest models for reaching
    and gazing movements.
        
    Attributes:
        conf (NaoConfiguration): a configuration class object
        expl_dims (list[int]): a list of the dimensions to be explored
        in_model_head (InterestModel): an Explauto interest model
        in_model_arm (InterestModel): an Explauto interest model
    """
    def __init__(self, conf, expl_dims,
                 model_type_head='discretized_progress',
                 config_str_head='default',
                 model_type_arm= 'discretized_progress',
                 config_str_arm ='default'):
        """NaoInterestModel init method.

        Args:
            conf (NaoConfiguration): a configuration class object
            expl_dims (list[int]): dimensions to explored
            model_type_head (str, optional)
            config_str_head (str, optional)
            model_type_arm (str, optional)
            config_str_arm (str, optional)
        """
        InterestModel.__init__(self, expl_dims)
        self.conf = conf

        conf_in_model_head = make_configuration(conf.mins[self.conf.head_dims],
                                                conf.maxs[self.conf.head_dims],
                                                conf.mins[self.conf.image_dims],
                                                conf.maxs[self.conf.image_dims])

        conf_in_model_arm  = make_configuration(conf.mins[self.conf.arm_dims],
                                                conf.maxs[self.conf.arm_dims],
                                                conf.mins[self.conf.space_dims],
                                                conf.maxs[self.conf.space_dims])

        # the next section organizes the expl_dims and inf_dims for the two
        # sub interest models. Any combination of goal and motor babbling is
        # possible

        # are image_dims part of expl_dims?
        # --> goal configuration for head
        if set(self.conf.image_dims).issubset(set(self.expl_dims)): 
            # explore sensory dimensions for head
            expl_dims_head = conf_in_model_head.s_dims

            # are space_dims part of expl_dims?
            # --> goal configuration for arm as well
            if set(self.conf.space_dims).issubset(set(self.expl_dims)): 
                # explore sensory dimensions for arm
                expl_dims_arm = conf_in_model_arm.s_dims

            # are arm_dims part of the exploration? 
            # --> motor configuration for arm           
            elif set(self.conf.arm_dims).issubset(set(self.expl_dims)):
                # explore motor dimensions for arm
                expl_dims_arm = conf_in_model_arm.m_dims

        # are head_dims part of the exploration?
        # --> motor configuration for head
        elif set(self.conf.head_dims).issubset(set(self.expl_dims)):
            # explore motor dimensions for head
            expl_dims_head = conf_in_model_head.m_dims

            # are space_dims part of the exploration?
            # --> goal configuration for arm
            if set(self.conf.space_dims).issubset(set(self.expl_dims)):
                # explore sensory dimensions for arm
                expl_dims_arm = conf_in_model_arm.s_dims

            # are arm_dims part of the exploration?
            # --> motor configuration for arm
            elif set(self.conf.arm_dims).issubset(set(self.expl_dims)):
                # explore motor dimensions for arm
                expl_dims_arm = conf_in_model_arm.m_dims

        # instaniate the sub-models
        self.in_model_head = InterestModel.from_configuration(conf_in_model_head,
                                                              expl_dims_head,
                                                              model_type_head,
                                                              config_str_head)
        self.in_model_arm = InterestModel.from_configuration(conf_in_model_arm,
                                                             expl_dims_arm,
                                                             model_type_arm,
                                                             config_str_arm)
        self.t = 0
        #TODO actually needs to be computed from the models or in agent somehow?
        #self.n_bootstrap = 0 

    @classmethod
    def from_configuration(cls, conf, config):
        """NaoInterestModel classmethod instantiation method.

        Args:
            conf (NaoConfiguration): a configuration class object
            config (str): a config string
        """
        return NaoInterestModel(conf, **config)

    @classmethod
    def available_configurations(cls):
        """Classmethod display possible configurations."""
        return configurations.keys()

    def sample(self):
        """Sample a target.

        Returns:
            An array of length len(self.expl_dims).
        """
        xHead = self.in_model_head.sample()
        xArm  = self.in_model_arm.sample()
        x = np.hstack((xHead,xArm))
        return x

    def update(self, xy, ms):
        """Update the model with collected samples.

        Args:
            xy (ndarray): (target,prediction)
            ms (ndarray): (motors,sensors)
        """
        # organize the updates. arm model is updated on arm_dims and space_dims
        # while head model is updated on head_dims and image_dims
        if not np.any(np.isnan(ms)) and len(ms)==self.conf.ndims:
            self.in_model_arm.update(xy[self.conf.arm_dims
                                        + self.conf.space_dims],
                                     ms[self.conf.arm_dims
                                        + self.conf.space_dims])
            self.in_model_head.update(xy[self.conf.head_dims
                                         + self.conf.image_dims],
                                      ms[self.conf.head_dims
                                         + self.conf.image_dims])
            self.t += 1        


configurations = {'discretized_progress':\
                    {'model_type_head':'discretized_progress',
                     'config_str_head':'default',
                     'model_type_arm':'discretized_progress',
                     'config_str_arm':'default'},
                  'random':\
                    {'model_type_head':'random',
                     'config_str_head':'default',
                     'model_type_arm':'random',
                     'config_str_arm':'default'},
                  'progress-random':\
                    {'model_type_head':'discretized_progress',
                     'config_str_head':'default',
                     'model_type_arm':'random',
                     'config_str_arm':'default'},
                  'random-progress':\
                    {'model_type_head':'random',
                     'config_str_head':'default',
                     'model_type_arm':'discretized_progress',
                     'config_str_arm':'default'}
                  }

        
interest_models = {'nao_interest_model': (NaoInterestModel,configurations)}

if __name__=="__main__":
    pass
