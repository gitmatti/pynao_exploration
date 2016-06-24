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
import numpy as np


all_sensory_dims_names = ['imageX','imageY','spaceX','spaceY','spaceZ']
all_sensory_dims_bounds = np.array([[-0.0,0.0],
                                    [-0.0,0.0],
                                    [0.0, 0.25],
                                    [0.00, 0.3],
                                    [-0.1, 0.3]])
all_joint_names = ['HeadPitch','HeadYaw',
                   'LShoulderRoll','LShoulderPitch',
                   'LElbowRoll','LElbowYaw',
                   'LWristYaw',
                   'RShoulderRoll','RShoulderPitch',
                   'RElbowRoll','RElbowYaw',
                   'RWristYaw']
lArm = ['LShoulderRoll','LShoulderPitch','LElbowRoll','LElbowYaw','LWristYaw']
head = ['HeadPitch','HeadYaw']

all_joint_bounds_deg = np.array([[-38.5, 29.5], [-90.0, 90.0], 
                                 [-18, 76], [-119.5, 119.5], 
                                 [-88.5, -2], [-119.5, 119.5], 
                                 [-104.5, 104.5],
                                 [-76, 18], [-119.5, 119.5], 
                                 [2, 88.5], [-119.5, 119.5], 
                                 [-104.5, 104.5]])
all_joint_bounds_rad = np.deg2rad(all_joint_bounds_deg)

def make_bounds_from_names(joint_names):
    """Returns m_mins and m_maxs for a list of joint_names"""
    idx_joints = [all_joint_names.index(item) for item in joint_names]
    m_mins = all_joint_bounds_rad[idx_joints,0]
    m_maxs = all_joint_bounds_rad[idx_joints,1]
    return m_mins, m_maxs

default_camera_parameters = {'width':320, 'height':240, 'followMarker':False}

# configuration 1
LArm3DoF_Head2DoF = {}
LArm3DoF_Head2DoF['joint_names'] = ['HeadPitch','HeadYaw',
                                    'LShoulderRoll','LShoulderPitch',
                                    'LElbowRoll']
LArm3DoF_Head2DoF['sensor_names'] = ['imageX','imageY',
                                     'spaceX','spaceY','spaceZ']
m_mins, m_maxs = make_bounds_from_names(LArm3DoF_Head2DoF['joint_names'])
LArm3DoF_Head2DoF['m_mins'] = m_mins
LArm3DoF_Head2DoF['m_maxs'] = m_maxs
LArm3DoF_Head2DoF['s_mins'] = np.array([-0.0, -0.0, 0.00,  0.0, -0.1,])
LArm3DoF_Head2DoF['s_maxs'] = np.array([ 0.0,  0.0, 0.25 , 0.3,  0.3 ])
LArm3DoF_Head2DoF['camera_params'] = default_camera_parameters
LArm3DoF_Head2DoF['dims'] = {'head_dims':[0,1],
                             'arm_dims':[2,3,4],
                             'image_dims':[5,6],
                             'space_dims':[7,8,9]}

# configuration 2
LArm4DoF_Head2DoF = {}
LArm4DoF_Head2DoF['joint_names'] = ['HeadPitch','HeadYaw',
                                    'LShoulderRoll','LShoulderPitch',
                                    'LElbowRoll', 'LElbowYaw']
LArm4DoF_Head2DoF['sensor_names'] = ['imageX','imageY',
                                     'spaceX','spaceY','spaceZ']
m_mins, m_maxs = make_bounds_from_names(LArm4DoF_Head2DoF['joint_names'])
LArm4DoF_Head2DoF['m_mins'] = m_mins
LArm4DoF_Head2DoF['m_maxs'] = m_maxs
LArm4DoF_Head2DoF['s_mins'] = np.array([-0.0, -0.0, 0.00,  0.0, -0.1,])
LArm4DoF_Head2DoF['s_maxs'] = np.array([ 0.0,  0.0, 0.25 , 0.3,  0.3 ])
LArm4DoF_Head2DoF['camera_params'] = default_camera_parameters
LArm4DoF_Head2DoF['dims'] = {'head_dims':[0,1],
                             'arm_dims':[2,3,4,5],
                             'image_dims':[6,7],
                             'space_dims':[8,9,10]}

# configuration 3
LArmAndHeadCombined_6DoF= {}
LArmAndHeadCombined_6DoF['joint_names'] = ['HeadPitch','HeadYaw',
                                           'LShoulderRoll','LShoulderPitch',
                                           'LElbowRoll','LElbowYaw']
m_mins, m_maxs = make_bounds_from_names(LArmAndHeadCombined_6DoF['joint_names'])
LArmAndHeadCombined_6DoF['sensor_names'] = ['spaceX','spaceY','spaceZ']
LArmAndHeadCombined_6DoF['m_mins'] = m_mins
LArmAndHeadCombined_6DoF['m_maxs'] = m_maxs
LArmAndHeadCombined_6DoF['s_maxs'] = np.array([0.0, 0.0, -0.1])
LArmAndHeadCombined_6DoF['s_mins'] = np.array([0.25, 0.3, 0.3])
LArmAndHeadCombined_6DoF['camera_params'] = default_camera_parameters
LArmAndHeadCombined_6DoF['dims'] = {'head_dims':[0,1],
                                    'arm_dims':[2,3,4,5],
                                    'image_dims':[],
                                    'space_dims':[6,7,8]}

configurations = {'LArm3DoF_Head2DoF': LArm3DoF_Head2DoF,
                  'LArm4DoF_Head2DoF':LArm4DoF_Head2DoF,
                  'LArmAndHeadCombined_6DoF':LArmAndHeadCombined_6DoF}












