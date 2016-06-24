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
Configuration module. Slightly modifies Explauto's (https://github.com/flowersteam/explauto)
configuration objects to also include attributes head_dims, arm_dims,
image_dims and space_dims.
"""
from numpy import hstack, vstack, array
from numpy.linalg import norm
from collections import namedtuple

NaoConfiguration = namedtuple('NaoConfiguration', ('m_mins', 'm_maxs',
                                                   's_mins', 's_maxs',
                                                   'mins', 'maxs',
                                                   'm_ndims', 's_ndims',
                                                   'ndims',
                                                   'm_dims', 's_dims',
                                                   'dims',
                                                   'head_dims', 'arm_dims',
                                                   'image_dims', 'space_dims',
                                                   'm_bounds', 's_bounds',
                                                   'bounds',
                                                   'm_centers', 's_centers',
                                                   'centers',
                                                   'm_ranges', 's_ranges',
                                                   'ranges'))


def make_nao_configuration(m_mins, m_maxs, s_mins, s_maxs,
                           head_dims, arm_dims, image_dims, space_dims,
                           **kwargs):
    """Make a configuration object. Slightly modifies Explauto's (https://github.com/flowersteam/explauto)
    configuration objects to also include attributes head_dims, arm_dims,
    image_dims and space_dims.

    Args:
        m_mins (list[float]): as in Explauto
        m_maxs (list[float]): as in Explauto
        s_mins (list[float]): as in Explauto
        s_maxs (list[float]): as in Explauto
        head_dims (list[int]): list of the head motor dimensions
        arm_dims (list[int]): list of the arm motor dimensions
        image_dims (list[int]): list of the image dimensions
        space_dims (list[int]): list of the cartesian space dimensions

    Returns:
        An instance of NaoConfiguration
    """
    mins = hstack((m_mins, s_mins))
    maxs = hstack((m_maxs, s_maxs))

    m_ndims = len(m_mins)
    s_ndims = len(s_mins)
    ndims = m_ndims + s_ndims

    m_dims = range(m_ndims)
    s_dims = range(m_ndims, ndims)
    dims = m_dims + s_dims

    m_bounds = vstack((m_mins, m_maxs))
    s_bounds = vstack((s_mins, s_maxs))
    bounds = hstack((m_bounds, s_bounds))

    m_ranges = array(m_maxs) - array(m_mins)
    s_ranges = array(s_maxs) - array(s_mins)
    ranges = hstack((m_ranges, s_ranges))

    m_centers = array(m_mins) + m_ranges / 2.
    s_centers = array(s_mins) + s_ranges / 2.
    centers = hstack((m_centers, s_centers))

    assert head_dims+arm_dims == m_dims
    assert image_dims+space_dims == s_dims

    return NaoConfiguration(m_mins, m_maxs, s_mins, s_maxs, mins, maxs,
                         m_ndims, s_ndims, ndims,
                         m_dims, s_dims, dims,
                         head_dims, arm_dims, image_dims, space_dims,
                         m_bounds, s_bounds, bounds,
                         m_centers, s_centers, centers,
                         m_ranges, s_ranges, ranges)


if __name__=="__main__":
    pass
