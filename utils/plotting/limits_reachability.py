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
For plotting some heuristic reachability limits.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os

axes_limits = np.array([[0.0,0.25],[0.0,0.3],[-0.1,0.3]])

limits = []
this_path = os.path.dirname(os.path.realpath(__file__))


def reachability_limit(data_x, data_y, res_x, res_y, width_x, width_y, line, bounds):
    bins_x = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / res_x)
    if bins_x[-1] < bounds[1] - 0.00001:
        bins_x = np.append(bins_x, bounds[1])
    bins_y = np.arange(bounds[2], bounds[3], (bounds[3] - bounds[2]) / res_y)
    if bins_y[-1] < bounds[3] - 0.00001:
        bins_y = np.append(bins_y, bounds[3])
    H, x_grid, y_grid = np.histogram2d(data_x, data_y, (bins_x, bins_y), normed=True)
    H = np.where(H>0, 1, H)
    kde = ndi.gaussian_filter(H, [width_x, width_y], order=0)
    cn = plt.contour(bins_x[1:], bins_y[1:], kde.T)#, [line,1])
    path = cn.collections[line].get_paths()[0]
    vert = path.vertices
    return vert

try:
    for name in ['limit_xy','limit_xz', 'limit_yz']:
        limit = np.loadtxt(os.path.join(this_path, name+".dat"))
        limits.append(limit)
except IOError:
    knowledge_base = \
    np.loadtxt(os.path.join(this_path, "..", "..",
                            "data", "knowledge", "other",
                            "knowledge_base_33539samples_resampling.txt"))
    knowledge_base = knowledge_base[:,8:11]

    bounds = np.hstack((axes_limits[0,:],axes_limits[1,:]))
    limit_xy = reachability_limit(knowledge_base[:,0],knowledge_base[:,1],
                                  100, 100,
                                  3, 3, 1,
                                  bounds=bounds)

    bounds = np.hstack((axes_limits[0,:],axes_limits[2,:]))
    limit_xz = reachability_limit(knowledge_base[:,0],knowledge_base[:,2],
                                  100, 100,
                                  3, 3, 1,
                                  bounds=bounds)

    bounds = np.hstack((axes_limits[1,:],axes_limits[2,:]))
    limit_yz = reachability_limit(knowledge_base[:,1],knowledge_base[:,2],
                                  100, 100,
                                  3, 3, 1,
                                  bounds=bounds)

    plt.close('all')

    limits = [limit_xy, limit_xz, limit_yz]

    for name, limit in zip(['limit_xy','limit_xz', 'limit_yz'],limits):
        np.savetxt(os.path.join(this_path,name+".dat"),limit)


