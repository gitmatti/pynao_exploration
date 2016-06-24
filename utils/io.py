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
Some IO functions.
"""
import cPickle
import json
import os
import numpy as np


def save_obj(obj, fname):
    if not fname[-4:] == '.pkl':
        fname += '.pkl'
    with open(fname, 'wb') as f:
        cPickle.dump(obj, f, 0)#cPickle.HIGHEST_PROTOCOL)

def load_obj(fname):
    if not fname[-4:] == '.pkl':
        fname += '.pkl'
    with open(fname, 'r') as f:
        return cPickle.load(f)


def make_files(data, directory):
    """Writes experiment data to disk.

    Args:
        data (dict): a data object as maintained by class NaoExplorationLog
        directory (str): a directory
    """
    if not directory[-1]=="/":
        directory += "/"
    with open(directory +"config.json", "w") as config:
        json.dump(data["config"], config)
    np.savetxt(directory + "marker_detected.dat", data["marker_detected"])
    np.savetxt(directory +"evaluated_at.dat", data["evaluated_at"])
    for i, e_at in enumerate(data["evaluated_at"]):
        np.savetxt(directory
                   + "forward_error_at_"
                   + str(int(e_at))
                   + ".dat",
                   data["forward_error"][i])
        np.savetxt(directory
                   + "inverse_error_at_"
                   + str(int(e_at))
                   + ".dat",
                   data["inverse_error"][i])
    np.savetxt(directory +"testcases.dat", data["testcases"])
    np.savetxt(directory +"experience.dat", data["experience"])
    np.savetxt(directory +"sensors.dat", data["sensors"])
    np.savetxt(directory +"inferences.dat", data["inferences"])
    np.savetxt(directory +"movements.dat", data["movements"])
    np.savetxt(directory +"choices.dat", data["choices"])


def from_files(directory):
    """Load experiment data from disk that was saved by make_files.

    Args:
        directory (str): a directory to load from
    """
    data = dict()
    if not directory[-1]=="/":
        directory += "/"
    with open(directory +"config.json", "r") as config:
        data["config"] = json.load(config)
    data["marker_detected"] = np.loadtxt(directory + "marker_detected.dat")
    data["evaluated_at"] = np.loadtxt(directory +"evaluated_at.dat" )
    data["forward_error"] = []
    data["inverse_error"] = []
    for i, e_at in enumerate(data["evaluated_at"]):
        data["forward_error"].append(np.loadtxt(directory
                                                + "forward_error_at_"
                                                + str(int(e_at))
                                                + ".dat"))
        data["inverse_error"].append(np.loadtxt(directory
                                                +"inverse_error_at_"
                                                + str(int(e_at))
                                                + ".dat"))
    data["testcases"] = np.loadtxt(directory +"testcases.dat")
    data["experience"] = np.loadtxt(directory +"experience.dat")
    data["sensors"] = np.loadtxt(directory +"sensors.dat")
    data["inferences"] = np.loadtxt(directory +"inferences.dat")
    data["movements"] = np.loadtxt(directory +"movements.dat")
    data["choices"] = np.loadtxt(directory +"choices.dat")
    return data



