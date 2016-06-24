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
"""PyNaoDetection is a python module based on naoqi 1.14.5 that provides marker 
detection with the ArUco library 1.2.4. 

Please find the ArUco and naoqi project at:
naoqi:
http://doc.aldebaran.com/1-14/dev/naoqi/index.html
ArUco:
http://www.uco.es/investiga/grupos/ava/node/26
https://sourceforge.net/projects/aruco/files/

Find my python wrapper for ArUco at:
TODO

Example Use
-----------
from naoqi import ALProxy

detectionProxy = ALProxy("NaoDetection", "127.0.0.1", 9559)
detectionProxy.init(['HeadPitch', 'HeadYaw'],
                    ['imageX', 'imageY', 'spaceX', 'spaceY', 'spaceZ'],
                    320, 240)
detectionID = detectionProxy.post.run()

data = detectionProxy.get_data()

marker_detected = data[0]
m = data[1:4]
s = data[4:7]
"""

# generic imports
import sys
import time
import copy
import numpy as np
from optparse import OptionParser
# third party
import cv2
# application specific
import pyAruco
from naoqi import ALProxy
from naoqi import ALModule
from naoqi import ALBroker
import vision_definitions

# NAO_IP
NAO_IP = "127.0.0.1"

# Global variable to store the MarkerTracking module instance
NaoDetection = None

class PyNaoDetection(ALModule):
    """PyNaoDetection is an ALModule written in python that provides marker
    detection with the AruCo library. 
    http://www.uco.es/investiga/grupos/ava/node/26
    http://doc.aldebaran.com/1-14/dev/naoqi/index.html
        
    Attributes:
        detector (pyAruco.detector): instance of python-wrapped marker detector
        marker_detected (boolean): True if a marker has been detected
        absolute_position (list[float]): absolute 3d-position of marker
        image_position (list[float]): relative 2d image position
        image (ndarray): last fetched image
        data (list[float]): last recorded data
    """
    def __init__(self, name):
        """PyNaoDetection init method.

        Args:
            name: Must match the name of the global module instance, 
            i.e. "NaoDetection"
        """
        # metaclass init
        ALModule.__init__(self, name)

        # Proxy Setup
        try:
            self._cam_proxy = ALProxy("ALVideoDevice")# IP, PORT)
        except Exception, e:
            print "Could not create proxy to ALVideoDevice"
            print "Error was: ", e
        
        try:
            self._motion_proxy = ALProxy("ALMotion")#, IP, PORT)
        except Exception, e:
            print "Could not create proxy to ALMotion"
            print "Error was: ", e
        self._requested_to_stop = False
        self.requested_to_exit = False
        self._default_width = 320
        self._default_height = 240
        self._default_joint_names = ['HeadPitch', 'HeadYaw', 'LShoulderRoll',
                                    'LShoulderPitch', 'LElbowRoll', 'LElbowYaw']
        self._default_sensor_names = ['imageX', 'imageY',
                                     'spaceX', 'spaceY', 'spaceZ']
        self.init(self._default_joint_names, self._default_sensor_names,
                  self._default_width, self._default_height)


    def init(self, joint_names, sensor_names, width, height):
        """ Callable init method.

        Args:
            joint_names (list[str]): list of names of tracked Nao joints
            sensor_names (list[str]): list of names of tracked marker dimensions
            width (int): camera width in pixels
            height (int): camera height in pixels

        Examples:
        ---------
        detection_proxy = ALProxy("NaoDetection", IP, PORT)
        detection_proxy.init(['HeadPitch','HeadYaw'],
                             ['imageX','imageY'],320,240)
        """
        self._joint_names = joint_names
        self._sensor_names = sensor_names
        self._move_speed = 0.05

        # camera setup
        self._image_width = width
        self._image_height = height
        self._image_half_width = width/2
        self._image_half_height = height/2 
        self._max_distance_to_center = np.sqrt(self._image_half_width**2
                                               + self._image_half_height**2)
        self._return_relative_position = True

        self._roboconfig = self._motion_proxy.getRobotConfig()
        if self._roboconfig[1][1]=='VERSION_40':
            self._opening_angle_diagonal = 72.6
            #self.hfov = np.rad2deg(1.064)
        elif self._roboconfig[1][1]=='VERSION_33':
            self._opening_angle_diagonal = 58
            #self.hfov = np.rad2deg(0.8342)

        self.kOriginalName = "SimulatorCam"
        self.kBottomCamera = 1
        self.kTopCamera = 0
        self.resolution = vision_definitions.kQVGA
        self.color_space = vision_definitions.kBGRColorSpace
        self.fps = 30

        self._video_client = self._cam_proxy.subscribeCamera(self.kOriginalName,
                                                             self.kBottomCamera,
                                                             self.resolution,
                                                             self.color_space,
                                                             self.fps)
        self._focal_length = self.compute_focal_length()
        self._optical_center = np.array([self._image_width/2,
                                         self._image_height/2])

        self._camera_matrix = self._motion_proxy.getTransform("CameraBottom",
                                                             0, True)

        # ArUco setup
        self.detector = pyAruco.PyDetector()
        self.marker_detected = False
        #self.markerTranslation = np.array([0.075, -0.033, -0.005])
        #self.markerJoint = "LElbowRoll"
        #self.analyticalPosition = []
        self.absolute_position = []
        self.image_position = []
        self.data = None
        self.image = None

        params_cam_matrix= np.zeros((3,3));
        params_cam_matrix[0,0]= self.compute_focal_length()
        params_cam_matrix[1,0]=0
        params_cam_matrix[2,0]=0
        params_cam_matrix[0,1]=0
        params_cam_matrix[1,1]= self.compute_focal_length()
        params_cam_matrix[2,1]=0
        params_cam_matrix[0,2]=self._optical_center[0]
        params_cam_matrix[1,2]=self._optical_center[1]
        params_cam_matrix[2,2]=1

        # no distorsion in the simulator
        params_distortion= np.zeros((4,1))

        self.detector.setCamParam(params_cam_matrix, params_distortion, 
                                  np.array([0, 0]))

    def stop(self, idx):
        """Stop method. Call to stop PyNaoDetection.run()"""
        self._requested_to_stop = True

    def cleanup(self):
        """Unsubscribes camera"""
        self._cam_proxy.unsubscribe(self._video_client)

    def visual(self):
        """ Visualizes the image seen by the module and a white cross where
        the marker center is assumed to be.
        """
        if not self.image==None:
            if not self.image_position==[]:
                center = self.image_position
                self.image[int(center[1]),int(center[0]),:] = (255,255,255)
                self.image[int(center[1])-1,int(center[0]),:] = (255,255,255)
                self.image[int(center[1])+1,int(center[0]),:] = (255,255,255)
                self.image[int(center[1]),int(center[0])-1,:] = (255,255,255)
                self.image[int(center[1]),int(center[0])+1,:] = (255,255,255)
            cv2.imshow('frame',self.image)

    def compute_focal_length(self):
        """Computes the focal length of the camera in pixels"""
        d2 = self._image_width**2 + self._image_height**2 
        halfDiagonalLength = 0.5 * np.sqrt(d2)
        if halfDiagonalLength > 0 and self._opening_angle_diagonal > 0:
            return (halfDiagonalLength
                    / np.tan(0.5 * np.deg2rad(self._opening_angle_diagonal)))
        else:
            raise ValueError

    def _ARcoords2Torso(self, coords):
        """References the ArUco coordinates to Nao's frame Torso."""
        new_coords = coords.flatten().copy()
        new_coords[0] = coords[2]
        new_coords[1] = -1.*coords[0]
        new_coords[2] = -1.*coords[1]
        output = self._coords2Torso(new_coords, self._camera_matrix)
        return output

    def _coords2Torso(self, coords, trans):
        """Transforms any coordinates"""
        out = np.zeros_like(coords)

        out[0] = trans[0]*coords[0] + trans[1]*coords[1] + trans[2]*coords[2]
        out[1] = trans[4]*coords[0] + trans[5]*coords[1] + trans[6]*coords[2]
        out[2] = trans[8]*coords[0] + trans[9]*coords[1] + trans[10]*coords[2]

        out[0] += trans[3]
        out[1] += trans[7]
        out[2] += trans[11]
        return out

    def _fetch_image(self):
        """Fetches one image from the camera and saves it in self.image"""
        nao_image = self._cam_proxy.getImageRemote(self._video_client)
        self._camera_matrix = self._motion_proxy.getTransform("CameraBottom",
                                                              0, False)
        #self.markerMatrix = self._motion_proxy.getTransform(self.markerJoint,
        #                                                    0, False)
        self._cam_proxy.releaseImage(self._video_client)
        self.image = np.reshape(np.fromstring(nao_image[6], np.uint8),
                                (nao_image[1],nao_image[0],3))

    def _look_at_point(self, point, move_speed=0.5):
        """Tries to adjust head joints to focus on a point 
        in (x,y) pixel space.
        """
        center = self._optical_center
        dev_x = center[0] - point[0]
        divAngleX = np.arctan2(dev_x, self._focal_length)
        dev_y = point[1] - center[1]
        divAngleY = np.arctan2(dev_y, self._focal_length)

        current_angles = self._motion_proxy.getAngles(["HeadYaw", "HeadPitch"],
                                                      True)
        new_angles = [current_angles[0] + divAngleX, 
                      current_angles[1] + divAngleY]
        self._motion_proxy.setStiffnesses(["HeadYaw", "HeadPitch"], 1.0)
        self._motion_proxy.setAngles(["HeadYaw", "HeadPitch"],
                                     new_angles, move_speed)
          
    def follow_marker(self):
        """TODO"""
        while True:
            if self.marker_detected:
                current_pos = self.image_position.copy()
                self._look_at_point(current_pos)
            time.sleep(0.035)

    def run(self):
        """The main method of the NaoDetection module. Starts the module to make
        the detection available.

        Examples
        --------
        detection_proxy = ALProxy("NaoDetection", IP, PORT)
        detection_proxy.init(['HeadPitch','HeadYaw'],['imageX','imageY'],320,240)
        detection_proxy.post.run()
        while True:
            print detection_proxy.get_data()

        """
        while not self._requested_to_stop:
            t1 = time.clock()
            self._fetch_image()
            self.current_angles = self._motion_proxy.getAngles(self._joint_names,
                                                               True)
            self.detector.detect(self.image)

            if self.detector.getID()== []:
                self.marker_detected = 0
                self.image_position = []
                self.absolute_position = []
                self.analytical_position = []
            else:
                self.marker_detected = 1
                img_coords = self.detector.getCenter()[0]
                self.image_position = list(np.array(img_coords,dtype='float32'))
                if self._return_relative_position:
                    # normalizes pixel image position to [-1,1]
                    self.image_position[0] = ((self.image_position[0]
                                               - self._optical_center[0])
                                               / self._image_half_width)
                    self.image_position[1] = ((self.image_position[1]
                                               - self._optical_center[1])
                                               / self._image_half_height)
                abs_coords = self._ARcoords2Torso(self.detector.getTvec()[0])
                self.absolute_position = self._dummy_translation(abs_coords)
                #self.analyticalPosition = list(self._coords2Torso(self.markerTranslation, self.markerMatrix))

            new_data = [self.marker_detected] + self.current_angles
            if set(['imageX','imageY']).issubset(self._sensor_names):
                new_data += self.image_position 
            if set(['spaceX','spaceY','spaceZ']).issubset(self._sensor_names):
                new_data += self.absolute_position

            self.data = copy.deepcopy(new_data)
            t2 = time.clock()
            time.sleep(0.035-(t2-t1))
            
    def exit(self):
        """ Exit method. Call to stop module. """
        self._requested_to_stop = True
        self.cleanup()
        self.requested_to_exit = True

    def get_data(self):
        """Get the essential data acquired by the run() method.

        Examples
        --------
        detection_proxy = ALProxy("NaoDetection", IP, PORT)
        detection_proxy.init(['HeadPitch', 'HeadYaw'],
                             ['imageX', 'imageY'], 320, 240)
        detection_proxy.post.run()
        while True:
            data = detection_proxy.get_data()
            if data[0]:                
                print 'marker detected!'
        """
        return self.data

    def _dummy_translation(self, array):
        """Certain kinds of floats/arrays do not seem to be properly transmitted
        through the ALNetwork. This is a dummy conversion to avoid that.
        """
        outlist = []
        for i in range(array.shape[0]):
            outlist.append(float(array[i])) 
        return outlist



def main():
    """ Main entry point
    """
    parser = OptionParser()
    parser.add_option("--pip",
        help="Parent broker port. The IP address or your robot",
        dest="pip")
    parser.add_option("--pport",
        help="Parent broker port. The port NAOqi is listening to",
        dest="pport",
        type="int")
    parser.set_defaults(
        pip=NAO_IP,
        pport=9559)

    (opts, args_) = parser.parse_args()
    pip   = opts.pip
    pport = opts.pport

    # We need this broker to be able to construct
    # NAOqi modules and subscribe to other modules
    # The broker must stay alive until the program exists
    myBroker = ALBroker("myBroker",
       "0.0.0.0",   # listen to anyone
       0,           # find a free port and use it
       pip,         # parent broker IP
       pport)       # parent broker port

    # Warning: NaoDetection must be a global variable
    # The name given to the constructor must be the name of the
    # variable
    global NaoDetection
    NaoDetection = PyNaoDetection("NaoDetection") #, 320, 240, 72.6)

    try:
        while True:
            time.sleep(1)
            #MarkerTracking.visual()
            if cv2.waitKey(1) & 0xFF == ord('q')\
            or NaoDetection.requested_to_exit:
                break
    except KeyboardInterrupt:
        print "Interrupted by user, shutting down"
        NaoDetection.exit()
        myBroker.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()




