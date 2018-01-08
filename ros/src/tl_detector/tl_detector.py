#!/usr/bin/env python
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier, TLKeras
import tf
import cv2
import yaml

# Set LOOP_ONCE = False to loop indefinitely around the track
LOOP_ONCE = False
USE_CLASSIFICATION = True
USE_KERAS_MODEL = True
STATE_COUNT_THRESHOLD = 3

# These three can be used for data capture
COLLECT_IMAGES = False
CAPTURE_DISTANCE = 40
CAPTURE_DIR = "/udacity/capture/"

#  These points can be anywhere, but I used to collect images 
#  around the traffic lights. For the record, the stop lines for
#  the lights are located at the following indices:
#  [293, 754, 2047, 2580, 6294, 7008, 8540, 9733]
capture_idxs = [6294, 7008, 9733]


# This is just used for naming the captured images of the traffic
# lights.  This effectively labels them as well (at least for a
# pure classification NN)
TRAFFIC_LIGHT_2_NAME = {
    TrafficLight.RED: "red",
    TrafficLight.YELLOW: "yellow",
    TrafficLight.GREEN: "green",
    TrafficLight.UNKNOWN: "unkn"
}

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pos = -1
        self.pose = None
        self.waypoints = None
        self.x_ave = 0.0
        self.y_ave = 0.0
        self.cos_rotate = 0.0
        self.sin_rotate = 0.0
        self.phi = []
        self.stop_lines = None        
        self.camera_image = None

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        if USE_KERAS_MODEL:
            self.light_classifier = TLKeras()
        else:
            self.light_classifier = TLClassifier()

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.stop_idxs = []
        
        # For image capture...
        self.last_capture_idx = 0
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        
        rospy.spin()

    def pose_cb(self, msg):
        if msg:
            self.pos = self.get_index(msg.pose.position.x, msg.pose.position.y)
            self.pose = msg

    def get_index(self, x, y):
        rho = self.get_angle(x, y)
        # special case of wrap around when past the last waypoint
        if not self.phi or rho > self.phi[-1]:
            return 0
        
        idx = 0
        while rho > self.phi[idx]:
            idx += 1

        return idx

    def get_angle(self, x, y):
        # First center
        xc = x - self.x_ave
        yc = y - self.y_ave
        
        # and now rotate
        xr = xc * self.cos_rotate - yc * self.sin_rotate
        yr = yc * self.cos_rotate + xc * self.sin_rotate
        
        # rho now starts at 0 and goes to 2pi for the track waypoints
        rho = math.pi - math.atan2(xr, yr)
        return rho
        
    def waypoints_cb(self, lane):
        #self.waypoints.extend(lane.waypoints)
        self.waypoints = lane.waypoints
        x_tot = 0.0
        y_tot = 0.0
        for p in self.waypoints:
            x_tot += p.pose.pose.position.x
            y_tot += p.pose.pose.position.y

        # We use the average values to recenter the self.waypoints
        self.x_ave = x_tot / len(self.waypoints)
        self.y_ave = y_tot / len(self.waypoints)
        
        # The very first waypoint determines the angle we need to rotate
        # all waypoints by
        xc = self.waypoints[0].pose.pose.position.x - self.x_ave
        yc = self.waypoints[0].pose.pose.position.y - self.y_ave
        rotate = math.atan2(xc, yc) + math.pi
        self.cos_rotate = math.cos(rotate)
        self.sin_rotate = math.sin(rotate)

        for p in self.waypoints:
            rho = self.get_angle(p.pose.pose.position.x, p.pose.pose.position.y)
            self.phi.append(rho)

        # We can only process the stop_lines after the waypoints
        stop_line_positions = self.config['stop_line_positions']
        for stop_line in stop_line_positions:
            idx = self.get_index(stop_line[0], stop_line[1])
            self.stop_idxs.append(idx)
            
        rospy.loginfo(self.stop_idxs)
            

    def traffic_cb(self, msg):
        # It is possible that traffic_cb is called before we've had a
        # chance to process the waypoints, so do nothing (returns None
        # so we know if anyone tries to use this prematurely)
        if not self.waypoints or len(msg.lights) != len(self.stop_idxs):
            return
        
        # Note that we depend on the fact that the stop_lines and the
        # traffic lights appear in the same order in their config files
        
        self.stop_lines = []
        for i, light in enumerate(msg.lights):
            sidx = self.stop_idxs[i]
            self.stop_lines.append((sidx, light.state, light))
        self.stop_lines.sort()

    def get_next_stop_line(self):
        if not self.stop_lines or not self.stop_idxs:
            return (None, None, None)
        elif self.pos > self.stop_lines[-1][0]:
            return self.stop_lines[0]
        idx = 0
        num_lights = len(self.stop_lines)
        while self.pos > self.stop_lines[idx][0]:
            idx += 1
            if idx >= len(self.stop_lines):
                rospy.logerr("stop lines idx: %d" % idx)

        # for debug. a positions past the last stop_line will trigger
        # if self.pos > self.stop_lines[idx][0]:
        #     rospy.loginfo("get_next_stop_line self.pos %d  stop_lines: %d" % \
        #                   (self.pos, self.stop_lines[idx][0]))
            
        return self.stop_lines[idx]

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        incoming_light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
            light_wp = self.last_wp
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = incoming_light_wp if state == TrafficLight.RED or \
                       state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp
        else:
            light_wp = self.last_wp

        # We refactored Udacity's code to make sure we publish on every cycle
        self.upcoming_red_light_pub.publish(Int32(light_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.get_index(pose.position.x, pose.position.y)


    def save_image(self, image):
        ''' Image capture for generating training images'''
        if not COLLECT_IMAGES:
            return
        sl_idx, state, light = self.get_next_stop_line()

        if sl_idx in capture_idxs and \
           sl_idx - self.pos < CAPTURE_DISTANCE and \
           self.pos > self.last_capture_idx + 10:
            self.last_capture_idx = self.pos
            fname = CAPTURE_DIR + TRAFFIC_LIGHT_2_NAME[state] + \
                    str(sl_idx) + "_" + str(self.pos) + ".jpg"
            # fake out cv2 to write rgb
            img = image[..., ::-1]
            cv2.imwrite(fname, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        # Changed this to rgb since that is default for tensorflow / keras
        # Avoids unnecessary color conversions
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # only for data collecton
        self.save_image(cv_image)
            
        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color.  If LOOP_ONCE is True, then alway return the last waypoint
            as a red light.

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if  self.pose:
            # car_position is now maintained by pose_cb as self.pos
            # Find the closest visible traffic light (if one exists)
            stop_line_wp, state, light = self.get_next_stop_line()
	    outstr = "Car position : " + str(self.pos) + " stop_line_wp : " + str(stop_line_wp)
            #rospy.loginfo(outstr)
            

        if light:
            if USE_CLASSIFICATION:
                state = self.get_light_state(light)
                #rospy.logwarn("light state: %d" % state)
                
            if LOOP_ONCE:
                # Only if our position is past the last traffic light, we return the
                # index of the very last waypoint as a red light
                if self.pos > self.stop_idxs[-1]:
                    state = TrafficLight.RED
                    stop_line_wp = len(self.waypoints) - 1
                
            return stop_line_wp, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
