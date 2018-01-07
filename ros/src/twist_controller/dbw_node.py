#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float64
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from lowpass import LowPassFilter
import math

CONTROL_RATE = 50  # Htz
CONTROL_PERIOD = 1.0 / CONTROL_RATE

from twist_controller import Controller

'''You can build this node only after you have built (or partially built)
the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed
linear and angular velocities.  You can subscribe to any other message
that you find important or refer to the document for list of messages
subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the
`twist_controller` class is the status of `dbw_enabled`. While in the
simulator, its enabled all the time, in the real car, that will not be
the case. This may cause your PID controller to accumulate error
because the car could temporarily be driven by a human instead of your
controller.

We have provided two launch files with this node. Vehicle specific
values (like vehicle_mass, wheel_base) etc should not be altered in
these files.

We have also provided some reference implementations for PID
controller and other utility classes.  You are free to use them or
build your own.

Once you have the proposed throttle, brake, and steer values, publish
it on the various publishers that we have created in the `__init__`
function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # Instance values
        self.dbw_enabled = True

        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)  # check
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)   # check
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)   # check

        self.decel_limit = -650.0
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)   # check
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)       # check
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)       # check
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.) # check
        
        self.current_velocity = 0.0
        self.current_angular = 0.0
        self.proposed_linear = 0.0
        self.proposed_angular = 0.0
        self.cte = 0.0
        self.lp_filter = LowPassFilter(0.5, CONTROL_PERIOD)
        
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # NOTE: I pass the dbw_node object to the controller so I don't
        #       have huge ugly param list for initialization.  Perhaps
        #       not the best practice...
        self.controller = Controller(self)

        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        # rospy.Subscriber('/cross_track_error', Float64, self.cross_track_cb)        
        self.loop()

    def loop(self):
        rate = rospy.Rate(CONTROL_RATE) # 50Hz
        while not rospy.is_shutdown():
            throttle, brake, steering = self.controller.control(self.proposed_linear,
                                                                self.proposed_angular,
                                                                self.current_velocity,
                                                                self.current_angular)

            if self.dbw_enabled:
                self.publish(throttle, brake, steering)
                                                                
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def twist_cmd_cb(self, msg):
        # There is a bug in the pure_pursuit code that sometimes returns a negative
        # linear.x, so we just take the abs() here as suggested on Slack
        self.proposed_linear = abs(msg.twist.linear.x)
        self.proposed_angular = msg.twist.angular.z
        return
    
    def current_velocity_cb(self, msg):
        # lowpass filter the finite difference approximation to the acceleration
        self.lp_filter.filt((self.current_velocity - msg.twist.linear.x) / CONTROL_RATE)
        self.current_velocity = msg.twist.linear.x
        self.current_angular = msg.twist.angular.z
        return

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg
        return

    def cross_track_cb(self, msg):
        self.cte = msg.data
        return
    
if __name__ == '__main__':
    DBWNode()
