import rospy
import numpy as np
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
CONTROL_RATE = 50  # Htz
CONTROL_PERIOD = 1.0 / CONTROL_RATE

# Two people in car?
PASSENGER_MASS = 150
SPEED_EPS = 0.1
ANGLE_EPS = 0.001

class Controller(object):
    def __init__(self, dbw_node):
        self.dbw_node = dbw_node
        self.min_speed = 0.1

        # This assumes that the gas tank is always full
        self.vehicle_mass = self.dbw_node.vehicle_mass + \
                            GAS_DENSITY * self.dbw_node.fuel_capacity + \
                            PASSENGER_MASS
        
        self.yaw_control = YawController(self.dbw_node.wheel_base,
                                         self.dbw_node.steer_ratio,
                                         self.min_speed,
                                         self.dbw_node.max_lat_accel,
                                         self.dbw_node.max_steer_angle)


        # Values of Kp, Ki, and Kd are from DataSpeed example
        # This is really only a proportional filter where the
        # output is contrained to [-1g, +1g]
        # These values are intended only for the two stage controller
        self.velo_pid = PID(2.0, 0.0, 0.0, -9.8, 9.8)

        # Throttle is between 0.0 and 1.0
        # Values of Kp, Ki, and Kd are from DataSpeed example
        self.accel_pid = PID(0.4, 0.1, 0.0, 0.0, 1.0)

        # PID for cross track error.  Meant to replace yaw_control
        # This did not perform well... not sure whether it was problems
        # with the simulator or the PID parameter values.  Did not
        # investigate deeply
        self.cte_pid = PID(0.2, 0.001, 0.85)

    # This only does yaw control at constant throttle.  Now used
    # for debugging only
    def simple_control(self, proposed_linear, proposed_angular,
                       current_linear, current_angular):
        #rospy.logwarn("current_linear: %f" % current_linear)
        throttle = 0.6
        steering = self.yaw_control.get_steering(proposed_linear,
                                                 proposed_angular,
                                                 current_linear)
        brake = 0.0
        return throttle, brake, steering
    
    def control(self, proposed_linear, proposed_angular,
                     current_linear, current_angular):

        '''
        #--------- The Udacity provided steering
        steering = self.yaw_control.orig_get_steering(proposed_linear,
                                                 proposed_angular,
                                                 current_linear)
        '''
        
        #--------- Modified steering for better low velocity control
        steering = self.yaw_control.get_steering(proposed_linear,
                                                 proposed_angular,
                                                 current_linear)

        '''
        #--------- CTE-PID Steering Controller-----------
        # Looks good until it goes completely haywire

        if abs(self.dbw_node.cte) < ANGLE_EPS:
            self.cte_pid.reset()

        steering = self.cte_pid.step(self.dbw_node.cte, CONTROL_PERIOD)
        '''

        velo_error = proposed_linear - current_linear

        # Reset integration sum if error is small
        if abs(velo_error) < SPEED_EPS or proposed_linear < self.min_speed:
            self.velo_pid.reset()

        accel_est = self.velo_pid.step(velo_error, CONTROL_PERIOD)

        # 2-stage controller

        # rospy.loginfo("proposed: %f  current: %f  error: %f  accel: %f" % \
        #               (proposed_linear, current_linear, velo_error, accel_est))

        if accel_est >= 0.05:
            filtered_accel = self.dbw_node.lp_filter.get()
            delta_accel = accel_est - filtered_accel
            tctrl = self.accel_pid.step(delta_accel, CONTROL_PERIOD)
            throttle = tctrl
        else:
            self.accel_pid.reset()
            throttle = 0.0

        if accel_est < 0.0:
            # braking actually takes a positive value
            calc_brake = -accel_est * self.vehicle_mass * self.dbw_node.wheel_radius
            # Should move this magic number somewhere else
            brake = min(calc_brake, 650)
        else:
            brake = 0.0
            
        # dbw not engaged, don't accumulate error
        if  not self.dbw_node.dbw_enabled:
            self.velo_pid.reset()
            self.accel_pid.reset()

        return throttle, brake, steering
