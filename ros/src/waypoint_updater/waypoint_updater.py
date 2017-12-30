#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32, Float64
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from copy import deepcopy

import math

ONE_MPH = 0.44704
CREEP_VELOCITY = 1.5
CREEP_RANGE = 30
MIN_DISTANCE = 40  

# Back down on the max velocity to ensure we do not exceed
VELOCITY_MARGIN = 0.965

'''This node will publish waypoints from the car's current position
to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version
which does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the
status of traffic lights too.

Please note that our simulator also provides the exact location of
traffic lights and their current status in `/vehicle/traffic_lights`
message. You can use this message to build this node as well as to
verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.

'''

# This is all in the launch files now.  Use rospy.get_param
# If it is not the highway, then it is the churchlot.  Highway is the
# default
# HIGHWAY = False

# if HIGHWAY:
#     LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
#     BRAKING_RANGE = 14.0 # This is in meters
# else:
#     # My version of churchlot has only 55 way points
#     LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number
#     BRAKING_RANGE = 15.0 # This _must_ be smaller than lookahead


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        self.waypoints = []
        self.x_ave = 0.0
        self.y_ave = 0.0
        self.cos_rotate = 0.0
        self.sin_rotate = 0.0
        self.phi = []
        self.current_velocity = 0.0
        self.red_light = -1
        self.lap_count = 0
        self.lap_toggle = False
        self.end_idx = -1

        # the param is in km per hour. Dividing by 3.6 give mps
        max_velocity = float(rospy.get_param("/waypoint_loader/velocity")) / 3.6
        self.target_velocity = max_velocity * VELOCITY_MARGIN
        rospy.loginfo("target velocity %f" % self.target_velocity)

        self.lookahead_wps = rospy.get_param("/waypoint_loader/num_wps")
        self.braking_range = float(rospy.get_param("/waypoint_loader/brake_range"))
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        
        # There is no topic (yet) for /obstacle_waypoint
        #rospy.Subscriber('/obstacle_waypoint', ???, self.obstacle_cb)

        # cross_track_error topic... disabling for now
        #self.cte_publisher = rospy.Publisher('cross_track_error', Float64, queue_size=1)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        rospy.spin()

    def lap(self, rho):
        # Ugly but reasonably robust
        if rho < 0.2 and self.lap_toggle:
            self.lap_count += 1
            rospy.logwarn("Completed lap %d" % self.lap_count)
            self.lap_toggle = False
        elif rho > 6.0:
            self.lap_toggle = True
        return
    
    def get_index(self, x, y):
        rho = self.get_angle(x, y)
        # special case of wrap around when past last waypoint
        if not self.phi or rho > self.phi[-1]:
            return 0
        
        self.lap(rho)
        
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

    def get_cte(self, idx, x0, y0):
        x1 = self.waypoints[idx].pose.pose.position.x
        y1 = self.waypoints[idx].pose.pose.position.y
        x2 = self.waypoints[idx-1].pose.pose.position.x
        y2 = self.waypoints[idx-1].pose.pose.position.y
        cte = -((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) /   \
                 math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        #self.cte_publisher.publish(Float64(cte))
        return cte

    def pose_cb(self, msg):
        # pose_cb might be called before waypoints_cb
        if self.waypoints == []:
            return None

        idx = self.get_index(msg.pose.position.x, msg.pose.position.y)
            
        # For now, do not calculate or publish the cte because the other
        # methods for generating steering angles seem to be better than
        # a PID controller
        #cte = self.get_cte(idx, msg.pose.position.x, msg.pose.position.y)
        
        #rospy.logwarn("position: %d   cte: %f  stop_line %d" % \
        #              (idx, cte, self.red_light.data))
        self.publish(idx)

    def waypoints_cb(self, lane):
        wp = lane.waypoints
        x_tot = 0.0
        y_tot = 0.0
        for p in wp:
            x_tot += p.pose.pose.position.x
            y_tot += p.pose.pose.position.y

        # We use the average values to recenter the waypoints
        self.x_ave = x_tot / len(wp)
        self.y_ave = y_tot / len(wp)
        
        # The very first waypoint determines the angle we need to rotate
        # all waypoints by
        xc = wp[0].pose.pose.position.x - self.x_ave
        yc = wp[0].pose.pose.position.y - self.y_ave
        rot_angle = math.atan2(xc, yc) + math.pi
        self.cos_rotate = math.cos(rot_angle)
        self.sin_rotate = math.sin(rot_angle)

        for p in wp:
            rho = self.get_angle(p.pose.pose.position.x, p.pose.pose.position.y)
            self.phi.append(rho)
                
        #self.waypoints.extend(wp)
        self.waypoints = wp
        self.end_idx = len(wp)
        
        # make wrap around easier by extending waypoints
        # (I guess this is overkill since we are supposed to stop at last wp)
        self.waypoints.extend(wp[0:self.lookahead_wps])

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x
        return

    def traffic_cb(self, msg):
        self.red_light = msg.data
        outstr = "Red light msg : " + str(self.red_light)
        rospy.loginfo(outstr)
        return
    
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        if not (0 <= waypoint <= len(waypoints)):
            rospy.logwarn("WTF: %d %d" % (waypoint, len(waypoints)))
        try:
            waypoints[waypoint].twist.twist.linear.x = velocity
        except IndexError:
            rospy.logwarn("Exception: %d %d" % (waypoint, len(waypoints)))

    def distance(self, waypoints, wp1, wp2):
        dist = 0.0
        dl = lambda a, b: math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
        if wp1 > wp2:
            for i in range(wp1, len(self.waypoints)):
                dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
                wp1 = i
            for i in range(0, wp2+1):
                dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
                wp1 = i
        else:
            for i in range(wp1, wp2+1):
                dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
                wp1 = i
        return dist

    def publish(self, idx):
        # The publish topic might start up before we have processed the waypoints
        if idx >= len(self.waypoints):
            return
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = []
        lane.waypoints.extend(self.waypoints[idx:idx+self.lookahead_wps])
        
        # it looks like we sometimes get bad red_light data
        # dist = self.red_light - idx
        # if self.lap == 0 and self.red_light > -1 and dist < self.braking_range:
        #     if idx < 280 and dist < CREEP_RANGE and self.current_velocity < CREEP_VELOCITY:
        #         for i in range(CREEP_RANGE - 5):
        #             self.set_waypoint_velocity(lane.waypoints, i, CREEP_VELOCITY)
        #         for i in range(CREEP_RANGE - 5, self.lookahead_wps):
        #             self.set_waypoint_velocity(lane.waypoints, i, 0.0)
        #     else:                
        #         # incr = 1.1 * (self.current_velocity / max(0.1, float(dist)))
        #         # velo = self.current_velocity - incr
        #         # for i in range(dist):
        #         #     self.set_waypoint_velocity(lane.waypoints, i, max(0.0, velo))
        #         #     velo -= incr
        #         # #rospy.logwarn("%d  %d  %d  %d" % (dist, len(lane.waypoints), self.lookahead_wps))
        #         # for i in range(dist, self.lookahead_wps):
        #         #     self.set_waypoint_velocity(lane.waypoints, i, 0.0)

        #         # let's just try a hard stop
        #         for i in range(self.lookahead_wps):
        #             self.set_waypoint_velocity(lane.waypoints, i, 0.0)
        # else:
        #     for i in range(self.lookahead_wps):
        #         self.set_waypoint_velocity(lane.waypoints, i, self.target_velocity)


        # Dead nuts simple braking strategy.  This brakes at max value
        dist = self.distance(self.waypoints, idx, self.red_light)
        if self.red_light != -1 and dist < self.braking_range:
            for i in range(self.lookahead_wps):
                self.set_waypoint_velocity(lane.waypoints, i, 0.0)
        else:
            for i in range(self.lookahead_wps):
                self.set_waypoint_velocity(lane.waypoints, i, self.target_velocity)
        

        # Karsten's patch
        # if self.red_light != -1:

        #     rampwidth = 100

        #     for i in range(self.lookahead_wps):
        #         xd = self.red_light - idx - i
        #         if xd < 0:
        #             rampspeed = 0.0
        #         elif xd > rampwidth:
        #             rampspeed = self.target_velocity
        #         else:
        #             rampspeed = self.target_velocity * xd / rampwidth
                            
        #         self.set_waypoint_velocity(lane.waypoints, i, rampspeed )
        # else:
        #     for i in range(self.lookahead_wps):
        #         self.set_waypoint_velocity(lane.waypoints, i, self.target_velocity)
        
        self.final_waypoints_pub.publish(lane)
        

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
