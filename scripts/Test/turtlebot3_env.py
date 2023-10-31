#! /usr/bin/env python
# coding=utf-8
## version 2:
## 1, navigate the robot using a constant heading angle
## 2, add the ddpg neural network
## 3, 24 laser data and just heading
## 4, added potential collisions

import rospy
import rospkg
import tf
from geometry_msgs.msg import Twist, Point
from math import sqrt, pow,  atan2
from tf.transformations import euler_from_quaternion
import threading
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
import time
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import random
from std_srvs.srv import Empty
import os
from nav_msgs.msg import Path
import pandas as pd

np.random.seed(1234)

class InfoGetter(object):
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        self._event.wait(timeout)
        return self._msg

class GameState:

    def __init__(self):
        self.talker_node = rospy.init_node('turtlebot3_original_ddpg', anonymous=True)
        self.pose_ig = InfoGetter()
        self.laser_ig = InfoGetter()
        self.collision_ig = InfoGetter()


        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.position = Point()
        self.move_cmd = Twist()
        self.goal_postion = Point()
        self.robot_pose = []
        self.df = pd.DataFrame()

        self.laser_info = rospy.Subscriber("/scan_filtered1", LaserScan, self.laser_ig)

        # tf
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'
        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
        (self.position, self.rotation) = self.get_odom()


        self.rate = rospy.Rate(10) # 10hz

        # Create a Twist message and add linear x and angular z values
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.6 #linear_x
        self.move_cmd.angular.z = 0.2 #angular_z

        # crush default value
        self.crash_indicator = 0
        self.goal_position = Pose()

        # observation_space and action_space
        # original
        self.state_num = 28 # when you change this value, remember to change the reset default function as well
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)


        self.laser_reward = 0
        # set target position
        self.target_x = 10
        self.target_y = -0.3

        # set turtlebot index in gazebo world
        self.model_index = 10 #25

    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def print_odom(self):
        while True:
            (self.position, self.rotation) = self.get_odom()
            print("position is %s, %s, %s, ", self.position.x, self.position.y, self.position.z)
            print("rotation is %s, ", self.rotation)

    def stop_robot(self):
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub.publish(self.move_cmd)
        time.sleep(1)

    def reset(self):
        index_list = [-1, 1]
        index_x = random.choice(index_list)
        index_y = random.choice(index_list)

        self.target_x = 10.0
        self.target_y = -0.0

        # goal box pose
        self.goal_position.position.x = self.target_x
        self.goal_position.position.y = self.target_y
        self.crash_indicator = 0

        initial_state = np.ones(self.state_num)
        initial_state[self.state_num-1] = 0
        initial_state[self.state_num-2] = 0
        initial_state[self.state_num-3] = 0
        initial_state[self.state_num-4] = 0

        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub.publish(self.move_cmd)
        time.sleep(0.1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()

        return initial_state

    def turtlebot_distance_reward(self,turtlebot_x_previous,turtlebot_y_previous,turtlebot_x,turtlebot_y):
        self.distance_reward = 0.0
        # make distance reward
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        # 0.1s时间内，robot移动的距离
        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous)**2 + (self.target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target
        return self.distance_reward


    def turtlebot_is_crashed(self, laser_values, range_limit):
        self.laser_crashed_value = False
        self.laser_crashed_reward = 0
        for i in range(len(laser_values)):
            if(laser_values[i] >= 2*range_limit):
                self.laser_crashed_reward = 0
            if (laser_values[i] < 2*range_limit):
                self.laser_crashed_reward = -80
            if (laser_values[i] < range_limit):
                print("min_laser:",min(laser_values))
                print(laser_values)
                self.laser_crashed_value = True
                self.laser_crashed_reward = -200
                self.stop_robot()
                break
        return self.laser_crashed_reward


    def game_step(self, time_step,linear_x, angular_z):
        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        # print("linear_x:",linear_x,"angular_z:",angular_z)
        if linear_x!=0.1:
            linear_x = 0.1
        else:
            rospy.logwarn("Velocity not set to 0.26 m/s")
        self.move_cmd.linear.x = linear_x
        if angular_z>0.5:
            angular_z = 0.5
        if angular_z<=-0.5:
            angular_z = -0.5
        self.move_cmd.angular.z = angular_z
        self.rate.sleep()

        (self.position, self.rotation) = self.get_odom()
        turtlebot_x_previous = self.position.x
        turtlebot_y_previous = self.position.y
        while (record_time_step < time_step) and (self.crash_indicator==0):
            self.pub.publish(self.move_cmd)
            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time

        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        angle_turtlebot = self.rotation
        self.robot_pose.append([turtlebot_x,turtlebot_y])
        self.df = pd.DataFrame( self.robot_pose)
        self.df.to_csv('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/robot_position_data.csv')
        # make input, angle between the turtlebot and the target
        angle_turtlebot_target = atan2(self.target_y - turtlebot_y, self.target_x- turtlebot_x)
        # print("angle_turtlebot_target:",angle_turtlebot_target)

        if angle_turtlebot < 0:
            angle_turtlebot = angle_turtlebot + 2*math.pi

        if angle_turtlebot_target < 0:
            angle_turtlebot_target = angle_turtlebot_target + 2*math.pi

        angle_diff = angle_turtlebot_target - angle_turtlebot
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2*math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2*math.pi

        # prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges

        normalized_laser =[]
        for i in range(len(laser_values)):
            if laser_values[i] == float('Inf'):
                normalized_laser.append(3.5/3.5)
            elif np.isnan(laser_values[i]):
                normalized_laser.append(0)
            else:
                normalized_laser.append(laser_values[i]/3.5)

        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        # print("current_distance_turtlebot_target:",cusrrent_distance_turtlebot_target)
        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x)
        state = np.append(state, angular_z)
        state = state.reshape(1, self.state_num)
        state = state.flatten()

        # make distance reward
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        # 0.1s时间内，robot移动的距离
        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous)**2 + (self.target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target

        # self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.15)
        self.laser_reward = sum(normalized_laser)-24
        # print("self.laser_reward:",self.laser_reward)
        self.collision_reward = self.laser_crashed_reward + self.laser_reward

        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        if angular_z > 0.8:
            self.angular_punish_reward = -1
        if angular_z < -0.8:
            self.angular_punish_reward = -1

        if linear_x < 0.2:
            self.linear_punish_reward = -2


        self.arrive_reward = 0
        if distance_turtlebot_target<1:
            # rospy.loginfo("Robot Arrive Goal!!!!!!!!!")
            self.arrive_reward = 200
            # self.reset()
            self.stop_robot()


        reward  = distance_reward*(5/time_step)*1.2*7 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward

        return reward, state, self.laser_crashed_value,self.arrive_reward





