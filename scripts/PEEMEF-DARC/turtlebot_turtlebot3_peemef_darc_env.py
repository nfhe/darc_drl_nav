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
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState,SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
import time
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import random
from std_srvs.srv import Empty
import os

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
        self.talker_node = rospy.init_node('turtlebot3_peemef_ddpg', anonymous=True)
        self.pose_ig = InfoGetter()
        self.laser_ig = InfoGetter()
        self.collision_ig = InfoGetter()


        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

        self.position = Point()
        self.move_cmd = Twist()
        self.goal_postion = Point()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.laser_info = rospy.Subscriber("/laserscan_filtered", LaserScan, self.laser_ig)

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


        self.rate = rospy.Rate(100) # 100hz

        # Create a Twist message and add linear x and angular z values
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.6 #linear_x
        self.move_cmd.angular.z = 0.2 #angular_z

        # crush default value
        self.crash_indicator = 0

        # observation_space and action_space
        # original
        self.state_num = 28 # when you change this value, remember to change the reset default function as well
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)


        self.laser_reward = 0
        # set target position
        self.target_x = 10
        self.target_y = 10

        # set turtlebot index in gazebo world
        self.model_index = 10 #25

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # goal box parameter set
        self.modelPath =r'/home/he/catkin_nav/src/turtlebot3_nav/turtlebot_ddpg/model/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.goal_position = Pose()
        self.init_goal_x = 0.6
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.check_model = False
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)

    def checkModel(self,model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
        rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,self.goal_position.position.y)
        self.check_model = True

    def deleteModel(self):
        if self.check_model == False:
            return
        rospy.wait_for_service('gazebo/delete_model')
        delete_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        delete_model_prox(self.modelName)

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

        # for maze
        # self.target_x = (np.random.random()-0.5)*5 + 12*index_x
        # self.target_y = (np.random.random()-0.5)*5 + 12*index_y
        self.target_x = (np.random.random()-0.5)*3 + 10*index_x
        self.target_y = (np.random.random()-0.5)*3 + 10*index_y
        if self.target_x>11.5:
            self.target_x = 11.5
        if self.target_x<-11.5:
            self.target_x = -11.5
        if self.target_y>11.5:
            self.target_y = 11.5
        if self.target_y<-11.5:
            self.target_y = -11.5

        # for corridor
        # self.target_x = (np.random.random()-0.5)*5 + 12*index_x
        # self.target_y = (np.random.random()-0.5)*3
        # self.target_x = 12*index_x
        # self.target_y = 0

        # goal box pose
        self.goal_position.position.x = self.target_x
        self.goal_position.position.y = self.target_y
        self.crash_indicator = 0

        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_waffle'
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0 #random_turtlebot_y
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = -0.2
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # set goal box
        self.deleteModel()
        time.sleep(0.5)
        self.respawnModel()
        time.sleep(0.5)

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
                self.laser_crashed_value = True
                self.laser_crashed_reward = -200
                self.stop_robot()
                # self.reset()
                # time.sleep(1)
                break
        return self.laser_crashed_reward


    def game_step(self, time_step,linear_x, angular_z):
        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        if linear_x!=0.26:
            linear_x = 0.26
        else:
            rospy.logwarn("Velocity not set to 0.26 m/s")
        self.move_cmd.linear.x = linear_x
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

        # make input, angle between the turtlebot and the target
        angle_turtlebot_target = atan2(self.target_y - turtlebot_y, self.target_x- turtlebot_x)

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

        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.15)
        self.laser_reward = sum(normalized_laser)-24
        # print("self.laser_reward:",self.laser_reward)
        self.collision_reward = self.laser_crashed_reward + self.laser_reward

        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        if abs(angular_z)<0.1:
            self.angular_punish_reward=-0.5
        elif abs(angular_z)<0.6 and abs(angular_z)>0.1:
            self.angular_punish_reward=-2
        elif abs(angular_z)>0.6:
            self.angular_punish_reward=-100
        else:
            self.angular_punish_reward=-500

        if abs(angular_z-0.5)>0.35:
            self.angular_osa_punish_reward=-5*abs(angular_z-0.5)
        else:
            self.angular_osa_punish_reward=0

        if linear_x < 0.2:
            self.linear_punish_reward = 10*linear_x
        elif linear_x >= 0.2:
            self.linear_punish_reward = -2
        else:
            self.linear_punish_reward = 0

        self.arrive_reward = 0
        if distance_turtlebot_target<1:
            # rospy.loginfo("Robot Arrive Goal!!!!!!!!!")
            self.arrive_reward = 400
            # self.reset()
            self.stop_robot()


        reward  = distance_reward*(5/time_step)*1.2*7 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward+self.angular_osa_punish_reward

        return reward, state, self.laser_crashed_value,self.arrive_reward





