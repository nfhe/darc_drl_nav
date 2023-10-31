# coding=utf-8
#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist,Pose2D, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import  pi, atan2
import time

class turtlebot3_pi:
    def __init__(self):
        self.node = rospy.init_node('reading_laser', anonymous=False)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.callback_pose, queue_size=10)
        self.sub = rospy.Subscriber('/scan', LaserScan, self.callback_laser)
        self.target_point = rospy.Subscriber('/target_point', Point, self.callback_target_point,queue_size=10)

        self.robot_pose = Pose2D()
        self.laser = LaserScan()
        self.rate = rospy.Rate(10)
        self.target_x = 12
        self.target_y = 00
        self.threshold_dist = 1.0
        self.linear_speed = 0.26
        self.angular_speed = 0.5
        self.k =0.3
        self.msg = Twist()
        self.linear_x = 0.0
        self.angular_z = 0
        self.regions={'right':1,'front':1,'left':1}

    def callback_target_point(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y

    def callback_laser(self, msg):
        # 120 degrees into 3 regions
        # 分为右、中、左三个区域，取每一个区域最小的值，然后再与数值10作比较，取最小值
        regions = {
        'right':  min(min(msg.ranges[3:8]), 10),
        'front':  min(min(msg.ranges[9:15]), 10),
        'left':   min(min(msg.ranges[16:20]), 10),
        }
        self.regions = regions
        # self.take_action(regions)

    def callback_pose(self, msg_odometry):
        self.robot_pose.x = msg_odometry.pose.pose.position.x
        self.robot_pose.y = msg_odometry.pose.pose.position.y
        quat_orig = msg_odometry.pose.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        self.robot_pose.theta = yaw

    def take_action(self, regions):
        # # 计算移动机器人当前点到目标点的角度偏差
        angle_turtlebot_target = atan2(self.target_y - self.robot_pose.y, self.target_x- self.robot_pose.x)
        angle_diff = angle_turtlebot_target - self.robot_pose.theta
        position_diff_x = self.target_x - self.robot_pose.x
        position_diff_y = self.target_y - self.robot_pose.y
        dis_diff = pow(pow(position_diff_x, 2) + pow(position_diff_y, 2), 0.5)
        self.linear_x = self.linear_speed
        if angle_diff > pi:
            angle_diff = angle_diff - 2 * pi
        elif angle_diff < -pi:
            angle_diff = angle_diff + 2 * pi
        if angle_diff>0:
            self.angular_z = abs(angle_diff) * self.k
        else:
            self.angular_z = -abs(angle_diff) * self.k
        if angle_diff<0.1 and angle_diff>-0.1:
            self.angular_z = 0
        if dis_diff<1:
            self.msg.linear.x = 0
            self.msg.angular.z = 0
            self.pub.publish(self.msg)
            time.sleep(5)

        state_description = ''
        # 三个区域都没有障碍物
        if regions['front'] > self.threshold_dist and regions['left'] > self.threshold_dist and regions['right'] > self.threshold_dist:
            state_description = 'case 1 - no obstacle'
        # 三个区域都有障碍物
        elif regions['front'] < self.threshold_dist and regions['left'] < self.threshold_dist and regions['right'] < self.threshold_dist:
            state_description = 'case 7 - front and left and right'
            self.linear_x = self.linear_speed
            self.angular_z = self.angular_speed*2 # Increase this angular speed for avoiding obstacle faster
        # 中间有障碍物
        elif regions['front'] < self.threshold_dist and regions['left'] > self.threshold_dist and regions['right'] > self.threshold_dist:
            state_description = 'case 2 - front'
            self.linear_x = self.linear_speed
            self.angular_z = self.angular_speed
        # 右边有障碍物
        elif regions['front'] > self.threshold_dist and regions['left'] > self.threshold_dist and regions['right'] < self.threshold_dist:
            state_description = 'case 3 - right'
            self.linear_x = self.linear_speed
            self.angular_z = self.angular_speed
        # 左边有障碍物
        elif regions['front'] > self.threshold_dist and regions['left'] < self.threshold_dist and regions['right'] > self.threshold_dist:
            state_description = 'case 4 - left'
            self.linear_x = self.linear_speed
            self.angular_z = -self.angular_speed
        # 中间和右边有障碍物
        elif regions['front'] < self.threshold_dist and regions['left'] > self.threshold_dist and regions['right'] < self.threshold_dist:
            state_description = 'case 5 - front and right'
            self.linear_x = self.linear_speed
            self.angular_z = self.angular_speed
        # 中间和左边有障碍物
        elif regions['front'] < self.threshold_dist and regions['left'] < self.threshold_dist and regions['right'] > self.threshold_dist:
            state_description = 'case 6 - front and left'
            self.linear_x = self.linear_speed
            self.angular_z = -self.angular_speed
        # 左边和右边有障碍物
        elif regions['front'] > self.threshold_dist and regions['left'] < self.threshold_dist and regions['right'] < self.threshold_dist:
            state_description = 'case 8 - left and right'
            self.linear_x = self.linear_speed
            self.angular_z = 0
        else:
            state_description = 'unknown case'
            # rospy.loginfo(regions)

        # rospy.loginfo(state_description)
        # rospy.loginfo(self.angular_z)
        self.msg.linear.x = self.linear_x
        self.msg.angular.z = self.angular_z
        # self.msg.linear.x = 0
        # self.msg.angular.z = 0
        self.pub.publish(self.msg)
        self.rate.sleep()

    def main(self):
        while not rospy.is_shutdown():
            while 1:
                turtlebot3.take_action(turtlebot3.regions)
                turtlebot3.rate.sleep()
            rospy.spin()

if __name__ == '__main__':
    turtlebot3 = turtlebot3_pi()
    turtlebot3.main()
