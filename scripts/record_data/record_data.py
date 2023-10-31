#! /usr/bin/env python
# coding=utf-8

import gym
import numpy as np
import random
from collections import deque
import os.path
import timeit
import csv
import math
import time
import sys
import rospy
import scipy.io as sio
import turtlebot_turtlebot3_record_data_env
import datetime
import record_utils

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)

# 生成参数类
class Args:
	def __init__(self,env):
		self.env = env
		self.env_name = "Turtlebot3Turtlebot3Env-v0"
		self.policy = "record_data"
		self.seed = 1234
		self.start_steps = 1e4
		self.eval_freq = 5e3
		self.max_timesteps = 1e6
		self.discount = 0.99
		self.tau =  0.005
		self.policy_noise = 0.2
		self.noise_clip = 0.5
		self.policy_freq = 2
		self.batch_size = 256
		self.expl_noise = 0.1
		self.replay_size = 1e6
		self.save_freq = 1e4

		self.current_time = datetime.datetime.now().strftime("%Y-%m-%d")
		self.dir ='/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time
		self.record_data_path = r'/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy)
		self.record_data_mkdir()
		self.num_trials = 1000
		self.trial_len = 500
		self.counter = 0
		self.five_average_reward = 0
		self.five_average_Q_values = 0
		self.total_reward = 0
		self.step = 0
		self.buffer_size = 0

		# Extract environment information
		self.state_dim = env.observation_space.shape[0]	#28
		self.action_dim = env.action_space.shape[0]	#2
		self.max_action = float(1.0)

	def record_data_mkdir(self):
		record_data_mkdir_exists =os.path.exists(self.record_data_path)
		if record_data_mkdir_exists ==True:
			pass
		else:
			os.mkdir(self.record_data_path)

def main():
    # Create environment
    game_state = turtlebot_turtlebot3_record_data_env.GameState()
	# Create a parser for robot.
    args = Args(game_state)
    # Create a replay buffer
    replay_buffer = record_utils.ReplayBuffer(args.state_dim, args.action_dim)
	# Create environment
    time.sleep(0.5)
    current_state = game_state.reset()

    for i   in range(args.num_trials):
        print("************************************************")
        print("trials number:" + str(i))
		##############################################################################################
        for j in range(args.trial_len):
			# 发送当前目标点
            game_state.pub_target_point.publish(game_state.target_point)

			# 得到当前状态
            current = game_state.read_state()
            current_state = current_state.reshape((1, game_state.observation_space.shape[0]))

			# 读取当前动作
            action = game_state.read_action()
            action = action.reshape((1, game_state.action_space.shape[0]))

			# 读取当前奖励，下一时刻状态，是否碰撞，是否到达目标点
            reward, new_state, crashed_value,arrive_reward = game_state.read_game_step(0.1, action[0][1], action[0][0])

			# 将当前状态，动作，奖励，下一时刻状态，是否碰撞存入replay_buffer
            replay_buffer.remember(current_state, action, reward, new_state, crashed_value)

            time.sleep(2)
            if rospy.is_shutdown():
                print('shutdown')
                break


if __name__ == "__main__":
	main()