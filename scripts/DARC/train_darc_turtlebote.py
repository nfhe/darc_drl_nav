# coding=utf-8
# tensorboard --logdir "/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/DARC/2023-04-04" --port 6006

import numpy as np
import torch
import os
import random
import datetime
import darc_utils
import DARC
from torch.utils.tensorboard import SummaryWriter
import turtlebot_turtlebot3_darc_env
import rospy
import time

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

# 生成参数类
class Args:
	def __init__(self,env):
		self.env = env
		self.env_name = "Turtlebot3Env-v0"
		self.policy = "DARC"
		self.seed = 1234
		self.start_steps = 1e4
		self.eval_freq = 5e3
		self.max_timesteps = 1e6
		self.discount = 0.99
		self.tau =  0.005
		self.policy_noise = 0.2
		self.noise_clip = 0.5
		self.policy_freq = 2
		self.batch_size = 256	#不同
		self.expl_noise = 0.1
		self.replay_size = 2e6
		self.save_model = True
		self.save_freq = 1e4
		self.hidden_sizes = '300,400,400'
		self.actor_lr = 0.0003
		self.critic_lr = 0.0003
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.max_episode_steps = 1000
		self.load_model = False

		self.current_time = datetime.datetime.now().strftime("%Y-%m-%d")
		# self.model_path = '/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/DCEERDDPG/'
		self.model_path = '/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/2023-03-14/corridor/DARC/2023-04-13/DARC/'
		self.actor1_path = self.model_path + 'actor1/'
		self.actor2_path = self.model_path + 'actor2/'
		self.critic1_path = self.model_path + 'critic1/'
		self.critic2_path = self.model_path + 'critic2/'
		self.final_num_trials = 500
		self.final_trial_len = 30
		self.num_trials = 500
		self.trial_len = 700

		# Extract environment information
		self.state_dim = env.observation_space.shape[0]	#28
		self.action_dim = env.action_space.shape[0]	#2
		self.max_action = float(1.0)


if __name__ == "__main__":
		########################################################
	game_state= turtlebot_turtlebot3_darc_env.GameState()   # game_state has frame_step(action) function
	# Create a parser for robot.
	args = Args(env=game_state)
	print(args.actor1_path,args.actor2_path)
	print("------------------------------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env_name, args.seed))
	print("------------------------------------------------------------")

	kwargs = {
		"state_dim": args.state_dim,
		"action_dim": args.action_dim,
		"max_action": args.max_action,
		"discount": args.discount,
		"tau": args.tau,
		"hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
		"actor_lr": args.actor_lr,
		"critic_lr": args.critic_lr,
		"device": args.device,
		"env":game_state,
	}

	# Create a model for DARC.
	policy = DARC.DARC(**kwargs)

	for i in range(args.num_trials):
		print("************************************************")
		print("trials number:" + str(i))

		# reset the environment
		current = game_state.reset()

		policy.load(args.actor1_path,args.actor2_path,args.critic1_path,args.critic2_path,args.final_num_trials,args.final_trial_len)
		##############################################################################################
		args.total_reward = 0

		for j in range(args.trial_len):
			# print("trials length:" + str(j))
			###########################################################################################

			start_time = time.time()

			# select action randomly or according to policy
			action = policy.select_action(current)

			end_time = time.time()
			# if j%50 == 0:
			# 	print(1/(end_time - start_time), "fps for calculating next step")
			# print("action:",action)
			#Obtain the next  state based on the current action
			reward,next_state, crashed_value,arrive_reward = game_state.game_step(0.1, action[1], action[0])

			if j == (args.trial_len - 1):
				crashed_value = 1
				print("this is total reward:", args.total_reward)

			# Gain cumulative rewards
			args.total_reward += reward

			# update current state
			current = next_state

            # if the robot collides with obstacles or arrives the goal, the trial ends
			if crashed_value == 1:
				rospy.loginfo("Robot collides with obstacles!!!")
				break
			if arrive_reward  >= 100:
				rospy.loginfo("Robot arrives the goal!!!")
				break







