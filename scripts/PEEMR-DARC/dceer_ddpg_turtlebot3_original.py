# coding=utf-8
# tensorboard --logdir "/home/dog/catkin_avoidance/src/turtlebot3_ddpg_nav/weight/PEEMR-DARC/2023-10-25" --port 6006 --samples_per_plugin scalars=0

import numpy as np
import torch
import os
import random
import datetime
import peemr_darc_ddpg_utils
import PEEMR_DARC
from torch.utils.tensorboard import SummaryWriter
import turtlebot_turtlebot3_peemr_darcddpg_env
import rospy

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
		self.env_name = "Turtlebot3Turtlebot3Env-v0"
		self.policy = "PEEMR-DARC"
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
		self.dir ='/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time
		self.logs_dir ='/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/'
		self.peemr_darc_path = r'/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/PEEMR-DARC/'
		self.model_path = '/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/PEEMR-DARC/'
		self.actor_mkdir_path = r'/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/PEEMR-DARC/actor/'
		self.critic1_mkdir_path = r'/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/PEEMR-DARC/critic1/'
		self.critic2_mkdir_path = r'/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/{}/'.format(self.policy) + self.current_time +'/PEEMR-DARC/critic2/'
		self.actor_path = self.model_path + 'actor/'
		self.critic1_path = self.model_path + 'critic1/'
		self.critic2_path = self.model_path + 'critic2/'
		self.writer = SummaryWriter(self.dir)
		self.peemr_darc_mkdir()
		self.num_trials = 1000
		self.trial_len = 700
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

	def peemr_darc_mkdir(self):
		peemr_darc_mkdir_exists =os.path.exists(self.peemr_darc_path)
		if peemr_darc_mkdir_exists ==True:
			pass
		else:
			os.mkdir(self.peemr_darc_path)
			os.mkdir(self.actor_mkdir_path)
			os.mkdir(self.critic1_mkdir_path)
			os.mkdir(self.critic2_mkdir_path)

if __name__ == "__main__":
		########################################################
	game_state= turtlebot_turtlebot3_peemr_darcddpg_env.GameState()   # game_state has frame_step(action) function
	# Create a parser for robot.
	args = Args(env=game_state)
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

	# Create a model for PEEMR-DARC.
	policy = PEEMR_DARC.PEEMR_DARC(**kwargs)

	# check if the model is loaded from a file
	# if args.load_model is not None:
	# 	policy.load("./models/{}".format(args.load_model))

	## write logs to record training parameters
	with open(args.logs_dir + 'log.txt','w') as f:
		f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
		for item in kwargs.items():
			f.write('\n {}'.format(item))

    # Create a experience replay buffer.
	replay_buffer = peemr_darc_ddpg_utils.ReplayBuffer(args.state_dim, args.action_dim,args.device)

	for i in range(args.num_trials):
		print("************************************************")
		print("trials number:" + str(i))

		# reset the environment
		current = game_state.reset()
		##############################################################################################
		args.total_reward = 0

		for j in range(args.trial_len):
			args.counter += 1
			###########################################################################################
			# print("trials length:" + str(j))

			# select action randomly or according to policy
			action = policy.act_peerm_darc(current)

			#Obtain the next  state based on the current action
			reward,next_state, crashed_value,arrive_reward = game_state.game_step(0.1, action[1], action[0])

			# Gain cumulative rewards
			args.total_reward += reward

			# Add the experience to the replay buffer
			args.buffer_size  = replay_buffer.add(current, action, next_state, reward, crashed_value)

			if j == (args.trial_len - 1):
				crashed_value = 1
				print("this is total reward:", args.total_reward)

			args.five_average_reward = args.five_average_reward + reward
			args.five_average_Q_values = args.five_average_Q_values + policy.read_peemr_darc_Q_values(current, action)

			if args.counter % 5 == 0:
				args.step += 1
				args.five_average_reward = args.five_average_reward / 5
				args.five_average_Q_values = args.five_average_Q_values / 5
				args.writer.add_scalar('Main/Episode_reward', args.five_average_reward, args.step)
				args.writer.add_scalar('Main/Episode_Q_values', args.five_average_Q_values, args.step)
				args.five_average_reward = 0
				args.five_average_Q_values = 0

			if (j % 5 == 0):
				policy.train(replay_buffer, args.batch_size,args.buffer_size)

			if args.step % 200 == 0:
				rospy.loginfo("UPDATE TARGET NETWORK!")

			# update current state
			current = next_state

			if crashed_value == 1:
				rospy.loginfo("Robot collides with obstacles!!!")
				game_state.stop_robot()
				break
			if arrive_reward  >= 100:
				rospy.loginfo("Robot arrives the goal!!!")
				game_state.stop_robot()
				break

		# Save the total reward
		args.writer.add_scalar('Main/Sum_reward', args.total_reward, i)

		# Save the model
		if i % 10 == 0:
			policy.save(args.actor_path,args.critic1_path,args.critic2_path, args.num_trials,i)

# args.writer.close()








