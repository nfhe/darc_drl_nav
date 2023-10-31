import numpy as np
import torch
from collections import deque
import scipy.io as sio

SEED = 1234
np.random.seed(SEED)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim,  max_size=int(1e6)):
		self.memory = deque(maxlen=40000)

	def remember(self, cur_state, action, reward, new_state, done):
		cur_state = cur_state.reshape(28)
		action = action.reshape(2)
		self.array_reward = np.array(reward)
		self.array_reward = self.array_reward.reshape(1)
		new_state = new_state.reshape(28)
		done = np.array(done)
		done = done.reshape(1)
		self.memory_pack = np.concatenate((cur_state, action))
		self.memory_pack = np.concatenate((self.memory_pack, self.array_reward))
		self.memory_pack = np.concatenate((self.memory_pack, new_state))
		self.memory_pack = np.concatenate((self.memory_pack, done))
		self.memory.append(self.memory_pack)

		print("self.memory length is %s", len(self.memory))

		if len(self.memory)%10==0:
			sio.savemat('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/record_data/record_data.mat',{'data':self.memory},True,'5', False, False,'row')


