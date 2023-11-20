# coding=utf-8

import numpy as np
import torch
import scipy.io as sio


SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(SEED)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device,max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.device = device

		#添加参数
		self.mat_content = sio.loadmat('/home/he/catkin_nav/src/turtlebot3_ddpg_nav/weight/human_data/2023-02-21/human_data.mat')
		self.experience_pool_data = self.mat_content['data']
		self.data_length = len(self.experience_pool_data)
		self.insert_number = 0
		self.new_buffer = []
		self.experience_pool_sort()

	def experience_pool_sort(self):
		# new_buffer添加新数据
		for i in range(self.data_length):
			self.new_buffer.append(self.experience_pool_data[i])

		# 把数据添加到buffer中
		for i in range(self.data_length):
			cur_state = self.new_buffer[i][0:28]
			action = self.new_buffer[i][28:30]
			reward = self.new_buffer[i][30]
			new_state = self.new_buffer[i][31:59]
			done = self.new_buffer[i][59]
			# self.add_experience_buffer(cur_state, action, new_state, reward, done)
			self.add(cur_state, action, new_state, reward, done)

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		return self.size


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)