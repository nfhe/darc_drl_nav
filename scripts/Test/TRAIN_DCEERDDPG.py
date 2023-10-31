# coding=utf-8

import copy
import numpy as np
import train_utils
# import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(SEED)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.l4 = nn.Linear(hidden_sizes[2], action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.max_action * torch.tanh(self.l4(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.l4 = nn.Linear(hidden_sizes[2], 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        return self.l4(q)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        return q1

class DCEERDDPG(object):
	def __init__(
        self,
		state_dim,
		action_dim,
		max_action,
		device,
		env,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
        policy_freq=2,
		actor_lr=1e-3,
		critic_lr=1e-3,
		hidden_sizes=[400, 300],
		epsilon = 0.9,
		epsilon_decay = .99995,
		expl_noise = 0.1,

    ):

		self.device = device
		self.env = env

		self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic1 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

		self.critic2 = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

		self.max_action = max_action + 0.5
		self.min_action = -self.max_action

		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.action_dim = action_dim
		self.expl_noise = expl_noise
		self.policy_freq = policy_freq

		self.total_it = 0
		self.buffer_proportion = 1.75

	def read_dc_Q_values(self, state,action):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
		q1 = self.critic1(state, action)
		q2 = self.critic2(state, action)
		q = torch.min(q1,q2)
		return q

	def act_dc_ddpg(self, state):
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(0.01, self.epsilon)
		if np.random.rand() < self.epsilon:
			action = (self.max_action - self.min_action) * np.random.random(self.env.action_space.shape[0]) + self.min_action
		else:
			action = (
				self.select_action(np.array(state)) + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
			).clip(-self.max_action, self.max_action)
		return action

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size,buffer_size):
		batch_size = batch_size*self.buffer_proportion
		if buffer_size < batch_size:
			# raise ValueError("Not enough samples to perform a training step")
			return

        # Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.add_sample(batch_size,self.buffer_proportion)

		with torch.no_grad():
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
			target_Q1 = self.critic1_target(next_state, next_action)
			target_Q2 = self.critic2_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		current_Q1 = self.critic1(state, action)
		current_Q2 = self.critic2(state, action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)


		self.critic1_optimizer.zero_grad()
		self.critic2_optimizer.zero_grad()
		critic_loss.backward()
		self.critic1_optimizer.step()
		self.critic2_optimizer.step()

		actor_loss = -self.critic1.Q1(state, self.actor(state)).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, actor_path,critic1_path,critic2_path,num_trials,i):
		torch.save(self.critic1.state_dict(), critic1_path +'critic1_model'+ '-' + str(i)+ '-' +str(num_trials))
		torch.save(self.critic1_optimizer.state_dict(), critic1_path + 'critic1_optimizer'+ '-' + str(i)+ '-' +str(num_trials))
		torch.save(self.critic2.state_dict(), critic2_path +'critic2_model'+ '-' + str(i)+ '-' +str(num_trials))
		torch.save(self.critic2_optimizer.state_dict(), critic2_path + 'critic2_optimizer'+ '-' + str(i)+ '-' +str(num_trials))

		torch.save(self.actor.state_dict(), actor_path + 'actor_model'+ '-' + str(i)+ '-' +str(num_trials))
		torch.save(self.actor_optimizer.state_dict(), actor_path + 'actor_optimizer'+ '-' + str(i)+ '-' +str(num_trials))

	def load(self, actor_path,critic1_path,critic2_path,num_trials,i):
		self.critic1.load_state_dict(torch.load(critic1_path +'critic1_model'+ '-' + str(i)+ '-' +str(num_trials)))
		self.critic1_optimizer.load_state_dict(torch.load(critic1_path + 'critic1_optimizer'+ '-' + str(i)+ '-' +str(num_trials)))
		self.critic2.load_state_dict(torch.load(critic2_path +'critic2_model'+ '-' + str(i)+ '-' +str(num_trials)))
		self.critic2_optimizer.load_state_dict(torch.load(critic2_path + 'critic2_optimizer'+ '-' + str(i)+ '-' +str(num_trials)))

		self.actor.load_state_dict(torch.load(actor_path + 'actor_model'+ '-' + str(i)+ '-' +str(num_trials)))
		self.actor_optimizer.load_state_dict(torch.load(actor_path + 'actor_optimizer'+ '-' + str(i)+ '-' +str(num_trials)))




