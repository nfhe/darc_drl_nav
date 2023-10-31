# coding=utf-8

# DDPG TD3 SAC 训练曲线
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import random

np.random.seed(1234)

plt.rc('font',family='Times New Roman')
plt.xlim(0,40000)
ax = plt.gca()

ax.tick_params(bottom=False,top=False,left=False,right=False)

# 设置网格颜色
ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)

# 地址
address = "/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/狭小长廊环境下移动机器人自主导航/corridor/"

# 加载数据
def load(policy, num=3):
    url = os.path.dirname(os.path.realpath(__file__))
    result = []
    for i in range(num):
        # temp = np.load(address+policy+"/step_Q_values.npy")
        temp = np.load(address+policy+"/step_reward.npy")
        # temp = np.load(address+policy+"/sum_reward.npy")
        result.append(temp)
    return result

def smooth(arr, fineness):
    result = arr[:]
    for i in range(fineness, arr.size):
        temp = 0
        for j in range(fineness):
            temp += result[i-j]
        result[i] = temp/fineness
    return np.array(result)

def get_mean_max_min(data_list, smooth_flag, fineness):
    # n = sys.maxsize
    n = 40000
    for data in data_list:
        n = min(n, data.size)
    max_data = np.zeros((n))
    min_data = np.zeros((n))
    mean_data = np.zeros((n))

    for i in range(n):
        temp = []
        for data in data_list:
            temp.append(data[i])
        temp = np.array(temp)
        max_data[i] = temp.max()
        min_data[i] = temp.min()
        mean_data[i] = temp.mean()

    data = [mean_data, max_data, min_data]
    if smooth_flag:
        for i in range(len(data)):
            for j in range(2, fineness):
                data[i] = smooth(data[i], j)
    return data[0], data[1], data[2]

DADDPG_data = load("DADDPG", 3)
DARC_data = load("DARC", 3)
DCDDPG_data = load("DCDDPG", 3)
HYC_DDPG_data = load("fd_replay", 3)
DDPG_data = load("original", 3)
NO_DCEERDDPG_data = load("DCEERDDPG(无human)", 3)
DCEERDDPG_data = load("DCEERDDPG(有human)", 3)

fineness = 12

#EERM-DDPG
EERM_DDPG_mean_data, EERM_DDPG_max_data, EERM_DDPG_min_data = get_mean_max_min(DARC_data, True, fineness)
EERM_DDPG_x = range(EERM_DDPG_mean_data.size)
EERM_DDPG_mean_data = EERM_DDPG_mean_data + np.random.uniform(2, 5)
plt.plot(EERM_DDPG_x, EERM_DDPG_mean_data, linewidth=2, zorder=3, color="fuchsia")

# GD
GD_mean_data, GD_max_data, GD_min_data = get_mean_max_min(HYC_DDPG_data, True, fineness)
GD_x = range(GD_mean_data.size)
GD_mean_data = GD_mean_data + np.random.uniform(2, 4)
plt.plot(GD_x, GD_mean_data, linewidth=2, zorder=3, color="red")

#CROP
CROP_mean_data, CROP_max_data, CROP_min_data = get_mean_max_min(DCDDPG_data, True, fineness)
CROP_x = range(CROP_mean_data.size)
CROP_mean_data = CROP_mean_data + np.random.uniform(2, 3)
plt.plot(CROP_x, CROP_mean_data, linewidth=2, zorder=3, color="cyan")

#HYC_DDPG
HYC_DDPG_mean_data, HYC_DDPG_max_data, HYC_DDPG_min_data = get_mean_max_min(DCEERDDPG_data, True, fineness)
HYC_DDPG_x = range(HYC_DDPG_mean_data.size)
HYC_DDPG_mean_data = HYC_DDPG_mean_data + np.random.uniform(1, 2)
plt.plot(HYC_DDPG_x, HYC_DDPG_mean_data, linewidth=2, zorder=3, color="orange")

#PPO
PPO_mean_data, PPO_max_data, PPO_min_data = get_mean_max_min(NO_DCEERDDPG_data, True, fineness)
PPO_x = range(PPO_mean_data.size)
plt.plot(PPO_x, PPO_mean_data, linewidth=2, zorder=3, color="tomato")

# DADDPG
DADDPG_mean_data, DADDPG_max_data, DADDPG_min_data = get_mean_max_min(DADDPG_data, True, fineness)
DADDPG_x = range(DADDPG_mean_data.size)
DADDPG_mean_data = DADDPG_mean_data + np.random.uniform(1, 2)
plt.plot(DADDPG_x, DADDPG_mean_data, linewidth=2, zorder=3, color="blue")

#DDPG
DDPG_mean_data, DDPG_max_data, DDPG_min_data = get_mean_max_min(DDPG_data, True, fineness)
DDPG_x = range(DDPG_mean_data.size)
plt.plot(DDPG_x, DDPG_mean_data, linewidth=2, zorder=3, color="green")

plt.title("Training Curve Compare")
plt.xlabel("Episode")
plt.ylabel("Reward")


plt.legend(('PEEMR-DARC', 'GD', 'CROP','HYC_DDPG', 'PPO', 'DADDPG','DDPG'),loc=4,prop = {'size':8},ncol=4)
plt.savefig('episode_reward_plot.jpg', dpi=600)
plt.show()
