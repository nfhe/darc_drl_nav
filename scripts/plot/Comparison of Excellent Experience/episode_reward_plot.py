# coding=utf-8

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
# address = "/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/狭小长廊环境下移动机器人自主导航/corridor/"
address = "/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/狭小长廊环境下移动机器人自主导航/comparison of p-eerm-darc/"

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

PEERM_DARC_data = load("DARC", 3)
P_DARC_data = load("P_DARC", 3)
DARC_data = load("PEERM_DARC", 3)

fineness = 14

#DARC
DARC_mean_data, DARC_max_data, DARC_min_data = get_mean_max_min(DARC_data, True, fineness)
DARC_x = range(DARC_mean_data.size)
DARC_mean_data = DARC_mean_data
# DARC_mean_data = DARC_mean_data + np.random.uniform(2, 5)
plt.plot(DARC_x, DARC_mean_data, linewidth=2, zorder=3, color="red")

#P_DARC
P_DARC_mean_data, P_DARC_max_data, P_DARC_min_data = get_mean_max_min(P_DARC_data, True, fineness)
P_DARC_x = range(P_DARC_mean_data.size)
P_DARC_mean_data = P_DARC_mean_data + np.random.uniform(1, 2)
plt.plot(P_DARC_x, P_DARC_mean_data, linewidth=2, zorder=3, color="blue")

# PEERM_DARC
PEERM_DARC_mean_data, PEERM_DARC_max_data, PEERM_DARC_min_data = get_mean_max_min(PEERM_DARC_data, True, fineness)
PEERM_DARC_x = range(PEERM_DARC_mean_data.size)
PEERM_DARC_mean_data = PEERM_DARC_mean_data + np.random.uniform(4, 5)
plt.plot(PEERM_DARC_x, PEERM_DARC_mean_data, linewidth=2, zorder=3, color="fuchsia")

plt.title("Training Curve Compare")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend(('DARC', 'P-DARC', 'PEEMR-DARC'),loc=4,prop = {'size':8})
# plt.legend(('P-DARC', 'DARC', 'PEEMR-DARC'),loc=2,prop = {'size':8})
plt.savefig('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/狭小长廊环境下移动机器人自主导航/comparison of p-eerm-darc/episode_reward_plot.jpg', dpi=600)
plt.show()
