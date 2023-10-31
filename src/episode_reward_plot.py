# coding=utf-8

# DDPG TD3 SAC 训练曲线
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

# plt.rc('font',family='Times New Roman')
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

plt.figure(figsize=(5,3))
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率

ax = plt.gca()

ax.tick_params(bottom=False,top=False,left=False,right=False)

# 设置网格颜色
ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5, zorder=0)

# 设置背景颜色
# ax.patch.set_facecolor("#e9e9f2")

# 去除边框
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['right'].set_visible(False)

# 加载数据
def load(policy, num=3):
    url = os.path.dirname(os.path.realpath(__file__))
    result = []
    for i in range(num):
        temp = np.load("/home/he/下载/训练数据/corridor/"+policy+"/step_reward.npy")
        # temp = np.load("/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/DCDDPG对比/"+policy+"/step_reward.npy")
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
    n = sys.maxsize
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

# def get_mean_max_min(data_list, smooth_flag, fineness):
#     max_data = []
#     min_data = []
#     mean_data = []
#     a = []
#     m = 0
#     for i in range(len(data_list[0])):
#         m = m + data_list[0][i]
#         a.append(data_list[0][i])
#         if i % fineness == 0:
#             ma = np.max(a)
#             mi = np.min(a)
#             max_data.append(ma)
#             min_data.append(mi)
#             mean_data.append(m/fineness)
#             m = 0
#             a = []

#     data = [mean_data, max_data, min_data]
#     data = np.array(data)
#     if smooth_flag:
#         for i in range(len(data)):
#             for j in range(2, fineness):
#                 data[i] = smooth(data[i], j)
#     return data[0], data[1], data[2]

DC_data = load("DCDDPG", 3)
DCEER_data = load("DCEERDDPG(无human)", 3)
DCEERHUMAN_data = load("DCEERDDPG(有human)", 3)

fineness = 16

SAC_mean_data, SAC_max_data, SAC_min_data = get_mean_max_min(DC_data, True, fineness)
TD3_mean_data, TD3_max_data, TD3_min_data = get_mean_max_min(DCEER_data, True, fineness)
DDPG_mean_data, DDPG_max_data, DDPG_min_data = get_mean_max_min(DCEERHUMAN_data, True, fineness)

SAC_x = range(SAC_mean_data.size)
plt.fill_between(SAC_x, SAC_min_data, SAC_max_data, alpha=0.2, zorder=2, color="blue")
plt.plot(SAC_x, SAC_mean_data, linewidth=2, label="DC", zorder=3, color="blue")

TD3_x = range(TD3_mean_data.size)
plt.fill_between(TD3_x, TD3_min_data, TD3_max_data, alpha=0.2, zorder=2, color="red")
plt.plot(TD3_x, TD3_mean_data, linewidth=2, label="DCEER", zorder=3, color="red")

DDPG_x = range(DDPG_mean_data.size)
plt.fill_between(DDPG_x, DDPG_min_data, DDPG_max_data, alpha=0.2, zorder=2, color="green")
plt.plot(DDPG_x, DDPG_mean_data, linewidth=2, label="DCEERHUMAN", zorder=3, color="green")

# plt.title("Training curve", pad=15)
plt.xlabel("Episode", labelpad=8.5)
plt.ylabel("Accumulated Reward", labelpad=8.5)

plt.tight_layout()

plt.legend(loc="lower right", frameon=True)

# Q_values
# plt.savefig('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/DCDDPG对比/episode_Q_values_plot.jpg')
# reward
plt.savefig('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/DCDDPG对比/episode_reward_plot.jpg')
plt.show()
