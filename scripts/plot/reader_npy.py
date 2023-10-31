# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os

# //读取npy文件
def read_npy():
    # data = np.load('/home/he/catkin_nav/src/1/step_reward.npy')
    data = np.load('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/狭小长廊环境下移动机器人自主导航/corridor/fd_replay/step_Q_values.npy')
    # data = np.load('/home/he/catkin_nav/src/1/sum_reward.npy')
    return data

data = read_npy()
print("data:",len(data))
plt.plot(data, 'g')
plt.show()


