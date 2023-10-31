# coding=utf-8
# //读取mat文件
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

def read_mat():
    # dateFile = '/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/daddpg/2023-03-20/step_reward.mat'
    # dateFile = '/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/daddpg/2023-03-20/step_Q.mat'
    dateFile = '/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/daddpg/2023-03-20/step_rewards.mat'
    mat_contents = sio.loadmat(dateFile)
    data = mat_contents['data']
    data = np.array(data).flatten()
    return data

# //保存npy文件
def save_npy(data):
    # np.save('/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/daddpg/2023-03-20/step_reward.npy', data)
    # np.save('/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/daddpg/2023-03-20/step_Q_values.npy', data)
    np.save('/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/daddpg/2023-03-20/sum_reward.npy', data)

data = read_mat()
result = []
for i in range(1,len(data)):
    if i%2 != 0:
        result.append(float(data[i]))

save_npy(result)