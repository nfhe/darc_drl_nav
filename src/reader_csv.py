# coding=utf-8
import csv
import matplotlib.pyplot as plt
import numpy as np

# //读取scv文件
def read_csv(filename):
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

# //保存npy文件
def save_npy(data):
    # np.save('/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/DARC/2023-04-11/step_reward.npy', data)
    # np.save('/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/DARC/2023-04-11/step_Q_values.npy', data)
    np.save('/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/DARC/2023-04-11/sum_reward.npy', data)

# dateFile = '/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/DARC/2023-04-11/run-.-tag-Main_Episode_reward.csv'
# dateFile = '/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/DARC/2023-04-11/run-.-tag-Main_Episode_Q_values.csv'
dateFile = '/media/he/UBUNTU 18_0/避障/ddpg_collision/weight/2023-03-14/corridor/DARC/2023-04-11/run-.-tag-Main_Sum_reward.csv'
data = read_csv(dateFile)
result = []
for i in range(1,len(data)):
    result.append(float(data[i][2]))
save_npy(result)