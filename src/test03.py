# coding=utf-8
import csv
import matplotlib.pyplot as plt
import numpy as np

# //读取csv文件
def read_csv(filename):
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

# //保存npy文件
def save_npy(data):
    np.save('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/step_Q.npy', data)

dateFile = '/media/he/KINGSTON/避障/ddpg_collision/weight/2023-03-14/maze/dcddpg/run-.-tag-Main_Episode_Q_values.csv'
data = read_csv(dateFile)
result = []
for i in range(1,len(data)):
    result.append(data[i][2])
save_npy(result)