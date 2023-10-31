# coding=utf-8

from tensorboard.backend.event_processing import event_accumulator        # 导入tensorboard的事件解析器
import numpy as np
import matplotlib.pyplot as plt

# 地址
address = "/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/狭小长廊环境下移动机器人自主导航/comparison of p-eerm-darc/PEERM_DARC"
# 初始化EventAccumulator对象
ea=event_accumulator.EventAccumulator(address, size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE)

ea.Reload()    # 这一步是必须的，将事件的内容都导进去
print(ea.scalars.Keys())    # 我们知道tensorboard可以保存Image scalars等对象，我们主要关注scalars
Episode_reward_loss = ea.scalars.Items("Main/Episode_reward")    # 读取train_reward
Episode_Q_values_loss = ea.scalars.Items("Main/Episode_Q_values")    # 读取train_Q_values
Sum_reward_loss = ea.scalars.Items("Main/Sum_reward")    # 读取train_loss

# [u'Main/reward', u'Main/Q_value', u'Main/total_reward']-replay-original-DADDPG
# Episode_reward_loss = ea.scalars.Items("Main/reward")    # 读取train_reward
# Episode_Q_values_loss = ea.scalars.Items("Main/Q_value")    # 读取train_Q_values
# Sum_reward_loss = ea.scalars.Items("Main/total_reward")    # 读取train_total_reward

# [u'Main/Sum_reward', u'Main/Episode_reward]'-PPO
# Episode_reward_loss = ea.scalars.Items("Main/Episode_reward")    # 读取train_reward
# Sum_reward_loss = ea.scalars.Items("Main/Sum_reward")    # 读取train_reward

# //保存npy文件
def save_npy1(data):
    np.save('/home/he/catkin_nav/src/1/step_reward.npy', data)
def save_npy2(data):
    np.save('/home/he/catkin_nav/src/1/step_Q_values.npy', data)
def save_npy3(data):
    np.save('/home/he/catkin_nav/src/1/sum_reward.npy', data)
def save_npy():
    save_npy1([x.value for x in Episode_reward_loss])
    save_npy2([x.value for x in Episode_Q_values_loss])
    save_npy3([x.value for x in Sum_reward_loss])
save_npy()
