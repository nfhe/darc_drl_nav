# tensorboard --logdir=./logs
# coding=utf-8
import numpy as np
import scipy.io as sio
import datetime
import tensorflow as tf
import time
#tensorboard --logdir "/home/he/catkin_nav/src/Comparative_experiment/log/2023-02-24/logs/plot_1" --port 6006
#tensorboard --logdir "/home/he/catkin_nav/src/Comparative_experiment/log/2023-02-24/logs/plot_1" --port 6006

dateFile1 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/fd_replay/step_reward.mat'
mat_contents_a = sio.loadmat(dateFile1)
a = mat_contents_a['data']
a = np.array(a).flatten()

dateFile11 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/original/step_reward.mat'
mat_contents_aa = sio.loadmat(dateFile11)
aa = mat_contents_aa['data']
aa = np.array(aa).flatten()

dateFile2 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/fd_replay/step_Q.mat'
mat_contents_b = sio.loadmat(dateFile2)
b = mat_contents_b['data']
b = np.array(b).flatten()

dateFile22 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/original/step_Q.mat'
mat_contents_bb = sio.loadmat(dateFile22)
bb = mat_contents_bb['data']
bb = np.array(bb).flatten()

dateFile3 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/fd_replay/step_rewards.mat'
mat_contents_c = sio.loadmat(dateFile3)
c = mat_contents_c['data']
c = np.array(c).flatten()

dateFile33 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/original/step_rewards.mat'
mat_contents_cc = sio.loadmat(dateFile33)
cc = mat_contents_cc['data']
cc = np.array(cc).flatten()

time_steps = 'Contrast_experiment'
current_time = datetime.datetime.now().strftime("%Y-%m-%d")
# train_log_dir1 = '/home/he/catkin_nav/src/Comparative_experiment/log/{}/'.format(time_steps) + current_time + '/logs/plot_1'
# train_log_dir2 = '/home/he/catkin_nav/src/Comparative_experiment/log/{}/'.format(time_steps) + current_time + '/logs/plot_2'
train_log_dir1 = '/home/he/catkin_nav/src/Comparative_experiment/log/'+ current_time + '/logs/plot_1'
train_log_dir2 = '/home/he/catkin_nav/src/Comparative_experiment/log/' + current_time + '/logs/plot_2'

summary_writer1 = tf.summary.FileWriter(train_log_dir1)
summary_writer2 = tf.summary.FileWriter(train_log_dir2)


log_total_reward = tf.Variable(0.0)
aaa = tf.summary.scalar("total_reward", log_total_reward)

log_Q_value = tf.Variable(0.0)
bbb = tf.summary.scalar("Q_value", log_Q_value)

log_reward = tf.Variable(0.0)
ccc = tf.summary.scalar("reward", log_reward)

session = tf.Session()
session.run(tf.global_variables_initializer())


if len(a)>= len(aa):
    length_a = len(aa)
else:
    length_a = len(a)
episode = 0
for i in range(0,length_a):
    if i%2 ==0:
        pass
    else:
        summary = session.run(aaa, {log_total_reward: a[i]})
        summary_writer1.add_summary(summary,episode)
        summary_writer1.flush()
        episode = episode + 1

episode = 0
for i in range(0,length_a):
    if i%2 ==0:
        pass
    else:
        summary = session.run(aaa, {log_total_reward: aa[i]})
        summary_writer2.add_summary(summary,episode)
        summary_writer2.flush()
        episode = episode + 1


if len(b)>= len(bb):
    length_b = len(bb)
else:
    length_b = len(b)
if length_b >=200001:
    length_b = 200001
episode = 0
# length_b =length_c =  4000
for i in range(0,length_b):
    if i%2 ==0:
        pass
    else:
        summary = session.run(bbb, {log_Q_value: b[i]})
        summary_writer1.add_summary(summary,episode)
        summary_writer1.flush()
        episode = episode + 1
print("b")
episode = 0
for i in range(0,length_b):
    if i%2 ==0:
        pass
    else:
        summary = session.run(bbb, {log_Q_value: bb[i]})
        summary_writer2.add_summary(summary,episode)
        summary_writer2.flush()
        episode = episode + 1
print("bb")
if len(c)>= len(cc):
    length_c = len(cc)
else:
    length_c = len(c)
if length_c >=200001:
    length_c = 200001

episode = 0
for i in range(0,length_c):
    if i%2 ==0:
        pass
    else:
        summary = session.run(ccc, {log_reward: c[i]})
        summary_writer1.add_summary(summary,episode)
        summary_writer1.flush()
        episode = episode + 1
print("c")
episode = 0
for i in range(0,length_c):
    if i%2 ==0:
        pass
    else:
        summary = session.run(ccc, {log_reward: cc[i]})
        summary_writer2.add_summary(summary,episode)
        summary_writer2.flush()
        episode = episode + 1




# q_value = []
# reward = []
# time_q_value = []
# q_values = []
# rewards = []
# time_q_values = []

# for i in range(0,len(b)):
#     if i%2 ==0:
#         time_q_value.append(b[i])
#     else:
#         q_value.append(b[i])
#         reward.append(c[i])

# ii = 0
# for i in range(0,len(q_value)):
#     if i%200 ==0:
#         q_values.append(q_value[i])
#         rewards.append(reward[i])
#         time_q_values.append(ii)
#         ii+=1









