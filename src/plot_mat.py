# coding=utf-8
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# dateFile1 = '/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/fd_replay/2022-12-08/step_reward.mat'
dateFile1 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/fd_replay/step_reward.mat'
mat_contents_a = sio.loadmat(dateFile1)
a = mat_contents_a['data']
a = np.array(a).flatten()

# dateFile2 = '/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/fd_replay/2022-12-08/step_Q.mat'
dateFile2 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/fd_replay/step_Q.mat'
mat_contents_b = sio.loadmat(dateFile2)
b = mat_contents_b['data']
b = np.array(b).flatten()

# dateFile3 = '/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/fd_replay/2022-12-08/step_rewards.mat'
dateFile3 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/fd_replay/step_rewards.mat'
mat_contents_c = sio.loadmat(dateFile3)
c = mat_contents_c['data']
c = np.array(c).flatten()

# dateFile11 = '/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/original/2022-12-08/step_reward.mat'
dateFile11 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/original/step_reward.mat'
mat_contents_aa = sio.loadmat(dateFile11)
aa = mat_contents_aa['data']
aa = np.array(aa).flatten()

# dateFile22 = '/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/original/2022-12-08/step_Q.mat'
dateFile22 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/original/step_Q.mat'
mat_contents_bb = sio.loadmat(dateFile22)
bb = mat_contents_bb['data']
bb = np.array(bb).flatten()

# dateFile33 = '/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/original/2022-12-08/step_rewards.mat'
dateFile33 = '/home/he/catkin_nav/src/Comparative_experiment/2023-02-21/maze/original/step_rewards.mat'
mat_contents_cc = sio.loadmat(dateFile33)
cc = mat_contents_cc['data']
cc = np.array(cc).flatten()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(211)
ax3 = fig2.add_subplot(212, sharex=ax1)

reward_a = []
time_reward_a = []
reward_aa = []
time_reward_aa = []
if len(a) >= len(aa):
    length_a = len(aa)
else:
    length_a = len(a)
for i in range(0,length_a):
    if i%2 ==0:
        time_reward_a.append(a[i])
        time_reward_aa.append(aa[i])
    else:
        reward_a.append(a[i])
        reward_aa.append(aa[i])

ax1.plot(time_reward_a,reward_a,'g')
ax1.plot(time_reward_aa,reward_aa,'b')

q_value_b = []
reward_b = []
time_q_value_b = []
q_value_bb = []
reward_bb = []
time_q_value_bb = []

if len(b) >= len(bb):
    length_b = len(bb)
else:
    length_b = len(b)
if length_b >= 200000:
    length_b = 200000
for i in range(0,length_b):
    if i%2 ==0:
        time_q_value_b.append(b[i])
        time_q_value_bb.append(bb[i])
    else:
        q_value_b.append(b[i])
        reward_b.append(c[i])
        q_value_bb.append(bb[i])
        reward_bb.append(cc[i])


ax2.plot(time_q_value_bb,q_value_bb,'r')
ax2.plot(time_q_value_b,q_value_b,'g')

ax3.plot(time_q_value_bb,reward_bb,'r')
ax3.plot(time_q_value_b,reward_b,'g')
plt.show()





