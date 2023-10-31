# DDPG TD3 SAC 训练曲线

import numpy as np
import sys
import matplotlib.pyplot as plt
import os


plt.rc('font',family='Times New Roman')
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

plt.figure(figsize=(4,3))
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率

ax = plt.gca()
ax.set_xlim(0, 500)
# ax.set_ylim(-60, 80)

ax.tick_params(bottom=False,top=False,left=False,right=False)

# 设置网格颜色
ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
# ax.grid(color='w', linestyle='-', linewidth=2, alpha=1, zorder=0)

# 设置背景颜色
# ax.patch.set_facecolor("#e9e9f2")

# 去除边框
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['right'].set_visible(False)

# 加载数据
def load(name ):
    from os import path
    url = os.path.dirname(os.path.realpath(__file__)) 
    result = []
    if name == "SAC":
        temp_SAC = np.load(path.join(url ,  "SAC_train_data/episode_rewards.npy"))
        result.append(temp_SAC)
        
   
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

    max_data = []
    min_data = []
    mean_data = []
    a = []
    m = 0

    for i in range(len(data_list[0])):
        
        m = m + data_list[0][i]
        a.append(data_list[0][i])

        if i % fineness == 0: 

            ma = np.max(a)
            mi = np.min(a)
            max_data.append(ma)
            min_data.append(mi)
            mean_data.append(m/fineness)
            m = 0
            a = []
    
    data = [mean_data, max_data, min_data]
    data = np.array(data)
    if smooth_flag:
        for i in range(len(data)):
            for j in range(2, fineness):
                data[i] = smooth(data[i], j)
    return data[0], data[1], data[2]

SAC_data = load("SAC")


fineness = 2


SAC_mean_data, SAC_min_data, SAC_max_data = get_mean_max_min(SAC_data, True, fineness)



SAC_x = range(SAC_mean_data.size)
plt.fill_between(SAC_x, SAC_min_data, SAC_max_data,  alpha=0.2, zorder=2, color="blue")
plt.plot(SAC_x, SAC_mean_data, linewidth=1, label="SAC", zorder=3, color="blue")

# plt.title("Training curve", pad=15)
plt.xlabel("Episode", labelpad = 5 )
# plt.ylabel("Accumulated Reward", labelpad=8.5)
plt.ylabel("Avarage Reward/every 6 episodes", labelpad=5)
# plt.title('Episode Reward Analysis Chart', fontweight='bold', pad=12)

# plt.tight_layout()

# plt.legend( bbox_to_anchor=(0.755,1.265),borderaxespad = 0.05, frameon=False, ncol = 2)
# plt.legend(loc="lower right", frameon=True, ncol = 2)
plt.legend( bbox_to_anchor=(1.0, 1.12), ncol = 5, frameon = 0, handlelength = 0.9, columnspacing = 0.5, fontsize = 'small')


# plt.savefig('episode_reward_plot.jpg')
plt.tight_layout()
plt.savefig('episode_rewards_plot.jpg')
plt.show()
