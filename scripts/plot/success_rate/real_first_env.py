# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# 柱状图数据
x = ['PEEMR-DARC', 'GD', 'CROP', 'HYC-DDPG', 'PPO', 'DADDPG', 'DDPG']
y = [100, 80, 50, 60, 40, 20, 0]

# 颜色列表
colors = ['fuchsia', 'red', 'cyan', 'orange', 'tomato', 'blue', 'green']

# 绘制柱状图
for i in range(len(x)):
    plt.bar(x[i], y[i], color=colors[i % len(colors)])
    plt.text(x[i], y[i]+1, str(y[i])+'%', ha='center', va='bottom', fontsize=12)

# 设置图表标题和坐标轴标签
plt.title('Results of 10 tests in the first unknown environment',fontsize=12, fontweight='bold')
plt.xlabel('Algorithm name',fontsize=12, fontweight='bold')
plt.ylabel('Success Rate (%)',fontsize=12, fontweight='bold')

# # 更改x轴字体和大小
plt.xticks(fontsize=11, fontname='bold')
plt.yticks(fontsize=12, fontname='bold')

# 设置y轴范围
plt.ylim(0, 110)

# 显示图表
plt.savefig('/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/weight/测试成功率柱状图/实物/第一个未知环境.jpg', dpi=600)
plt.show()