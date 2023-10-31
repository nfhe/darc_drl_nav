# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import spline
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
plt.show()

