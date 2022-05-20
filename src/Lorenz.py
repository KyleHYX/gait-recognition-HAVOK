import numpy as np
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# **********
# description: reproducing code from HAVOK paper
# author: Hongye Xu
# create: 05/20/2022
# **********

# load Lorenz data
# t is 200000 * 1, xdat is 200000 * 3
tLoader = scipy.io.loadmat("../Data/Lorenz-Data/tMatrix.mat")
xdatLoader = scipy.io.loadmat("../Data/Lorenz-Data/xdatMatrix.mat")
t = tLoader.get('t')
xdat = xdatLoader.get('xdat')

# attractor
lrz_att_fig = plt.figure(figsize=(8,8))
lrz_att_ax = lrz_att_fig.add_subplot(111, projection='3d')
lrz_att_ax.plot3D(xdat[:, 0], xdat[:, 1], xdat[:, 2], color='blue')
plt.title('Lorenz Attractor')
plt.show()

# some parameters
stack_max = 100  # number of shift-stacked rows
lmd = 0  # threshold for sparse regression
r_max = 15  # maximum singular vectors to include

# **********
# eigen-time delay coordinates
# **********
# time delay embedding something something ...
# t1 t2 t3 ... t_end - 100
# t2 t3 t4 ...
# ...
# t100 t101 ...
H = np.zeros([stack_max, len(xdat[:, 0]) - stack_max])
for i in range(0, stack_max):
    H[i, :] = xdat[i: (len(xdat[:, 0]) - stack_max + i), 0]

print('done')