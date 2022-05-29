import numpy as np
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt

# **********
# description: reproducing code from HAVOK paper
# author: Hongye Xu
# create: 05/20/2022
# **********

# some helper function
def svht(beta, s):
    """Return SVHT value using equation from Gavish 2014"""
    return np.median(s) * (0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43)


# ********* Code starts here **********
# load Lorenz data
# t is 200000 * 1, xdat is 200000 * 3
tLoader = scipy.io.loadmat("../Data/Lorenz-Data/tMatrix.mat")
xdatLoader = scipy.io.loadmat("../Data/Lorenz-Data/xdatMatrix.mat")
t = tLoader.get('t')
xdat = xdatLoader.get('xdat')

#xdat, t = GenLorenzData.genLorenzData()

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
dt = 0.001

# ********** eigen-time delay coordinates **********
# H
# t1 t2 t3 ... t_end - 100
# t2 t3 t4 ...
# ...
# t100 t101 ...
H = np.zeros([stack_max, len(xdat[:, 0]) - stack_max], np.double)
for i in range(0, stack_max):
    H[i, :] = xdat[i: (len(xdat[:, 0]) - stack_max + i), 0]
# # do svd on H
U, S, V = np.linalg.svd(H, 0)
# transpose v
V = np.conj(V)
V = np.transpose(V)
# get threshold using SVHT
m, n = sorted(H.shape)
beta = m / n
thresh = svht(beta, S)
r = min(r_max, np.sum(S > thresh))

#%%
# ********** derivatives **********
# compute derivative
m, n = sorted(V.shape)
dV = np.zeros([n - 5, r])

for i in range(2, n-3):
    for k in range(0, r):
        dV[i - 2, k] = (1 / (12 * dt)) * (-V[i + 2, k] + 8 * V[i + 1, k] - 8 * V[i - 1, k] + V[i - 2, k])

#%%
x = V[0:-1, 0:r]
dx = V[1:, 0:r]

#%%
# ********** BUILD HAVOK REGRESSION MODEL ON TIME DELAY COORDINATES **********
Xi = np.dot(np.linalg.pinv(dx), x)
B = Xi[0: r - 1, r - 1]
A = Xi[0: r - 1, 0: r - 1]
B = B.reshape(-1, 1)
B_0 = 0*B
sys = scipy.signal.StateSpace(A, B, np.eye(r - 1), B_0, dt = dt)

L = []
for i in range(0, 199899):
    L.append(i * dt)

#%%
[t, x_out, y_out] = scipy.signal.dlsim(system = sys, u = x[:, r - 1], t = L, x0 = x[0, 0: r - 1])

# ********* graph **********
tspan = []
for i in range(0, 200000):
    tspan.append(0.001 * i)

#%%
# part 4 in matlab code
# attractor
fig4 = plt.figure(figsize=(8,3))
fig4_1 = fig4.add_subplot(111)
fig4_1.plot(tspan[300: 180000], x[300: 180000, 0])
fig4_1.plot(tspan[300: 180000: 50], y_out[300: 180000: 50, 0], 'r+')
plt.xlim((0, 25))
plt.ylim((-.0051, .005))
plt.title('fig4_1')
plt.show()