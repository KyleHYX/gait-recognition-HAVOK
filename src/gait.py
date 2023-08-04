"""Function for downloading data from HuGaDB file."""
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import scipy.signal
from src.get_usv import getUSV


def load_HuGaDB_file(path_to_file):
    """Read HuGaDB data from file and get numpy matrix
        Parameters:
            path_to_file: string
                path to HuGaDB file.
        Return: 2d-array
            Data in numpy format
    """
    return np.genfromtxt(path_to_file, delimiter='\t', skip_header=4)

def moving_avg(data, win_size):
    res = []
    m, n = sorted(data.shape)
    for n in range(win_size/2, m - win_size/2):
        res.append()

def smooth(a,WSZ):
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate(( start , out0, stop ))


bla = load_HuGaDB_file('../Data/HuGa/HuGaDB_v1_walking_01_00.txt')
bla1 = load_HuGaDB_file('../Data/HuGa/HuGaDB_v1_walking_01_01.txt')
#aaa = data = np.fromfile("../Data/HuGa/20190206_P0108_011_VP_130.dat", dtype=np.uint16)

win_size = 7
numObs = 2000
obs_range = range(0,numObs)

trainDat = bla[obs_range,36:39]
for n in range(0,3):
    trainDat[:,n] = smooth(trainDat[:,n], win_size)
dim = 1

# lrz_att_fig = plt.figure(figsize=(8,8))
# lrz_att_ax = lrz_att_fig.add_subplot(111, projection='3d')
# lrz_att_ax.plot3D(bla[:,36], bla[:,37], bla[:,38], color='blue')
# plt.title('Attractor')
# plt.show()
lrz_att_fig = plt.figure(figsize=(8,8))
lrz_att_ax = lrz_att_fig.add_subplot(111)
lrz_att_ax.plot(bla[0:200,36], bla[0:200,37], color='blue')
plt.title('Attractor')
plt.show()

# some parameters
stack_max = 100  # number of shift-stacked rows
r_max = 13  # maximum singular vectors to include
dt = 1

# ********** eigen-time delay coordinates **********
# do svd on H
U, S, V = getUSV(trainDat, stack_max, dim)
r = r_max

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
for i in range(0, numObs - (stack_max + 1)):
    L.append(i)

#%%
[t, x_out, y_out] = scipy.signal.dlsim(system = sys, u = x[:, r - 1], t = L, x0 = x[0, 0: r - 1])

# ********* graph **********
tspan = []
for i in range(0, numObs):
    tspan.append(i)

#%%
# part 4 in matlab code
# attractor
fig4 = plt.figure(figsize=(8,3))
fig4_1 = fig4.add_subplot(111)
fig4_1.plot(tspan[0: numObs - 201], x[0: numObs - 201, 0])
fig4_1.plot(tspan[0: numObs - 201 : 1], y_out[0: numObs - 201 : 1, 0], 'r+')
plt.title('fig4_1')
plt.show()

#%%
x_fig = plt.figure(figsize=(8,3))
x_fig = x_fig.add_subplot(111)
x_fig.plot(x[:, r - 1])
sth = []
for n in range(numObs):
    if(trainDat[n, 0] >= 150 and trainDat[n,1] >= 150):
        sth.append(1)
    else:
        sth.append(0)
x_fig.plot(sth, 'r+')
plt.xlim(0,300)
plt.title('Vr')
plt.show()

x_fig1 = plt.figure(figsize=(8,3))
x_fig1 = x_fig1.add_subplot(111)
x_fig1.plot(trainDat[0:numObs, dim])
#plt.xlim(0,250)
plt.title('dim')
plt.show()

numPred = 2000
test_range = range(numObs, numObs+numPred)
testDat = bla[test_range,36:39]
for n in range(0,3):
    testDat[:,n] = smooth(testDat[:,n], win_size)

#%%
#prediction
u_pred, s_pred, v_pred = getUSV(testDat, stack_max, dim)
x_pred = v_pred[0:-1, 0:r]

L_pred = []
for i in range(0, numPred - (stack_max + 1)):
    L_pred.append(i)
[t, x_pred_out, y_pred_out] = scipy.signal.dlsim(system = sys, u = x_pred[:, r - 1], t = L_pred, x0 = x_pred[0, 0: r - 1])
fig5 = plt.figure(figsize=(8,3))
fig5_1 = fig5.add_subplot(111)
tspan_pred = []
for i in range(0, numPred):
    tspan_pred.append(i)
fig5_1.plot(tspan_pred[0: numPred -1000 - 201], x_pred[0: numPred - 1000 - 201, 0])
fig5_1.plot(tspan_pred[0: numPred -1000 - 201 : 1], y_pred_out[0: numPred - 1000 - 201 : 1, 0], 'r+')

plt.title('fig5_1 prediction')
plt.show()