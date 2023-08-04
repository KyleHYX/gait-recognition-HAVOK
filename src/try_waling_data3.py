import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import scipy.signal
from src.get_usv import getUSV


def load_HuGaDB_file(path_to_file):
    return np.genfromtxt(path_to_file)

def smooth(a,WSZ):
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate(( start , out0, stop ))


bla = load_HuGaDB_file('../Data/Daphnet/S05R01.txt')

win_size = 7
start = 84000
numObs = 6000
obs_range = range(start,start + numObs)
bla_start = 4
bla_end = 8

trainDat = bla[obs_range, bla_start:bla_end]
for n in range(0, 3):
    trainDat[:, n] = smooth(trainDat[:, n], win_size)

dim = 1

# some parameters
stack_max = 100  # number of shift-stacked rows
lmd = 0  # threshold for sparse regression
r_max = 10  # maximum singular vectors to include
dt = 0.001

# ********** eigen-time delay coordinates **********
H = np.zeros([stack_max, len(trainDat[:, dim]) - stack_max], np.double)
for i in range(0, stack_max):
    H[i, :] = trainDat[i: (len(trainDat[:, dim]) - stack_max + i), dim]
# do svd on H
U, S, V = np.linalg.svd(H, 0)

# transpose v
V = np.conj(V)
V = np.transpose(V)
# get threshold using SVHT
m, n = sorted(H.shape)
beta = m / n
#r = min(r_max, np.sum(S > thresh))
SSRs = []
minSSR = 99999999
bestR = 0

for rcnt in range(3, 60):
    r = rcnt
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
        L.append(i * dt)

    #%%
    [t, x_out, y_out] = scipy.signal.dlsim(system = sys, u = x[:, r - 1], t = L, x0 = x[0, 0: r - 1])

    # ********* graph **********
    tspan = []
    for i in range(0, numObs):
        tspan.append(0.001 * i)

    ## SSR
    SSR = 0
    for i in range(0, len(x)):
        SSR += pow((x[i, 0] - y_out[i, 0]), 2);
    if SSR < minSSR:
        minSSR = SSR
        bestR = r

    SSRs.append(SSR)

# attractor
r_fig = plt.figure(figsize=(16,8))
r_fig_ax = r_fig.add_subplot(111)
r_fig_ax.plot(SSRs, color='blue')
plt.title('ssrs')
plt.show()

print(bestR)