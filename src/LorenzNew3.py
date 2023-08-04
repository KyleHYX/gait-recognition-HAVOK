import numpy as np
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt
from src.get_usv import getUSV

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
#tLoader = scipy.io.loadmat("../Data/Lorenz-Data/tMatrix18_1000.mat")
xdatLoader = scipy.io.loadmat("../Data/Lorenz-Data/xdatMatrix15_1000_3.mat")
#t = tLoader.get('t')
xdat = xdatLoader.get('xdat')
numObs = 10000
numPred = 10000
start = 40000
trainDat = xdat[start : start+numObs]
testDat = xdat[start+numObs : start+numObs+numPred]
dim = 0

#xdat, t = GenLorenzData.genLorenzData()

# attractor
lrz_att_fig = plt.figure(figsize=(8,8))
lrz_att_ax = lrz_att_fig.add_subplot(111, projection='3d')
lrz_att_ax.plot3D(xdat[:, 0], xdat[:, 1], xdat[:, 2], color='blue')
plt.title('Attractor')
plt.show()

# some parameters
stack_max = 100  # number of shift-stacked rows
lmd = 0  # threshold for sparse regression
r_max = 15  # maximum singular vectors to include
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
for i in range(0, numObs - (stack_max + 1)):
    L.append(i * dt)

#%%
[t, x_out, y_out] = scipy.signal.dlsim(system = sys, u = x[:, r - 1], t = L, x0 = x[0, 0: r - 1])

# ********* graph **********
tspan = []
for i in range(0, numObs):
    tspan.append(0.001 * i)

#%%
# part 4 in matlab code
# attractor
fig4 = plt.figure(figsize=(8,3))
fig4_1 = fig4.add_subplot(111)
fig4_1.plot(tspan[300: numObs - 5000], x[300: numObs - 5000, 0])
fig4_1.plot(tspan[300: numObs - 5000 : 25], y_out[300: numObs - 5000 : 25, 0], 'r+')
#plt.ylim((-.006, .006))
plt.title('fig4_1')
plt.show()

#%%
#prediction
u_pred, s_pred, v_pred = getUSV(testDat, stack_max, dim)
x_pred = v_pred[0:-1, 0:r]

L_pred = []
for i in range(0, numPred - (stack_max + 1)):
    L_pred.append(i * dt)
[t, x_pred_out, y_pred_out] = scipy.signal.dlsim(system = sys, u = x_pred[:, r - 1], t = L_pred, x0 = x_pred[0, 0: r - 1])
fig5 = plt.figure(figsize=(8,3))
fig5_1 = fig5.add_subplot(111)
tspan_pred = []
for i in range(0, numPred):
    tspan_pred.append(0.001 * i)
fig5_1.plot(tspan_pred[300: numPred - 5000], x_pred[300: numPred - 5000, 0], color='blue')
fig5_1.plot(tspan_pred[300: numPred - 5000 : 50], y_pred_out[300: numPred - 5000 : 50, 0], 'r+')

#plt.ylim((-.006, .006))
plt.title('fig5_1 prediction')
plt.show()

## graph show error trend
cnt = 300
diff = []
while cnt < numPred - 5000:
    diff.append(abs(x_pred[cnt, 0] - y_pred_out[cnt, 0]))
    cnt += 50

fig6 = plt.figure(figsize=(8,3))
fig6_1 = fig6.add_subplot(111)
fig6_1.plot(tspan_pred[300: numPred - 5000 : 50], diff, 'r+')
plt.title('fig6 difference over time')
plt.show()

#%%
# prediction of actual val
U_r = U[:, 0:r-1]
S_r = np.zeros([r-1, r-1])
for scnt in range(r-1):
    S_r[scnt][scnt] = S[scnt]
temp_H_pred = np.dot(U_r, S_r)
y_pred_out = y_pred_out.reshape([numPred - (stack_max + 1), r-1])
# transpose Y
#y_pred_out_tp = np.conj(y_pred_out)
y_pred_out_tp = np.transpose(y_pred_out)
H_pred = np.dot(temp_H_pred, y_pred_out_tp)

#%%
fig7 = plt.figure(figsize=(8,3))
fig7_1 = fig7.add_subplot(111)
fig7_1.plot(tspan_pred[0:numPred - (stack_max + 1)], testDat[0:numPred - (stack_max + 1), dim], color='blue')
fig7_1.plot(tspan_pred[0:numPred - (stack_max + 1) : 50], H_pred[0][::50], 'r+')
plt.ylim((-50, 50))
plt.title('fig7 -- 1st dim actual val vs predicted val')
plt.show()

#%%
dat_diff_cnt = 0
dat_diff = []
errorCnt = 0
while dat_diff_cnt < len(H_pred[0]):
    dat_diff.append(abs(H_pred[0][dat_diff_cnt] - testDat[dat_diff_cnt, dim]))
    if abs(H_pred[0][dat_diff_cnt] - testDat[dat_diff_cnt, dim]) > 1:
        errorCnt+=1
    dat_diff_cnt += 50

print("error")
print(errorCnt)

fig8 = plt.figure(figsize=(8,3))
fig8_1 = fig8.add_subplot(111)
fig8_1.plot(tspan_pred[0:len(H_pred[0]) : 50], dat_diff, 'r+')
#fig8_1.set_xlabel("over 100 time unit")
plt.ylim((0,20))
plt.title("fig8 -- 1st dim prediction: error over time")
plt.show()

#%%
# ## convolve
# first_100 = trainDat[5:105, 0]
# first_100_2 = first_100.reshape(100, 1)
# first_100_2 = first_100_2.T
# U_14 = U[:, r-2]
# U_0 = U[:, 0]
# Vr = first_100_2 * U_14.T
# plz = np.dot(first_100, U_14) / S[r-2]
# conv_res = scipy.signal.convolve(first_100, U_14)
# #conv_res0 = scipy.signal.convolve(first_100, U_0)
#
# fig9 = plt.figure(figsize=(8,3))
# fig9_1 = fig9.add_subplot(111)
# fig9_1.plot(plz, color='blue')
# #plt.ylim((-0.025, 0.025))
# plt.show()

#%%
## get V15 run time
window_size = stack_max
pred_Vr = []
Ur = U[:, r - 1]
Ur_t = Ur.T
#first_100 = trainDat[5:105, 0]
for win_ptr in range(0, numPred - window_size - 1):
    cur_win = testDat[win_ptr : win_ptr + window_size, dim]
    cur_Vr = np.dot(cur_win, Ur_t) / S[r - 1]
    pred_Vr.append(cur_Vr)

pred_Vr = np.array(pred_Vr)
[t, x_pred_out_run_time, y_pred_out_run_time] = scipy.signal.dlsim(system = sys, u = pred_Vr, t = L_pred, x0 = x_pred[0, 0: r - 1])

fig10 = plt.figure(figsize=(8,3))
fig10_1 = fig10.add_subplot(111)
tspan_pred_run_time = []
for i in range(0, numPred):
    tspan_pred_run_time.append(0.001 * i)
fig10_1.plot(tspan_pred[300: numPred - 5000], x_pred[300: numPred - 5000, 0], color='blue')
fig10_1.plot(tspan_pred[300: numPred - 5000 : 50], y_pred_out_run_time[300: numPred - 5000 : 50, 0], 'r+')

plt.ylim((-.006, .006))
plt.title('fig10_1 prediction')
plt.show()

#%%
fig11 = plt.figure(figsize=(20,3))
fig11_1 = fig11.add_subplot(111)
tspan_pred_run_time = []
for i in range(0, numPred):
    tspan_pred_run_time.append(0.001 * i)
fig11_1.plot(tspan_pred[0: numPred - stack_max - 1], x_pred[:, r - 1], color='blue')
fig11_1.plot(tspan_pred[0: numPred - stack_max - 1 : 100], pred_Vr[::100], 'r+')
#fig11_1.plot(testDat[:, dim], color='red')
#fig11_1.plot(x_pred[:, r - 1] * 500, color='blue')
#plt.xlim((300,(numPred - stack_max - 1)/2))
#plt.ylim((-.006, .006))
plt.title('Vr obtain by SVD on Hankel VS Vr obtain by convolve Ur with measurement')
plt.show()

#%%
# reconstructed attractor
rc_att = plt.figure(figsize=(8,8))
rc_att_ax = rc_att.add_subplot(111, projection='3d')
rc_att_ax.plot3D(y_out[7000:8000, 0], y_out[7000:8000, 1], y_out[7000:8000, 2], color='blue')
plt.title('reconstructed attractor')
plt.show()