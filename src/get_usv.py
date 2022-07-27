import numpy as np

def getUSV(xdat, stack_max, dim):
    H = np.zeros([stack_max, len(xdat[:, dim]) - stack_max], np.double)
    for i in range(0, stack_max):
        H[i, :] = xdat[i: (len(xdat[:, dim]) - stack_max + i), dim]
    # # do svd on H
    U, S, V = np.linalg.svd(H, 0)
    # transpose v
    V = np.conj(V)
    V = np.transpose(V)

    return [U, S, V]
