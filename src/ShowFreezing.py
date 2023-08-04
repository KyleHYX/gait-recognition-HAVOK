from matplotlib import pyplot as plt
import numpy as np

def load_HuGaDB_file(path_to_file):
    return np.genfromtxt(path_to_file)

bla = load_HuGaDB_file('../Data/Daphnet/S05R01.txt')

frez_fig = plt.figure(figsize=(16,3))
frez_fig = frez_fig.add_subplot(111)
frez_fig.plot(bla[84000:90000, 10])
plt.title('freeze or not')
plt.show()


dim = 2;
lrz_att_fig = plt.figure(figsize=(16,3))
lrz_att_ax = lrz_att_fig.add_subplot(111)
lrz_att_ax.plot(bla[84000:90000, dim])
plt.title('Attractor')
plt.show()