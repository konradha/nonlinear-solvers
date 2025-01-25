import numpy as np
import matplotlib.pyplot as plt

from sys import argv

def energy(u, dx, dy):
    u_x = (np.gradient(u, dx, axis=0))
    u_y = (np.gradient(u, dy, axis=1))
    u_bar = (1/4) * np.abs(u) ** 4
    grad = .5 * (np.abs(u_x) ** 2 + np.abs(u_y) ** 2)
    t = grad - u_bar
    return np.sum(t) * dx * dy

if __name__ == '__main__':
    fname = str(argv[1])
    L = float(argv[2])
    data = np.load(fname)
    nx = data.shape[1]
    dx = 2 * L / (nx - 1)
    nt = data.shape[0]
    E = [energy(data[i], dx, dx) for i in range(nt)]
    plt.plot(range(nt), E)
    plt.show()

