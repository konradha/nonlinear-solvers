import numpy as np
import matplotlib.pyplot as plt

from sys import argv

def energy(u, v, dx, dy):
    u_x = (np.gradient(u, dx, axis=0))
    u_y = (np.gradient(u, dy, axis=1))
    u_t = v
    gradu2 = u_x ** 2 + u_y ** 2
    v2 = u_t ** 2
    cos = 2 * (1 - np.cos(u))
    t = np.sum(.5 * (gradu2 + v2 + cos))
    return .5 * dx * dy * t

if __name__ == '__main__':
    fname_u = str(argv[1])
    fname_v = str(argv[2])

    L = float(argv[3])
    data_u = np.load(fname_u)
    data_v = np.load(fname_v)
    assert data_u.shape == data_v.shape

    nx = data_u.shape[1]
    dx = 2 * L / (nx - 1)
    nt = data_u.shape[0]

    E = [energy(data_u[i], data_v[i], dx, dx) for i in range(nt)]
    plt.plot(range(nt), E)
    plt.show()

