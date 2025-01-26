import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import argv
from PIL import Image as img

def animate(X, Y, data, nt, save=None):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                data[frame],
                cmap='viridis')
    fps = 1
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )

    if save and save == "gif":
        fname = f"evolution.gif"
        ani.save(fname,)
        #img.open(fname).save(fname, optimize=True)

    elif save and save ==  "mp4":
        fname = f"evolution.mp4"
        ani.save(fname,)
    else:
        plt.show()

if __name__ == '__main__':
    fname = str(argv[1])
    L = float(argv[2])
    nx = int(argv[3])
    ny = int(argv[4])
    save_ani = str(argv[5])

    data = np.load(fname)
    nt = data.shape[0]
    assert data.shape[1] == nx and data.shape[2] == ny

    xn = np.linspace(-L, L, nx)
    yn = np.linspace(-L, L, ny)
    X, Y = np.meshgrid(xn, yn)

    animate(X, Y, data, nt, save_ani)
