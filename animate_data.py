import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import argv
from PIL import Image as img

from pathlib import Path

def animate(X, Y, data, nt, save=None, fname=None):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                #data[frame],
                #np.imag(data[frame]),
                np.abs(data[frame]),
                cmap='coolwarm')
    fps = 2
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )

    if save and save == "gif":
        fname = f"evolution.gif"
        ani.save(fname,)
        #img.open(fname).save(fname, optimize=True)

    elif save and save ==  "mp4":
        if fname:
            fname = f"{fname}.mp4"
        ani.save(fname,)
    else:
        plt.show()

def animate_comparison(x, y, data, nt,):
    fig, axs = plt.subplots(ncols=2,subplot_kw={"projection":'3d'})
    dt = 5 / 100 
    def update(frame):
        axs[0].clear()
        axs[1].clear()

        axs[0].plot_surface(X, Y,
                data[frame],
                cmap='viridis')

        axs[1].plot_surface(X, Y,
                u_analytical(X, Y, frame * dt),
                cmap='viridis') 

    fps = 10
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )
    plt.show()

if __name__ == '__main__':
    fname = str(argv[1])
    L = float(argv[2])
    nx = int(argv[3])
    ny = int(argv[4])
    save_ani = str(argv[5])
    fname_save = str(argv[6])

    path_data = Path(fname)

    data = np.load(path_data) if path_data.suffix == '.npy' else np.load(path_data)['arr_0']

    nt = data.shape[0]
    assert data.shape[1] == nx and data.shape[2] == ny

    xn = np.linspace(-L, L, nx)
    yn = np.linspace(-L, L, ny)
    X, Y = np.meshgrid(xn, yn)

    #animate_comparison(X, Y, data, nt)
    animate(X, Y, data, nt, save_ani, fname_save)
