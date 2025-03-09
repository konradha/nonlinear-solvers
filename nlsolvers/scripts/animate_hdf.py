import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation
from sys import argv
from pathlib import Path
import os


def load_trajectory(filename):
    with h5py.File(filename, 'r') as f:
        u = f['u'][:]
        X = f['X'][:]
        Y = f['Y'][:]
        params = dict(f['u'].attrs)
    return u, X, Y, params


def animate(X, Y, data, nt, name):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data[0], cmap='viridis',)
    def update(frame):
        ax.clear()
        ax.plot_surface(X, Y,
                np.real(data[frame]),
                cmap='coolwarm')
    fps = 2
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )
    plt.show()
    #ani.save(name,)

def animate_diff(X, Y, data, nt, name, is_complex=False, title=""):
    fig = plt.figure(figsize=(12, 8))
    
    def update(frame):
        plt.clf()
        plt.imshow(np.abs(data[frame]) if is_complex else data[frame], 
                  extent=[X.min(), X.max(), Y.min(), Y.max()],
                  aspect='auto',
                  cmap='coolwarm' if not is_complex else 'viridis')
        plt.colorbar()
        fig.suptitle(title)
    
    fps = 7
    ani = FuncAnimation(fig, update, 
                       frames=nt,
                       interval=1000/fps,
                       blit=False)
    #plt.show()
    ani.save(name)
    
if __name__ == '__main__':
    fname = str(argv[1])
    is_complex = True if int(argv[2]) != 0 else False
    path_data = Path(fname)
    u, X, Y, params = load_trajectory(path_data)  
    nt = u.shape[0]
    
    fname_save = fname.replace(".h5", ".mp4")
    animate_diff(X, Y, u, nt, fname_save, is_complex)
    #animate(X, Y, u, nt, fname_save)

