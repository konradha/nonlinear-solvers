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
                np.abs(data[frame]),
                cmap='coolwarm')
    fps = 2
    ani = FuncAnimation(fig, update, frames=nt, interval=nt / fps, )
    ani.save(name,)

def animate_diff(X, Y, data, nt, name):
    fig = plt.figure(figsize=(12, 8))
    
    def update(frame):
        plt.clf()
        plt.imshow(np.abs(data[frame]), 
                  extent=[X.min(), X.max(), Y.min(), Y.max()],
                  aspect='auto',
                  cmap='coolwarm')
        plt.colorbar()
    
    fps = 7
    ani = FuncAnimation(fig, update, 
                       frames=nt,
                       interval=1000/fps,
                       blit=False)
    ani.save(name)
    
if __name__ == '__main__':
    fname = str(argv[1])
    path_data = Path(fname)
    u, X, Y, params = load_trajectory(path_data)  
    nt = u.shape[0]
    
    fname_save = fname.replace(".h5", ".mp4")
    animate_diff(X, Y, u, nt, fname_save)

