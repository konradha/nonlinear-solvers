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
        fig.suptitle(title, wrap=True, fontsize=12)
    
    fps = 7
    ani = FuncAnimation(fig, update, 
                       frames=nt,
                       interval=1000/fps,
                       blit=False)
    #plt.show()
    ani.save(name)


def animate__(X, Y, data, nt, name, is_complex=False, title=""):
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 1, height_ratios=[1, 9], hspace=0.05)
    
    title_ax = plt.subplot(grid[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=14, wrap=True)
    
    main_ax = plt.subplot(grid[1])
    
    processed_data = [np.abs(d) if is_complex else d for d in data]
    vmin = np.min([np.min(d) for d in processed_data])
    vmax = np.max([np.max(d) for d in processed_data])
    
    im = main_ax.imshow(processed_data[0],
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                aspect='auto',
                cmap='coolwarm' if not is_complex else 'viridis',
                vmin=vmin, vmax=vmax)
    
    cbar = fig.colorbar(im, ax=main_ax)
    main_ax.set_xlabel('X')
    main_ax.set_ylabel('Y')
    
    def update(frame):
        im.set_array(processed_data[frame])
        return [im]
    
    fps = 7
    ani = FuncAnimation(fig, update, 
                       frames=nt,
                       interval=1000/fps,
                       blit=True)
    
    ani.save(name, dpi=250)
    plt.close(fig)
    
if __name__ == '__main__':
    fname = str(argv[1])
    is_complex = True if int(argv[2]) != 0 else False
    path_data = Path(fname)
    u, X, Y, params = load_trajectory(path_data)  
    nt = u.shape[0]
    
    fname_save = fname.replace(".h5", ".mp4")
    animate_diff(X, Y, u, nt, fname_save, is_complex)
    #animate(X, Y, u, nt, fname_save)

