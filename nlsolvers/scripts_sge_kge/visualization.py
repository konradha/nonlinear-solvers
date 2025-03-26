import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation

from sys import argv
from pathlib import Path
import os


def animate_simulation(X, Y, data, nt, name, title=""):
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 1, height_ratios=[1, 9], hspace=0.05)
    title_ax = plt.subplot(grid[0])
    title_ax.axis('off')
    title_ax.text(
        0.5,
        0.5,
        title,
        ha='center',
        va='center',
        fontsize=14,
        wrap=True)

    main_ax = plt.subplot(grid[1])

    processed_data = data
    vmin = np.min([np.min(d) for d in processed_data])
    vmax = np.max([np.max(d) for d in processed_data])

    im = main_ax.imshow(processed_data[0],
                        extent=[X.min(), X.max(), Y.min(), Y.max()],
                        aspect='auto',
                        cmap='coolwarm',  # real space map only
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
                        interval=1000 / fps,
                        blit=True)

    ani.save(name, dpi=250)
    plt.close(fig)
