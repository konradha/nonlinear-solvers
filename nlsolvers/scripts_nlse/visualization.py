import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation
from sys import argv
from pathlib import Path
import os
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec



def animate_simulation(X, Y, data, nt, name, is_complex=True, title=""):
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


def animate_isosurface(X, Y, Z, data, title, fname, T, threshold=0.5, fps=10, duration=5):
    assert len(data.shape) == 4 # 3d trajectories only
    nt, nx, ny, nz = data.shape
    frames = min(nt, int(fps * duration))
    step = max(1, nt // frames)
    dt = T / nt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)

    # for now, only take |u| as proxy
    data = np.abs(data)

    vmin = np.min(data)
    vmax = np.max(data)
    norm = plt.Normalize(vmin, vmax)

    def update(frame):
        ax.clear()
        t_idx = frame * step

        mask = data[t_idx] > threshold
        x_plot = X[mask]
        y_plot = Y[mask]
        z_plot = Z[mask]
        c_plot = data[t_idx][mask]

        scatter = ax.scatter(x_plot, y_plot, z_plot, c=c_plot, cmap='viridis',
                            s=30, alpha=0.7, norm=norm)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title} - t={(t_idx * dt):.2f}")

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        return scatter,

    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(fname, writer='ffmpeg', fps=fps)
    plt.close(fig)

def extrema_tracking(X, Y, Z, data, title, fname, T, fps=10, duration=5):
    assert len(data.shape) == 4 # 3d trajectories only
    nt, nx, ny, nz = data.shape
    frames = min(nt, int(fps * duration))
    step = max(1, nt // frames)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    dt = T / nt

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)

    max_positions = []
    min_positions = []
    max_values = []
    min_values = []


    # for now, only take |u| as proxy
    data = np.abs(data)

    for t in range(0, nt, step):
        local_max_idx = np.argmax(data[t])
        local_min_idx = np.argmin(data[t])

        max_idx = np.unravel_index(local_max_idx, (nx, ny, nz))
        min_idx = np.unravel_index(local_min_idx, (nx, ny, nz))

        max_pos = (X[max_idx], Y[max_idx], Z[max_idx])
        min_pos = (X[min_idx], Y[min_idx], Z[min_idx])

        max_positions.append(max_pos)
        min_positions.append(min_pos)
        max_values.append(data[t][max_idx])
        min_values.append(data[t][min_idx])

    def update(frame):
        ax.clear()
        t_idx = frame * step
        current_idx = frame

        current_data = data[t_idx]
        mask = current_data > np.percentile(current_data, 85)

        x_plot = X[mask]
        y_plot = Y[mask]
        z_plot = Z[mask]
        c_plot = current_data[mask]

        ax.scatter(x_plot, y_plot, z_plot, c=c_plot, cmap='viridis',
                  s=10, alpha=0.3)

        if current_idx > 0:
            max_x, max_y, max_z = zip(*max_positions[:current_idx+1])
            min_x, min_y, min_z = zip(*min_positions[:current_idx+1])

            ax.plot(max_x, max_y, max_z, 'r-', linewidth=3, label='Max Path')
            ax.plot(min_x, min_y, min_z, 'b-', linewidth=3, label='Min Path')

            ax.scatter([max_positions[current_idx][0]],
                      [max_positions[current_idx][1]],
                      [max_positions[current_idx][2]],
                      color='red', s=100, marker='*')

            ax.scatter([min_positions[current_idx][0]],
                      [min_positions[current_idx][1]],
                      [min_positions[current_idx][2]],
                      color='blue', s=100, marker='*')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title} - t={(t_idx * dt):.2f}")

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        if current_idx:
            ax.legend()

        return []

    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(fname, writer='ffmpeg', fps=fps)
    plt.close(fig)

def comparative_animation(X, Y, Z, data, T, c, m, title, fname, fps=10, duration=5):
    assert len(data.shape) == 4 # 3d trajectories only
    nt, nx, ny, nz = data.shape
    frames = min(nt, int(fps * duration))
    step = max(1, nt // frames)

    dt = T / nt

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1], projection='3d')

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)


    # for now, only take |u| as proxy
    data = np.abs(data)


    vmin = np.min(data)
    vmax = np.max(data)
    threshold = (vmax - vmin) * 0.3 + vmin

    nonlin_values = np.zeros_like(data)

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dz = (z_max - z_min) / (nz - 1)

    for t in range(nt):
        u = data[t]

        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1] - 2*u[1:-1, 1:-1, 1:-1]) / (dx**2) +
            (u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1] - 2*u[1:-1, 1:-1, 1:-1]) / (dy**2) +
            (u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2] - 2*u[1:-1, 1:-1, 1:-1]) / (dz**2)
        )

        nonlin_term = m * u
        nonlin_values[t] = np.abs(nonlin_term) / (np.abs(laplacian) + np.abs(nonlin_term) + 1e-8)

    nonlin_threshold = 0.2

    def update(frame):
        ax1.clear()
        ax2.clear()

        t_idx = frame * step

        current_data = data[t_idx]
        current_nonlin = nonlin_values[t_idx]

        mask = current_data > threshold
        nonlin_mask = current_nonlin > nonlin_threshold

        x_plot = X[mask][1:-1]
        y_plot = Y[mask][1:-1]
        z_plot = Z[mask][1:-1]
        c_plot = current_data[mask][1:-1]

        x_nonlin = X[nonlin_mask][1:-1]
        y_nonlin = Y[nonlin_mask][1:-1]
        z_nonlin = Z[nonlin_mask][1:-1]
        c_nonlin = current_nonlin[nonlin_mask][1:-1]

        ax1.scatter(x_plot, y_plot, z_plot, c=c_plot, cmap='viridis',
                   s=30, alpha=0.7)

        ax2.scatter(x_nonlin, y_nonlin, z_nonlin, c=c_nonlin, cmap='plasma',
                   s=30, alpha=0.7)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f"wave amplitude at t={(t_idx * dt):.2f}")

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f"nonlinearity dominance at t={(t_idx * dt):.2f}")

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax1.set_zlim([z_min, z_max])

        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        ax2.set_zlim([z_min, z_max])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return []

    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(fname, writer='ffmpeg', fps=fps)
    plt.close(fig)
