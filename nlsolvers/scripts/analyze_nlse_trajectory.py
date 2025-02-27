from animate_hdf import animate_diff
from plot_nlse_conditions import plot_initial

import h5py
import numpy as np
from sys import argv

if __name__ == '__main__':
    input_hdf5 = str(argv[1])
    output_dir = str(argv[2])

    output_initials = f"{output_dir}/initial_condition.png"
    output_movie = f"{output_dir}/trajectory.mp4"

    # get initial conditions
    with h5py.File(input_hdf5, 'r') as f:
        u0 = f['initial_condition']['u0'][:]
        nx, Lx = f['grid'].attrs['nx'], f['grid'].attrs['Lx']
        ny, Ly = f['grid'].attrs['ny'], f['grid'].attrs['Ly']
        dx = 2 * Lx / (nx - 1)
        dy = 2 * Ly / (ny - 1)

        m  = f['focusing']['m'][:]
        times = [f['time'].attrs[l] for l in ['T', 'nt', 'num_snapshots']]
        [T, nt, num_snapshots] = times
        dt_approx = T / num_snapshots
        X  = f['X'][:]
        Y  = f['Y'][:]
        u  = f['u'][:]
        params = {
                "T": T, "nt": nt, "nx": nx, "Lx": Lx,
                "problem": f["metadata"].attrs["problem_type"],
                "bc": f["metadata"].attrs["boundary_condition"],
                }

    plot_initial(X, Y, u0, m, params, output_initials)    
    animate_diff(X, Y, u, num_snapshots, output_movie, is_complex=True)
