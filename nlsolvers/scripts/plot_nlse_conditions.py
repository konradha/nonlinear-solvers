import numpy as np
import matplotlib.pyplot as plt
import h5py

from sys import argv

def plot_initial(X, Y, u0, m, params, initials_save_path):
    fig, axs = plt.subplots(figsize=(22, 7), ncols=2, subplot_kw={"projection":'3d'})
    mp = [(u0, "u0"), (m, "m")]
    for i in range(len(mp)):
        axs[i].plot_surface(X, Y, mp[i][0], cmap='coolwarm')
        axs[i].set_title(mp[i][1])


    problem_type = params["problem"]
    bc = params["bc"]
    T, nt = params["T"], params["nt"]
    Lx, nx = params["Lx"], params["nx"]
   
    fig.suptitle(
            f"""
System: {problem_type}
$T = {T:.2f}, n_t = {nt}, L_x = L_y = {Lx:.2f}$
$n_x = n_y = {nx}$
boundary condition: {bc}
            """,
        fontsize=20
        )
    plt.tight_layout()
    plt.savefig(f"{initials_save_path}.png", dpi=300)

if __name__ == '__main__':
    res_file_path = str(argv[1])
    save_initial_summary_path = str(argv[2])
    with h5py.File(res_file_path, 'r') as f:
        u0 = f['initial_condition']['u0'][:]

        nx, Lx = f['grid'].attrs['nx'], f['grid'].attrs['Lx']
        ny, Ly = f['grid'].attrs['ny'], f['grid'].attrs['Ly']
        dx = 2 * Lx / (nx - 1)
        dy = 2 * Ly / (ny - 1)

        m  = f['coupling']['m'][:]
        times = [f['time'].attrs[l] for l in ['T', 'nt', 'num_snapshots']]
        [T, nt, num_snapshots] = times
        dt_approx = T / num_snapshots
        X  = f['X'][:]
        Y  = f['Y'][:]
        params = {
                "T": T, "nt": nt, "nx": nx, "Lx": Lx,
                "problem": f["metadata"].attrs["problem_type"],
                "bc": f["metadata"].attrs["boundary_condition"],
                }
    plot_initial(X, Y, u0, m, params, save_initial_summary_path) 
