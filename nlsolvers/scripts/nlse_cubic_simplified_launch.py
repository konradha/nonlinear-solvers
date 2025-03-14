import numpy as np
import os
import argparse
import subprocess
import time
from pathlib import Path
import uuid
import sys
import h5py
import datetime

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from downsampling import downsample_fft, downsample_interpolation
from complex_samplers import sampler_random_solitons, sampler_random_solitons_with_velocities, sampler_random_fourier_localized
from real_samplers import generate_grf, generate_wavelet
from improved_nlse_samplers import sample_nlse_initial_condition
from animate_hdf import animate_diff, animate__
from global_nlse_sampler import NLSESampler


###
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

def calculate_energy(u, dx, dy):
    assert isinstance(u, np.ndarray)
    nt, nx, ny = u.shape
    energy = np.zeros(nt)
    
    for t in range(nt):
        ux = np.zeros((nx-2, ny-2), dtype=complex)
        uy = np.zeros((nx-2, ny-2), dtype=complex) 
        ux = (u[t, 2:nx, 1:ny-1] - u[t, 0:nx-2, 1:ny-1]) / (2 * dx)
        uy = (u[t, 1:nx-1, 2:ny] - u[t, 1:nx-1, 0:ny-2]) / (2 * dy) 
        u_interior = u[t, 1:nx-1, 1:ny-1] 
        gradient_squared = np.abs(ux)**2 + np.abs(uy)**2
        u_squared = np.abs(u_interior)**2
        nonlinear_term = u_squared**2 / 2 
        integrand = gradient_squared - nonlinear_term
        energy[t] = np.sum(integrand) * dx * dy        
    return energy

def create_analysis_plots(traj_data, m, args, energies=None):
    rcParams['font.size'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    
    nt, nx, ny = traj_data.shape
    # n_t is num_snapshots when we call this function
    tn_actual = np.linspace(0, args.T, args.snapshots)
    
    if energies is None:
        dx = 2 * args.Lx / (args.nx - 1)
        dy = 2 * args.Ly / (args.ny - 1)
        energies = calculate_energy(traj_data, dx, dy)
    
    dx = 2 * args.Lx / (args.nx - 1) 
    dy = 2 * args.Ly / (args.ny - 1)
    dV = dx * dy
    norm_over_time = np.array([np.sum(np.abs(traj_data[i])**2) * dV for i in range(nt)])
    
    norm_diff = np.abs(norm_over_time - norm_over_time[0])
    norm_log_diff = np.log10(np.maximum(norm_diff, np.finfo(float).eps))
    norm_log_diff = [np.nan] + list(norm_log_diff[1:])

    energy_diff = np.abs(energies - energies[0])
    energy_log_diff = np.log10(np.maximum(energy_diff, np.finfo(float).eps))
    energy_log_diff = [np.nan] + list(energy_log_diff[1:])

    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[0, 2])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[2, 0])
    ax9 = fig.add_subplot(gs[2, 2])
    
    mid_idx = nt // 2
    
    ax1.plot(tn_actual, norm_over_time, linewidth=2)
    ax1.set_title(r"$\int |u|^2 dx dy$")
    #ax1.set_xlabel("T / [1]")
    ax1.set_ylabel("Norm / [1]")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("linear")
    ax1.set_ylim(bottom=0.5, top=1.5)

    
    ax2.plot(tn_actual, norm_log_diff, label="Norm (diff to 1.0)", linewidth=2, color='blue')
    ax2.plot(tn_actual, energy_log_diff, label="Energy (diff to $E_0$)", linewidth=2, color='orange')
    ax2.set_title("Conservation Errors")
    #ax2.set_xlabel("T / [1]")
    ax2.set_ylabel("Error [$\log$ diff] / [1]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    extent = [-args.Lx, args.Lx, -args.Ly, args.Ly]
    im3 = ax3.imshow(np.abs(traj_data[0])**2, extent=extent, cmap='viridis')
    ax3.set_title(f"$|u|^2$ at t=0")
    fig.colorbar(im3, ax=ax3)
    
    im4 = ax4.imshow(np.abs(traj_data[mid_idx])**2, extent=extent, cmap='viridis')
    ax4.set_title(f"$|u|^2$ at t={tn_actual[mid_idx]:.2f}")
    fig.colorbar(im4, ax=ax4)
    
    im5 = ax5.imshow(np.abs(traj_data[-1])**2, extent=extent, cmap='viridis')
    ax5.set_title(f"$|u|^2$ at t={args.T:.2f}")
    fig.colorbar(im5, ax=ax5)

    im6 = ax6.imshow(np.abs(traj_data[0]), extent=extent, cmap='viridis')
    ax6.set_title("Initial amplitude $|u_0|$")
    fig.colorbar(im6, ax=ax6)
    
    im7 = ax7.imshow(m, extent=extent, cmap='viridis')
    ax7.set_title("Anisotropy $m(x,y)$")
    fig.colorbar(im7, ax=ax7)
   
    term_ratio = np.zeros(nt)
    for t in range(nt):
        laplacian = np.zeros_like(traj_data[t], dtype=complex)
        laplacian[1:-1, 1:-1] = (traj_data[t, 2:, 1:-1] + traj_data[t, :-2, 1:-1] + 
                               traj_data[t, 1:-1, 2:] + traj_data[t, 1:-1, :-2] - 
                               4*traj_data[t, 1:-1, 1:-1]) / (dx*dy)
        nonlinear = m * np.abs(traj_data[t])**2 * traj_data[t]
        lap_mag = np.mean(np.abs(laplacian[2:-2, 2:-2]))
        nl_mag = np.mean(np.abs(nonlinear[2:-2, 2:-2]))
        
        if lap_mag > 0:
            term_ratio[t] = nl_mag / lap_mag
        else:
            term_ratio[t] = 0
   
    ax8.plot(tn_actual, term_ratio, linewidth=2)
    ax8.set_title("Nonlinear/Laplacian Term Ratio")
    ax8.set_xlabel("T / [1]")
    ax8.set_ylabel("r / [1]")
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)
   
    im9 = ax9.imshow(np.angle(traj_data[-1]), extent=extent, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax9.set_title(f"Phase at t={args.T:.2f}")
    fig.colorbar(im9, ax=ax9)
  
    
    return fig

###



def make_grid(Nx, Ny, L):
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y

def activate_inner(X, Y, L, psi):
    mask_x, mask_y = torch.abs(X) < .5 * L, torch.abs(Y) < .5 * L
    inner_mask = torch.logical_and(mask_x, mask_y)
    outer = ~inner_mask
    psi[outer] = 0.
    return psi


def save_to_hdf5(run_id, run_idx, args, u0, m, traj, X, Y, elapsed_time, actual_seed=0):
    output_dir = Path(args.output_dir)
    h5_dir = output_dir / "hdf5"
    h5_dir.mkdir(exist_ok=True)
    
    h5_file = h5_dir / f"run_{run_id}_{run_idx:04d}.h5" 
    with h5py.File(h5_file, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['problem_type'] = 'nlse_cubic'
        meta.attrs['boundary_condition'] = 'noflux'
        meta.attrs['run_id'] = run_id
        meta.attrs['run_index'] = run_idx
        meta.attrs['timestamp'] = str(datetime.datetime.now())
        meta.attrs['elapsed_time'] = elapsed_time
        grid = f.create_group('grid')
        grid.attrs['nx'] = args.nx
        grid.attrs['ny'] = args.ny
        grid.attrs['Lx'] = args.Lx
        grid.attrs['Ly'] = args.Ly
        time_grp = f.create_group('time')
        time_grp.attrs['T'] = args.T
        time_grp.attrs['nt'] = args.nt
        time_grp.attrs['num_snapshots'] = args.snapshots
        ic_grp = f.create_group('initial_condition')
        ic_grp.attrs['sampler'] = args.sampler
        ic_grp.create_dataset('u0', data=u0) 
        m_grp = f.create_group('focusing')
        m_grp.attrs['mean'] = args.m_mean
        m_grp.attrs['std'] = args.m_std
        m_grp.attrs['scale'] = args.m_scale
        m_grp.create_dataset('m', data=m)
        f.create_dataset('u', data=traj)
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
 
    return h5_file

def main():
    parser = argparse.ArgumentParser(description="Local NLSE cubic nonlinearity solver launcher")
    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y")
    parser.add_argument("--T", type=float, default=1.5, help="Simulation time")
    parser.add_argument("--nt", type=int, default=500, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=100, help="Number of snapshots")
    parser.add_argument("--sampler", choices=["programmatic", "random"], default="programmatic", 
                      help="Initial condition sampler")
    parser.add_argument("--sample-type", choices=['ground', 'vortex', 'multi_vortex', 'soliton',
                         'lattice', 'turbulence', 'rogue', 'breather',
                         'sound', 'spiral_vortex',
                         'multi_spiral', 'skyrmion', 'multi_skyrmion',
                         'vortex_dipole', 'dynamic_lattice', None], default=None, help="Specifying the phenomenon to sample")

    parser.add_argument("--m_type", choices=["one", "anisotropy"], default="one", help="m=m(x,y) term choice")
    parser.add_argument("--m-mean", type=float, default=1., help="Mean value of m(x,y)")
    parser.add_argument("--m-std", type=float, default=.5, help="Standard deviation of m(x,y)")
    parser.add_argument("--m-scale", type=float, default=2., help="Correlation scale of m(x,y)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--exe", type=str, default="./nlse_cubic", help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--dr-x", type=int, default=128, help="Number of gridpoints to sample down for in x-direction")
    parser.add_argument("--dr-y", type=int, default=128, help="Number of gridpoints to sample down for in x-direction")
    parser.add_argument("--dr-strategy", choices=["FFT", "interpolation", "none"], default="interpolation",
                      help="Downsampling strategy: Default is interpolation due to non-periodic boundary conditions. Choose 'none' if you want to keep the resolution")
    parser.add_argument("--delete-intermediates", type=bool, default=True,
            help="Save space by removing intermediate files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ic_dir = output_dir / "initial_conditions"
    ic_dir.mkdir(exist_ok=True)
    
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    
    focusing_dir = output_dir / "focusing"
    focusing_dir.mkdir(exist_ok=True)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    exe_path = Path(args.exe)
    if not exe_path.exists():
        print(f"Error: Executable {args.exe} not found")
        return 1

    run_id = str(uuid.uuid4())[:8] 
    params_file = output_dir / f"params_{run_id}.txt"
    with open(params_file, "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Grid: {args.nx}x{args.ny}\n")
        f.write(f"Domain: {args.Lx}x{args.Ly}\n")
        f.write(f"Time: T={args.T}, steps={args.nt}, snapshots={args.snapshots}\n")
        f.write(f"Sampler: {args.sampler}\n")
        f.write(f"Executable: {args.exe}\n")
    
    x = np.linspace(-args.Lx, args.Lx, args.nx)
    y = np.linspace(-args.Ly, args.Ly, args.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for i in range(args.num_runs):   
        # TODO: unifiy sampler call (+ ARGS!!!)
        if args.sample_type is None and args.sampler == "random": 
            u0 = sampler_random_fourier_localized(args.nx, args.ny, args.Lx, args.Ly, seed=seed)
        elif args.sample_type is None and args.sampler == "programmatic": 

            if np.random.choice([0,1]):
                num_packets = int(np.random.choice(np.linspace(1, 8)))
                u0 = sampler_random_solitons(X, Y,
                        num_solitons=num_packets,
                        soliton_width=.1 * 2 * args.Lx, amp_range=(0.5, 2.0), seed=None)
                sample_type = f"Colliding wavepackets, {num_packets=}"
            else:
                num_packets = int(np.random.choice(np.linspace(1, 8)))
                u0 = sampler_random_solitons_with_velocities(X, Y,
                        num_solitons=num_packets,
                        soliton_width=.1 * 2 * args.Lx, amp_range=(0.5, 2.0), seed=None)
                sample_type = f"Colliding wavepackets with velocities, {num_packets=}"
        
        if args.sample_type is not None:
            u0, sample_type = sample_nlse_initial_condition(args.nx, args.ny, args.Lx, condition_type=args.sample_type)
        
        
        #u0 = activate_inner(torch.from_numpy(X), torch.from_numpy(Y), args.Lx, u0)

        dx = 2 * args.Lx / (args.nx - 1)
        dy = 2 * args.Ly / (args.ny - 1)
        dV = dx * dy
        if isinstance(u0, np.ndarray):
            current_norm = np.sum(np.abs(u0)**2) * dV
            if current_norm > 0:
                u0 = u0 / np.sqrt(current_norm)
        
        elif isinstance(u0, torch.Tensor):
            current_norm = torch.sum(torch.abs(u0)**2) * dV
            if current_norm > 0:
                u0 = u0 / torch.sqrt(current_norm)
            
        else:
            raise TypeError(f"Unsupported type: {type(u0)}. Expected numpy.ndarray or torch.Tensor")
    
        u0 = u0.detach().numpy().astype(np.complex128) if isinstance(u0, torch.Tensor) else u0.astype(np.complex128)
        print(f"Run {i+1}/{args.num_runs}, type={sample_type}")
        

        ic_file = ic_dir / f"ic_{run_id}_{i:04d}.npy"
        np.save(ic_file, u0) 

        if args.m_type == "one":
            m = np.ones_like(u0).astype(np.float64)
        elif args.m_type == "anisotropy":
            m = generate_wavelet(args.nx, args.ny, args.Lx, args.Ly, 
                       scale=args.m_scale, mean=args.m_mean, std=args.m_std, ).astype(np.float64)
            m = m - np.mean(m)
            m = m * 25
        else:
            raise NotImplemented
            
        m_file = focusing_dir / f"m_{run_id}_{i:04d}.npy"
        np.save(m_file, m) 
        traj_file = traj_dir / f"traj_{run_id}_{i:04d}.npy"
        cmd = [
            str(exe_path),
            str(args.nx),
            str(args.ny),
            str(args.Lx),
            str(args.Ly),
            str(ic_file),
            str(traj_file),
            str(args.T),
            str(args.nt),
            str(args.snapshots),
            str(m_file)
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings/Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            print(f"Command: {' '.join(cmd)}")
            continue
        
        end_time = time.time()
        walltime = end_time - start_time
        traj_data = np.load(traj_file)

        fig = create_analysis_plots(traj_data, m, args)
        fig.savefig(analysis_dir / f"analysis_{run_id}_{i:04d}.png", dpi=300)


        if args.dr_strategy == 'none':
            traj_data = traj_data
        elif args.dr_strategy == 'FFT':
            traj_data = downsample_fft(traj_data, target_shape=(args.dr_x, args.dr_y)) 
        elif args.dr_strategy == 'interpolation':
            traj_data = downsample_interpolation(traj_data, target_shape=(args.dr_x, args.dr_y), Lx=args.Lx, Ly=args.Ly)
        else:
            raise NotImplemented    
        try:
            h5_file = save_to_hdf5(run_id, i, args, u0, m, traj_data, X, Y, walltime)
        except Exception as e:
            # catastrophic abort if writing hdf5 fails
            raise e

        cuda_avail = torch.cuda.is_available() 
        cuda_string = "device (presumably)" if cuda_avail else "host"
        m_string = f"GRF: $s={args.m_scale:.2f}$, $\mu={args.m_mean:.2f}$, $\sigma= {args.m_std:.2f}$\n" \
                if args.m_type != "one" else "$m=1 \\forall (x,y) \in [-L_x, L_x] x [-L_y, L_y]$\n"

        animation_title = "NLSE $i u_t + \Delta u + m(x,y) |u|^2 u = 0$\n" + f"m: {args.m_type}; " + m_string + \
        f"domain: $[0, T = {args.T}] x [-{args.Lx:.2f}, {args.Lx:.2f}] x [-{args.Ly:.2f}, {args.Ly:.2f}]$\n" + \
        f"solution type: {sample_type}, boundary conditions: no-flux\n" + \
        f"resolution $n_t = {args.nt}, n_x = {args.nx}, n_y = {args.ny}$\n" + \
        f"downsampled to: {args.dr_x} x {args.dr_y} using strategy '{args.dr_strategy}'\n" +\
        f"samples collected: {args.snapshots}, walltime={walltime:.2f} seconds on: {cuda_string}\n\n"

        postproc_start = time.time()
        animation_output = traj_dir / f"{run_id}_{i:04d}.mp4"  
        animate__(X, Y, traj_data, args.snapshots, animation_output, is_complex=True, title=animation_title)
        postproc_end = time.time()
        postproc_time = postproc_end - postproc_start 

        print(f"Walltime: {walltime:.4f}")
        print(f"Postproc: {postproc_time:.4f}")

        # delete all intermediate files after successfully writing hdf5 and the movie 
        if args.delete_intermediates:
            os.unlink(traj_file)
            os.unlink(ic_file)
            os.unlink(m_file)

        

    if args.delete_intermediates:
        os.unlink(params_file)
 
if __name__ == "__main__":
    sys.exit(main())
