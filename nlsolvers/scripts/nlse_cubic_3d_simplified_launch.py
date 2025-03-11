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

from downsampling import downsample_fft_3d, downsample_interpolation_3d
from tentative_nlse_3d_sampler import sample_nlse_initial_condition_3d
from animate_3d import animate_isosurface, animate_isosurface_dual
from real_samplers import generate_grf


def save_to_hdf5(run_id, run_idx, args, u0, m, traj, X, Y, Z, elapsed_time, sample_type, actual_seed=0):
    output_dir = Path(args.output_dir)
    h5_dir = output_dir / "hdf5"
    h5_dir.mkdir(exist_ok=True)
    
    h5_file = h5_dir / f"run_{run_id}_{run_idx:04d}.h5" 
    with h5py.File(h5_file, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['problem_type'] = 'nlse_cubic_3d'
        meta.attrs['boundary_condition'] = 'noflux'
        meta.attrs['run_id'] = run_id
        meta.attrs['run_index'] = run_idx
        meta.attrs['timestamp'] = str(datetime.datetime.now())
        meta.attrs['elapsed_time'] = elapsed_time
        meta.attrs['sample_type'] = sample_type
        grid = f.create_group('grid')
        grid.attrs['nx'] = args.nx
        grid.attrs['ny'] = args.ny
        grid.attrs['nz'] = args.nz
        grid.attrs['Lx'] = args.Lx
        grid.attrs['Ly'] = args.Ly
        grid.attrs['Lz'] = args.Lz
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
        f.create_dataset('Z', data=Z)
 
    return h5_file

def main():
    parser = argparse.ArgumentParser(description="Local NLSE cubic nonlinearity solver 3D launcher")
    parser.add_argument("--nx", type=int, default=64, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=64, help="Grid points in y")
    parser.add_argument("--nz", type=int, default=64, help="Grid points in z")
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y")
    parser.add_argument("--Lz", type=float, default=10.0, help="Domain half-width in z")
    parser.add_argument("--T", type=float, default=1.5, help="Simulation time")
    parser.add_argument("--nt", type=int, default=500, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=100, help="Number of snapshots")
    parser.add_argument("--sampler", choices=["programmatic", "random"], default="programmatic", 
                      help="Initial condition sampler")
    parser.add_argument("--m_type", choices=["one", "anisotropy"], default="one", help="m=m(x,y,z) term choice")
    parser.add_argument("--m-mean", type=float, default=1.0, help="Mean value of m(x,y,z)")
    parser.add_argument("--m-std", type=float, default=0.5, help="Standard deviation of m(x,y,z)")
    parser.add_argument("--m-scale", type=float, default=2.0, help="Correlation scale of m(x,y,z)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results_3d", help="Output directory")
    parser.add_argument("--exe", type=str, default="./nlse_cubic_3d", help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1,
            help="Number of runs to perform")
    parser.add_argument("--dr-x", type=int, default=50,
            help="Number of gridpoints to sample down for in x-direction")
    parser.add_argument("--dr-y", type=int, default=50,
            help="Number of gridpoints to sample down for in y-direction")
    parser.add_argument("--dr-z", type=int, default=50,
            help="Number of gridpoints to sample down for in z-direction")
    parser.add_argument("--dr-strategy", choices=["FFT", "interpolation", "none"], default="interpolation",
            help="Downsampling strategy: Default is interpolation due to non-periodic boundary conditions. Choose 'none' if you want to keep the resolution")
    parser.add_argument("--iso-level", type=float, default=None,
            help="Isosurface level (default: 0.3 * max amplitude)")
    parser.add_argument("--view-angles", type=float, nargs=2, default=[30, 45],
            help="View angles for 3D visualization (elevation, azimuth)")
    parser.add_argument("--delete-intermediates", type=bool, default=True,
            help="Save space by removing intermediate files")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random_seed = args.seed
    else:
        random_seed = np.random.randint(0, (1 << 31) - 1)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ic_dir = output_dir / "initial_conditions"
    ic_dir.mkdir(exist_ok=True)
    
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    
    focusing_dir = output_dir / "focusing"
    focusing_dir.mkdir(exist_ok=True)
    
    anim_dir = output_dir / "animations"
    anim_dir.mkdir(exist_ok=True)
    
    exe_path = Path(args.exe)
    if not exe_path.exists():
        print(f"Error: Executable {args.exe} not found")
        return 1

    run_id = str(uuid.uuid4())[:8] 
    params_file = output_dir / f"params_{run_id}.txt"
    with open(params_file, "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Grid: {args.nx}x{args.ny}x{args.nz}\n")
        f.write(f"Domain: {args.Lx}x{args.Ly}x{args.Lz}\n")
        f.write(f"Time: T={args.T}, steps={args.nt}, snapshots={args.snapshots}\n")
        f.write(f"Sampler: {args.sampler}\n")
        f.write(f"Executable: {args.exe}\n")
        f.write(f"Random Seed: {random_seed}\n")
    
    x = np.linspace(-args.Lx, args.Lx, args.nx)
    y = np.linspace(-args.Ly, args.Ly, args.ny)
    z = np.linspace(-args.Lz, args.Lz, args.nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    for i in range(args.num_runs):  
        if args.sampler == "random": 
            u0, m, sample_type = sample_nlse_initial_condition_3d(args.nx, args.ny, args.nz, args.Lx)
        elif args.sampler == "programmatic":
            u0, m, sample_type = sample_nlse_initial_condition_3d(args.nx, args.ny, args.nz, args.Lx)
        else:
            raise NotImplemented

        u0 = u0.detach().numpy().astype(np.complex128) if isinstance(u0, torch.Tensor) else u0
        m = m.detach().numpy().astype(np.float64) if isinstance(m, torch.Tensor) else m
        
        print(f"Run {i+1}/{args.num_runs}, type={sample_type}")
        
        ic_file = ic_dir / f"ic_{run_id}_{i:04d}.npy"
        np.save(ic_file, u0) 

        if args.m_type == "one":
            m = np.ones_like(u0, dtype=u0.dtype)
        elif args.m_type == "anisotropy":
            if isinstance(m, np.ndarray) and args.sampler == "programmatic":
                m = m.astype(u0.dtype)
            else:
                L = max(args.Lx, args.Ly, args.Lz)
                nx, ny, nz = args.nx, args.ny, args.nz
                m_tensor = generate_grf(nx, ny, L, L, 
                                      scale=args.m_scale, mean=args.m_mean, std=args.m_std)
                m_expanded = np.repeat(m_tensor[:, :, np.newaxis], nz, axis=2)
                m = m_expanded
        else:
            raise NotImplemented
        
        m_file = focusing_dir / f"m_{run_id}_{i:04d}.npy"
        np.save(m_file, m) 
        traj_file = traj_dir / f"traj_{run_id}_{i:04d}.npy"
        cmd = [
            str(exe_path),
            str(args.nx),
            str(args.ny),
            str(args.nz),
            str(args.Lx),
            str(args.Ly),
            str(args.Lz),
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

        if args.dr_strategy == 'none':
            traj_data = traj_data
        elif args.dr_strategy == 'FFT':
            traj_data = downsample_fft_3d(traj_data, target_shape=(args.dr_x, args.dr_y, args.dr_z)) 
        elif args.dr_strategy == 'interpolation':
            traj_data = downsample_interpolation_3d(traj_data, 
                                                 target_shape=(args.dr_x, args.dr_y, args.dr_z), 
                                                 Lx=args.Lx, Ly=args.Ly, Lz=args.Lz)
        else:
            raise NotImplemented
            
        try:
            h5_file = save_to_hdf5(run_id, i, args, u0, m, traj_data, X, Y, Z, walltime, sample_type)
        except Exception as e:
            print(f"Error saving HDF5: {e}")
            raise e

        cuda_avail = torch.cuda.is_available() 
        cuda_string = "device (presumably)" if cuda_avail else "host"
        m_string = f"GRF: $s={args.m_scale:.2f}$, $\mu={args.m_mean:.2f}$, $\sigma= {args.m_std:.2f}$\n" \
                if args.m_type != "one" else "$m=1 \\forall (x,y,z) \in [-L_x, L_x] \\times [-L_y, L_y] \\times [-L_z, L_z]$\n"

        animation_title = "3D NLSE $i u_t + \Delta u + m(x,y,z) |u|^2 u = 0$\n" + f"m: {args.m_type}; " + m_string + \
        f"domain: $[0, T = {args.T}] \\times [-{args.Lx:.2f}, {args.Lx:.2f}] \\times [-{args.Ly:.2f}, {args.Ly:.2f}] \\times [-{args.Lz:.2f}, {args.Lz:.2f}]$\n" + \
        f"solution type: {sample_type}, boundary conditions: no-flux\n" + \
        f"resolution $n_t = {args.nt}, n_x = {args.nx}, n_y = {args.ny}, n_z = {args.nz}$\n" + \
        f"downsampled to: ${args.dr_x} \\times {args.dr_y} \\times {args.dr_z}$ using strategy '{args.dr_strategy}'\n" +\
        f"samples collected: {args.snapshots}, walltime={walltime:.2f} seconds on: {cuda_string}\n\n"

        postproc_start = time.time()
        animation_output = anim_dir / f"iso_{run_id}_{i:04d}.mp4"  
        
        X_ds, Y_ds, Z_ds = np.meshgrid(
            np.linspace(-args.Lx, args.Lx, args.dr_x),
            np.linspace(-args.Ly, args.Ly, args.dr_y),
            np.linspace(-args.Lz, args.Lz, args.dr_z),
            indexing='ij'
        )
        
        animate_isosurface(X_ds, Y_ds, Z_ds, traj_data, args.snapshots, animation_output, 
                         is_complex=True, title=animation_title, 
                         iso_level=args.iso_level, view_angles=tuple(args.view_angles))
        
        
        postproc_end = time.time()
        postproc_time = postproc_end - postproc_start 

        print(f"Walltime: {walltime:.4f}")
        print(f"Postproc: {postproc_time:.4f}")

        if args.delete_intermediates:
            os.unlink(traj_file)
            os.unlink(ic_file)
            os.unlink(m_file)

    if args.delete_intermediates:
        os.unlink(params_file)
 
if __name__ == "__main__":
    sys.exit(main())
