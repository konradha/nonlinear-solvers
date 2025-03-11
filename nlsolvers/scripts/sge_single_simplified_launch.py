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
from real_samplers import generate_grf
from general_real_soliton_samplers import sample_sine_gordon_solution
from animate_hdf import animate_diff, animate__


def save_to_hdf5(run_id, run_idx, args, u0, m, traj, X, Y, elapsed_time, actual_seed=0):
    output_dir = Path(args.output_dir)
    h5_dir = output_dir / "hdf5"
    h5_dir.mkdir(exist_ok=True)
    
    h5_file = h5_dir / f"run_{run_id}_{run_idx:04d}.h5" 
    with h5py.File(h5_file, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['problem_type'] = 'single sine-Gordon'
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
    parser.add_argument("--sampler", choices=["programmatic"], default="programmatic", 
                      help="Initial condition sampler")
    parser.add_argument("--m_type", choices=["one", "anisotropy"], default="one", help="m=m(x,y) term choice")
    parser.add_argument("--m-mean", type=float, default=1.0, help="Mean value of m(x,y)")
    parser.add_argument("--m-std", type=float, default=0.5, help="Standard deviation of m(x,y)")
    parser.add_argument("--m-scale", type=float, default=2.0, help="Correlation scale of m(x,y)")
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

    vel_dir = output_dir / "velocities"
    vel_dir.mkdir(exist_ok=True)
    
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    
    focusing_dir = output_dir / "focusing"
    focusing_dir.mkdir(exist_ok=True)
    
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
        if args.sampler == "programmatic":
            u0, v0, m, sample_type = sample_sine_gordon_solution(args.nx, args.ny, args.Lx) 
            u0, v0 = u0.cpu().numpy().astype(np.float64), v0.cpu().numpy().astype(np.float64)

        else:
            raise NotImplemented

        print(f"Run {i+1}/{args.num_runs}, type={sample_type}") 
        ic_file = ic_dir / f"ic_{run_id}_{i:04d}.npy"
        vel_file = vel_dir / f"vel_{run_id}_{i:04d}.npy"
        np.save(ic_file, u0) 
        np.save(vel_file, v0)


        if args.m_type == "one":
            m = np.ones_like(u0)
        elif args.m_type == "anisotropy":
            m = generate_grf(args.nx, args.ny, args.Lx, args.Ly, 
                       scale=args.m_scale, mean=args.m_mean, std=args.m_std, ).astype(u0.dtype)
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
            str(vel_file),
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

        animation_title = "sG $u_{tt} = \Delta u - m(x,y) sin(u)$\n" + f"m: {args.m_type}; " + m_string + \
        f"domain: $[0, T = {args.T}] x [-{args.Lx:.2f}, {args.Lx:.2f}] x [-{args.Ly:.2f}, {args.Ly:.2f}]$\n" + \
        f"solution type: {sample_type}, boundary conditions: no-flux\n" + \
        f"resolution $n_t = {args.nt}, n_x = {args.nx}, n_y = {args.ny}$\n" + \
        f"downsampled to: {args.dr_x} x {args.dr_y} using strategy '{args.dr_strategy}'\n" +\
        f"samples collected: {args.snapshots}, walltime={walltime:.2f} seconds on: {cuda_string}\n\n"

        postproc_start = time.time()
        animation_output = traj_dir / f"{run_id}_{i:04d}.mp4"  
        animate__(X, Y, traj_data, args.snapshots, animation_output, is_complex=False, title=animation_title)
        postproc_end = time.time()
        postproc_time = postproc_end - postproc_start 

        print(f"Walltime: {walltime:.4f}")
        print(f"Postproc: {postproc_time:.4f}", flush=True)

        # delete all intermediate files after successfully writing hdf5 and the movie 
        if args.delete_intermediates:
            os.unlink(traj_file)
            os.unlink(ic_file)
            os.unlink(m_file)

    if args.delete_intermediates:
        os.unlink(params_file)
 
if __name__ == "__main__":
    sys.exit(main())
