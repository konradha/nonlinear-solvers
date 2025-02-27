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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from complex_samplers import sampler_random_solitons, sampler_random_fourier_localized
from real_samplers import generate_grf

def save_to_hdf5(run_id, run_idx, args, u0, m, traj, X, Y, elapsed_time, kappa, actual_seed=0):
    output_dir = Path(args.output_dir)
    h5_dir = output_dir / "hdf5"
    h5_dir.mkdir(exist_ok=True)
    
    h5_file = h5_dir / f"run_{run_id}_{run_idx:04d}.h5" 
    with h5py.File(h5_file, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['problem_type'] = 'nlse_saturating'
        meta.attrs['kappa'] = kappa
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
    parser.add_argument("--sampler", choices=["solitons", "fourier"], default="solitons", 
                      help="Initial condition sampler")
    parser.add_argument("--num-solitons", type=int, default=5, help="Number of solitons")
    parser.add_argument("--m-mean", type=float, default=1.0, help="Mean value of m(x,y)")
    parser.add_argument("--m-std", type=float, default=0.5, help="Standard deviation of m(x,y)")
    parser.add_argument("--m-scale", type=float, default=2.0, help="Correlation scale of m(x,y)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--exe", type=str, default="./nlse_cubic", help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--kappa", type=float, default=.5, help="kappa parameter for equation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ic_dir = output_dir / "initial_conditions"
    ic_dir.mkdir(exist_ok=True)
    
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
        f.write(f"Sampler: {args.sampler}" + (f", solitons={args.num_solitons}" if args.sampler == "solitons" else "") + "\n")
        f.write(f"m parameters: mean={args.m_mean}, std={args.m_std}, scale={args.m_scale}\n")
        f.write(f"Executable: {args.exe}\n")
    
    x = np.linspace(-args.Lx, args.Lx, args.nx)
    y = np.linspace(-args.Ly, args.Ly, args.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for i in range(args.num_runs):
        print(f"Run {i+1}/{args.num_runs}")
        seed = args.seed + i if args.seed is not None else np.random.randint(0, 10000) 
        if args.sampler == "solitons":
            u0 = sampler_random_solitons(X, Y, args.num_solitons, seed=seed)
        else:
            u0 = sampler_random_fourier_localized(args.nx, args.ny, args.Lx, args.Ly, seed=seed)
        norm = np.sqrt(np.sum(np.abs(u0)**2) * (2*args.Lx/(args.nx-1)) * (2*args.Ly/(args.ny-1)))
        u0 = u0 / norm
        ic_file = ic_dir / f"ic_{run_id}_{i:04d}.npy"
        np.save(ic_file, u0) 
        m = generate_grf(args.nx, args.ny, args.Lx, args.Ly, 
                       scale=args.m_scale, mean=args.m_mean, std=args.m_std, seed=seed)
        
        m_file = focusing_dir / f"m_{run_id}_{i:04d}.npy"
        np.save(m_file, m) 
        traj_file = traj_dir / f"traj_{run_id}_{i:04d}.npy"
        cmd = [
            str(exe_path),
            str(args.nx),
            str(args.ny),
            str(args.Lx),
            str(args.Ly),
            str(args.kappa),
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
            continue
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        traj_data = np.load(traj_file)
        h5_file = save_to_hdf5(run_id, i, args, u0, m, traj_data, X, Y, elapsed_time, args.kappa)
            
if __name__ == "__main__":
    sys.exit(main())
