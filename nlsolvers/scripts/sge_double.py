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
from real_initial_samplers import sample_initial_conditions


def save_to_hdf5(run_id, run_idx, args, u0, v0, m, traj, elapsed_time, X, Y, actual_seed):
    output_dir = Path(args.output_dir)
    h5_dir = output_dir / "hdf5"
    h5_dir.mkdir(exist_ok=True) 
    h5_file = h5_dir / f"run_{run_id}_{run_idx:04d}.h5"
    
    with h5py.File(h5_file, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['problem_type'] = 'sge_double'
        meta.attrs['boundary_condition'] = 'noflux' # TODO: more precise for all
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
        ic_grp.attrs['seed'] = actual_seed
        ic_grp.create_dataset('u0', data=u0)
        ic_grp.create_dataset('v0', data=v0)
        if m is not None:
            m_grp = f.create_group('coupling')
            m_grp.create_dataset('m', data=m)
        f.create_dataset('u', data=traj)
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
 
    return h5_file

def main():
    parser = argparse.ArgumentParser(description="Double sine-Gordon solver launcher")
    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y")
    parser.add_argument("--T", type=float, default=10., help="Simulation time")
    parser.add_argument("--nt", type=int, default=500, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=100, help="Number of snapshots")

    parser.add_argument("--m-mean", type=float, default=1.0, help="Mean value of m(x,y)")
    parser.add_argument("--m-std", type=float, default=0.5, help="Standard deviation of m(x,y)")
    parser.add_argument("--m-scale", type=float, default=2.0, help="Correlation scale of m(x,y)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--exe", type=str, default="./sg_single", help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to perform")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ic_dir = output_dir / "initial_conditions"
    ic_dir.mkdir(exist_ok=True)
    
    vel_dir = output_dir / "velocities"
    vel_dir.mkdir(exist_ok=True)
    
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    
    coupling_dir = output_dir / "coupling"
    coupling_dir.mkdir(exist_ok=True)
    
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
        f.write(f"Executable: {args.exe}\n")
    
    x = np.linspace(-args.Lx, args.Lx, args.nx)
    y = np.linspace(-args.Ly, args.Ly, args.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for i in range(args.num_runs):
        print(f"\nRun {i+1}/{args.num_runs}")
        seed = args.seed + i if args.seed is not None else np.random.randint(0, 10000)
        u0, v0, m, _, _ = sample_initial_conditions(args.nx, args.ny, args.Lx)
        u0, v0, m = u0.detach().numpy().astype(np.float64), v0.detach().numpy().astype(np.float64), m.detach().numpy().astype(np.float64)
         
        ic_file = ic_dir / f"ic_{run_id}_{i:04d}.npy"
        vel_file = vel_dir / f"vel_{run_id}_{i:04d}.npy"
        np.save(ic_file, u0)
        np.save(vel_file, v0)
    
        m_file = coupling_dir / f"m_{run_id}_{i:04d}.npy"
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
            if result.stderr:
                print(f"Warnings/Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error in run: {e}")
            continue 

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("walltime", elapsed_time)
        try:
            traj_data = np.load(traj_file)
            h5_file = save_to_hdf5(run_id, i, args, u0, v0, m, traj_data, elapsed_time, X, Y, seed)
        except Exception as e:
            print(f"Error loading trajectory or saving HDF5: {e}")


if __name__ == "__main__":
    sys.exit(main())
