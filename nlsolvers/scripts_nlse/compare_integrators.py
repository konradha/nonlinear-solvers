import argparse
import os
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

from spatial_amplification_3d import make_grid, create_constant_m
from nlse_sampler import NLSE3DSampler
from valid_spaces import get_parameter_spaces_3d
from nonlinearity_profiles_3d import highlight_profiles

def calculate_energy(u, c, m, dx, dy, dz):
    nx, ny, nz = u.shape
    dV = dx * dy * dz
    
    ux = np.zeros_like(u, dtype=np.complex128)
    uy = np.zeros_like(u, dtype=np.complex128)
    uz = np.zeros_like(u, dtype=np.complex128)
    
    ux[1:-1, :, :] = (u[2:, :, :] - u[0:-2, :, :]) / (2 * dx)
    uy[:, 1:-1, :] = (u[:, 2:, :] - u[:, 0:-2, :]) / (2 * dy)
    uz[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, 0:-2]) / (2 * dz)
    
    kinetic_term = np.sum(c * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2)) * dV
    potential_term = -np.sum(m * np.abs(u)**4) * dV / 2
    
    return kinetic_term + potential_term

def run_integrator(exe_path, nx, ny, nz, Lx, Ly, Lz, u0_file, output_file, T, nt, num_snapshots, m_file, c_file):
    cmd = [
        str(exe_path),
        str(nx),
        str(ny),
        str(nz),
        str(Lx),
        str(Ly),
        str(Lz),
        str(u0_file),
        str(output_file),
        str(T),
        str(nt),
        str(num_snapshots),
        str(m_file),
        str(c_file)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    end_time = time.time()
    
    if result.stderr:
        print(f"Warnings/Errors from {exe_path}: {result.stderr}")
    
    return np.load(output_file), end_time - start_time

def plot_solution_comparison(u1, u2, T, output_dir, run_id):
    nt = u1.shape[0]
    time_points = [0.25, 0.5, 0.75, 1.0]
    time_indices = [int(t * (nt - 1)) for t in time_points]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    
    for i, t_idx in enumerate(time_indices):
        mid_slice = u1.shape[3] // 2
        u1_slice = np.abs(u1[t_idx, :, :, mid_slice])
        u2_slice = np.abs(u2[t_idx, :, :, mid_slice])
        
        vmax = max(np.max(u1_slice), np.max(u2_slice))
        
        im1 = axes[i, 0].imshow(u1_slice, cmap='viridis', vmin=0, vmax=vmax)
        axes[i, 0].set_title(f"{args.name1} at t={time_points[i]*T:.2f}")
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(u2_slice, cmap='viridis', vmin=0, vmax=vmax)
        axes[i, 1].set_title(f"{args.name2} at t={time_points[i]*T:.2f}")
        plt.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/solution_states_{run_id}.png")
    plt.close()

def plot_metrics_comparison(metrics, T, output_dir, run_id):
    time = np.linspace(0, T, len(metrics["l1_diff"]))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(time, metrics["l1_diff"])
    axes[0, 0].set_title("L1 Difference")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("L1 Norm")
    
    axes[0, 1].plot(time, metrics["l2_diff"])
    axes[0, 1].set_title("L2 Difference")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("L2 Norm")
    
    axes[1, 0].plot(time, metrics["energy1"], label=args.name1)
    axes[1, 0].plot(time, metrics["energy2"], label=args.name2)
    axes[1, 0].set_title("Energy Behavior")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Energy")
    axes[1, 0].legend()
    
    axes[1, 1].plot(time, metrics["max_amp1"], label=args.name1)
    axes[1, 1].plot(time, metrics["max_amp2"], label=args.name2)
    axes[1, 1].set_title("Maximum Amplitude")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Max |u|")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison_{run_id}.png")
    plt.close()

def compute_metrics(u1, u2, c, m, dx, dy, dz):
    nt = u1.shape[0]
    dV = dx * dy * dz
    
    metrics = {
        "l1_diff": np.zeros(nt),
        "l2_diff": np.zeros(nt),
        "energy1": np.zeros(nt),
        "energy2": np.zeros(nt),
        "max_amp1": np.zeros(nt),
        "max_amp2": np.zeros(nt)
    }
    
    for t in range(nt):
        abs_diff = np.abs(u1[t] - u2[t])
        metrics["l1_diff"][t] = np.sum(abs_diff) * dV
        metrics["l2_diff"][t] = np.sqrt(np.sum(abs_diff**2) * dV)
        metrics["energy1"][t] = calculate_energy(u1[t], c, m, dx, dy, dz)
        metrics["energy2"][t] = calculate_energy(u2[t], c, m, dx, dy, dz)
        metrics["max_amp1"][t] = np.max(np.abs(u1[t]))
        metrics["max_amp2"][t] = np.max(np.abs(u2[t]))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare NLSE integrators")
    parser.add_argument("--exe1", type=str, required=True, help="Path to first integrator executable")
    parser.add_argument("--exe2", type=str, required=True, help="Path to second integrator executable")
    parser.add_argument("--name1", type=str, default="Integrator 1", help="Name of first integrator")
    parser.add_argument("--name2", type=str, default="Integrator 2", help="Name of second integrator")
    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
    parser.add_argument("--nz", type=int, default=128, help="Grid points in z")
    parser.add_argument("--Lx", type=float, default=3.0, help="Domain half-width in x")
    parser.add_argument("--Ly", type=float, default=3.0, help="Domain half-width in y")
    parser.add_argument("--Lz", type=float, default=3.0, help="Domain half-width in z")
    parser.add_argument("--T", type=float, default=1.5, help="Simulation time")
    parser.add_argument("--nt", type=int, default=500, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=100, help="Number of snapshots")
    parser.add_argument("--phenomenon", type=str, default="multi_soliton_state", 
                        choices=["skyrmion_tube", "multi_soliton_state"], help="Phenomenon type")
    parser.add_argument("--c-m-pair", type=str, choices=['optimal', 'resonant_cavity', 
                        'focusing_soliton', 'sharp_interfaces', 'multi_scale', 'fractal_nonlinear', 
                        'waveguide', 'grf_threshold', 'anisotropic', 'maybe_blowup'], 
                        default=None, help="Choose pair between c(x,y,z) and m(x,y,z)")
    parser.add_argument("--m-type", type=str, default="constant", choices=["constant"], 
                        help="Type of spatial amplification function m(x,y,z)")
    parser.add_argument("--m-mean", type=float, default=1.0, help="Mean value for m")
    parser.add_argument("--c-type", type=str, default="constant", choices=["constant"], 
                        help="Type of anisotropy c(x,y,z)")
    parser.add_argument("--c-mean", type=float, default=1.0, help="Mean value for c")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    global args
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed(int(time.time()))
    
    run_id = str(uuid.uuid4())[:8]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    X, Y, Z = make_grid(args.nx, args.Lx)
    dx = 2 * args.Lx / (args.nx - 1)
    dy = 2 * args.Ly / (args.ny - 1)
    dz = 2 * args.Lz / (args.nz - 1)
    
    sampler = NLSE3DSampler(args.nx, args.ny, args.nz, args.Lx)
    parameter_spaces = get_parameter_spaces_3d()
    
    for run_idx in range(args.num_runs):
        print(f"Run {run_idx+1}/{args.num_runs}")
        
        u0_file = tmp_dir / f"u0_{run_id}_{run_idx}.npy"
        m_file = tmp_dir / f"m_{run_id}_{run_idx}.npy"
        c_file = tmp_dir / f"c_{run_id}_{run_idx}.npy"
        traj1_file = tmp_dir / f"traj1_{run_id}_{run_idx}.npy"
        traj2_file = tmp_dir / f"traj2_{run_id}_{run_idx}.npy"
        
        if args.phenomenon in parameter_spaces:
            space = parameter_spaces[args.phenomenon]
            params = {}
            
            for key, values in space.items():
                if isinstance(values[0], tuple) or isinstance(values[0], list):
                    l = len(values)
                    idx = np.random.randint(0, l)
                    params[key] = values[idx]
                else:
                    params[key] = np.random.choice(values)
            
            params["system_type"] = "cubic"
            print(f"Parameters: {params}")
            
            u0 = sampler.generate_initial_condition(phenomenon_type=args.phenomenon, **params)
            
            if hasattr(u0, 'detach'):
                u0 = u0.detach().numpy()
        else:
            raise ValueError(f"Unknown phenomenon type: {args.phenomenon}")
        
        if args.c_m_pair is None:
            if args.m_type == "constant":
                m = create_constant_m(X, value=args.m_mean)
            else:
                raise ValueError(f"Unknown m_type: {args.m_type}")
            
            if args.c_type == "constant":
                c = create_constant_m(X, value=args.c_mean)
            else:
                raise ValueError(f"Unknown c_type: {args.c_type}")
        else:
            profiles = highlight_profiles(args.nx, args.Lx)
            c, m = profiles[args.c_m_pair]
        
        np.save(u0_file, u0)
        np.save(m_file, m)
        np.save(c_file, c)
        
        print(f"Running {args.name1}: {args.exe1}")
        traj1, time1 = run_integrator(
            args.exe1, args.nx, args.ny, args.nz, args.Lx, args.Ly, args.Lz,
            u0_file, traj1_file, args.T, args.nt, args.snapshots, m_file, c_file
        )
        
        print(f"Running {args.name2}: {args.exe2}")
        traj2, time2 = run_integrator(
            args.exe2, args.nx, args.ny, args.nz, args.Lx, args.Ly, args.Lz,
            u0_file, traj2_file, args.T, args.nt, args.snapshots, m_file, c_file
        )
        
        print(f"{args.name1} runtime: {time1:.2f}s, {args.name2} runtime: {time2:.2f}s")
        
        metrics = compute_metrics(traj1, traj2, c, m, dx, dy, dz)
        
        plot_metrics_comparison(metrics, args.T, output_dir, f"{run_id}_{run_idx}")
        plot_solution_comparison(traj1, traj2, args.T, output_dir, f"{run_id}_{run_idx}")
        
        if args.num_runs > 1:
            for file in [u0_file, m_file, c_file, traj1_file, traj2_file]:
                if file.exists():
                    os.unlink(file)
    
    print(f"Comparison complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
