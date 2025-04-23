#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import argparse
import time
from pathlib import Path

def create_initial_condition(n, dim, equation_type):
    """Create initial condition for the given equation type."""
    if dim == 2:
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2
        
        if equation_type.startswith("nlse"):
            # Gaussian for NLSE
            u0 = np.exp(-r2)
            return u0.astype(np.complex128)
        else:
            # Gaussian for real-space equations
            u0 = np.exp(-r2)
            v0 = np.zeros_like(u0)
            return u0.astype(np.float64), v0.astype(np.float64)
    elif dim == 3:
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        z = np.linspace(-5, 5, n)
        X, Y, Z = np.meshgrid(x, y, z)
        r2 = X**2 + Y**2 + Z**2
        
        if equation_type.startswith("nlse"):
            # Gaussian for NLSE
            u0 = np.exp(-r2)
            return u0.astype(np.complex128)
        else:
            # Gaussian for real-space equations
            u0 = np.exp(-r2)
            v0 = np.zeros_like(u0)
            return u0.astype(np.float64), v0.astype(np.float64)

def calculate_energy(u, v=None, equation_type="kg", dx=0.1):
    """Calculate energy for the given equation type."""
    if equation_type.startswith("nlse"):
        # For NLSE: E = ∫(|∇u|² + |u|⁴) dx
        grad_u_squared = np.abs(np.gradient(u, dx)[0])**2
        if len(u.shape) > 1:
            grad_u_squared += np.abs(np.gradient(u, dx)[1])**2
        if len(u.shape) > 2:
            grad_u_squared += np.abs(np.gradient(u, dx)[2])**2
        
        return np.sum(grad_u_squared + np.abs(u)**4) * dx**len(u.shape)
    else:
        # For real-space: E = ∫(v² + |∇u|² + V(u)) dx
        # where V(u) depends on the equation type
        grad_u_squared = np.abs(np.gradient(u, dx)[0])**2
        if len(u.shape) > 1:
            grad_u_squared += np.abs(np.gradient(u, dx)[1])**2
        if len(u.shape) > 2:
            grad_u_squared += np.abs(np.gradient(u, dx)[2])**2
        
        if equation_type == "kg":
            potential = u**2
        elif equation_type == "sg":
            potential = 1 - np.cos(u)
        elif equation_type == "phi4":
            potential = 0.25 * (u**2 - 1)**2
        else:
            potential = u**2  # Default
        
        return np.sum(v**2 + grad_u_squared + potential) * dx**len(u.shape)

def run_comparison(equation_type, dim, methods, device_types, n=128, T=10.0, nt=1000, snapshots=100):
    """Run comparison between different methods and device types."""
    # Create temporary directory for files
    tmp_dir = Path("./tmp_comparison")
    tmp_dir.mkdir(exist_ok=True)
    
    # Create initial condition
    if equation_type.startswith("nlse"):
        u0 = create_initial_condition(n, dim, equation_type)
        np.save(tmp_dir / "initial_u.npy", u0)
        initial_v_arg = ""
    else:
        u0, v0 = create_initial_condition(n, dim, equation_type)
        np.save(tmp_dir / "initial_u.npy", u0)
        np.save(tmp_dir / "initial_v.npy", v0)
        initial_v_arg = f"--initial-v={tmp_dir}/initial_v.npy"
    
    # Run simulations
    results = {}
    for device in device_types:
        for method in methods:
            # Skip incompatible method-equation combinations
            if equation_type.startswith("nlse") and method in ["gautschi", "sv"]:
                if (equation_type == "nlse-cubic" and method == "gautschi") or \
                   (equation_type == "nlse-cubic-quintic" and method == "gautschi") or \
                   (equation_type == "nlse-saturating" and method == "gautschi"):
                    pass  # These combinations are supported
                else:
                    continue
            
            if not equation_type.startswith("nlse") and method in ["ss2", "sewi"]:
                continue
            
            # Set output files
            output_file = tmp_dir / f"traj_{equation_type}_{dim}d_{device}_{method}.npy"
            velocity_file_arg = ""
            if not equation_type.startswith("nlse"):
                velocity_file = tmp_dir / f"vel_{equation_type}_{dim}d_{device}_{method}.npy"
                velocity_file_arg = f"--velocity-file={velocity_file}"
            
            # Determine which binary to use
            if equation_type.startswith("nlse"):
                binary = "./nlwave_bin_complex"
            else:
                binary = "./nlwave_bin_real"
            
            # Build command
            cmd = [
                binary,
                f"--device={device}",
                f"--system-type={equation_type}",
                f"--dim={dim}",
                f"--n={n}",
                f"--L=10.0",
                f"--T={T}",
                f"--nt={nt}",
                f"--snapshots={snapshots}",
                f"--method={method}",
                f"--initial-u={tmp_dir}/initial_u.npy",
                f"--trajectory-file={output_file}"
            ]
            
            if initial_v_arg:
                cmd.append(initial_v_arg)
            
            if velocity_file_arg:
                cmd.append(velocity_file_arg)
            
            # Run command
            print(f"Running {equation_type} {dim}D with {device}/{method}...")
            start_time = time.time()
            try:
                subprocess.run(cmd, check=True)
                end_time = time.time()
                
                # Load results and calculate energy
                traj = np.load(output_file)
                
                if not equation_type.startswith("nlse"):
                    vel = np.load(velocity_file)
                    energies = [calculate_energy(traj[i], vel[i], equation_type) for i in range(len(traj))]
                else:
                    energies = [calculate_energy(traj[i], None, equation_type) for i in range(len(traj))]
                
                results[f"{device}_{method}"] = {
                    "runtime": end_time - start_time,
                    "energies": energies
                }
                
                print(f"  Completed in {end_time - start_time:.2f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"  Failed: {e}")
    
    return results

def plot_results(results, equation_type, dim, output_dir="./plots"):
    """Plot comparison results."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot energy conservation
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        device, method = label.split("_")
        plt.plot(data["energies"], label=f"{device}/{method}")
    
    plt.title(f"Energy Conservation: {equation_type} {dim}D")
    plt.xlabel("Snapshot")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"energy_{equation_type}_{dim}d.png", dpi=300)
    
    # Plot runtime comparison
    plt.figure(figsize=(10, 6))
    labels = []
    runtimes = []
    for label, data in results.items():
        labels.append(label)
        runtimes.append(data["runtime"])
    
    plt.bar(labels, runtimes)
    plt.title(f"Runtime Comparison: {equation_type} {dim}D")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"runtime_{equation_type}_{dim}d.png", dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Compare different methods and devices for nonlinear wave equations")
    parser.add_argument("--equation", choices=["kg", "sg", "nlse-cubic", "nlse-saturating"], default="kg", help="Equation type")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimension")
    parser.add_argument("--n", type=int, default=128, help="Grid size")
    parser.add_argument("--T", type=float, default=10.0, help="Simulation time")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=100, help="Number of snapshots")
    parser.add_argument("--methods", nargs="+", default=["gautschi", "sv", "ss2", "sewi"], help="Methods to compare")
    parser.add_argument("--devices", nargs="+", default=["host", "cuda"], help="Devices to compare")
    parser.add_argument("--output-dir", default="./plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Run all comparisons
    if args.equation == "all":
        equations = ["kg", "sg", "nlse-cubic", "nlse-saturating"]
    else:
        equations = [args.equation]
    
    for eq in equations:
        results = run_comparison(
            equation_type=eq,
            dim=args.dim,
            methods=args.methods,
            device_types=args.devices,
            n=args.n,
            T=args.T,
            nt=args.nt,
            snapshots=args.snapshots
        )
        
        plot_results(results, eq, args.dim, args.output_dir)

if __name__ == "__main__":
    main()
