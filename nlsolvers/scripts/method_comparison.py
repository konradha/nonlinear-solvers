#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import argparse
from pathlib import Path

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def calculate_energy(traj, vel=None, system_type="kge", dim=2):
    if system_type in ["kge", "sge"]:
        # For real-space wave equations, energy is kinetic + potential
        # E = 0.5 * (v^2 + (grad u)^2 + V(u))
        # We'll just use a simple approximation here
        if vel is None:
            print("Warning: Velocity data required for energy calculation")
            return None
        
        kinetic_energy = 0.5 * np.sum(vel**2, axis=tuple(range(1, dim+1)))
        gradient_energy = 0.5 * np.sum(np.gradient(traj, axis=tuple(range(1, dim+1)))**2)
        
        if system_type == "kge":
            potential_energy = 0.5 * np.sum(traj**2, axis=tuple(range(1, dim+1)))
        elif system_type == "sge":
            potential_energy = np.sum(1 - np.cos(traj), axis=tuple(range(1, dim+1)))
        
        return kinetic_energy + gradient_energy + potential_energy
    
    elif system_type == "nlse":
        # For NLSE, energy is |grad u|^2 - |u|^4/2 (for cubic nonlinearity)
        gradient_energy = np.sum(np.abs(np.gradient(traj, axis=tuple(range(1, dim+1))))**2, axis=tuple(range(1, dim+1)))
        nonlinear_energy = -0.5 * np.sum(np.abs(traj)**4, axis=tuple(range(1, dim+1)))
        return gradient_energy + nonlinear_energy
    
    return None

def create_initial_conditions(system_type, dim, n):
    if dim == 2:
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        if system_type in ["kge", "sge"]:
            # Gaussian pulse for real-space waves
            u0 = np.exp(-R**2)
            v0 = np.zeros_like(u0)
            return u0, v0
        elif system_type == "nlse":
            # Gaussian for NLSE
            u0 = np.exp(-R**2) * np.exp(1j * 0)
            return u0
    
    elif dim == 3:
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        z = np.linspace(-10, 10, n)
        X, Y, Z = np.meshgrid(x, y, z)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        if system_type in ["kge", "sge"]:
            # Gaussian pulse for real-space waves
            u0 = np.exp(-R**2)
            v0 = np.zeros_like(u0)
            return u0, v0
        elif system_type == "nlse":
            # Gaussian for NLSE
            u0 = np.exp(-R**2) * np.exp(1j * 0)
            return u0
    
    return None

def compare_kge_host_vs_device(dim, method="gautschi"):
    print(f"Comparing KGE host vs device in {dim}D using {method} method")
    
    # Create directory for results
    output_dir = Path(f"comparison_results/kge_{dim}d_{method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create initial conditions
    n = 100
    if dim == 2:
        u0, v0 = create_initial_conditions("kge", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        np.save(output_dir / "initial_v.npy", v0)
        
        # Run host simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "klein-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", method,
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_host.npy"),
            "--velocity-file", str(output_dir / "vel_host.npy")
        ])
        
        # Run device simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "cuda",
            "--system-type", "klein-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", method,
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_device.npy"),
            "--velocity-file", str(output_dir / "vel_device.npy")
        ])
        
        # Load results and calculate energy
        traj_host = np.load(output_dir / "traj_host.npy")
        vel_host = np.load(output_dir / "vel_host.npy")
        traj_device = np.load(output_dir / "traj_device.npy")
        vel_device = np.load(output_dir / "vel_device.npy")
        
        energy_host = calculate_energy(traj_host, vel_host, "kge", dim)
        energy_device = calculate_energy(traj_device, vel_device, "kge", dim)
        
        # Plot energy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(energy_host, label="Host")
        plt.plot(energy_device, label="Device")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"KGE {dim}D Energy Comparison: Host vs Device ({method})")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_comparison.png")
        
    elif dim == 3:
        u0, v0 = create_initial_conditions("kge", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        np.save(output_dir / "initial_v.npy", v0)
        
        # Run host simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "klein-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", method,
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_host.npy"),
            "--velocity-file", str(output_dir / "vel_host.npy")
        ])
        
        # Load results and calculate energy
        traj_host = np.load(output_dir / "traj_host.npy")
        vel_host = np.load(output_dir / "vel_host.npy")
        
        energy_host = calculate_energy(traj_host, vel_host, "kge", dim)
        
        # Plot energy
        plt.figure(figsize=(10, 6))
        plt.plot(energy_host, label="Host")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"KGE {dim}D Energy: Host ({method})")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_host.png")

def compare_sge_host_methods(dim):
    print(f"Comparing SGE host SV vs Gautschi in {dim}D")
    
    # Create directory for results
    output_dir = Path(f"comparison_results/sge_{dim}d_methods")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create initial conditions
    n = 100
    if dim == 2:
        u0, v0 = create_initial_conditions("sge", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        np.save(output_dir / "initial_v.npy", v0)
        
        # Run Gautschi simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "sine-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "gautschi",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_gautschi.npy"),
            "--velocity-file", str(output_dir / "vel_gautschi.npy")
        ])
        
        # Run SV simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "sine-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "strang",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_sv.npy"),
            "--velocity-file", str(output_dir / "vel_sv.npy")
        ])
        
        # Load results and calculate energy
        traj_gautschi = np.load(output_dir / "traj_gautschi.npy")
        vel_gautschi = np.load(output_dir / "vel_gautschi.npy")
        traj_sv = np.load(output_dir / "traj_sv.npy")
        vel_sv = np.load(output_dir / "vel_sv.npy")
        
        energy_gautschi = calculate_energy(traj_gautschi, vel_gautschi, "sge", dim)
        energy_sv = calculate_energy(traj_sv, vel_sv, "sge", dim)
        
        # Plot energy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(energy_gautschi, label="Gautschi")
        plt.plot(energy_sv, label="Strang-Splitting")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"SGE {dim}D Energy Comparison: Gautschi vs Strang-Splitting")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_comparison.png")
        
    elif dim == 3:
        u0, v0 = create_initial_conditions("sge", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        np.save(output_dir / "initial_v.npy", v0)
        
        # Run Gautschi simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "sine-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "gautschi",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_gautschi.npy"),
            "--velocity-file", str(output_dir / "vel_gautschi.npy")
        ])
        
        # Run SV simulation
        run_command([
            "./bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "sine-gordon",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "strang",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--initial-v", str(output_dir / "initial_v.npy"),
            "--trajectory-file", str(output_dir / "traj_sv.npy"),
            "--velocity-file", str(output_dir / "vel_sv.npy")
        ])
        
        # Load results and calculate energy
        traj_gautschi = np.load(output_dir / "traj_gautschi.npy")
        vel_gautschi = np.load(output_dir / "vel_gautschi.npy")
        traj_sv = np.load(output_dir / "traj_sv.npy")
        vel_sv = np.load(output_dir / "vel_sv.npy")
        
        energy_gautschi = calculate_energy(traj_gautschi, vel_gautschi, "sge", dim)
        energy_sv = calculate_energy(traj_sv, vel_sv, "sge", dim)
        
        # Plot energy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(energy_gautschi, label="Gautschi")
        plt.plot(energy_sv, label="Strang-Splitting")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"SGE {dim}D Energy Comparison: Gautschi vs Strang-Splitting")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_comparison.png")

def compare_nlse_cubic_host_vs_device(dim):
    print(f"Comparing NLSE cubic host vs device in {dim}D")
    
    # Create directory for results
    output_dir = Path(f"comparison_results/nlse_cubic_{dim}d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create initial conditions
    n = 100
    if dim == 2:
        u0 = create_initial_conditions("nlse", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        
        # Run host simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "cubic",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_host.npy")
        ])
        
        # Run device simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "cuda",
            "--system-type", "cubic",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_device.npy")
        ])
        
        # Load results and calculate energy
        traj_host = np.load(output_dir / "traj_host.npy")
        traj_device = np.load(output_dir / "traj_device.npy")
        
        energy_host = calculate_energy(traj_host, None, "nlse", dim)
        energy_device = calculate_energy(traj_device, None, "nlse", dim)
        
        # Plot energy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(energy_host, label="Host")
        plt.plot(energy_device, label="Device")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"NLSE Cubic {dim}D Energy Comparison: Host vs Device")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_comparison.png")
        
    elif dim == 3:
        u0 = create_initial_conditions("nlse", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        
        # Run host simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "cubic",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_host.npy")
        ])
        
        # Load results and calculate energy
        traj_host = np.load(output_dir / "traj_host.npy")
        
        energy_host = calculate_energy(traj_host, None, "nlse", dim)
        
        # Plot energy
        plt.figure(figsize=(10, 6))
        plt.plot(energy_host, label="Host")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"NLSE Cubic {dim}D Energy: Host")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_host.png")

def compare_nlse_saturable_host_methods(dim):
    print(f"Comparing NLSE saturable host SV vs Gautschi in {dim}D")
    
    # Create directory for results
    output_dir = Path(f"comparison_results/nlse_saturable_{dim}d_methods")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create initial conditions
    n = 100
    if dim == 2:
        u0 = create_initial_conditions("nlse", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        
        # Run Gautschi simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "saturable",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "gautschi",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_gautschi.npy")
        ])
        
        # Run SV simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "saturable",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "strang",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_sv.npy")
        ])
        
        # Load results and calculate energy
        traj_gautschi = np.load(output_dir / "traj_gautschi.npy")
        traj_sv = np.load(output_dir / "traj_sv.npy")
        
        energy_gautschi = calculate_energy(traj_gautschi, None, "nlse", dim)
        energy_sv = calculate_energy(traj_sv, None, "nlse", dim)
        
        # Plot energy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(energy_gautschi, label="Gautschi")
        plt.plot(energy_sv, label="Strang-Splitting")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"NLSE Saturable {dim}D Energy Comparison: Gautschi vs Strang-Splitting")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_comparison.png")
        
    elif dim == 3:
        u0 = create_initial_conditions("nlse", dim, n)
        np.save(output_dir / "initial_u.npy", u0)
        
        # Run Gautschi simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "saturable",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "gautschi",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_gautschi.npy")
        ])
        
        # Run SV simulation
        run_command([
            "./bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "saturable",
            "--dim", str(dim),
            "--L", "10.0",
            "--n", str(n),
            "--T", "10.0",
            "--nt", "1000",
            "--snapshots", "32",
            "--method", "strang",
            "--initial-u", str(output_dir / "initial_u.npy"),
            "--trajectory-file", str(output_dir / "traj_sv.npy")
        ])
        
        # Load results and calculate energy
        traj_gautschi = np.load(output_dir / "traj_gautschi.npy")
        traj_sv = np.load(output_dir / "traj_sv.npy")
        
        energy_gautschi = calculate_energy(traj_gautschi, None, "nlse", dim)
        energy_sv = calculate_energy(traj_sv, None, "nlse", dim)
        
        # Plot energy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(energy_gautschi, label="Gautschi")
        plt.plot(energy_sv, label="Strang-Splitting")
        plt.xlabel("Snapshot")
        plt.ylabel("Energy")
        plt.title(f"NLSE Saturable {dim}D Energy Comparison: Gautschi vs Strang-Splitting")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "energy_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Compare different methods and implementations")
    parser.add_argument("--all", action="store_true", help="Run all comparisons")
    parser.add_argument("--kge-2d", action="store_true", help="Compare KGE host vs device in 2D")
    parser.add_argument("--kge-3d", action="store_true", help="Compare KGE host vs device in 3D")
    parser.add_argument("--sge-2d", action="store_true", help="Compare SGE host SV vs Gautschi in 2D")
    parser.add_argument("--sge-3d", action="store_true", help="Compare SGE host SV vs Gautschi in 3D")
    parser.add_argument("--nlse-cubic-2d", action="store_true", help="Compare NLSE cubic host vs device in 2D")
    parser.add_argument("--nlse-cubic-3d", action="store_true", help="Compare NLSE cubic host vs device in 3D")
    parser.add_argument("--nlse-saturable-2d", action="store_true", help="Compare NLSE saturable host SV vs Gautschi in 2D")
    parser.add_argument("--nlse-saturable-3d", action="store_true", help="Compare NLSE saturable host SV vs Gautschi in 3D")
    
    args = parser.parse_args()
    
    if args.all or args.kge_2d:
        compare_kge_host_vs_device(2)
    
    if args.all or args.kge_3d:
        compare_kge_host_vs_device(3)
    
    if args.all or args.sge_2d:
        compare_sge_host_methods(2)
    
    if args.all or args.sge_3d:
        compare_sge_host_methods(3)
    
    if args.all or args.nlse_cubic_2d:
        compare_nlse_cubic_host_vs_device(2)
    
    if args.all or args.nlse_cubic_3d:
        compare_nlse_cubic_host_vs_device(3)
    
    if args.all or args.nlse_saturable_2d:
        compare_nlse_saturable_host_methods(2)
    
    if args.all or args.nlse_saturable_3d:
        compare_nlse_saturable_host_methods(3)

if __name__ == "__main__":
    main()
