#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import argparse
from pathlib import Path

def create_initial_condition(dim, n, type='gaussian'):
    """Create initial condition for the simulation."""
    if dim == 2:
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        if type == 'gaussian':
            u = np.exp(-(X**2 + Y**2) / 2)
        elif type == 'sine':
            u = np.sin(X) * np.sin(Y)
        else:
            raise ValueError(f"Unknown initial condition type: {type}")
    elif dim == 3:
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        z = np.linspace(-10, 10, n)
        X, Y, Z = np.meshgrid(x, y, z)
        if type == 'gaussian':
            u = np.exp(-(X**2 + Y**2 + Z**2) / 2)
        elif type == 'sine':
            u = np.sin(X) * np.sin(Y) * np.sin(Z)
        else:
            raise ValueError(f"Unknown initial condition type: {type}")
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    return u

def create_initial_velocity(dim, n, type='zero'):
    """Create initial velocity for the simulation."""
    if dim == 2:
        if type == 'zero':
            v = np.zeros((n, n))
        elif type == 'gaussian':
            x = np.linspace(-10, 10, n)
            y = np.linspace(-10, 10, n)
            X, Y = np.meshgrid(x, y)
            v = 0.1 * np.exp(-(X**2 + Y**2) / 4)
        else:
            raise ValueError(f"Unknown initial velocity type: {type}")
    elif dim == 3:
        if type == 'zero':
            v = np.zeros((n, n, n))
        elif type == 'gaussian':
            x = np.linspace(-10, 10, n)
            y = np.linspace(-10, 10, n)
            z = np.linspace(-10, 10, n)
            X, Y, Z = np.meshgrid(x, y, z)
            v = 0.1 * np.exp(-(X**2 + Y**2 + Z**2) / 4)
        else:
            raise ValueError(f"Unknown initial velocity type: {type}")
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    return v

def create_complex_initial_condition(dim, n, type='gaussian'):
    """Create complex initial condition for the simulation."""
    if dim == 2:
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, y)
        if type == 'gaussian':
            u = np.exp(-(X**2 + Y**2) / 2) * np.exp(1j * (X + Y))
        elif type == 'vortex':
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            u = np.tanh(r) * np.exp(1j * theta)
        else:
            raise ValueError(f"Unknown initial condition type: {type}")
    elif dim == 3:
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        z = np.linspace(-10, 10, n)
        X, Y, Z = np.meshgrid(x, y, z)
        if type == 'gaussian':
            u = np.exp(-(X**2 + Y**2 + Z**2) / 2) * np.exp(1j * (X + Y + Z))
        elif type == 'vortex':
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            u = np.tanh(r) * np.exp(1j * theta) * np.exp(-Z**2 / 4)
        else:
            raise ValueError(f"Unknown initial condition type: {type}")
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    return u

def calculate_energy(u, v=None, dx=0.1, system_type='klein-gordon'):
    """Calculate energy of the system."""
    if system_type in ['klein-gordon', 'sine-gordon']:
        # For real-valued wave equations
        if v is None:
            raise ValueError("Velocity is required for real-valued wave equations")
        
        # Kinetic energy
        kinetic = 0.5 * np.sum(v**2) * dx**u.ndim
        
        # Potential energy
        if u.ndim == 2:
            ux = np.gradient(u, dx, axis=0)
            uy = np.gradient(u, dx, axis=1)
            gradient_term = np.sum(ux**2 + uy**2) * dx**2
        elif u.ndim == 3:
            ux = np.gradient(u, dx, axis=0)
            uy = np.gradient(u, dx, axis=1)
            uz = np.gradient(u, dx, axis=2)
            gradient_term = np.sum(ux**2 + uy**2 + uz**2) * dx**3
        
        if system_type == 'klein-gordon':
            potential = 0.5 * (gradient_term + np.sum(u**2) * dx**u.ndim)
        elif system_type == 'sine-gordon':
            potential = 0.5 * gradient_term + np.sum(1 - np.cos(u)) * dx**u.ndim
        
        return kinetic + potential
    
    elif system_type == 'nlse-cubic':
        # For NLSE
        if u.ndim == 2:
            ux = np.gradient(u, dx, axis=0)
            uy = np.gradient(u, dx, axis=1)
            gradient_term = np.sum(np.abs(ux)**2 + np.abs(uy)**2) * dx**2
        elif u.ndim == 3:
            ux = np.gradient(u, dx, axis=0)
            uy = np.gradient(u, dx, axis=1)
            uz = np.gradient(u, dx, axis=2)
            gradient_term = np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2) * dx**3
        
        # Nonlinear term
        nonlinear_term = -0.5 * np.sum(np.abs(u)**4) * dx**u.ndim
        
        return gradient_term + nonlinear_term
    
    else:
        raise ValueError(f"Unsupported system type: {system_type}")

def run_comparison():
    """Run comparison between different methods."""
    # Create output directory
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    n = 100
    L = 10.0
    T = 10.0
    nt = 1000
    snapshots = 32
    
    # Create initial conditions
    print("Creating initial conditions...")
    
    # KGE: host versus device in both 2d and 3d (Stormer-Verlet)
    for dim in [2, 3]:
        print(f"Running KGE comparison in {dim}D...")
        
        # Create initial conditions
        u_init = create_initial_condition(dim, n, 'gaussian')
        v_init = create_initial_velocity(dim, n, 'zero')
        
        # Save initial conditions
        np.save(output_dir / f"kge_{dim}d_initial_u.npy", u_init)
        np.save(output_dir / f"kge_{dim}d_initial_v.npy", v_init)
        
        # Run host simulation
        print(f"  Running KGE on host...")
        host_cmd = [
            "./build/bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "klein-gordon",
            "--dim", str(dim),
            "--L", str(L),
            "--n", str(n),
            "--T", str(T),
            "--nt", str(nt),
            "--snapshots", str(snapshots),
            "--method", "gautschi",
            "--initial-u", str(output_dir / f"kge_{dim}d_initial_u.npy"),
            "--initial-v", str(output_dir / f"kge_{dim}d_initial_v.npy"),
            "--trajectory-file", str(output_dir / f"kge_{dim}d_host_traj.npy"),
            "--velocity-file", str(output_dir / f"kge_{dim}d_host_vel.npy")
        ]
        subprocess.run(host_cmd)
        
        # Run device simulation if CUDA is available
        try:
            print(f"  Running KGE on device...")
            device_cmd = [
                "./build/bin/nlwave_bin_real",
                "--device", "cuda",
                "--system-type", "klein-gordon",
                "--dim", str(dim),
                "--L", str(L),
                "--n", str(n),
                "--T", str(T),
                "--nt", str(nt),
                "--snapshots", str(snapshots),
                "--method", "gautschi",
                "--initial-u", str(output_dir / f"kge_{dim}d_initial_u.npy"),
                "--initial-v", str(output_dir / f"kge_{dim}d_initial_v.npy"),
                "--trajectory-file", str(output_dir / f"kge_{dim}d_device_traj.npy"),
                "--velocity-file", str(output_dir / f"kge_{dim}d_device_vel.npy")
            ]
            subprocess.run(device_cmd)
        except Exception as e:
            print(f"  Error running on device: {e}")
    
    # SGE: host SV versus Gautschi in both 2d and 3d
    for dim in [2, 3]:
        print(f"Running SGE comparison in {dim}D...")
        
        # Create initial conditions
        u_init = create_initial_condition(dim, n, 'gaussian')
        v_init = create_initial_velocity(dim, n, 'zero')
        
        # Save initial conditions
        np.save(output_dir / f"sge_{dim}d_initial_u.npy", u_init)
        np.save(output_dir / f"sge_{dim}d_initial_v.npy", v_init)
        
        # Run with Stormer-Verlet
        print(f"  Running SGE with Stormer-Verlet...")
        sv_cmd = [
            "./build/bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "sine-gordon",
            "--dim", str(dim),
            "--L", str(L),
            "--n", str(n),
            "--T", str(T),
            "--nt", str(nt),
            "--snapshots", str(snapshots),
            "--method", "stormer-verlet",
            "--initial-u", str(output_dir / f"sge_{dim}d_initial_u.npy"),
            "--initial-v", str(output_dir / f"sge_{dim}d_initial_v.npy"),
            "--trajectory-file", str(output_dir / f"sge_{dim}d_sv_traj.npy"),
            "--velocity-file", str(output_dir / f"sge_{dim}d_sv_vel.npy")
        ]
        subprocess.run(sv_cmd)
        
        # Run with Gautschi
        print(f"  Running SGE with Gautschi...")
        gautschi_cmd = [
            "./build/bin/nlwave_bin_real",
            "--device", "host",
            "--system-type", "sine-gordon",
            "--dim", str(dim),
            "--L", str(L),
            "--n", str(n),
            "--T", str(T),
            "--nt", str(nt),
            "--snapshots", str(snapshots),
            "--method", "gautschi",
            "--initial-u", str(output_dir / f"sge_{dim}d_initial_u.npy"),
            "--initial-v", str(output_dir / f"sge_{dim}d_initial_v.npy"),
            "--trajectory-file", str(output_dir / f"sge_{dim}d_gautschi_traj.npy"),
            "--velocity-file", str(output_dir / f"sge_{dim}d_gautschi_vel.npy")
        ]
        subprocess.run(gautschi_cmd)
    
    # NLSE cubic: host versus device in both 2d and 3d (SS2)
    for dim in [2, 3]:
        print(f"Running NLSE cubic comparison in {dim}D...")
        
        # Create initial conditions
        u_init = create_complex_initial_condition(dim, n, 'gaussian')
        
        # Save initial conditions
        np.save(output_dir / f"nlse_cubic_{dim}d_initial_u.npy", u_init)
        
        # Run host simulation
        print(f"  Running NLSE cubic on host...")
        host_cmd = [
            "./build/bin/nlwave_bin_nlse",
            "--device", "host",
            "--system-type", "nlse-cubic",
            "--dim", str(dim),
            "--L", str(L),
            "--n", str(n),
            "--T", str(T),
            "--nt", str(nt),
            "--snapshots", str(snapshots),
            "--method", "strang",
            "--initial-u", str(output_dir / f"nlse_cubic_{dim}d_initial_u.npy"),
            "--trajectory-file", str(output_dir / f"nlse_cubic_{dim}d_host_traj.npy")
        ]
        subprocess.run(host_cmd)
        
        # Run device simulation if CUDA is available
        try:
            print(f"  Running NLSE cubic on device...")
            device_cmd = [
                "./build/bin/nlwave_bin_nlse",
                "--device", "cuda",
                "--system-type", "nlse-cubic",
                "--dim", str(dim),
                "--L", str(L),
                "--n", str(n),
                "--T", str(T),
                "--nt", str(nt),
                "--snapshots", str(snapshots),
                "--method", "strang",
                "--initial-u", str(output_dir / f"nlse_cubic_{dim}d_initial_u.npy"),
                "--trajectory-file", str(output_dir / f"nlse_cubic_{dim}d_device_traj.npy")
            ]
            subprocess.run(device_cmd)
        except Exception as e:
            print(f"  Error running on device: {e}")
    
    # Plot energy comparison
    print("Plotting energy comparison...")
    
    # KGE energy comparison
    for dim in [2, 3]:
        print(f"  Plotting KGE energy comparison in {dim}D...")
        
        # Load trajectories
        host_traj = np.load(output_dir / f"kge_{dim}d_host_traj.npy")
        host_vel = np.load(output_dir / f"kge_{dim}d_host_vel.npy")
        
        # Calculate energy
        host_energy = np.zeros(host_traj.shape[0])
        for i in range(host_traj.shape[0]):
            host_energy[i] = calculate_energy(host_traj[i], host_vel[i], dx=2*L/(n-1), system_type='klein-gordon')
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, T, host_traj.shape[0]), host_energy, label='Host')
        
        try:
            # Load device trajectory if available
            device_traj = np.load(output_dir / f"kge_{dim}d_device_traj.npy")
            device_vel = np.load(output_dir / f"kge_{dim}d_device_vel.npy")
            
            # Calculate energy
            device_energy = np.zeros(device_traj.shape[0])
            for i in range(device_traj.shape[0]):
                device_energy[i] = calculate_energy(device_traj[i], device_vel[i], dx=2*L/(n-1), system_type='klein-gordon')
            
            # Plot
            plt.plot(np.linspace(0, T, device_traj.shape[0]), device_energy, label='Device')
        except:
            pass
        
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title(f'KGE Energy Comparison ({dim}D)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"kge_{dim}d_energy_comparison.png")
        plt.close()
    
    # SGE energy comparison
    for dim in [2, 3]:
        print(f"  Plotting SGE energy comparison in {dim}D...")
        
        # Load trajectories
        sv_traj = np.load(output_dir / f"sge_{dim}d_sv_traj.npy")
        sv_vel = np.load(output_dir / f"sge_{dim}d_sv_vel.npy")
        gautschi_traj = np.load(output_dir / f"sge_{dim}d_gautschi_traj.npy")
        gautschi_vel = np.load(output_dir / f"sge_{dim}d_gautschi_vel.npy")
        
        # Calculate energy
        sv_energy = np.zeros(sv_traj.shape[0])
        gautschi_energy = np.zeros(gautschi_traj.shape[0])
        
        for i in range(sv_traj.shape[0]):
            sv_energy[i] = calculate_energy(sv_traj[i], sv_vel[i], dx=2*L/(n-1), system_type='sine-gordon')
            gautschi_energy[i] = calculate_energy(gautschi_traj[i], gautschi_vel[i], dx=2*L/(n-1), system_type='sine-gordon')
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, T, sv_traj.shape[0]), sv_energy, label='Stormer-Verlet')
        plt.plot(np.linspace(0, T, gautschi_traj.shape[0]), gautschi_energy, label='Gautschi')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title(f'SGE Energy Comparison ({dim}D)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"sge_{dim}d_energy_comparison.png")
        plt.close()
    
    # NLSE cubic energy comparison
    for dim in [2, 3]:
        print(f"  Plotting NLSE cubic energy comparison in {dim}D...")
        
        # Load trajectories
        host_traj = np.load(output_dir / f"nlse_cubic_{dim}d_host_traj.npy")
        
        # Calculate energy
        host_energy = np.zeros(host_traj.shape[0])
        for i in range(host_traj.shape[0]):
            host_energy[i] = calculate_energy(host_traj[i], dx=2*L/(n-1), system_type='nlse-cubic')
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, T, host_traj.shape[0]), host_energy, label='Host')
        
        try:
            # Load device trajectory if available
            device_traj = np.load(output_dir / f"nlse_cubic_{dim}d_device_traj.npy")
            
            # Calculate energy
            device_energy = np.zeros(device_traj.shape[0])
            for i in range(device_traj.shape[0]):
                device_energy[i] = calculate_energy(device_traj[i], dx=2*L/(n-1), system_type='nlse-cubic')
            
            # Plot
            plt.plot(np.linspace(0, T, device_traj.shape[0]), device_energy, label='Device')
        except:
            pass
        
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title(f'NLSE Cubic Energy Comparison ({dim}D)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"nlse_cubic_{dim}d_energy_comparison.png")
        plt.close()
    
    print("Comparison completed. Results saved in the 'comparison_results' directory.")

if __name__ == "__main__":
    run_comparison()
