import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import uuid
import subprocess
from pathlib import Path
import h5py
import sys

from visualization import animate_simulation

try:
    from real_sampler import RealWaveSampler
except ImportError:
    print("Warning: real_sampler not found. Using default IC.")
    RealWaveSampler = None

def compute_gradient_sq_norm(u, dx, dy):
    u = u.reshape(int(np.sqrt(u.size)), -1)
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    return np.sum(ux**2 + uy**2) * dx * dy

def compute_hamiltonian_sge(u, ut, dx, dy, m, **kwargs):
    energy_ut = 0.5 * np.sum(ut**2) * dx * dy
    energy_grad = 0.5 * compute_gradient_sq_norm(u, dx, dy)
    energy_pot = np.sum(m * (1.0 - np.cos(u.reshape(m.shape)))) * dx * dy
    return energy_ut, energy_grad, energy_pot

def compute_hamiltonian_kg(u, ut, dx, dy, m, **kwargs):
    energy_ut = 0.5 * np.sum(ut**2) * dx * dy
    energy_grad = 0.5 * compute_gradient_sq_norm(u, dx, dy)
    energy_pot = 0.5 * np.sum(m * u**2) * dx * dy
    return energy_ut, energy_grad, energy_pot

def compute_hamiltonian_phi4(u, ut, dx, dy, m, **kwargs):
    energy_ut = 0.5 * np.sum(ut**2) * dx * dy
    energy_grad = 0.5 * compute_gradient_sq_norm(u, dx, dy)
    energy_pot = np.sum(m * (0.25 * u**4 - 0.5 * u**2)) * dx * dy
    return energy_ut, energy_grad, energy_pot

def compute_hamiltonian_sge_double(u, ut, dx, dy, m, **kwargs):
    energy_ut = 0.5 * np.sum(ut**2) * dx * dy
    energy_grad = 0.5 * compute_gradient_sq_norm(u, dx, dy)
    u_reshaped = u.reshape(m.shape)
    energy_pot = np.sum(m * (1.0 - np.cos(u_reshaped)) + 0.3 * m * (1.0 - np.cos(2.0*u_reshaped))) * dx * dy
    return energy_ut, energy_grad, energy_pot

def compute_hamiltonian_sge_hyperbolic(u, ut, dx, dy, m, **kwargs):
    energy_ut = 0.5 * np.sum(ut**2) * dx * dy
    energy_grad = 0.5 * compute_gradient_sq_norm(u, dx, dy)
    energy_pot = np.sum(m * (np.cosh(u.reshape(m.shape)) - 1.0)) * dx * dy
    return energy_ut, energy_grad, energy_pot

SYSTEM_CONFIG_WAVE = {
    'SGE': {
        'dtype': np.float64, 'params': ['m_value'], 'hamiltonian_func': compute_hamiltonian_sge,
        'needs_v0': True, 'sampler_class': 'RealWaveSampler', 'extra_args_spec': []
    },
    'KG': {
        'dtype': np.float64, 'params': ['m_value'], 'hamiltonian_func': compute_hamiltonian_kg,
        'needs_v0': True, 'sampler_class': 'RealWaveSampler', 'extra_args_spec': []
    },
    'Phi4': {
        'dtype': np.float64, 'params': ['m_value'], 'hamiltonian_func': compute_hamiltonian_phi4,
        'needs_v0': True, 'sampler_class': 'RealWaveSampler', 'extra_args_spec': []
    },
    'SGE_double': {
        'dtype': np.float64, 'params': ['m_value'], 'hamiltonian_func': compute_hamiltonian_sge_double,
        'needs_v0': True, 'sampler_class': 'RealWaveSampler', 'extra_args_spec': []
    },
    'SGE_hyperbolic': {
        'dtype': np.float64, 'params': ['m_value'], 'hamiltonian_func': compute_hamiltonian_sge_hyperbolic,
        'needs_v0': True, 'sampler_class': 'RealWaveSampler', 'extra_args_spec': []
    },
}

class WaveIntegratorComparer:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.system_config = SYSTEM_CONFIG_WAVE[args.system_type]
        self.dtype = self.system_config['dtype']
        self.needs_v0 = self.system_config['needs_v0']
        self.sampler = None
        self.setup_directories()
        self.sampler = self._get_sampler()

    def setup_directories(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir = self.output_dir / f"temp_{self.run_id}"
        self.temp_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def _get_sampler(self):
        if self.sampler is None:
             if RealWaveSampler:
                 self.sampler = RealWaveSampler(self.args.nx, self.args.ny, self.args.Lx)
             else:
                 raise RuntimeError("RealWaveSampler not available")
        return self.sampler

    def _generate_initial_conditions(self):
        u0_path = self.temp_dir / "u0.npy"
        v0_path = self.temp_dir / "v0.npy"
        m_path = self.temp_dir / "m.npy"

        if self.sampler:
            u0_np, v0_np = self.sampler.generate_initial_condition(phenomenon_type=self.args.ic_type)
            if hasattr(u0_np, 'numpy'): u0_np = u0_np.numpy()
            if hasattr(v0_np, 'numpy'): v0_np = v0_np.numpy()
            u0_np = u0_np.squeeze().astype(self.dtype)
            v0_np = v0_np.squeeze().astype(self.dtype)

        else:
            X, Y = np.meshgrid(np.linspace(-self.args.Lx, self.args.Lx, self.args.nx),
                               np.linspace(-self.args.Ly, self.args.Ly, self.args.ny), indexing='ij')
            sigma = self.args.Lx / 5.0
            u0_np = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            u0_np = u0_np.astype(self.dtype)
            v0_np = np.zeros_like(u0_np)

        m_np = np.full((self.args.ny, self.args.nx), self.args.m_value, dtype=np.float64)

        np.save(u0_path, u0_np)
        np.save(m_path, m_np)
        if self.needs_v0:
            np.save(v0_path, v0_np)

        return u0_path, v0_path, m_path

    def _build_command(self, exe_path, u0_path, v0_path, m_path, traj_path, vel_path):
        base_cmd = [
            str(exe_path),
            str(self.args.nx), str(self.args.ny),
            str(self.args.Lx), str(self.args.Ly),
            str(u0_path),
        ]
        if self.needs_v0: base_cmd.append(str(v0_path))
        base_cmd.append(str(traj_path))
        if self.needs_v0: base_cmd.append(str(vel_path))
        base_cmd.extend([
            str(self.args.T), str(self.args.nt),
            str(self.args.num_snapshots), str(m_path)
        ])
        return base_cmd

    def _run_single_simulation(self, integrator_name, exe_path, u0_path, v0_path, m_path):
        traj_path = self.temp_dir / f"traj_{integrator_name}.npy"
        vel_path = self.temp_dir / f"vel_{integrator_name}.npy"
        cmd = self._build_command(exe_path, u0_path, v0_path, m_path, traj_path, vel_path)
        print(f"Running {integrator_name}: {' '.join(cmd)}")
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stderr: print(f"STDERR ({integrator_name}):\n{result.stderr}\n---------")
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {integrator_name}:\nCmd: {' '.join(e.cmd)}\nCode: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            raise
        except Exception as e:
             raise Exception(e)

        end_time = time.time()
        walltime = end_time - start_time
        print(f"{integrator_name} finished in {walltime:.2f}s")
        return traj_path, vel_path, walltime

    def _load_trajectory(self, file_path):
         if not file_path.exists(): raise FileNotFoundError(f"Trajectory file not found: {file_path}")
         return np.load(file_path).astype(self.dtype)

    def _calculate_hamiltonian(self, traj_data, vel_data, m_np):
        hamiltonian_func = self.system_config['hamiltonian_func']
        num_snapshots = traj_data.shape[0]
        hamiltonian_values = np.zeros(num_snapshots)
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        dt_snap = self.args.T / (self.args.num_snapshots - 1) if self.args.num_snapshots > 1 else 0
        params_dict = {k: getattr(self.args, k) for k in self.system_config['params']}
        if vel_data is None and self.needs_v0:
             raise Exception
        elif vel_data is not None and vel_data.shape[0] != num_snapshots:
             raise Exception
        elif not self.needs_v0:
             vel_data = np.zeros_like(traj_data)

        for i in range(num_snapshots):
            u_snap = traj_data[i].flatten()
            m_flat = m_np.flatten()
            ut_snap = vel_data[i].flatten()
            h_args = {'u': u_snap, 'ut': ut_snap, 'dx': dx, 'dy': dy, 'm': m_flat}
            h_args.update(params_dict)
            u_sq, ut, pot = hamiltonian_func(**h_args)
            hamiltonian_values[i] = u_sq + ut + pot 
        return hamiltonian_values

    def _analyze_trajectory(self, traj_path, vel_path):
        traj_data = self._load_trajectory(traj_path)
        vel_data = self._load_trajectory(vel_path) if self.needs_v0 and vel_path.exists() else None
        m_path = self.temp_dir / "m.npy"
        m_np = np.load(m_path)
        hamiltonian = self._calculate_hamiltonian(traj_data, vel_data, m_np)
        h0 = hamiltonian[0]
        hamiltonian_error = np.array([np.nan] + list(np.log(np.abs(hamiltonian[1:] - h0))))
        metrics = {'hamiltonian': hamiltonian, 'hamiltonian_error': hamiltonian_error,
                   'time_points': np.linspace(0, self.args.T, self.args.num_snapshots)}
        return metrics, traj_data

    def _compute_differences(self, traj1, traj2):
        if traj1.shape != traj2.shape: raise ValueError("Trajectory shapes do not match.")
        num_snapshots = traj1.shape[0]
        l1_diff = np.zeros(num_snapshots)
        l2_diff = np.zeros(num_snapshots)
        rms_diff = np.zeros(num_snapshots)
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        N = self.args.nx * self.args.ny

        for i in range(num_snapshots):
            diff = traj1[i] - traj2[i]
            diff_sq = diff**2
            l1_diff[i] = np.sum(np.abs(diff)) * dx * dy
            l2_diff[i] = np.sqrt(np.sum(diff_sq) * dx * dy)
            rms_diff[i] = np.sqrt(np.sum(diff_sq) / N)

        diff_metrics = {'l1_difference': l1_diff, 'l2_difference': l2_diff, 'rms_difference': rms_diff,
                       'time_points': np.linspace(0, self.args.T, self.args.num_snapshots)}
        return diff_metrics

    def _compute_hamiltonians_closer(self, traj_data, vel_data):
        # redundant calculation but important to separate concerns ... maybe TODO)
        hamiltonian_func = self.system_config['hamiltonian_func']
        num_snapshots = traj_data.shape[0]
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        dt_snap = self.args.T / (self.args.num_snapshots - 1) if self.args.num_snapshots > 1 else 0
        params_dict = {k: getattr(self.args, k) for k in self.system_config['params']}
        if vel_data is None and self.needs_v0:
             raise Exception
        elif vel_data is not None and vel_data.shape[0] != num_snapshots:
             raise Exception
        elif not self.needs_v0:
             vel_data = np.zeros_like(traj_data)
        m_path = self.temp_dir / "m.npy"
        m_np = np.load(m_path)

        sq  = np.zeros(num_snapshots)
        vel = np.zeros(num_snapshots)
        pot = np.zeros(num_snapshots)
        for i in range(num_snapshots):
            u_snap = traj_data[i].flatten()
            m_flat = m_np.flatten()
            ut_snap = vel_data[i].flatten()
            h_args = {'u': u_snap, 'ut': ut_snap, 'dx': dx, 'dy': dy, 'm': m_flat}
            h_args.update(params_dict)
            u_sq, ut, potential = hamiltonian_func(**h_args)
            sq[i]  = u_sq
            vel[i] = ut 
            pot[i] = potential 
        return sq, vel, pot

    def _plot_energies_closer(self, metrics, traj1, vel1, traj2, vel2, name1, name2):
        sq1, ut1, pot1 = self._compute_hamiltonians_closer(traj1, vel1)
        sq2, ut2, pot2 = self._compute_hamiltonians_closer(traj2, vel2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 12))
        ax1, ax2, ax_diff = axes
        colors  = ["red", "green", "blue"]
        markers = ['x', 10]
        time    = metrics['time_points']
        labels  = ["$|u|^2$", "$|u_t|^2$", "pot"]
        for i, m in enumerate([sq1, ut1, pot1]):
            ax1.plot(time, m, color=colors[i], marker=markers[0], label=labels[i])
        for i, m in enumerate([sq2, ut2, pot2]):
            ax2.plot(time, m, color=colors[i], marker=markers[1], label=labels[i])

        for i, m in enumerate([np.abs(sq1-sq2), np.abs(ut1-ut2), np.abs(pot1-pot2)]):
            ax_diff.plot(time, m, color=colors[i], label=labels[i], linestyle='-.')

        for ax in [ax1, ax2, ax_diff]:
            ax.grid(True)
            ax.legend()

        ax_diff.set_yscale("log")
        ax1.set_ylabel("E / [1]")
        ax1.set_xlabel("T / [1]")

        fig.suptitle(f"Comparison: {name1} vs {name2} ({self.args.system_type}, nx={self.args.nx}, nt={self.args.nt}, T={self.args.T})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = self.plots_dir / f"close_energy_comparison_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {plot_filename}")

    def _plot_comparison(self, metrics1, name1, metrics2, name2, diff_metrics):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        ax_ham_err, ax_diff_norms, ax_ham_abs, ax_l2_diff = axes.flatten()
        time = metrics1['time_points']

        ax_ham_err.plot(time, metrics1['hamiltonian_error'], label=f"{name1} H Rel Err")
        ax_ham_err.plot(time, metrics2['hamiltonian_error'], label=f"{name2} H Rel Err", linestyle='--')
        ax_ham_err.set_title(f"$\log |E_0 - E|$")
        ax_ham_err.set_xlabel("T / [1]"); ax_ham_err.set_ylabel("log diff")
        ax_ham_err.legend(); ax_ham_err.grid(True)

        ax_diff_norms.plot(time, diff_metrics['l1_difference'], label=f"L1 ||{name1} - {name2}||")
        #ax_diff_norms.plot(time, diff_metrics['l2_difference'], label=f"L2 ||{name1} - {name2}||", linestyle='-.')
        ax_diff_norms.plot(time, diff_metrics['rms_difference'], label=f"RMS({name1} - {name2})", linestyle=':')
        ax_diff_norms.set_title(f"Diff (norms between solutions)")
        ax_diff_norms.set_xlabel("T / [1]"); ax_diff_norms.set_ylabel("Error Norm Value")
        ax_diff_norms.set_yscale('log'); ax_diff_norms.legend(); ax_diff_norms.grid(True)

        ax_ham_abs.plot(time, metrics1['hamiltonian'], label=f"{name1} H(t)")
        ax_ham_abs.plot(time, metrics2['hamiltonian'], label=f"{name2} H(t)", linestyle='--')
        ax_ham_abs.set_title(f"Absolute Hamiltonian Evolution")
        ax_ham_abs.set_xlabel("T / [1]"); ax_ham_abs.set_ylabel("")
        ax_ham_abs.legend(); ax_ham_abs.grid(True)

        ax_l2_diff.plot(time, diff_metrics['l2_difference'], label=f"L2 Norm ||{name1} - {name2}||")
        ax_l2_diff.set_title(f"L2 Difference Evolution")
        ax_l2_diff.set_xlabel("T / [1]"); ax_l2_diff.set_ylabel("L2 Norm")
        ax_l2_diff.set_yscale('log'); ax_l2_diff.legend(); ax_l2_diff.grid(True)

        fig.suptitle(f"Comparison: {name1} vs {name2} ({self.args.system_type}, nx={self.args.nx}, nt={self.args.nt}, T={self.args.T})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = self.plots_dir / f"comparison_summary_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {plot_filename}")

    def _plot_state_differences(self, traj1, traj2, name1, name2, ic_type):
        num_snapshots = traj1.shape[0]
        indices = [num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4]
        times = [self.args.T / 4, self.args.T / 2, 3 * self.args.T / 4]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Absolute Difference |{name1} - {name2}| at different times ({ic_type})", fontsize=16)

        for i, idx in enumerate(indices):
            if idx >= num_snapshots: continue
            diff_data = np.abs(traj1[idx] - traj2[idx])
            im = axes[i].imshow(diff_data, cmap='viridis', origin='lower',
                                extent=[-self.args.Lx, self.args.Lx, -self.args.Ly, self.args.Ly])
            axes[i].set_title(f"t = {times[i]:.2f}")
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            fig.colorbar(im, ax=axes[i])

        plot_filename = self.plots_dir / f"state_difference_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"State difference plot saved to {plot_filename}")

    def _plot_state(self, traj, name, ic_type):
        num_snapshots = traj.shape[0]
        indices = [num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4]
        times = [self.args.T / 4, self.args.T / 2, 3 * self.args.T / 4]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{name} at different times ({ic_type})", fontsize=16)

        for i, idx in enumerate(indices):
            if idx >= num_snapshots: continue
            diff_data = traj[idx]
            im = axes[i].imshow(diff_data, cmap='coolwarm', origin='lower',
                                extent=[-self.args.Lx, self.args.Lx, -self.args.Ly, self.args.Ly])
            axes[i].set_title(f"t = {times[i]:.2f}")
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")
            fig.colorbar(im, ax=axes[i])

        plot_filename = self.plots_dir / f"state_{name}_{self.args.system_type}_{self.run_id}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"State {name} plot saved to {plot_filename}")


    def execute(self):
        print(f"Starting Wave comparison for {self.args.system_type}")
        print(f"Integrator 1: {self.args.name1} ({self.args.exe1})")
        print(f"Integrator 2: {self.args.name2} ({self.args.exe2})")
        print(f"Grid: nx={self.args.nx}, nt={self.args.nt}, T={self.args.T}, L={self.args.Lx}")
        u0_path, v0_path, m_path = self._generate_initial_conditions()
        traj1_path, vel1_path, walltime1 = self._run_single_simulation(self.args.name1, self.args.exe1, u0_path, v0_path, m_path)
        traj2_path, vel2_path, walltime2 = self._run_single_simulation(self.args.name2, self.args.exe2, u0_path, v0_path, m_path)
        metrics1, traj1_data = self._analyze_trajectory(traj1_path, vel1_path)
        metrics2, traj2_data = self._analyze_trajectory(traj2_path, vel2_path)
        diff_metrics = self._compute_differences(traj1_data, traj2_data)
        self._plot_comparison(metrics1, self.args.name1, metrics2, self.args.name2, diff_metrics)
        self._plot_state_differences(traj1_data, traj2_data, self.args.name1, self.args.name2, self.args.ic_type)
        self._plot_state(traj1_data, self.args.name1, self.args.ic_type)
        self._plot_state(traj2_data, self.args.name2, self.args.ic_type)
        vel1_data, vel2_data = np.load(vel1_path), np.load(vel2_path)
        self._plot_energies_closer(metrics1, traj1_data, vel1_data, traj2_data, vel2_data,
                self.args.name1, self.args.name2)

        if self.args.visualize:
            X, Y = np.meshgrid(np.linspace(-self.args.Lx, self.args.Lx, self.args.nx),
                                   np.linspace(-self.args.Ly, self.args.Ly, self.args.ny), indexing='ij')
            animate_simulation(X, Y, traj1_data, nt=traj1_data.shape[0],
                    name=self.plots_dir / f"{self.args.name1}_{self.run_id}.mp4", title=f"{self.args.name1}")
            animate_simulation(X, Y, traj2_data, nt=traj2_data.shape[0],
                    name=self.plots_dir / f"{self.args.name2}_{self.run_id}.mp4", title=f"{self.args.name2}")

        if not getattr(self.args, 'keep_temps', False):
            try:
                for f in self.temp_dir.glob('*'): os.unlink(f)
                os.rmdir(self.temp_dir)
            except OSError as e: print(f"Error removing temp dir {self.temp_dir}: {e}")
        else: print(f"Temporary files kept in {self.temp_dir}")
