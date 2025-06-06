import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import uuid
import subprocess
from pathlib import Path


try:
    from nlse_sampler import NLSE3DSampler as NLSEPhenomenonSampler
    from valid_spaces import get_parameter_spaces_3d as get_parameter_spaces
except ImportError:
    print("Warning: nlse_sampler not found.")
    raise Exception

def compute_gradient_sq_norm(u, dx, dy, dz):
    ux = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2. * dx)
    uy = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2. * dy)
    uz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2. * dz)
    return np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2) * dx * dy * dz

def compute_hamiltonian_nlse_cubic(u, dx, dy, dz, **kwargs):
    kinetic = 0.5 * compute_gradient_sq_norm(u, dx, dy, dz)
    potential = -0.5 * np.sum(np.abs(u[1:-1, 1:-1, 1:-1])**4) * dx * dy * dz
    return kinetic, potential


SYSTEM_CONFIG_NLSE = {
    'NLSE_cubic': {
        'dtype': np.complex128, 'params': ['m_value'], 'hamiltonian_func': compute_hamiltonian_nlse_cubic,
        'extra_args_spec': []
    },
}

class NlseComparer3d:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.system_config = SYSTEM_CONFIG_NLSE[args.system_type]
        self.dtype = self.system_config['dtype']
        self.sampler = None
        self.setup_directories()
        sampler = self._get_sampler()
        self.param_spaces = get_parameter_spaces()

    def setup_directories(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir = self.output_dir / f"temp_{self.run_id}"
        self.temp_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def _get_sampler(self):
        if self.sampler is None:
            if NLSEPhenomenonSampler:
                self.sampler = NLSEPhenomenonSampler(self.args.nx, self.args.ny,
                                                     self.args.nz, self.args.Lx)
            else:
                raise RuntimeError("NLSEPhenomenonSampler not available")
        return self.sampler

    def _generate_initial_conditions(self):
        u0_path = self.temp_dir / "u0.npy"
        m_path = self.temp_dir / "m.npy"
        c_path = self.temp_dir / "c.npy"

        if self.sampler: 
            ic_params = self.param_spaces[self.args.ic_type]  
            import random # quick fix: np.random.choice has issues with non-1-dimensional data
            ic_params_choice = {k: random.choice(v) for k, v in ic_params.items()}
            self.params = ic_params_choice
            u0_np = self.sampler.generate_ensemble(self.args.ic_type, n_samples=1, **ic_params_choice)
            if hasattr(u0_np, 'numpy'):
                u0_np = u0_np.numpy()
            u0_np = u0_np.squeeze().astype(self.dtype)
        else:
            raise NotImplemented
        
        m_np = np.full((self.args.nz, self.args.ny, self.args.nx), self.args.m_value, dtype=np.float64)
        c_np = np.full((self.args.nz, self.args.ny, self.args.nx), self.args.m_value, dtype=np.float64)

        np.save(u0_path, u0_np)
        np.save(m_path, m_np)
        np.save(c_path, c_np)

        return u0_path, m_path, c_path

    def _build_command(self, exe_path, u0_path, m_path, c_path, traj_path):
        base_cmd = [
            str(exe_path),
            str(self.args.nx), str(self.args.ny), str(self.args.nz),
            str(self.args.Lx), str(self.args.Ly), str(self.args.Lz)
        ]

        extra_args_map = {}
        for name, pos in self.system_config.get('extra_args_spec', []):
             param_value = getattr(self.args, name, None)
             if param_value is None:
                 raise ValueError(f"Missing required parameter '{name}' for {self.args.system_type}")
             extra_args_map[pos] = str(param_value)

        final_cmd = list(base_cmd)
        arg_insert_index = 5

        sorted_positions = sorted(extra_args_map.keys())
        for insert_pos in sorted_positions:
            if insert_pos != arg_insert_index:
                 raise ValueError(f"Argument position mismatch definition for {self.args.system_type}")
            final_cmd.insert(insert_pos, extra_args_map[insert_pos])
            arg_insert_index += 1

        final_cmd.extend([
            str(u0_path), str(traj_path),
            str(self.args.T), str(self.args.nt),
            str(self.args.num_snapshots), str(m_path),
            str(c_path)
        ])
        return final_cmd

    def _run_single_simulation(self, integrator_name, exe_path, u0_path, m_path, c_path):
        traj_path = self.temp_dir / f"traj_{integrator_name}.npy"
        cmd = self._build_command(exe_path, u0_path, m_path, c_path, traj_path)
        print(f"Running {integrator_name}: {' '.join(cmd)}")
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stderr: print(f"STDERR ({integrator_name}):\n{result.stderr}\n---------")
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {integrator_name}:\nCmd: {' '.join(e.cmd)}\nCode: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            raise
        except Exception:
             raise Exception

        end_time = time.time()
        walltime = end_time - start_time
        print(f"{integrator_name} finished in {walltime:.2f}s")
        return traj_path, walltime

    def _load_trajectory(self, file_path):
        if not file_path.exists(): raise FileNotFoundError(f"Trajectory file not found: {file_path}")
        return np.load(file_path).astype(self.dtype)

    def _calculate_hamiltonian(self, traj_data, m_np):
        hamiltonian_func = self.system_config['hamiltonian_func']
        num_snapshots = traj_data.shape[0]
        hamiltonian_values = np.zeros(num_snapshots)
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        dz = 2 * self.args.Lz / (self.args.nz - 1)
        params_dict = {k: getattr(self.args, k) for k in self.system_config['params']}
        for i in range(num_snapshots):
            u_snap = traj_data[i]
            m_flat = m_np # Assume m is already shaped correctly for func
            h_args = {'u': u_snap, 'dx': dx, 'dy': dy, 'dz': dz,
                      'm': m_flat}
            h_args.update(params_dict)
            k, p = hamiltonian_func(**h_args)
            hamiltonian_values[i] = k.real + p.real
        return hamiltonian_values

    def _calculate_hamiltonian_closer(self, traj_data, m_np):
        hamiltonian_func = self.system_config['hamiltonian_func']
        num_snapshots = traj_data.shape[0]
        kinetic_values   = np.zeros(num_snapshots)
        potential_values = np.zeros(num_snapshots)

        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        dz = 2 * self.args.Lz / (self.args.nz - 1)
        params_dict = {k: getattr(self.args, k) for k in self.system_config['params']}
        for i in range(num_snapshots):
            u_snap = traj_data[i]
            m_flat = m_np # Assume m is already shaped correctly for func
            h_args = {'u': u_snap, 'dx': dx, 'dy': dy, 'dz': dz,
                      'm': m_flat}
            h_args.update(params_dict)
            k, p = hamiltonian_func(**h_args)
            kinetic_values[i] = k.real
            potential_values[i] = p.real 
        return kinetic_values, potential_values

    def _plot_ic(self, u0, m):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig = plt.figure(figsize=(20, 12))
        grid = plt.GridSpec(4, 3, height_ratios=[1, 5, 5, 1])
        
        param_ax = fig.add_subplot(grid[0, :])
        param_ax.axis('off')
        
        ax_u0_angle = fig.add_subplot(grid[1:3, 0], projection='3d')
        ax_u0_abs = fig.add_subplot(grid[1:3, 1], projection='3d')
        ax_m = fig.add_subplot(grid[1:3, 2], projection='3d')
        
        x = np.linspace(-self.args.Lx, self.args.Lx, u0.shape[0])
        y = np.linspace(-self.args.Ly, self.args.Ly, u0.shape[1])
        z = np.linspace(-self.args.Lz, self.args.Lz, u0.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        u0_angle = np.angle(u0)
        u0_abs = np.abs(u0)
        
        iso_level_angle = np.percentile(u0_angle, 75)
        ax_u0_angle.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=u0_angle.flatten(), 
                            alpha=0.02, s=2, cmap='viridis')
        ax_u0_angle.contour3D(X[:,:,u0.shape[2]//2], Y[:,:,u0.shape[2]//2], 
                            u0_angle[:,:,u0.shape[2]//2], levels=10, cmap='viridis')
        ax_u0_angle.contour3D(X[:,u0.shape[1]//2,:], Z[:,u0.shape[1]//2,:], 
                            u0_angle[:,u0.shape[1]//2,:], levels=10, cmap='viridis')
        ax_u0_angle.contour3D(Y[u0.shape[0]//2,:,:], Z[u0.shape[0]//2,:,:], 
                            u0_angle[u0.shape[0]//2,:,:], levels=10, cmap='viridis')
        iso_level_abs = np.percentile(u0_abs, 75)
        ax_u0_abs.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=u0_abs.flatten(), 
                        alpha=0.02, s=2, cmap='RdBu_r')
        ax_u0_abs.contour3D(X[:,:,u0.shape[2]//2], Y[:,:,u0.shape[2]//2], 
                        u0_abs[:,:,u0.shape[2]//2], levels=10, cmap='RdBu_r')
        ax_u0_abs.contour3D(X[:,u0.shape[1]//2,:], Z[:,u0.shape[1]//2,:], 
                        u0_abs[:,u0.shape[1]//2,:], levels=10, cmap='RdBu_r')
        ax_u0_abs.contour3D(Y[u0.shape[0]//2,:,:], Z[u0.shape[0]//2,:,:], 
                        u0_abs[u0.shape[0]//2,:,:], levels=10, cmap='RdBu_r')
        
        iso_level_m = np.percentile(m, 75)
        ax_m.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=m.flatten(), 
                    alpha=0.02, s=2, cmap='twilight')
        ax_m.contour3D(X[:,:,m.shape[2]//2], Y[:,:,m.shape[2]//2], 
                    m[:,:,m.shape[2]//2], levels=10, cmap='twilight')
        ax_m.contour3D(X[:,m.shape[1]//2,:], Z[:,m.shape[1]//2,:], 
                    m[:,m.shape[1]//2,:], levels=10, cmap='twilight')
        ax_m.contour3D(Y[m.shape[0]//2,:,:], Z[m.shape[0]//2,:,:], 
                    m[m.shape[0]//2,:,:], levels=10, cmap='twilight')
        
        ax_u0_angle.set_title("$u_0$ angle")
        ax_u0_abs.set_title("$|u_0|$")
        ax_m.set_title("$m(x,y,z)$")
        
        for ax in [ax_u0_angle, ax_u0_abs, ax_m]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(-self.args.Lx, self.args.Lx)
            ax.set_ylim(-self.args.Ly, self.args.Ly)
            ax.set_zlim(-self.args.Lz, self.args.Lz)
        
        import json
        dict_str = json.dumps(self.params)
        dict_str = '\n'.join(dict_str[i:i+45] for i in range(0, len(dict_str), 45))
        
        param_ax.text(0.5, 0.5, f"Parameters:\n{dict_str}", 
                    ha='center', va='center', 
                    fontfamily='monospace', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgrey', alpha=0.8))
        fig.add_subplot(grid[3, :]).axis('off')
        fig.suptitle("Initial conditions (3D)")
        plot_filename = self.plots_dir / f"IC_{self.args.system_type}_{self.run_id}_3d.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D initial condition view saved to {plot_filename}")

    def _plot_energy_closer(self, time, traj1, traj2, name1, name2, m_np):
        fig = plt.figure(figsize=(15, 8))
        grid = plt.GridSpec(4, 3, height_ratios=[1, 5, 5, 1])
        param_ax = fig.add_subplot(grid[0, :])
        param_ax.axis('off')
        ax1 = fig.add_subplot(grid[1:3, 0])
        ax2 = fig.add_subplot(grid[1:3, 1])
        ax_diff = fig.add_subplot(grid[1:3, 2])
        k1, p1 = self._calculate_hamiltonian_closer(traj1, m_np) 
        k2, p2 = self._calculate_hamiltonian_closer(traj2, m_np)
        
        colors = ["red", "blue"]
        markers = [10, 'x']
        labels = ["kinetic", "potential"]
        for i, d in enumerate([k1, p1]):
            ax1.plot(time, d, color=colors[i], marker=markers[0], label=labels[i])
        for i, d in enumerate([k2, p2]):
            ax2.plot(time, d, color=colors[i], marker=markers[1], label=labels[i])
        for i, d in enumerate([np.log(np.abs(k1 - k2))[1:], np.log(np.abs(p1 - p2))[1:]]):
            ax_diff.plot(time, [np.nan] + list(d), color=colors[i], linestyle='-.', label=labels[i])
        ax1.set_title(f"{name1}")
        ax2.set_title(f"{name2}")
        ax_diff.set_title("$\log |E_{1} - E_{2}|$")
        for ax in [ax1, ax2, ax_diff]:
            ax.legend()
            ax.grid(True)
        
        import json
        dict_str = json.dumps(self.params)
        dict_str = '\n'.join(dict_str[i:i+45] for i in range(0, len(dict_str), 45))
        param_ax.text(0.5, 0.5, f"Parameters:\n{dict_str}", 
                     ha='center', va='center', 
                     fontfamily='monospace', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgrey', alpha=0.8))
        title_ax = fig.add_subplot(grid[3, 1])
        title_ax.axis('off')
        title_ax.text(0.5, 0.2, f"{name1} vs {name2}", 
                     ha='center', va='center', 
                     fontsize=12, fontweight='bold')

        fig.text(0.5, 0.01, "Time", ha='center', fontsize=10)
        plot_filename = self.plots_dir / f"closer_energy_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Closer energy comparison plot saved to {plot_filename}")

    def _analyze_trajectory(self, traj_path):
        traj_data = self._load_trajectory(traj_path)
        m_path = self.temp_dir / "m.npy"
        m_np = np.load(m_path)
        hamiltonian = self._calculate_hamiltonian(traj_data, m_np)
        h0 = hamiltonian[0]
        hamiltonian_error = np.array([np.nan] + list(np.log(np.abs(hamiltonian[1:] - h0))))
        metrics = {'hamiltonian': hamiltonian, 'hamiltonian_error': hamiltonian_error,
                   'time_points': np.linspace(0, self.args.T, self.args.num_snapshots)}
        return metrics, traj_data

    def _compute_differences(self, traj1, traj2):
        if traj1.shape != traj2.shape: raise ValueError("Trajectory shapes do not match.")
        num_snapshots = traj1.shape[0]
        l2_diff = np.zeros(num_snapshots)
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)
        dz = 2 * self.args.Lz / (self.args.nz - 1)
        for i in range(num_snapshots):
            diff = traj1[i] - traj2[i]
            l2_diff[i] = np.sqrt(np.sum(np.abs(diff)**2) * dx * dy * dz)
        diff_metrics = {'l2_difference': l2_diff,
                       'time_points': np.linspace(0, self.args.T, self.args.num_snapshots)}
        return diff_metrics

    def _plot_state(self, traj, name):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_snapshots = traj.shape[0]
        indices = [num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4, num_snapshots - 1]
        times = [self.args.T / 4, self.args.T / 2, 3 * self.args.T / 4, self.args.T]
        
        fig = plt.figure(figsize=(20, 15))
        
        for i, idx in enumerate(indices):
            if idx >= num_snapshots: continue
            
            diff_data = np.abs(traj[idx])
            threshold = np.percentile(diff_data, 90)
            
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            x = np.linspace(-self.args.Lx, self.args.Lx, diff_data.shape[0])
            y = np.linspace(-self.args.Ly, self.args.Ly, diff_data.shape[1])
            z = np.linspace(-self.args.Lz, self.args.Lz, diff_data.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            mask = diff_data > threshold
            ax.scatter(X[mask], Y[mask], Z[mask], c=diff_data[mask], 
                    s=5, alpha=0.7, cmap='viridis')
            
            ax.set_title(f"t = {times[i]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(-self.args.Lx, self.args.Lx)
            ax.set_ylim(-self.args.Ly, self.args.Ly)
            ax.set_zlim(-self.args.Lz, self.args.Lz)
        
        fig.suptitle(f"state |{name}| at different times", fontsize=16)
        plot_filename = self.plots_dir / f"state_3d_{name}_{self.args.system_type}_{self.run_id}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"3D state plot saved to {plot_filename}")

    def _plot_state_differences(self, traj1, traj2, name1, name2):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_snapshots = traj1.shape[0]
        indices = [num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4, num_snapshots - 1]
        times = [self.args.T / 4, self.args.T / 2, 3 * self.args.T / 4, self.args.T]
        
        fig = plt.figure(figsize=(20, 15))
        
        for i, idx in enumerate(indices):
            if idx >= num_snapshots: continue
            
            diff_data = np.abs(traj1[idx] - traj2[idx])
            threshold = np.percentile(diff_data, 95)
            
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            x = np.linspace(-self.args.Lx, self.args.Lx, diff_data.shape[0])
            y = np.linspace(-self.args.Ly, self.args.Ly, diff_data.shape[1])
            z = np.linspace(-self.args.Lz, self.args.Lz, diff_data.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            mask = diff_data > threshold
            ax.scatter(X[mask], Y[mask], Z[mask], c=diff_data[mask], 
                    s=5, alpha=0.7, cmap='viridis')
            
            ax.set_title(f"t = {times[i]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(-self.args.Lx, self.args.Lx)
            ax.set_ylim(-self.args.Ly, self.args.Ly)
            ax.set_zlim(-self.args.Lz, self.args.Lz)
        
        fig.suptitle(f"Absolute Difference |{name1} - {name2}| at different times (significant differences only)", fontsize=16)
        plot_filename = self.plots_dir / f"state_difference_3d_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"3D state difference plot saved to {plot_filename}")

    def _plot_state_differences_3d_other(self, traj1, traj2, name1, name2):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_snapshots = traj1.shape[0]
        indices = [num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4, num_snapshots - 1]
        times = [self.args.T / 4, self.args.T / 2, 3 * self.args.T / 4, self.args.T]
        
        fig = plt.figure(figsize=(20, 15))
        
        for i, idx in enumerate(indices):
            if idx >= num_snapshots: continue
            
            diff_data = np.abs(traj1[idx] - traj2[idx])
            
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            
            x = np.linspace(-self.args.Lx, self.args.Lx, diff_data.shape[0])
            y = np.linspace(-self.args.Ly, self.args.Ly, diff_data.shape[1])
            z = np.linspace(-self.args.Lz, self.args.Lz, diff_data.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            mid_x = diff_data.shape[0] // 2
            mid_y = diff_data.shape[1] // 2
            mid_z = diff_data.shape[2] // 2
            
            levels = np.linspace(diff_data.min(), diff_data.max(), 10)
            
            ax.contourf(X[:, :, mid_z], Y[:, :, mid_z], diff_data[:, :, mid_z], 
                    zdir='z', offset=0, levels=levels, cmap='viridis', alpha=0.7)
            
            ax.contourf(X[:, mid_y, :], Z[:, mid_y, :], diff_data[:, mid_y, :], 
                    zdir='y', offset=0, levels=levels, cmap='viridis', alpha=0.7)
            
            ax.contourf(Y[mid_x, :, :], Z[mid_x, :, :], diff_data[mid_x, :, :], 
                    zdir='x', offset=0, levels=levels, cmap='viridis', alpha=0.7)
            
            iso_val = np.percentile(diff_data, 90)
            ax.contour3D(X, Y, Z, diff_data, levels=[iso_val], cmap='hot')
            
            ax.set_title(f"t = {times[i]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(-self.args.Lx, self.args.Lx)
            ax.set_ylim(-self.args.Ly, self.args.Ly)
            ax.set_zlim(-self.args.Lz, self.args.Lz)
        
        fig.suptitle(f"Absolute Difference |{name1} - {name2}| at different times (orthogonal slices and isosurface)", fontsize=16)
        plot_filename = self.plots_dir / f"state_difference_3d_slices_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"3D state difference plot with slices saved to {plot_filename}")

    def _plot_comparison(self, metrics1, name1, metrics2, name2, diff_metrics):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax_ham, ax_diff = axes.flatten()
        time = metrics1['time_points']
        ax_ham.plot(time, metrics1['hamiltonian_error'], label=f"{name1} H Error")
        ax_ham.plot(time, metrics2['hamiltonian_error'], label=f"{name2} H Error", linestyle='--')
        ax_ham.set_title(f"Hamiltonian Error ($log |E_0 - E |$) ({self.args.system_type})")
        ax_ham.set_xlabel("T / [1]"); ax_ham.set_ylabel("log |H(t)-H(0)|")
        ax_ham.legend(); ax_ham.grid(True)
        ax_diff.plot(time, diff_metrics['l2_difference'], label=f"L2 Norm ||{name1} - {name2}||")
        ax_diff.set_title(f"$L_2$ (difference between solutions)")
        ax_diff.set_xlabel("T / [1]"); ax_diff.set_ylabel("$L_2$")
        ax_diff.set_yscale('log'); ax_diff.legend(); ax_diff.grid(True)
        fig.suptitle(
                f"Comparison: {name1} vs {name2} ({self.args.system_type}, nx={self.args.nx}," +\
                f"nt={self.args.nt}, T={self.args.T})\n{self.params}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = self.plots_dir / f"comparison_{name1}_vs_{name2}_{self.args.system_type}_{self.run_id}.png"
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {plot_filename}")

    def execute(self):
        print(f"Starting NLSE comparison for {self.args.system_type}")
        print(f"Integrator 1: {self.args.name1} ({self.args.exe1})")
        print(f"Integrator 2: {self.args.name2} ({self.args.exe2})")
        print(f"Grid: nx={self.args.nx}, nt={self.args.nt}, T={self.args.T}, L={self.args.Lx}")
        u0_path, m_path, c_path = self._generate_initial_conditions()
        u0 = np.load(u0_path)
        m_np = np.load(m_path)
        c_np = np.load(c_path)
        self._plot_ic(u0, m_np)
        traj1_path, walltime1 = self._run_single_simulation(self.args.name1, self.args.exe1, u0_path, m_path, c_path)
        traj2_path, walltime2 = self._run_single_simulation(self.args.name2, self.args.exe2, u0_path, m_path, c_path)
        metrics1, traj1_data = self._analyze_trajectory(traj1_path)
        metrics2, traj2_data = self._analyze_trajectory(traj2_path)
        diff_metrics = self._compute_differences(traj1_data, traj2_data)
        self._plot_comparison(metrics1, self.args.name1, metrics2, self.args.name2, diff_metrics)
        self._plot_state_differences(traj1_data, traj2_data, self.args.name1, self.args.name2)
        self._plot_energy_closer(metrics1['time_points'], traj1_data, traj2_data, self.args.name1, self.args.name2, m_np)
        self._plot_state(traj1_data, self.args.name1)
        self._plot_state(traj2_data, self.args.name2)

        if not getattr(self.args, 'keep_temps', False):
            try:
                for f in self.temp_dir.glob('*'): os.unlink(f)
                os.rmdir(self.temp_dir)
            except OSError as e: print(f"Error removing temp dir {self.temp_dir}: {e}")
        else: print(f"Temporary files kept in {self.temp_dir}")
