import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import os
import time
import uuid
import subprocess
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fft2, fftshift
import torch
from skimage.metrics import structural_similarity as ssim
    
from downsampling import downsample_interpolation, reconstruct_interpolation
from visualization import animate_simulation
from nlse_sampler import NLSEPhenomenonSampler
from valid_spaces import get_parameter_spaces 

def compute_mass(u, dx, dy):
    return np.sum(np.abs(u)**2) * dx * dy

def compute_energy(u, dx, dy, m):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    
    kinetic = 0.5 * np.sum(np.abs(ux)**2 + np.abs(uy)**2) * dx * dy
    potential = -0.5 * np.sum(m[1:-1, 1:-1] * np.abs(u[1:-1, 1:-1])**4) * dx * dy
    
    return kinetic, potential


def compute_modal_energy(data, n_modes=32):
    nt, nx, ny = data.shape
    modal_energies = np.zeros((nt, n_modes, n_modes))

    for t in range(nt):
        spectrum = fftshift(fft2(data[t]))
        center_x, center_y = nx // 2, ny // 2
        for i in range(n_modes):
            for j in range(n_modes):
                i_idx = center_x - n_modes // 2 + i
                j_idx = center_y - n_modes // 2 + j
                modal_energies[t, i, j] = np.abs(spectrum[i_idx, j_idx])**2

    return modal_energies


def compute_structure_similarity(data, reference_frame=None):
    if reference_frame is None:
        reference_frame = np.abs(data[0])

    nt = data.shape[0]
    ssim_values = np.zeros(nt)

    for t in range(nt):
        ssim_values[t] = ssim(
            np.abs(reference_frame),
            np.abs(data[t]),
            data_range=np.abs(data).max() - np.abs(data).min())

    return ssim_values


def compute_phase_error(data, reference_data=None):
    if reference_data is None:
        reference_data = data[0]
    
    nt = data.shape[0]
    phase_errors = np.zeros(nt)
    
    for t in range(nt):
        phase_diff = np.angle(data[t]) - np.angle(reference_data)
        phase_errors[t] = np.mean(np.abs(np.exp(1j * phase_diff) - 1))
    
    return phase_errors


def compute_amplitude_fidelity(data, reference_data=None):
    if reference_data is None:
        reference_data = data[0]
    
    nt = data.shape[0]
    amplitude_errors = np.zeros(nt)
    
    for t in range(nt):
        amplitude_diff = np.abs(data[t]) - np.abs(reference_data)
        amplitude_errors[t] = np.sqrt(np.mean(amplitude_diff**2))
    
    return amplitude_errors


def compute_spectral_dispersion(data, dx, dt):
    nt, nx, ny = data.shape
    if nx != ny:
        raise ValueError("Expected square grid for dispersion analysis")

    k_max = np.pi / dx
    k_values = np.fft.fftfreq(nx, dx) * 2 * np.pi
    k_mag = np.sqrt(k_values[:, None]**2 + k_values[None, :]**2)

    dispersion_observed = np.zeros((nt // 2, nx, ny))
    for t in range(1, nt // 2 + 1):
        fft_u = np.fft.fft2(data[t])
        fft_u0 = np.fft.fft2(data[0])

        fft_ratio = fft_u / (fft_u0 + 1e-10)
        phase = np.angle(fft_ratio)
        dispersion_observed[t - 1] = phase / (t * dt)

    dispersion_avg = np.mean(dispersion_observed, axis=0)
    k_bins = np.linspace(0, k_max, 50)
    dispersion_radial = np.zeros(len(k_bins) - 1)
    dispersion_std = np.zeros(len(k_bins) - 1)

    for i in range(len(k_bins) - 1):
        mask = (k_mag > k_bins[i]) & (k_mag <= k_bins[i + 1])
        if np.sum(mask) > 0:
            dispersion_radial[i] = np.mean(dispersion_avg[mask])
            dispersion_std[i] = np.std(dispersion_avg[mask])

    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    return k_centers, dispersion_radial, dispersion_std, dispersion_avg

def focussing(n, L):
    xn = yn = np.linspace(-L, L, n)
    X, Y = np.meshgrid(xn, yn)
    field = np.exp(-(X ** 2 + Y ** 2)/4.) * np.exp(-1j*(X ** 2 + Y ** 2)/2.)
    return field


class NLSEIntegratorComparison:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.executables = {
            'gautschi': args.gautschi_exe,
            'ss2': args.ss2_exe
        }
        self.nx_values = args.nx_values
        self.nt_values = args.nt_values
        self.Lx = args.Lx
        self.Ly = args.Ly
        self.T = args.T
        self.m_value = args.m_value
        self.setup_directories()
        max_n = max(args.nx_values)
        self.sampler = NLSEPhenomenonSampler(max_n, max_n, args.Lx)

        self.traj_files = list() # let's keep track of them to remove after done

    def setup_directories(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.ic_dir = self.output_dir / "initial_conditions"
        self.ic_dir.mkdir(exist_ok=True)

        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(exist_ok=True)

        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def _sample_phenomenon_params(self):
        parameter_spaces = get_parameter_spaces()
        if self.args.ic_type in parameter_spaces:
            space = parameter_spaces[self.args.ic_type]
            params = {}
            for key, values in space.items():
                if isinstance(values[0], tuple) or isinstance(values[0], list):
                    l = len(values)
                    idx = np.random.randint(0, l)
                    params[key] = values[idx]
                else:
                    params[key] = np.random.choice(values)
            return params
        else:
            raise ValueError(
                f"Unknown phenomenon type: {self.args.phenomenon}")

    def generate_base_conditions(self):
        max_nx = max(self.nx_values)
        u0_file = self.ic_dir / f"u0_{max_nx}_{self.run_id}.npy"
        m_file = self.ic_dir / f"m_{max_nx}_{self.run_id}.npy"

        try:
            phenomenon_params = self._sample_phenomenon_params()
            u0 = self.sampler.generate_ensemble(
                self.args.ic_type,
                n_samples=1,
                **phenomenon_params
            )
            if isinstance(u0, torch.Tensor):
                u0 = u0.detach().numpy()

        except Exception as e:
            raise ValueError(f"Unknown ic_type: {self.args.ic_type} or other error {e}")
        
        #u0 = focussing(max_nx, self.Lx)
        m = np.ones((max_nx, max_nx)) * self.m_value
        np.save(u0_file, u0)
        np.save(m_file, m)
        return u0, m, u0_file, m_file

    def run_simulation(self, integrator_type, nx, nt, u0, m):
        u0_nx_file = self.ic_dir / f"u0_{nx}_{self.run_id}.npy"
        m_nx_file = self.ic_dir / f"m_{nx}_{self.run_id}.npy"
        
        if nx != u0.shape[0]: 
            u0_nx = downsample_interpolation(u0.reshape(
                1, *u0.shape), (nx, nx), self.Lx, self.Ly)[0]
            m_nx = downsample_interpolation(
                m.reshape(1, *m.shape), (nx, nx), self.Lx, self.Ly)[0] 
        else:
            u0_nx, m_nx = u0, m
        """
                    self.nx_values = args.nx_values
                    self.nt_values = args.nt_values
                    self.Lx = args.Lx
                    self.Ly = args.Ly
                    self.T = args.T
        """
        dt = self.T / nt
        dx = dy = 2 * self.Lx / (nx + 1)
        k_nyquist_x = np.pi / dx  
        k_nyquist_y = np.pi / dy
        k_max = np.sqrt(k_nyquist_x**2 + k_nyquist_y**2)
        dt_max = np.pi / k_max
        if dt > min(dx**2 / 4., dt_max):
            print(f"Warning: {nx=}, {nt=} yields {dt_max=:.2f} and {(dx**2 / 4)=:.2e}.,"
                   "ie. augmented CFL condition violated")


        np.save(u0_nx_file, u0_nx)
        np.save(m_nx_file, m_nx)

        traj_file = self.traj_dir / f"{integrator_type}_{nx}_{nt}_{self.run_id}.npy"
        self.traj_files.append(traj_file)

        exe_path = Path(self.executables[integrator_type])
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable {exe_path} not found")

        snapshots = nt // 10 # min(100, nt // 10)
        self.snapshots = snapshots

        cmd = [
            str(exe_path),
            str(nx),
            str(nx),
            str(self.Lx),
            str(self.Ly),
            str(u0_nx_file),
            str(traj_file),
            str(self.T),
            str(nt),
            str(snapshots),
            str(m_nx_file)
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True)
        end_time = time.time()

        walltime = end_time - start_time

        return traj_file, walltime

    def analyze_simulation(self, integrator, nx, nt, traj_file):
        traj_data = np.load(traj_file, mmap_mode='r')
        snapshots = traj_data.shape[0]

        time_points = np.linspace(0, self.T, snapshots)
        dx = 2 * self.Lx / (nx - 1)
        dy = 2 * self.Ly / (nx - 1)
        dt = self.T / (nt - 1)
        
        m_nx_file = self.ic_dir / f"m_{nx}_{self.run_id}.npy"
        m = np.load(m_nx_file)
        
        mass_values = np.array([compute_mass(u, dx, dy) for u in traj_data])
        mass_error = np.abs(mass_values - mass_values[0]) / mass_values[0]
        
        energy_components = [compute_energy(u, dx, dy, m) for u in traj_data]
        kinetic_energy = np.array([e[0] for e in energy_components])
        potential_energy = np.array([e[1] for e in energy_components])
        total_energy = kinetic_energy + potential_energy
        energy_error = np.abs(total_energy - total_energy[0]) / np.abs(total_energy[0])
        
        modal_energy = compute_modal_energy(traj_data)
        ssim_values = compute_structure_similarity(traj_data)
        phase_errors = compute_phase_error(traj_data)
        amplitude_errors = compute_amplitude_fidelity(traj_data)
        
        k_centers, dispersion_radial, dispersion_std, dispersion_full = compute_spectral_dispersion(
            traj_data, dx, dt)
        
        conservation_metric_mass = np.array(
            [np.nan] + [np.log(np.abs(mass_values[i] - mass_values[0]) + 1e-16)
                        for i in range(1, snapshots)]
        )
        
        conservation_metric_energy = np.array(
            [np.nan] + [np.log(np.abs(total_energy[i] - total_energy[0]) + 1e-16)
                        for i in range(1, snapshots)]
        )

        metrics = {
            'time_points': time_points,
            'mass_values': mass_values,
            'mass_error': mass_error,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'total_energy': total_energy,
            'energy_error': energy_error,
            'modal_energy': modal_energy,
            'structure_similarity': ssim_values,
            'phase_errors': phase_errors,
            'amplitude_errors': amplitude_errors,
            'dispersion_radial': dispersion_radial,
            'dispersion_std': dispersion_std,
            'dispersion_full': dispersion_full,
            'dispersion_k': k_centers,
            'conservation_metric_mass': conservation_metric_mass,
            'conservation_metric_energy': conservation_metric_energy
        }

        return metrics, traj_data

    def plot_conservation_comparison(self, nt, metrics_by_nx):
        for nx in sorted(metrics_by_nx.keys()):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
            ax_mass_evol, ax_energy_evol, ax_log_mass_err, ax_log_energy_err = axes.flatten()

            metrics_dict = metrics_by_nx[nx]

            for integrator, metrics in metrics_dict.items():
                time_points = metrics['time_points']
                ax_mass_evol.plot(
                    time_points,
                    metrics['mass_values'],
                    label=f"{integrator}"
                )
                ax_energy_evol.plot(
                    time_points,
                    metrics['total_energy'],
                    label=f"{integrator}"
                )
                ax_log_mass_err.plot(
                    time_points[1:],
                    metrics['conservation_metric_mass'][1:],
                    label=f"{integrator}"
                )
                ax_log_energy_err.plot(
                    time_points[1:],
                    metrics['conservation_metric_energy'][1:],
                    label=f"{integrator}"
                )

            ax_mass_evol.set_title(f"Mass Evolution (nx={nx}, nt={nt})")
            ax_mass_evol.set_ylabel("Total Mass")
            ax_mass_evol.grid(True)
            ax_mass_evol.legend()

            ax_energy_evol.set_title(f"Energy Evolution (nx={nx}, nt={nt})")
            ax_energy_evol.set_ylabel("Total Energy")
            ax_energy_evol.grid(True)
            ax_energy_evol.legend()

            ax_log_mass_err.set_title(f"Log Mass Error (nx={nx}, nt={nt})")
            ax_log_mass_err.set_xlabel("T / [1]")
            ax_log_mass_err.set_ylabel("Log Abs Mass Error")
            ax_log_mass_err.grid(True)
            ax_log_mass_err.legend()

            ax_log_energy_err.set_title(f"Log Energy Error (nx={nx}, nt={nt})")
            ax_log_energy_err.set_xlabel("T / [1]")
            ax_log_energy_err.set_ylabel("Log Abs Energy Error")
            ax_log_energy_err.grid(True)
            ax_log_energy_err.legend()

            fig.tight_layout()
            plot_filename = self.plots_dir / f"conservation_comparison_nt{nt}_nx{nx}_{self.run_id}.png"
            fig.savefig(plot_filename, dpi=300)
            plt.close(fig)

    def plot_structure_comparison(self, nt, metrics_by_nx):
        for i, nx in enumerate(sorted(metrics_by_nx.keys())):
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['structure_similarity'],
                    label=f"{integrator}")
            plt.title(f"Structure Similarity (nx={nx}, nt={nt})")
            plt.xlabel("T / [1]")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 2, 2)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['phase_errors'],
                    label=f"{integrator}")
            plt.title(f"Phase Error (nx={nx}, nt={nt})")
            plt.xlabel("T / [1]")
            plt.ylabel("Mean Phase Error")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 2, 3)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['amplitude_errors'],
                    label=f"{integrator}")
            plt.title(f"Amplitude Error (nx={nx}, nt={nt})")
            plt.xlabel("T / [1]")
            plt.ylabel("RMS Amplitude Error")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 2, 4)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['kinetic_energy'] / metrics['kinetic_energy'][0],
                    label=f"KE {integrator}")
                plt.plot(
                    metrics['time_points'],
                    metrics['potential_energy'] / metrics['potential_energy'][0],
                    '--',
                    label=f"PE {integrator}")
            plt.title(f"Energy Components (nx={nx}, nt={nt})")
            plt.xlabel("T / [1]")
            plt.ylabel("Relative Energy")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(
                self.plots_dir /
                f"structure_comparison_nt{nt}_nx{nx}_{self.run_id}.png",
                dpi=300)
            plt.close()

    def plot_modal_energy(self, nx, nt, metrics_by_integrator):
        plt.figure(figsize=(12, 10))
        for i, (integrator, metrics) in enumerate(
                metrics_by_integrator.items()):
            modal_e = metrics['modal_energy']
            snapshots = modal_e.shape[0]
            time_points = metrics['time_points']

            plt.subplot(2, 2, i + 1)
            plt.imshow(np.log(modal_e[snapshots // 2] + 1), cmap='viridis')
            plt.colorbar()
            plt.title(f'{integrator} Modal Energy t=T/2')
        plt.subplot(2, 1, 2)
        modes_to_plot = 2

        for integrator, metrics in metrics_by_integrator.items():
            modal_e = metrics['modal_energy']
            time_points = metrics['time_points']

            linestyles = '-' if integrator == 'gautschi' else '--'

            for i in range(modes_to_plot):
                for j in range(modes_to_plot):
                    if i == 0 and j == 0:
                        continue
                    plt.plot(time_points, modal_e[:, i, j] / modal_e[:, 0, 0],
                             label=f'Mode ({i},{j}) - {integrator}', linestyle=linestyles)

        plt.xlabel('Time')
        plt.ylabel('Relative Modal Energy')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"modal_energy_nx{nx}_nt{nt}_{self.run_id}.png",
            dpi=300)
        plt.close()

    def plot_dispersion_relation(self, nx, nt, metrics_by_integrator):
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        for integrator, metrics in metrics_by_integrator.items():
            k_values = metrics['dispersion_k']
            dispersion = metrics['dispersion_radial']
            dispersion_std = metrics['dispersion_std']

            plt.plot(k_values, dispersion, label=f'{integrator}')
            plt.fill_between(
                k_values,
                dispersion -
                dispersion_std,
                dispersion +
                dispersion_std,
                alpha=0.3)

        plt.title(f'Dispersion Relation (nx={nx}, nt={nt})')
        plt.xlabel('Wavenumber k')
        plt.ylabel('Frequency ω')
        plt.grid(True)
        plt.legend()

        for i, (integrator, metrics) in enumerate(
                metrics_by_integrator.items()):
            plt.subplot(2, 2, i + 3)
            plt.imshow(metrics['dispersion_full'], cmap='viridis', origin='lower',
                       extent=[0, np.pi / self.Lx * nx, 0, np.pi / self.Lx * nx])
            plt.colorbar()
            plt.title(f'{integrator} Dispersion Relation')
            plt.xlabel('kx')
            plt.ylabel('ky')

        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"dispersion_nx{nx}_nt{nt}_{self.run_id}.png",
            dpi=300)
        plt.close()

    def plot_metric_vs_resolution(self, metrics_all):
        metrics_to_plot = [
            ('mass_error', 'Max Mass Error',
             lambda x: np.max(x['mass_error'])),
            ('energy_error', 'Max Energy Error',
             lambda x: np.max(x['energy_error'])),
            ('structure_similarity', 'Final Structure Similarity',
             lambda x: x['structure_similarity'][-1])
        ]

        for nt in self.nt_values:
            plt.figure(figsize=(15, 5))

            for i, (metric_key, metric_name, metric_func) in enumerate(
                    metrics_to_plot):
                plt.subplot(1, 3, i + 1)

                for integrator in ['gautschi', 'ss2']:
                    x_values = []
                    y_values = []

                    for nx in self.nx_values:
                        if (nx, nt) in metrics_all and integrator in metrics_all[(
                                nx, nt)]:
                            x_values.append(nx)
                            y_values.append(metric_func(
                                metrics_all[(nx, nt)][integrator]))

                    plt.plot(x_values, y_values, 'o-', label=integrator)

                plt.title(metric_name)
                plt.xlabel('Grid Size (nx)')
                plt.grid(True)
                plt.legend()

            plt.tight_layout()
            plt.savefig(
                self.plots_dir /
                f"metrics_vs_nx_nt{nt}_{self.run_id}.png",
                dpi=300)
            plt.close()

        for nx in self.nx_values:
            plt.figure(figsize=(15, 5))

            for i, (metric_key, metric_name, metric_func) in enumerate(
                    metrics_to_plot):
                plt.subplot(1, 3, i + 1)

                for integrator in ['gautschi', 'ss2']:
                    x_values = []
                    y_values = []

                    for nt in self.nt_values:
                        if (nx, nt) in metrics_all and integrator in metrics_all[(
                                nx, nt)]:
                            x_values.append(nt)
                            y_values.append(metric_func(
                                metrics_all[(nx, nt)][integrator]))

                    plt.plot(x_values, y_values, 'o-', label=integrator)

                plt.title(metric_name)
                plt.xlabel('Time Steps (nt)')
                plt.grid(True)
                plt.legend()

            plt.tight_layout()
            plt.savefig(
                self.plots_dir /
                f"metrics_vs_nt_nx{nx}_{self.run_id}.png",
                dpi=300)
            plt.close()

    def plot_detailed_comparison(self, nx, nt, gautschi_traj, ss2_traj, gautschi_metrics, ss2_metrics):
        time_points = gautschi_metrics['time_points']

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.plot(
            time_points,
            gautschi_metrics['kinetic_energy'],
            label='Kinetic (G)')
        plt.plot(
            time_points,
            gautschi_metrics['potential_energy'],
            label='Potential (G)')
        plt.title("Energy Components (Gautschi)")
        plt.xlabel("T / [1]")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(
            time_points,
            ss2_metrics['kinetic_energy'],
            '--',
            label='Kinetic (SS2)')
        plt.plot(
            time_points,
            ss2_metrics['potential_energy'],
            '--',
            label='Potential (SS2)')
        plt.title("Energy Components (SS2)")
        plt.xlabel("T / [1]")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(
            time_points,
            gautschi_metrics['structure_similarity'],
            label='SSIM (G)')
        plt.plot(
            time_points,
            ss2_metrics['structure_similarity'],
            '--',
            label='SSIM (SS2)')
        plt.title("Structure Similarity")
        plt.xlabel("T / [1]")
        plt.grid(True)
        plt.legend()

        nt_steps = gautschi_traj.shape[0]
        indices = [nt_steps // 2, nt_steps - 1]

        for i, idx in enumerate(indices):
            plt.subplot(2, 3, i + 4)
            plt.imshow(
                np.abs(
                    gautschi_traj[idx] -
                    ss2_traj[idx]),
                cmap='viridis')
            plt.colorbar()
            plt.title(f'Difference at t={time_points[idx]:.2f}')

        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"detailed_comparison_nx{nx}_nt{nt}_{self.run_id}.png",
            dpi=300)
        plt.close()
        
        plt.figure(figsize=(15, 10))
        
        for i, idx in enumerate([0, nt_steps // 4, nt_steps // 2, 3 * nt_steps // 4, nt_steps - 1]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(np.abs(gautschi_traj[idx]), cmap='viridis')
            plt.title(f'|Gautschi| t={time_points[idx]:.2f}')
            plt.colorbar()
            
            plt.subplot(2, 5, i + 6)
            plt.imshow(np.abs(ss2_traj[idx]), cmap='viridis')
            plt.title(f'|SS2| t={time_points[idx]:.2f}')
            plt.colorbar()
            
        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"solution_evolution_nx{nx}_nt{nt}_{self.run_id}.png",
            dpi=300)
        plt.close()

    def plot_computational_efficiency(self, walltime_data, metrics_all):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        for nx in self.nx_values:
            ss2_times = [walltime_data[nx][nt]['ss2'] for nt in self.nt_values if nt in walltime_data[nx]]
            gautschi_times = [walltime_data[nx][nt]['gautschi'] for nt in self.nt_values if nt in walltime_data[nx]]
            nt_filtered = [nt for nt in self.nt_values if nt in walltime_data[nx]]
            if ss2_times and gautschi_times:
                plt.plot(nt_filtered, ss2_times, 'o-', label=f'SS2 nx={nx}')
                plt.plot(
                    nt_filtered,
                    gautschi_times,
                    's--',
                    label=f'Gautschi nx={nx}')

        plt.title("Computation Time vs Time Steps")
        plt.xlabel("Time Steps (nt)")
        plt.ylabel("Computation Time (s)")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 2)
        for nt in self.nt_values:
            nx_filtered = [nx for nx in self.nx_values if nt in walltime_data[nx]]
            if nx_filtered:
                ss2_times = [walltime_data[nx][nt]['ss2'] for nx in nx_filtered]
                gautschi_times = [walltime_data[nx][nt]['gautschi'] for nx in nx_filtered]
                plt.plot(nx_filtered, ss2_times, 'o-', label=f'SS2 nt={nt}')
                plt.plot(
                    nx_filtered,
                    gautschi_times,
                    's--',
                    label=f'Gautschi nt={nt}')

        plt.title("Computation Time vs Grid Size")
        plt.xlabel("Grid Size (nx)")
        plt.ylabel("Computation Time (s)")
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 3)

        for nt in self.nt_values:
            ss2_efficiency = []
            gautschi_efficiency = []
            nx_values = []

            for nx in self.nx_values:
                if (nx, nt) in metrics_all and 'ss2' in metrics_all[(
                        nx, nt)] and 'gautschi' in metrics_all[(nx, nt)] and nt in walltime_data[nx]:
                    ss2_accuracy = 1.0 / \
                        (1e-10 +
                         np.max(metrics_all[(nx, nt)]['ss2']['mass_error']))
                    gautschi_accuracy = 1.0 / \
                        (1e-10 +
                         np.max(metrics_all[(nx, nt)]['gautschi']['mass_error']))

                    ss2_efficiency.append(
                        ss2_accuracy / walltime_data[nx][nt]['ss2'])
                    gautschi_efficiency.append(
                        gautschi_accuracy / walltime_data[nx][nt]['gautschi'])
                    nx_values.append(nx)

            if nx_values:
                plt.plot(nx_values, ss2_efficiency, 'o-', label=f'SS2 nt={nt}')
                plt.plot(
                    nx_values,
                    gautschi_efficiency,
                    's--',
                    label=f'Gautschi nt={nt}')

        plt.title("Mass Conservation Efficiency")
        plt.xlabel("Grid Size (nx)")
        plt.ylabel("Accuracy per Compute Time")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 4)

        for nt in self.nt_values:
            ss2_efficiency = []
            gautschi_efficiency = []
            nx_values = []

            for nx in self.nx_values:
                if (nx, nt) in metrics_all and 'ss2' in metrics_all[(
                        nx, nt)] and 'gautschi' in metrics_all[(nx, nt)] and nt in walltime_data[nx]:
                    ss2_accuracy = metrics_all[(
                        nx, nt)]['ss2']['structure_similarity'][-1]
                    gautschi_accuracy = metrics_all[(
                        nx, nt)]['gautschi']['structure_similarity'][-1]

                    ss2_efficiency.append(
                        ss2_accuracy / walltime_data[nx][nt]['ss2'])
                    gautschi_efficiency.append(
                        gautschi_accuracy / walltime_data[nx][nt]['gautschi'])
                    nx_values.append(nx)

            if nx_values:
                plt.plot(nx_values, ss2_efficiency, 'o-', label=f'SS2 nt={nt}')
                plt.plot(
                    nx_values,
                    gautschi_efficiency,
                    's--',
                    label=f'Gautschi nt={nt}')

        plt.title("Structure Preservation Efficiency")
        plt.xlabel("Grid Size (nx)")
        plt.ylabel("Accuracy per Compute Time")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"computational_efficiency_{self.run_id}.png",
            dpi=300)
        plt.close()
    def generate_stability_map(self, metrics_all):
        fig, ax = plt.subplots(figsize=(10, 8))

        cell_size = 1.0
        nx_values = self.nx_values
        nt_values = self.nt_values

        for i, nx in enumerate(nx_values):
            for j, nt in enumerate(nt_values):
                x = i * cell_size
                y = j * cell_size

                gautschi_result = metrics_all.get((nx, nt), {}).get('gautschi', {})
                ss2_result = metrics_all.get((nx, nt), {}).get('ss2', {})

                has_gautschi_data = len(gautschi_result) > 0
                has_ss2_data = len(ss2_result) > 0

                g_stable = has_gautschi_data and not (np.any(np.isnan(gautschi_result.get('mass_values', [0]))) or
                                                   np.any(np.isnan(gautschi_result.get('total_energy', [0]))))
                ss2_stable = has_ss2_data and not (np.any(np.isnan(ss2_result.get('mass_values', [0]))) or
                                                np.any(np.isnan(ss2_result.get('total_energy', [0]))))

                ax.add_patch(plt.Rectangle((x, y), 0.5, 1.0, color='green' if g_stable else 'red'))
                ax.add_patch(plt.Rectangle((x + 0.5, y), 0.5, 1.0, color='green' if ss2_stable else 'red'))

                ax.text(x + 0.25, y + 0.5, "G", ha='center', va='center', fontsize=12,
                      color='black' if g_stable else 'white')
                ax.text(x + 0.75, y + 0.5, "S", ha='center', va='center', fontsize=12,
                      color='black' if ss2_stable else 'white')

        ax.set_xlim(0, len(nx_values) * cell_size)
        ax.set_ylim(0, len(nt_values) * cell_size)
        ax.set_xticks(np.arange(len(nx_values)) * cell_size + 0.5)
        ax.set_yticks(np.arange(len(nt_values)) * cell_size + 0.5)
        ax.set_xticklabels(nx_values)
        ax.set_yticklabels(nt_values)
        ax.set_xlabel('Grid Size (nx)')
        ax.set_ylabel('Time Steps (nt)')

        ax.set_title('Stability Map: Gautschi (G) vs SS2 (S)')

        plt.tight_layout()
        fig.savefig(self.plots_dir / f"stability_map_{self.run_id}.png", dpi=300)
        plt.close(fig)

    def generate_metrics_map(self, metrics_all):
        fig, ax = plt.subplots(figsize=(12, 10))

        cell_size = 1.0
        nx_values = self.nx_values
        nt_values = self.nt_values

        for i, nx in enumerate(nx_values):
            for j, nt in enumerate(nt_values):
                x = i * cell_size
                y = j * cell_size

                gautschi_result = metrics_all.get((nx, nt), {}).get('gautschi', {})
                ss2_result = metrics_all.get((nx, nt), {}).get('ss2', {})

                has_gautschi_data = len(gautschi_result) > 0
                has_ss2_data = len(ss2_result) > 0

                g_stable = has_gautschi_data and not (np.any(np.isnan(gautschi_result.get('mass_values', [0]))) or
                                                   np.any(np.isnan(gautschi_result.get('total_energy', [0]))))
                ss2_stable = has_ss2_data and not (np.any(np.isnan(ss2_result.get('mass_values', [0]))) or
                                                np.any(np.isnan(ss2_result.get('total_energy', [0]))))

                if has_gautschi_data and has_ss2_data and g_stable and ss2_stable:
                    g_l2 = np.max(gautschi_result.get('amplitude_errors', [0]))
                    ss2_l2 = np.max(ss2_result.get('amplitude_errors', [0]))

                    best_l2 = min(g_l2, ss2_l2)
                    g_l2_norm = g_l2 / best_l2 if best_l2 > 0 else 1.0
                    ss2_l2_norm = ss2_l2 / best_l2 if best_l2 > 0 else 1.0

                    g_energy_err = np.max(gautschi_result.get('energy_error', [0]))
                    ss2_energy_err = np.max(ss2_result.get('energy_error', [0]))

                    l2_cmap = plt.cm.coolwarm
                    ax.add_patch(plt.Rectangle((x, y), 0.5, 0.5, color=l2_cmap(min(g_l2_norm, 2.0)/2.0)))
                    ax.add_patch(plt.Rectangle((x + 0.5, y), 0.5, 0.5, color=l2_cmap(min(ss2_l2_norm, 2.0)/2.0)))

                    ax.add_patch(plt.Rectangle((x, y+0.5), 0.5, 0.5,
                                             color=plt.cm.viridis(min(1.0, g_energy_err*10))))
                    ax.add_patch(plt.Rectangle((x + 0.5, y+0.5), 0.5, 0.5,
                                             color=plt.cm.viridis(min(1.0, ss2_energy_err*10))))

                    ax.text(x + 0.25, y + 0.75, f"{g_energy_err:.1e}", ha='center', va='center', fontsize=6)
                    ax.text(x + 0.75, y + 0.75, f"{ss2_energy_err:.1e}", ha='center', va='center', fontsize=6)
                    ax.text(x + 0.25, y + 0.25, f"{g_l2:.1e}", ha='center', va='center', fontsize=6)
                    ax.text(x + 0.75, y + 0.25, f"{ss2_l2:.1e}", ha='center', va='center', fontsize=6)
                else:
                    ax.add_patch(plt.Rectangle((x, y), 1.0, 1.0, color='lightgray'))

        ax.set_xlim(0, len(nx_values) * cell_size)
        ax.set_ylim(0, len(nt_values) * cell_size)
        ax.set_xticks(np.arange(len(nx_values)) * cell_size + 0.5)
        ax.set_yticks(np.arange(len(nt_values)) * cell_size + 0.5)
        ax.set_xticklabels(nx_values)
        ax.set_yticklabels(nt_values)
        ax.set_xlabel('Grid Size (nx)')
        ax.set_ylabel('Time Steps (nt)')

        ax.set_title('Metrics Map: Gautschi vs SS2')

        l2_sm = plt.cm.ScalarMappable(cmap=l2_cmap, norm=Normalize(0, 1))
        l2_sm.set_array([])
        l2_cbar = plt.colorbar(l2_sm, ax=ax, location='right', pad=0.1)
        l2_cbar.set_label('L2 Distance (normalized)')

        e_sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(0, 0.1))
        e_sm.set_array([])
        e_cbar = plt.colorbar(e_sm, ax=ax, location='right', pad=0.15)
        e_cbar.set_label('Energy Error')

        plt.tight_layout()
        fig.savefig(self.plots_dir / f"metrics_map_{self.run_id}.png", dpi=300)
        plt.close(fig)

    def execute_comparison(self):
        u0, m, u0_file, m_file = self.generate_base_conditions()

        metrics_all = {}
        walltime_data = {}

        for nx in self.nx_values:
            walltime_data[nx] = {}
            for nt in self.nt_values:
                print(f"Running comparison for nx={nx}, nt={nt}")
                


                metrics_by_integrator = {}
                walltime_data[nx][nt] = {}

                for integrator in ['gautschi', 'ss2']:
                    traj_file, walltime = self.run_simulation(
                        integrator, nx, nt, u0, m)
                    print(f"    {integrator} completed in {walltime:.2f}s")
                    walltime_data[nx][nt][integrator] = walltime

                    metrics, traj_data = self.analyze_simulation(
                        integrator, nx, nt, traj_file)
                    metrics_by_integrator[integrator] = metrics

                    if (nx, nt) not in metrics_all:
                        metrics_all[(nx, nt)] = {}

                    metrics_all[(nx, nt)][integrator] = metrics

                    if nx == max(self.nx_values) and nt == max(self.nt_values):
                        xn = yn = np.linspace(-self.Lx, self.Lx, nx)
                        X, Y = np.meshgrid(xn, yn)
                        if integrator == 'gautschi':
                            gautschi_traj = traj_data
                            gautschi_metrics = metrics 
                            
                            animate_simulation(X, Y,
                                              np.abs(gautschi_traj), self.snapshots,
                                              self.analysis_dir / f"gautschi_{self.run_id}.mp4",
                                              title="Gautschi trajectory")
                        else:                    
                            ss2_traj = traj_data
                            ss2_metrics = metrics 
                            animate_simulation(X, Y,
                                              np.abs(ss2_traj), self.snapshots,
                                              self.analysis_dir / f"ss2_{self.run_id}.mp4",
                                              title="SS2 trajectory")

                self.plot_modal_energy(nx, nt, metrics_by_integrator)
                self.plot_dispersion_relation(nx, nt, metrics_by_integrator)
                metrics_by_nx = {nx: metrics_by_integrator}
                self.plot_conservation_comparison(nt, metrics_by_nx)
                self.plot_structure_comparison(nt, metrics_by_nx)
               
        self.plot_metric_vs_resolution(metrics_all)
        self.plot_computational_efficiency(walltime_data, metrics_all)
        self.generate_stability_map(metrics_all)
        self.generate_metrics_map(metrics_all)

        
        if 'gautschi_traj' in locals() and 'ss2_traj' in locals():
            self.plot_detailed_comparison(
                max(self.nx_values), max(self.nt_values),
                gautschi_traj, ss2_traj, gautschi_metrics, ss2_metrics)
 
        return metrics_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Gautschi and SS2 integrators for NLSE")
    parser.add_argument("--gautschi-exe", type=str, required=True,
                        help="Path to Gautschi integrator executable")
    parser.add_argument("--ss2-exe", type=str, required=True,
                        help="Path to SS2 integrator executable")

    parser.add_argument("--ic-type", type=str, default="multi_soliton",                        
                        choices=["multi_soliton", "spectral", "chaotic", "vortex_lattice",
                                 "dark_soliton", "multi_ring", "solitary_wave_with_ambient",
                                 "logarithmic_singularity_adapted", "turbulent_condensate",
                                 "topological_defect_network", "akhmediev_breather",
                                 "self_similar_pattern"],
                        required=True,
                        help="Phenomenon type to simulate")
    parser.add_argument("--m-value", type=float, default=1.0,
                        help="Value for constant m or amplitude for spatially varying m")
    parser.add_argument("--nx-values", type=int, nargs='+', required=True,
                        help="List of spatial resolution values to test")
    parser.add_argument("--nt-values", type=int, nargs='+', required=True,
                        help="List of temporal resolution values to test")
    parser.add_argument(
        "--Lx",
        type=float,
        default=10.0,
        help="Domain half-width in x")
    parser.add_argument(
        "--Ly",
        type=float,
        default=10.0,
        help="Domain half-width in y")
    parser.add_argument(
        "--T",
        type=float,
        default=10.0,
        help="Simulation end time")
    parser.add_argument("--output-dir", type=str, default="integrator_comparison_results",
                        help="Directory for output data")

    return parser.parse_args()


def main():
    args = parse_args()
    comparison = NLSEIntegratorComparison(args)
    metrics_all = comparison.execute_comparison()
    print(f"results in {args.output_dir}")
    for traj_file in comparison.traj_files:
        os.unlink(traj_file)


if __name__ == "__main__":
    main()
