import numpy as np
import matplotlib.pyplot as plt
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

from real_sampler import RealWaveSampler
from downsampling import downsample_interpolation, reconstruct_interpolation
from valid_spaces import get_parameter_spaces
from visualization import animate_simulation

from spatial_amplification import (
    create_constant_m,
    create_periodic_boxes,
    create_periodic_gaussians,
    create_grf,
    create_wavelet_modulated_grf,
    scale_m_to_range
)


def compute_energy(u, v, dx, dy):
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
    ut = v[1:-1, 1:-1]
    ux2 = .5 * ux ** 2
    uy2 = .5 * uy ** 2
    ut2 = .5 * ut ** 2
    cos = (1 - np.cos(u[1:-1, 1:-1]))
    return ux2 + uy2, ut2, cos


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
        reference_frame = data[0]

    nt = data.shape[0]
    ssim_values = np.zeros(nt)

    for t in range(nt):
        ssim_values[t] = ssim(
            reference_frame,
            data[t],
            data_range=data.max() -
            data.min())

    return ssim_values


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


def compute_local_conservation(data, vel_data, dx, dy, dt):
    nt, nx, ny = data.shape
    conservation_metric = np.zeros(nt)

    for t in range(1, nt):
        u = data[t]
        u_prev = data[t - 1]
        v = vel_data[t]

        laplacian = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] +
                     u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / (dx * dy)
        ut_numerical = (u - u_prev) / dt
        ut_physical = v

        conservation_err = np.mean(
            np.abs(ut_numerical[1:-1, 1:-1] - ut_physical[1:-1, 1:-1]))
        conservation_metric[t] = conservation_err

    return conservation_metric


class IntegratorComparison:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.executables = {
            'gautschi': args.gautschi_exe,
            'sv': args.sv_exe
        }
        self.nx_values = args.nx_values
        self.nt_values = args.nt_values
        self.system_type = args.system_type
        self.phenomenon_type = args.phenomenon_type
        self.Lx = args.Lx
        self.Ly = args.Ly
        self.T = args.T
        self.setup_directories()

        max_n = max(args.nx_values)
        self.sampler = RealWaveSampler(max_n, max_n, args.Lx)

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

    def sample_phenomenon_params(self, phenomenon_type):
        parameter_spaces = get_parameter_spaces(self.args.Lx)
        if phenomenon_type in parameter_spaces:
            space = parameter_spaces[phenomenon_type]
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
            raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")

    def generate_base_conditions(self):
        max_nx = max(self.nx_values)
        u0_file = self.ic_dir / f"u0_{max_nx}_{self.run_id}.npy"
        v0_file = self.ic_dir / f"v0_{max_nx}_{self.run_id}.npy"
        m_file = self.ic_dir / f"m_{max_nx}_{self.run_id}.npy"

        phenomenon_params = self.sample_phenomenon_params(
            self.args.phenomenon_type)
        phenomenon_params["system_type"] = "sine_gordon"
        phenomenon_params["velocity_type"] = self.args.velocity_type

        print(f"Parameters: {phenomenon_params}")

        u0, v0 = self.sampler.generate_initial_condition(
            phenomenon_type=self.args.phenomenon_type, **phenomenon_params)

        if isinstance(u0, torch.Tensor):
            u0 = u0.detach().numpy()
        if isinstance(v0, torch.Tensor):
            v0 = v0.detach().numpy()

        u0_file = self.ic_dir / f"reference_u0_{self.run_id}.npy"
        v0_file = self.ic_dir / f"reference_v0_{self.run_id}.npy"

        m = np.ones((max_nx, max_nx))
        # u0 = u0 / np.max(np.abs(u0))
        # v0 = v0 / np.max(np.abs(v0))

        np.save(u0_file, u0)
        np.save(v0_file, v0)
        np.save(m_file, m)

        self.reference_ic = (u0, v0)
        self.reference_ic_params = phenomenon_params

        return u0, v0, m, u0_file, v0_file, m_file

    def run_simulation(self, integrator_type, nx, nt, u0, v0, m):
        u0_nx_file = self.ic_dir / f"u0_{nx}_{self.run_id}.npy"
        v0_nx_file = self.ic_dir / f"v0_{nx}_{self.run_id}.npy"
        m_nx_file = self.ic_dir / f"m_{nx}_{self.run_id}.npy"
        if nx != u0.shape[0]:
            u0_nx = downsample_interpolation(u0.reshape(
                1, *u0.shape), (nx, nx), self.Lx, self.Ly)[0]
            v0_nx = downsample_interpolation(v0.reshape(
                1, *v0.shape), (nx, nx), self.Lx, self.Ly)[0]
            m_nx = downsample_interpolation(
                m.reshape(1, *m.shape), (nx, nx), self.Lx, self.Ly)[0]
        else:
            u0_nx, v0_nx, m_nx = u0, v0, m

        np.save(u0_nx_file, u0_nx)
        np.save(v0_nx_file, v0_nx)
        np.save(m_nx_file, m_nx)

        traj_file = self.traj_dir / \
            f"{integrator_type}_{nx}_{nt}_{self.run_id}.npy"
        vel_file = self.traj_dir / \
            f"{integrator_type}_{nx}_{nt}_{self.run_id}_vel.npy"

        exe_path = Path(self.executables[integrator_type])
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable {exe_path} not found")

        snapshots = min(100, nt // 10)
        self.snapshots = snapshots

        cmd = [
            str(exe_path),
            str(nx),
            str(nx),
            str(self.Lx),
            str(self.Ly),
            str(u0_nx_file),
            str(v0_nx_file),
            str(traj_file),
            str(vel_file),
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

        return traj_file, vel_file, walltime

    def analyze_simulation(self, integrator, nx, nt, traj_file, vel_file):
        traj_data = np.load(traj_file, mmap_mode='r')
        vel_data = np.load(vel_file, mmap_mode='r')
        snapshots = traj_data.shape[0]

        time_points = np.linspace(0, self.T, snapshots)
        dx = 2 * self.Lx / (nx - 1)
        dy = 2 * self.Ly / (nx - 1)
        dt = self.T / (nt - 1)
        energy_components = [
            compute_energy(
                u,
                vel_data[i],
                dx,
                dy) for i,
            u in enumerate(traj_data)]

        e_gradient = np.zeros(len(energy_components))
        e_kinetic = np.zeros(len(energy_components))
        e_potential = np.zeros(len(energy_components))

        for i, (grad, kin, pot) in enumerate(energy_components):
            e_gradient[i] = np.sum(grad) * dx * dy * 0.5
            e_kinetic[i] = np.sum(kin) * dx * dy * 0.5
            e_potential[i] = np.sum(pot) * dx * dy * 0.5

        total_energy = e_gradient + e_kinetic + e_potential
        energy_drift = np.abs(total_energy - total_energy[0]) / total_energy[0]

        modal_energy = compute_modal_energy(traj_data)
        ssim_values = compute_structure_similarity(traj_data)
        k_centers, dispersion_radial, dispersion_std, dispersion_full = compute_spectral_dispersion(
            traj_data, dx, dt)

        conservation_metric = np.array(
            [np.nan] + [np.log(np.abs(total_energy[i] - total_energy[0]))
                        for i in range(1, snapshots)]
        )

        metrics = {
            'time_points': time_points,
            'energy_gradient': e_gradient,
            'energy_kinetic': e_kinetic,
            'energy_potential': e_potential,
            'total_energy': total_energy,
            'energy_drift': energy_drift,
            'modal_energy': modal_energy,
            'structure_similarity': ssim_values,
            'dispersion_radial': dispersion_radial,
            'dispersion_std': dispersion_std,
            'dispersion_full': dispersion_full,
            'dispersion_k': k_centers,
            'conservation_metric': conservation_metric
        }

        return metrics, traj_data, vel_data

    def plot_energy_comparison(self, nt, metrics_by_nx):
        #plt.figure(figsize=(15, 18))

        print(list(metrics_by_nx.keys()))
        for i, nx in enumerate(sorted(metrics_by_nx.keys())):
            plt.subplot(len(metrics_by_nx), 2, 2 * i + 1)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['total_energy'],
                    label=f"{integrator}")
            plt.title(f"Energy (nx={nx}, nt={nt})")
            plt.xlabel("T / [1]")
            plt.ylabel("Total Energy")
            plt.grid(True)
            plt.legend()

            plt.subplot(len(metrics_by_nx), 2, 2 * i + 2)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['energy_drift'],
                    label=f"{integrator}")
            plt.title(f"Energy Drift (nx={nx}, nt={nt})")
            plt.xlabel("T / [1]")
            plt.ylabel("Relative Energy Drift")
            plt.grid(True)
            plt.legend()

        # plt.tight_layout()
        if len(list(metrics_by_nx.keys())):
            nx = list(metrics_by_nx.keys())[0]
            plt.savefig(
                self.plots_dir /
                f"energy_comparison_nt{nt}_nx{nx}_{self.run_id}.png",
                dpi=300)
        else:
            plt.savefig(
                self.plots_dir /
                f"energy_comparison_nt{nt}_{self.run_id}.png",
                dpi=300)
        plt.close()

    def plot_structure_comparison(self, nt, metrics_by_nx):
        fig = plt.figure()

        for i, nx in enumerate(sorted(metrics_by_nx.keys())):
            plt.subplot(len(metrics_by_nx), 2, 2 * i + 1)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['structure_similarity'],
                    label=f"{integrator}")
            plt.title(f"Structure Similarity")
            plt.xlabel("T / [1]")
            plt.ylabel("SSIM")
            plt.grid(True)
            plt.legend()

            plt.subplot(len(metrics_by_nx), 2, 2 * i + 2)
            for integrator, metrics in metrics_by_nx[nx].items():
                plt.plot(
                    metrics['time_points'],
                    metrics['conservation_metric'],
                    label=f"{integrator}")
            plt.title(f"Local Conservation Error ")
            plt.xlabel("T / [1]")
            plt.ylabel("Mean Error")
            plt.grid(True)
            plt.legend()
        fig.suptitle(f"(nx={nx}, nt={nt})")
        # plt.tight_layout()
        if len(list(metrics_by_nx.keys())):
            nx = list(metrics_by_nx.keys())[0]
            plt.savefig(
                self.plots_dir /
                f"structure_comparison_nt{nt}_nx{nx}_{self.run_id}.png",
                dpi=300)
        else:
            plt.savefig(
                self.plots_dir /
                f"structure_comparison_nt{nt}_{self.run_id}.png",
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

        # plt.plot(k_values, np.sqrt(k_values**2 + 1), 'k--', label='Theoretical')
        plt.subplot(2, 1, 2)

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
            ('energy_drift', 'Max Energy Drift',
             lambda x: np.max(x['energy_drift'])),
            ('structure_similarity', 'Final Structure Similarity',
             lambda x: x['structure_similarity'][-1]),
            ('conservation_metric', 'Mean Conservation Error',
             lambda x: np.mean(x['conservation_metric']))
        ]

        for nt in self.nt_values:
            plt.figure(figsize=(15, 5))

            for i, (metric_key, metric_name, metric_func) in enumerate(
                    metrics_to_plot):
                plt.subplot(1, 3, i + 1)

                for integrator in ['gautschi', 'sv']:
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

                for integrator in ['gautschi', 'sv']:
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

    def plot_detailed_comparison(
            self, nx, nt, gautschi_traj, sv_traj, gautschi_metrics, sv_metrics):
        time_points = gautschi_metrics['time_points']

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.plot(
            time_points,
            gautschi_metrics['energy_gradient'],
            label='Gradient (G)')
        plt.plot(
            time_points,
            gautschi_metrics['energy_kinetic'],
            label='Kinetic (G)')
        plt.plot(
            time_points,
            gautschi_metrics['energy_potential'],
            label='Potential (G)')
        plt.title("Energy Components (Gautschi)")
        plt.xlabel("T / [1]")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(
            time_points,
            sv_metrics['energy_gradient'],
            '--',
            label='Gradient (SV)')
        plt.plot(
            time_points,
            sv_metrics['energy_kinetic'],
            '--',
            label='Kinetic (SV)')
        plt.plot(
            time_points,
            sv_metrics['energy_potential'],
            '--',
            label='Potential (SV)')
        plt.title("Energy Components (Stormer-Verlet)")
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
            sv_metrics['structure_similarity'],
            '--',
            label='SSIM (SV)')
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
                    sv_traj[idx]),
                cmap='viridis')
            plt.colorbar()
            plt.title(f'Difference at t={time_points[idx]:.2f}')

        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"detailed_comparison_nx{nx}_nt{nt}_{self.run_id}.png",
            dpi=300)
        plt.close()

    def execute_comparison(self):
        u0, v0, m, u0_file, v0_file, m_file = self.generate_base_conditions()

        metrics_all = {}
        walltime_data = {}

        for nx in self.nx_values:
            walltime_data[nx] = {}
            for nt in self.nt_values:
                print(f"Running comparison for nx={nx}, nt={nt}")
                metrics_by_integrator = {}
                walltime_data[nx][nt] = {}

                for integrator in ['gautschi', 'sv']:
                    traj_file, vel_file, walltime = self.run_simulation(
                        integrator, nx, nt, u0, v0, m)
                    print(f"    {integrator} completed in {walltime:.2f}s")
                    walltime_data[nx][nt][integrator] = walltime

                    metrics, traj_data, vel_data = self.analyze_simulation(
                        integrator, nx, nt, traj_file, vel_file)
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
                                               gautschi_traj, self.snapshots,
                                               self.analysis_dir / "gautschi.mp4",
                                               title="Gautschi trajectory")
                        else:
                            sv_traj = traj_data
                            sv_metrics = metrics
                            animate_simulation(X, Y,
                                               sv_traj, self.snapshots,
                                               self.analysis_dir / "sv.mp4",
                                               title="Stormer-Verlet trajectory")

                self.plot_modal_energy(nx, nt, metrics_by_integrator)
                self.plot_dispersion_relation(nx, nt, metrics_by_integrator)
                metrics_by_nx = {nx: metrics_by_integrator}
                self.plot_energy_comparison(nt, metrics_by_nx)
                self.plot_structure_comparison(nt, metrics_by_nx)

        self.plot_metric_vs_resolution(metrics_all)
        self.plot_computational_efficiency(walltime_data, metrics_all)
        if 'gautschi_traj' in locals() and 'sv_traj' in locals():
            self.plot_detailed_comparison(
                max(self.nx_values), max(self.nt_values),
                gautschi_traj, sv_traj, gautschi_metrics, sv_metrics)

        return metrics_all

    def plot_computational_efficiency(self, walltime_data, metrics_all):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        for nx in self.nx_values:
            sv_times = [walltime_data[nx][nt]['sv'] for nt in self.nt_values]
            gautschi_times = [walltime_data[nx][nt]['gautschi']
                              for nt in self.nt_values]
            plt.plot(self.nt_values, sv_times, 'o-', label=f'SV nx={nx}')
            plt.plot(
                self.nt_values,
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
            sv_times = [walltime_data[nx][nt]['sv'] for nx in self.nx_values]
            gautschi_times = [walltime_data[nx][nt]['gautschi']
                              for nx in self.nx_values]
            plt.plot(self.nx_values, sv_times, 'o-', label=f'SV nt={nt}')
            plt.plot(
                self.nx_values,
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
            sv_efficiency = []
            gautschi_efficiency = []
            nx_values = []

            for nx in self.nx_values:
                if (nx, nt) in metrics_all and 'sv' in metrics_all[(
                        nx, nt)] and 'gautschi' in metrics_all[(nx, nt)]:
                    sv_accuracy = 1.0 / \
                        (1e-10 +
                         np.max(metrics_all[(nx, nt)]['sv']['energy_drift']))
                    gautschi_accuracy = 1.0 / \
                        (1e-10 +
                         np.max(metrics_all[(nx, nt)]['gautschi']['energy_drift']))

                    sv_efficiency.append(
                        sv_accuracy / walltime_data[nx][nt]['sv'])
                    gautschi_efficiency.append(
                        gautschi_accuracy / walltime_data[nx][nt]['gautschi'])
                    nx_values.append(nx)

            if nx_values:
                plt.plot(nx_values, sv_efficiency, 'o-', label=f'SV nt={nt}')
                plt.plot(
                    nx_values,
                    gautschi_efficiency,
                    's--',
                    label=f'Gautschi nt={nt}')

        plt.title("Energy Conservation Efficiency")
        plt.xlabel("Grid Size (nx)")
        plt.ylabel("Accuracy per Compute Time")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 4)

        for nt in self.nt_values:
            sv_efficiency = []
            gautschi_efficiency = []
            nx_values = []

            for nx in self.nx_values:
                if (nx, nt) in metrics_all and 'sv' in metrics_all[(
                        nx, nt)] and 'gautschi' in metrics_all[(nx, nt)]:
                    sv_accuracy = metrics_all[(
                        nx, nt)]['sv']['structure_similarity'][-1]
                    gautschi_accuracy = metrics_all[(
                        nx, nt)]['gautschi']['structure_similarity'][-1]

                    sv_efficiency.append(
                        sv_accuracy / walltime_data[nx][nt]['sv'])
                    gautschi_efficiency.append(
                        gautschi_accuracy / walltime_data[nx][nt]['gautschi'])
                    nx_values.append(nx)

            if nx_values:
                plt.plot(nx_values, sv_efficiency, 'o-', label=f'SV nt={nt}')
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Gautschi and Störmer-Verlet integrators")
    parser.add_argument("--gautschi-exe", type=str, required=True,
                        help="Path to Gautschi integrator executable")
    parser.add_argument("--sv-exe", type=str, required=True,
                        help="Path to Störmer-Verlet integrator executable")
    parser.add_argument("--system-type", type=str, required=True,
                        choices=[
                            "klein_gordon",
                            "sine_gordon",
                            "double_sine_gordon",
                            "hyperbolic_sine_gordon"],
                        help="Type of PDE system to solve")
    parser.add_argument("--phenomenon_type", type=str, default="kink_field",
                        choices=["kink_solution", "kink_field", "kink_array_field",
                                 "multi_breather_field", "spiral_wave_field", "multi_spiral_state",
                                 "ring_soliton", "colliding_rings", "multi_ring_state",
                                 "skyrmion_solution", "skyrmion_lattice", "skyrmion_like_field",
                                 "q_ball_solution", "multi_q_ball", "breather_solution",
                                 "breather_field", "elliptical_soliton", "grf_modulated_soliton_field"])
    parser.add_argument("--velocity-type", type=str, default="zero",
                        choices=["zero", "fitting", "random"])
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
    comparison = IntegratorComparison(args)
    metrics_all = comparison.execute_comparison()
    print(f"results in {args.output_dir}")


if __name__ == "__main__":
    main()
