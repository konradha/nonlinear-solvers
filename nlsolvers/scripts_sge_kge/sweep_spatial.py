from valid_spaces import get_parameter_spaces
from real_sampler import RealWaveSampler
from spatial_amplification import (
    create_constant_m,
    create_periodic_boxes,
    create_periodic_gaussians,
    create_grf,
    create_wavelet_modulated_grf,
    scale_m_to_range
)
import argparse
import os
import time
import uuid
import subprocess
import numpy as np
import h5py
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.interpolate import griddata
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import find_peaks
import gc
import warnings
warnings.filterwarnings('ignore')


class ParameterSweep:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.executables = {
            'klein_gordon': args.klein_gordon_exe,
            'sine_gordon': args.sine_gordon_exe,
            'double_sine_gordon': args.double_sine_gordon_exe,
            'hyperbolic_sine_gordon': args.hyperbolic_sine_gordon_exe
        }
        self.system_types = list(self.executables.keys())
        self.m_types = [
            'constant',
            'periodic_boxes',
            'periodic_gaussians',
            'grf',
            'wavelet_grf']

        self.setup_directories()
        self.configure_grid()

        self.sampler = RealWaveSampler(args.nx, args.ny, args.Lx)

        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))

        self.reference_ic = None
        self.reference_ic_params = None
        self.parameter_grid = self.generate_parameter_grid()
        self.trajectory_features = {}
        self.outliers = {}

    def setup_directories(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.ic_dir = self.output_dir / "initial_conditions"
        self.ic_dir.mkdir(exist_ok=True)

        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(exist_ok=True)

        self.focusing_dir = self.output_dir / "focusing"
        self.focusing_dir.mkdir(exist_ok=True)

        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.h5_dir = self.output_dir / "hdf5"
        self.h5_dir.mkdir(exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.ensemble_dir = self.output_dir / "ensemble"
        self.ensemble_dir.mkdir(exist_ok=True)

        self.cluster_dir = self.output_dir / "clustering"
        self.cluster_dir.mkdir(exist_ok=True)

        self.outlier_dir = self.output_dir / "outliers"
        self.outlier_dir.mkdir(exist_ok=True)

    def configure_grid(self):
        self.x = np.linspace(-self.args.Lx, self.args.Lx, self.args.nx)
        self.y = np.linspace(-self.args.Ly, self.args.Ly, self.args.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.dx = 2 * self.args.Lx / (self.args.nx - 1)
        self.dy = 2 * self.args.Ly / (self.args.ny - 1)
        self.dV = self.dx * self.dy

    def generate_parameter_grid(self):
        grid = []

        m_constant_values = [-10., -5., -1., 1., 5., 10.]

        m_grf_params = [
            (0.0, 1, 2.), (0.0, 10, 2.), (0.0, 40, 2.),
            (0.0, 1, .4), (0.0, 10, .4), (0.0, 40, .4),
        ]

        m_periodic_params = [
            (10., 2, 0.05), (-10., 2, 0.1), (10., 2, 0.2),
            (10., 4, 0.05), (-10., 4, 0.1), (10., 4, 0.2),
        ]

        for system_type in self.system_types:
            for m_value in m_constant_values:
                grid.append({
                    'system_type': system_type,
                    'm_type': 'constant',
                    'm_params': {'value': m_value}
                })

            for mean, std, scale in m_grf_params:
                grid.append({
                    'system_type': system_type,
                    'm_type': 'grf',
                    'm_params': {'mean': mean, 'std': std, 'scale': scale}
                })

                for wavelet_scale in [1.0, 2.0]:
                    grid.append({
                        'system_type': system_type,
                        'm_type': 'wavelet_grf',
                        'm_params': {
                            'mean': mean,
                            'std': std,
                            'scale': scale,
                            'wavelet_scale': wavelet_scale
                        }
                    })

            for factor, boxes_per_dim, box_length in m_periodic_params:
                grid.append({
                    'system_type': system_type,
                    'm_type': 'periodic_boxes',
                    'm_params': {
                        'factor': factor,
                        'num_boxes_per_dim': boxes_per_dim,
                        'box_length': box_length,
                        'wall_dist': 0.1
                    }
                })

                grid.append({
                    'system_type': system_type,
                    'm_type': 'periodic_gaussians',
                    'm_params': {
                        'factor': factor,
                        'num_gaussians_per_dim': boxes_per_dim,
                        'sigma': box_length,
                        'wall_dist': 0.1
                    }
                })

        return grid

    def generate_reference_ic(self):
        phenomenon_params = self.sample_phenomenon_params(self.args.phenomenon)
        phenomenon_params["system_type"] = "sine_gordon"
        phenomenon_params["velocity_type"] = self.args.velocity_type

        print(f"Parameters: {phenomenon_params}")

        u0, v0 = self.sampler.generate_initial_condition(
            phenomenon_type=self.args.phenomenon, **phenomenon_params)

        if isinstance(u0, torch.Tensor):
            u0 = u0.detach().numpy()
        if isinstance(v0, torch.Tensor):
            v0 = v0.detach().numpy()

        self.reference_ic = (u0, v0)
        self.reference_ic_params = phenomenon_params

        u0_file = self.ic_dir / f"reference_u0_{self.run_id}.npy"
        v0_file = self.ic_dir / f"reference_v0_{self.run_id}.npy"
        np.save(u0_file, u0)
        np.save(v0_file, v0)

        plt.figure(figsize=(10, 8))
        plt.imshow(u0, extent=[-self.args.Lx, self.args.Lx, -self.args.Ly, self.args.Ly],
                   origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title(f"IC: {self.args.phenomenon}")
        plt.savefig(
            self.plots_dir /
            f"reference_ic_{self.run_id}.png",
            dpi=300)
        plt.close()

        return u0_file, v0_file

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

    def generate_spatial_amplification(self, m_type, **params):
        if m_type == "constant":
            m = create_constant_m(self.X, self.Y, value=params['value'])
        elif m_type == "periodic_gaussians":
            m = create_periodic_gaussians(
                self.args.nx,
                self.args.Lx,
                factor=params['factor'],
                num_gaussians_per_dim=params['num_gaussians_per_dim'],
                sigma=params['sigma'],
                wall_dist=params['wall_dist']
            )
        elif m_type == "periodic_boxes":
            m = create_periodic_boxes(
                self.args.nx,
                self.args.Lx,
                factor=params['factor'],
                num_boxes_per_dim=params['num_boxes_per_dim'],
                box_length=params['box_length'],
                wall_dist=params['wall_dist']
            )
        elif m_type == "grf":
            m = create_grf(
                self.args.nx,
                self.args.ny,
                self.args.Lx,
                self.args.Ly,
                mean=params['mean'],
                std=params['std'],
                scale=params['scale']
            )
        elif m_type == "wavelet_grf":
            m = create_wavelet_modulated_grf(
                self.args.nx,
                self.args.ny,
                self.args.Lx,
                self.args.Ly,
                wavelet_scale=params['wavelet_scale'],
                grf_scale=params['scale'],
                mean=params['mean'],
                std=params['std']
            )
        else:
            raise ValueError(f"Unknown m_type: {m_type}")

        return m.astype(np.float64)

    def run_simulation(self, system_type, m_file, u0_file, v0_file, run_idx):
        traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"
        vel_file = self.traj_dir / f"vel_{self.run_id}_{run_idx:04d}.npy"

        exe_path = Path(self.executables[system_type])
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable {exe_path} not found")

        cmd = [
            str(exe_path),
            str(self.args.nx),
            str(self.args.ny),
            str(self.args.Lx),
            str(self.args.Ly),
            str(u0_file),
            str(v0_file),
            str(traj_file),
            str(vel_file),
            str(self.args.T),
            str(self.args.nt),
            str(self.args.snapshots),
            str(m_file)
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")

        end_time = time.time()
        walltime = end_time - start_time

        return traj_file, vel_file, walltime

    def compute_metrics_from_trajectory(
            self, traj_file, vel_file, m, chunk_size=10):
        traj_data = np.load(traj_file, mmap_mode='r')
        vel_data = np.load(vel_file, mmap_mode='r')

        T, nx, ny = traj_data.shape

        lyapunov_data = np.zeros(T - 1)
        entropy_data = np.zeros(T)
        localization_data = np.zeros(T)
        instability_data = np.zeros(T)
        energy_data = np.zeros(T)

        grad_m_x = np.zeros_like(m)
        grad_m_y = np.zeros_like(m)

        grad_m_x[1:-1, :] = (m[2:, :] - m[:-2, :]) / (2 * self.dx)
        grad_m_y[:, 1:-1] = (m[:, 2:] - m[:, :-2]) / (2 * self.dy)

        grad_m_magnitude = np.sqrt(grad_m_x**2 + grad_m_y**2)

        num_chunks = T // chunk_size + (1 if T % chunk_size else 0)
        first_frame = None

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, T)

            chunk_traj = traj_data[start_idx:end_idx]
            chunk_vel = vel_data[start_idx:end_idx]

            if chunk_idx == 0:
                first_frame = chunk_traj[0].copy()

                dx_0 = (first_frame[1:, :] - first_frame[:-1, :])[:, 1:]
                dy_0 = (first_frame[:, 1:] - first_frame[:, :-1])[1:, :]
                delta_0 = np.sqrt(dx_0**2 + dy_0**2)
                delta_avg_0 = np.mean(delta_0)

            for t_idx, (frame, vel_frame) in enumerate(
                    zip(chunk_traj, chunk_vel)):
                global_t_idx = start_idx + t_idx

                if global_t_idx > 0:
                    dx = (frame[1:, :] - frame[:-1, :])[:, 1:]
                    dy = (frame[:, 1:] - frame[:, :-1])[1:, :]
                    delta = np.sqrt(dx**2 + dy**2)
                    delta_avg = np.mean(delta)

                    lyapunov_data[global_t_idx -
                                  1] = np.log(delta_avg / delta_avg_0)

                u_fft = np.fft.fft2(frame)
                u_fft = np.fft.fftshift(u_fft)
                power_spectrum = np.abs(u_fft)**2 * \
                    self.dx * self.dy / (2 * np.pi)**2
                total_energy = np.sum(power_spectrum)

                normalized_spectrum = power_spectrum / total_energy
                valid_indices = normalized_spectrum > 1e-10
                entropy_data[global_t_idx] = -np.sum(normalized_spectrum[valid_indices] *
                                                     np.log(normalized_spectrum[valid_indices]))

                u_squared = frame**2
                numerator = np.sum(u_squared**2)
                denominator = np.sum(u_squared)**2
                localization_data[global_t_idx] = numerator / \
                    denominator * (nx * ny)

                grad_u_x = np.zeros_like(frame)
                grad_u_y = np.zeros_like(frame)

                grad_u_x[1:-1, :] = (frame[2:, :] -
                                     frame[:-2, :]) / (2 * self.dx)
                grad_u_y[:, 1:-1] = (frame[:, 2:] -
                                     frame[:, :-2]) / (2 * self.dy)

                grad_u_squared = grad_u_x**2 + grad_u_y**2
                integrand = grad_u_squared * grad_m_magnitude
                instability_data[global_t_idx] = np.sum(
                    integrand) * self.dx * self.dy

                kinetic_energy = 0.5 * np.sum(vel_frame**2) * self.dV
                potential_energy_grad = 0.5 * np.sum(grad_u_squared) * self.dV
                energy_data[global_t_idx] = kinetic_energy + \
                    potential_energy_grad

        mode_data = self.extract_principal_modes(traj_file)
        feature_vector = self.extract_trajectory_features(traj_data, vel_data)

        del traj_data, vel_data
        gc.collect()

        return {
            'lyapunov': lyapunov_data,
            'entropy': entropy_data,
            'localization': localization_data,
            'instability': instability_data,
            'energy': energy_data,
            'modes': mode_data,
            'features': feature_vector
        }

    def extract_principal_modes(self, traj_file):
        traj_data = np.load(traj_file, mmap_mode='r')
        T, nx, ny = traj_data.shape

        sample_indices = np.linspace(0, T - 1, min(T, 20), dtype=int)
        sampled_frames = traj_data[sample_indices].reshape(
            len(sample_indices), -1)

        pca = PCA(n_components=min(5, len(sample_indices) - 1))
        pca.fit(sampled_frames)

        mode_energy = pca.explained_variance_ratio_

        del traj_data, sampled_frames
        gc.collect()

        return mode_energy

    def extract_trajectory_features(self, traj_data, vel_data):
        T, nx, ny = traj_data.shape

        sample_points = [0, T // 4, T // 2, 3 * T // 4, T - 1]

        selected_frames = []
        for t in sample_points:
            frame = traj_data[t].flatten()
            selected_frames.append(frame)

        pca = PCA(n_components=min(len(selected_frames) - 1, 4))
        features = pca.fit_transform(np.array(selected_frames))

        energy_stats = self.compute_field_statistics(traj_data)
        velocity_stats = self.compute_field_statistics(vel_data)

        mean_frame = np.mean(selected_frames, axis=0)
        std_frame = np.std(selected_frames, axis=0)

        downsampled_mean = mean_frame.reshape(nx, ny)[::10, ::10].flatten()
        downsampled_std = std_frame.reshape(nx, ny)[::10, ::10].flatten()

        return np.concatenate([
            features.flatten(),
            energy_stats,
            velocity_stats,
            downsampled_mean[:10],
            downsampled_std[:10]
        ])

    def compute_field_statistics(self, field_data):
        max_val = np.max(field_data)
        min_val = np.min(field_data)
        mean_val = np.mean(field_data)
        std_val = np.std(field_data)
        abs_max = np.max(np.abs(field_data))
        skew = np.mean(((field_data - mean_val) / std_val)
                       ** 3) if std_val > 0 else 0
        kurtosis = np.mean(((field_data - mean_val) / std_val)
                           ** 4) if std_val > 0 else 0

        return np.array([max_val, min_val, mean_val,
                        std_val, abs_max, skew, kurtosis])

    def save_to_hdf5(self, run_idx, system_type, m_type, m_params,
                     u0, v0, m, traj_file, vel_file, metrics, elapsed_time):
        h5_file = self.h5_dir / f"run_{self.run_id}_{run_idx:04d}.h5"
        with h5py.File(h5_file, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['problem_type'] = system_type
            meta.attrs['boundary_condition'] = 'noflux'
            meta.attrs['run_id'] = self.run_id
            meta.attrs['run_index'] = run_idx
            meta.attrs['timestamp'] = str(time.strftime("%Y-%m-%d %H:%M:%S"))
            meta.attrs['elapsed_time'] = elapsed_time
            meta.attrs['phenomenon'] = self.args.phenomenon
            meta.attrs['traj_file'] = str(traj_file)
            meta.attrs['vel_file'] = str(vel_file)

            for key, value in self.reference_ic_params.items():
                meta.attrs[f'phenomenon_{key}'] = str(value)

            grid = f.create_group('grid')
            grid.attrs['nx'] = self.args.nx
            grid.attrs['ny'] = self.args.ny
            grid.attrs['Lx'] = self.args.Lx
            grid.attrs['Ly'] = self.args.Ly

            time_grp = f.create_group('time')
            time_grp.attrs['T'] = self.args.T
            time_grp.attrs['nt'] = self.args.nt
            time_grp.attrs['num_snapshots'] = self.args.snapshots

            ic_grp = f.create_group('initial_condition')
            ic_grp.create_dataset('u0', data=u0)
            ic_grp.create_dataset('v0', data=v0)

            m_grp = f.create_group('focusing')
            m_grp.attrs['type'] = m_type
            for param_name, param_value in m_params.items():
                m_grp.attrs[param_name] = param_value
            m_grp.create_dataset('m', data=m)

            metrics_grp = f.create_group('metrics')
            for key, value in metrics.items():
                if key != 'features':
                    metrics_grp.create_dataset(key, data=value)

            metrics_grp.create_dataset('features', data=metrics['features'])

        return h5_file

    def analyze_metrics(self, metrics, run_idx, system_type, m_type, m_params):
        lyapunov = metrics['lyapunov']
        entropy = metrics['entropy']
        localization = metrics['localization']
        instability = metrics['instability']
        energy = metrics['energy']
        modes = metrics['modes']

        time_points = np.linspace(0, self.args.T, len(entropy))
        lyapunov_time = np.linspace(0, self.args.T, len(lyapunov))

        metrics_file = self.analysis_dir / \
            f"metrics_{self.run_id}_{run_idx:04d}.npz"
        np.savez(metrics_file,
                 time_points=time_points,
                 lyapunov_time=lyapunov_time,
                 lyapunov=lyapunov,
                 entropy=entropy,
                 localization=localization,
                 instability=instability,
                 energy=energy,
                 modes=modes)

        scalar_metrics = {
            'max_lyapunov': np.max(lyapunov),
            'final_lyapunov': lyapunov[-1],
            'mean_entropy': np.mean(entropy),
            'max_entropy': np.max(entropy),
            'mean_localization': np.mean(localization),
            'max_localization': np.max(localization),
            'total_instability': np.sum(instability) * (self.args.T / len(instability)),
            'entropy_slope': np.polyfit(time_points, entropy, 1)[0],
            'localization_stability': np.std(localization[len(localization) // 2:]),
            'lyapunov_growth': np.mean(np.diff(lyapunov)),
            'energy_final': energy[-1],
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'energy_max': np.max(energy),
            'energy_growth': (energy[-1] - energy[0]) / self.args.T,
            'mode1_energy': modes[0] if len(modes) > 0 else 0,
            'mode2_energy': modes[1] if len(modes) > 1 else 0,
            'mode_ratio': modes[0] / modes[1] if len(modes) > 1 and modes[1] > 0 else 0
        }

        oscillation_data = self.analyze_oscillations(
            metrics, run_idx, system_type, m_type, m_params)
        if oscillation_data:
            scalar_metrics.update(oscillation_data)

        return scalar_metrics

    def analyze_oscillations(self, metrics, run_idx,
                             system_type, m_type, m_params):
        energy = metrics['energy']

        try:
            peak_indices, _ = find_peaks(
                energy, height=np.mean(energy), distance=5)
            valley_indices, _ = find_peaks(-energy,
                                           height=-np.mean(energy), distance=5)

            if len(peak_indices) > 1:
                peak_values = energy[peak_indices]
                peak_times = peak_indices * self.args.T / len(energy)
                peak_spacing = np.diff(peak_times)

                dominant_frequency = 1.0 / \
                    np.mean(peak_spacing) if np.mean(peak_spacing) > 0 else 0

                return {
                    'num_peaks': len(peak_indices),
                    'peak_value_std': np.std(peak_values),
                    'peak_spacing_std': np.std(peak_spacing) if len(peak_spacing) > 0 else 0,
                    'peak_value_mean': np.mean(peak_values),
                    'dominant_frequency': dominant_frequency,
                    'oscillation_regularity': 1.0 / (1.0 + np.std(peak_spacing) / np.mean(peak_spacing))
                    if len(peak_spacing) > 0 and np.mean(peak_spacing) > 0 else 0
                }

        except Exception as e:
            print(f"Error in oscillation analysis for run {run_idx}: {e}")

        return None

    def execute_sweep(self):
        u0_file, v0_file = self.generate_reference_ic()
        u0, v0 = self.reference_ic

        results_data = []

        for i, params in enumerate(self.parameter_grid):
            system_type = params['system_type']
            m_type = params['m_type']
            m_params = params['m_params']

            try:
                print(
                    f"Run {i+1}/{len(self.parameter_grid)}: System={system_type}, m_type={m_type}")

                m = self.generate_spatial_amplification(m_type, **m_params)
                m_file = self.focusing_dir / f"m_{self.run_id}_{i:04d}.npy"
                np.save(m_file, m)

                traj_file, vel_file, walltime = self.run_simulation(
                    system_type, m_file, u0_file, v0_file, i)

                metrics = self.compute_metrics_from_trajectory(
                    traj_file, vel_file, m)

                h5_file = self.save_to_hdf5(i, system_type, m_type, m_params,
                                            u0, v0, m, traj_file, vel_file, metrics, walltime)

                scalar_metrics = self.analyze_metrics(
                    metrics, i, system_type, m_type, m_params)

                self.trajectory_features[i] = {
                    'run_idx': i,
                    'system_type': system_type,
                    'm_type': m_type,
                    'm_params': m_params,
                    'features': metrics['features'],
                    'metrics': scalar_metrics,
                    'h5_file': str(h5_file),
                    'traj_file': str(traj_file),
                    'vel_file': str(vel_file)
                }

                result = {
                    'run_idx': i,
                    'system_type': system_type,
                    'm_type': m_type,
                    'm_params': m_params,
                    'walltime': walltime,
                    'h5_file': str(h5_file),
                    'metrics': scalar_metrics
                }

                results_data.append(result)
                print(f"  Completed in {walltime:.2f}s")

            except Exception as e:
                print(f"Error in run {i}: {e}")
                import traceback
                traceback.print_exc()

        self.create_summary_results(results_data)
        self.create_ensemble_visualizations()
        self.perform_clustering_analysis()
        self.detect_outliers(results_data)

        return results_data

    def create_summary_results(self, results_data):
        rows = []
        for result in results_data:
            if 'metrics' not in result:
                continue

            row = {
                'run_idx': result['run_idx'],
                'system_type': result['system_type'],
                'm_type': result['m_type'],
                'h5_file': result['h5_file'],
                'walltime': result['walltime']
            }

            for param_name, param_value in result['m_params'].items():
                row[f'm_{param_name}'] = param_value

            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(
            self.output_dir /
            f"sweep_results_{self.run_id}.csv",
            index=False)

        self.create_violin_plots(df)

    def create_violin_plots(self, results_df):
        key_metrics = ['max_lyapunov', 'mean_entropy', 'energy_mean', 'energy_growth',
                       'localization_stability', 'mode1_energy', 'mode_ratio']

        plt.figure(figsize=(20, 15))

        for i, metric in enumerate(key_metrics):
            plt.subplot(3, 3, i + 1)

            if metric in results_df.columns:
                try:
                    ax = sns.violinplot(x='system_type', y=metric, hue='m_type',
                                        data=results_df, split=False, inner='box',
                                        palette='Set3', cut=0)
                    plt.title(f"Distribution of {metric}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    if i == 0:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        plt.legend([], [], frameon=False)
                except Exception as e:
                    print(f"Error creating violin plot for {metric}: {e}")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"metric_distributions_{self.run_id}.png",
            dpi=300,
            bbox_inches='tight')
        plt.close()

        self.create_correlation_matrix(results_df)

    def create_correlation_matrix(self, results_df):
        metric_columns = [col for col in results_df.columns if col not in
                          ['run_idx', 'system_type', 'm_type', 'h5_file', 'walltime'] and
                          not col.startswith('m_')]

        if len(metric_columns) < 2:
            return

        corr_matrix = results_df[metric_columns].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
                    center=0, square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('Correlation Matrix of Metrics')
        plt.tight_layout()
        plt.savefig(
            self.plots_dir /
            f"correlation_matrix_{self.run_id}.png",
            dpi=300)
        plt.close()

    def create_ensemble_visualizations(self):
        system_m_combinations = {}

        for run_idx, data in self.trajectory_features.items():
            key = (data['system_type'], data['m_type'])
            if key not in system_m_combinations:
                system_m_combinations[key] = []
            system_m_combinations[key].append(run_idx)

        for (system_type, m_type), run_indices in system_m_combinations.items():
            if len(run_indices) < 2:
                continue

            self.plot_ensemble_metrics(system_type, m_type, run_indices)

    def plot_ensemble_metrics(self, system_type, m_type, run_indices):
        metrics_list = ['energy', 'entropy', 'localization', 'lyapunov']
        metric_time_data = {metric: [] for metric in metrics_list}
        time_points = {metric: None for metric in metrics_list}

        for run_idx in run_indices:
            metrics_file = self.analysis_dir / \
                f"metrics_{self.run_id}_{run_idx:04d}.npz"
            try:
                with np.load(metrics_file) as data:
                    for metric in metrics_list:
                        if metric == 'lyapunov':
                            metric_time_data[metric].append(data[metric])
                            if time_points[metric] is None:
                                time_points[metric] = data['lyapunov_time']
                        else:
                            metric_time_data[metric].append(data[metric])
                            if time_points[metric] is None:
                                time_points[metric] = data['time_points']
            except Exception as e:
                print(f"Error loading metrics for run {run_idx}: {e}")

        plt.figure(figsize=(20, 15))

        for i, metric in enumerate(metrics_list):
            plt.subplot(2, 2, i + 1)

            metric_data = np.array(metric_time_data[metric])
            if len(metric_data) == 0:
                continue

            x = time_points[metric]

            mean_curve = np.mean(metric_data, axis=0)
            std_curve = np.std(metric_data, axis=0)

            plt.plot(x, mean_curve, 'b-', lw=2, label='Mean')
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             color='b', alpha=0.2, label='±1σ')

            outlier_threshold = 2.0
            for j, trajectory in enumerate(metric_data):
                z_scores = np.abs(
                    (trajectory - mean_curve) / (std_curve + 1e-10))
                if np.any(z_scores > outlier_threshold):
                    run_idx = run_indices[j]
                    params_str = self.get_params_str(run_idx)
                    plt.plot(x, trajectory, 'r-', alpha=0.7, lw=1)
                    max_outlier_idx = np.argmax(z_scores)
                    plt.annotate(f"Run {run_idx}\n{params_str}",
                                 xy=(x[max_outlier_idx],
                                     trajectory[max_outlier_idx]),
                                 xytext=(0, 20), textcoords='offset points',
                                 arrowprops=dict(arrowstyle='->', color='red'),
                                 fontsize=8, color='red')

            plt.title(f"{system_type}, {m_type}: {metric.capitalize()}")
            plt.xlabel("Time")
            plt.ylabel(metric.capitalize())
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.ensemble_dir / f"ensemble_{system_type}_{m_type}_{self.run_id}.png",
                    dpi=300)
        plt.close()

    def get_params_str(self, run_idx):
        data = self.trajectory_features[run_idx]
        m_type = data['m_type']
        m_params = data['m_params']

        if m_type == 'constant':
            return f"m={m_params['value']}"
        elif m_type in ['grf', 'wavelet_grf']:
            return f"μ={m_params['mean']}, σ={m_params['std']}"
        else:
            return f"factor={m_params.get('factor', '')}"

    def perform_clustering_analysis(self):
        if len(self.trajectory_features) < 5:
            return

        features_list = []
        metadata = []

        for run_idx, data in self.trajectory_features.items():
            features_list.append(data['features'])
            metadata.append({
                'run_idx': run_idx,
                'system_type': data['system_type'],
                'm_type': data['m_type'],
                'm_params': data['m_params']
            })

        X = np.vstack(features_list)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(5, len(X) // 3)
        if n_clusters < 2:
            return

        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        perplexity = min(30, max(5, len(X) // 5))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca')
        X_tsne = tsne.fit_transform(X_scaled)

        self.plot_pca_clustering(X_pca, metadata, labels, n_clusters, pca)
        self.plot_tsne_clustering(X_tsne, metadata, labels, n_clusters)
        self.plot_tsne_by_parameters(X_tsne, metadata)

        self.analyze_clusters(metadata, labels, n_clusters)

    def plot_pca_clustering(self, X_pca, metadata, labels, n_clusters, pca):
        plt.figure(figsize=(12, 10))

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=f'Cluster {cluster_id}', s=80, alpha=0.7)

            for i, point in enumerate(X_pca[mask]):
                idx = np.where(mask)[0][i]
                plt.annotate(str(metadata[idx]['run_idx']),
                             xy=(point[0], point[1]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8)

        plt.title("PCA Clustering of Trajectory Features")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f}%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.cluster_dir /
            f"pca_clusters_{self.run_id}.png",
            dpi=300)
        plt.close()

    def plot_tsne_clustering(self, X_tsne, metadata, labels, n_clusters):
        plt.figure(figsize=(12, 10))

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        label=f'Cluster {cluster_id}', s=80, alpha=0.7)

            for i, point in enumerate(X_tsne[mask]):
                idx = np.where(mask)[0][i]
                plt.annotate(str(metadata[idx]['run_idx']),
                             xy=(point[0], point[1]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8)

        plt.title("t-SNE Clustering of Trajectory Features")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.cluster_dir /
            f"tsne_clusters_{self.run_id}.png",
            dpi=300)
        plt.close()

    def plot_tsne_by_parameters(self, X_tsne, metadata):
        plt.figure(figsize=(20, 16))

        plt.subplot(2, 2, 1)
        system_types = sorted(set(data['system_type'] for data in metadata))
        for i, system_type in enumerate(system_types):
            mask = [data['system_type'] == system_type for data in metadata]
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        label=system_type, s=80, alpha=0.7)
        plt.title("t-SNE by System Type")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        m_types = sorted(set(data['m_type'] for data in metadata))
        for i, m_type in enumerate(m_types):
            mask = [data['m_type'] == m_type for data in metadata]
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        label=m_type, s=80, alpha=0.7)
        plt.title("t-SNE by m_type")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        const_indices = [i for i, data in enumerate(
            metadata) if data['m_type'] == 'constant']
        if const_indices:
            const_values = [metadata[i]['m_params']['value']
                            for i in const_indices]
            value_colors = plt.cm.viridis(
                np.linspace(0, 1, len(set(const_values))))
            value_color_map = {
                val: color for val,
                color in zip(
                    sorted(
                        set(const_values)),
                    value_colors)}

            for i in const_indices:
                value = metadata[i]['m_params']['value']
                plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                            color=value_color_map[value], s=80, alpha=0.7)
                plt.annotate(f"{value}", xy=(X_tsne[i, 0], X_tsne[i, 1]), xytext=(5, 5),
                             textcoords='offset points', fontsize=8)
        plt.title("t-SNE for Constant m (values shown)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        grf_indices = [i for i, data in enumerate(metadata)
                       if data['m_type'] in ['grf', 'wavelet_grf']]

        if grf_indices:
            grf_std_values = [metadata[i]['m_params']['std']
                              for i in grf_indices]
            min_std = min(grf_std_values)
            max_std = max(grf_std_values)
            norm_values = [(v - min_std) / (max_std - min_std) if max_std > min_std else 0.5
                           for v in grf_std_values]

            for i, idx in enumerate(grf_indices):
                plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                            c=[plt.cm.plasma(norm_values[i])], s=80, alpha=0.7)

                grf_type = metadata[idx]['m_type']
                marker = 'o' if grf_type == 'grf' else 's'

                plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                            facecolors='none', edgecolors='k', marker=marker, s=100, alpha=0.5)

                plt.annotate(f"{grf_std_values[i]}", xy=(X_tsne[idx, 0], X_tsne[idx, 1]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.title("t-SNE for GRF (std values shown, squares=wavelet)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.cluster_dir /
            f"tsne_parameters_{self.run_id}.png",
            dpi=300)
        plt.close()

    def analyze_clusters(self, metadata, labels, n_clusters):
        cluster_stats = []

        for cluster_id in range(n_clusters):
            cluster_members = [
                metadata[i] for i in range(
                    len(metadata)) if labels[i] == cluster_id]

            system_types = [m['system_type'] for m in cluster_members]
            m_types = [m['m_type'] for m in cluster_members]

            system_counts = {st: system_types.count(
                st) for st in set(system_types)}
            m_type_counts = {mt: m_types.count(mt) for mt in set(m_types)}

            dominant_system = max(system_counts.items(), key=lambda x: x[1])[0]
            dominant_m_type = max(m_type_counts.items(), key=lambda x: x[1])[0]

            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_members),
                'dominant_system': dominant_system,
                'dominant_m_type': dominant_m_type,
                'system_distribution': system_counts,
                'm_type_distribution': m_type_counts,
                'members': [m['run_idx'] for m in cluster_members]
            })

        with open(self.cluster_dir / f"cluster_stats_{self.run_id}.txt", 'w') as f:
            for cluster in cluster_stats:
                f.write(f"Cluster {cluster['cluster_id']}:\n")
                f.write(f"  Size: {cluster['size']}\n")
                f.write(f"  Dominant system: {cluster['dominant_system']}\n")
                f.write(f"  Dominant m_type: {cluster['dominant_m_type']}\n")
                f.write(
                    f"  System distribution: {cluster['system_distribution']}\n")
                f.write(
                    f"  m_type distribution: {cluster['m_type_distribution']}\n")
                f.write(
                    f"  Members: {', '.join(map(str, cluster['members']))}\n\n")

    def detect_outliers(self, results_data):
        if len(results_data) < 5:
            return

        metrics_to_check = ['energy_mean', 'energy_growth', 'max_lyapunov', 'mean_entropy',
                            'max_localization', 'total_instability']

        outliers = {}

        for metric in metrics_to_check:
            values = [r['metrics'].get(metric, 0)
                      for r in results_data if 'metrics' in r]
            if not values:
                continue

            values = np.array(values)
            median = np.median(values)
            mad = np.median(np.abs(values - median))

            threshold = 3.5
            if mad == 0:
                continue

            z_scores = 0.6745 * (values - median) / mad
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

            for idx in outlier_indices:
                run_idx = results_data[idx]['run_idx']
                if run_idx not in outliers:
                    outliers[run_idx] = []

                outlier_value = values[idx]
                outliers[run_idx].append(
                    (metric, outlier_value, z_scores[idx]))

        if not outliers:
            return

        self.outliers = outliers
        self.visualize_outliers(results_data)

    def visualize_outliers(self, results_data):
        if not self.outliers:
            return

        for run_idx, outlier_metrics in self.outliers.items():
            if not outlier_metrics:
                continue

            try:
                run_data = next(
                    (r for r in results_data if r['run_idx'] == run_idx), None)
                if not run_data:
                    continue

                system_type = run_data['system_type']
                m_type = run_data['m_type']
                m_params = run_data['m_params']
                metrics = run_data['metrics']

                h5_file = run_data['h5_file']
                traj_file = None
                vel_file = None

                with h5py.File(h5_file, 'r') as f:
                    if 'metadata' in f and 'traj_file' in f['metadata'].attrs:
                        traj_file = f['metadata'].attrs['traj_file']
                    if 'metadata' in f and 'vel_file' in f['metadata'].attrs:
                        vel_file = f['metadata'].attrs['vel_file']

                if not traj_file:
                    continue

                traj_data = np.load(traj_file, mmap_mode='r')
                if 'energy' in metrics:
                    energy = metrics['energy']
                else:
                    metrics_file = self.analysis_dir / \
                        f"metrics_{self.run_id}_{run_idx:04d}.npz"
                    with np.load(metrics_file) as data:
                        energy = data['energy']

                plt.figure(figsize=(15, 10))

                plt.subplot(2, 2, 1)
                plt.imshow(traj_data[0], cmap='viridis')
                plt.title("Initial State")
                plt.colorbar()

                plt.subplot(2, 2, 2)
                plt.imshow(traj_data[-1], cmap='viridis')
                plt.title("Final State")
                plt.colorbar()

                plt.subplot(2, 2, 3)
                time_points = np.linspace(0, self.args.T, len(energy))
                plt.plot(time_points, energy)
                plt.grid(True)
                plt.title("Energy Evolution")
                plt.xlabel("Time")
                plt.ylabel("Energy")

                plt.subplot(2, 2, 4)
                plt.axis('off')
                outlier_text = f"System: {system_type}\nM Type: {m_type}\n\n"

                for param_name, param_value in m_params.items():
                    outlier_text += f"{param_name}: {param_value}\n"

                outlier_text += "\nOutlier Metrics:\n"
                for metric, value, z_score in outlier_metrics:
                    outlier_text += f"{metric}: {value:.4f} (z={z_score:.2f})\n"

                plt.text(
                    0.1,
                    0.9,
                    outlier_text,
                    ha='left',
                    va='top',
                    transform=plt.gca().transAxes)

                plt.tight_layout()
                plt.savefig(
                    self.outlier_dir /
                    f"outlier_{run_idx}_{self.run_id}.png",
                    dpi=300)
                plt.close()

                del traj_data
                gc.collect()

            except Exception as e:
                print(f"Error visualizing outlier for run {run_idx}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for nonlinear wave equations")

    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
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
        help="Simulation time")
    parser.add_argument(
        "--nt",
        type=int,
        default=1000,
        help="Number of time steps")
    parser.add_argument(
        "--snapshots",
        type=int,
        default=100,
        help="Number of snapshots to save")

    parser.add_argument("--phenomenon", type=str, default="kink_field",
                        choices=["kink_solution", "kink_field", "kink_array_field",
                                 "multi_breather_field", "spiral_wave_field", "multi_spiral_state",
                                 "ring_soliton", "colliding_rings", "multi_ring_state",
                                 "skyrmion_solution", "skyrmion_lattice", "skyrmion_like_field",
                                 "q_ball_solution", "multi_q_ball", "breather_solution",
                                 "breather_field", "elliptical_soliton"])

    parser.add_argument("--velocity-type", type=str, default="zero",
                        choices=["zero", "fitting", "random"])

    parser.add_argument("--klein-gordon-exe", type=str, required=True)
    parser.add_argument("--sine-gordon-exe", type=str, required=True)
    parser.add_argument("--double-sine-gordon-exe", type=str, required=True)
    parser.add_argument(
        "--hyperbolic-sine-gordon-exe",
        type=str,
        required=True)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="parameter_sweep_results")

    return parser.parse_args()


def main():
    args = parse_args()
    sweep = ParameterSweep(args)
    sweep.execute_sweep()


if __name__ == "__main__":
    main()
