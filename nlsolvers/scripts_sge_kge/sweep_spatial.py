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
import pandas as pd
from scipy.interpolate import griddata
import seaborn as sns

from spatial_amplification import (
    create_constant_m,
    create_periodic_boxes,
    create_periodic_gaussians,
    create_grf,
    create_wavelet_modulated_grf,
    scale_m_to_range
)
from real_sampler import RealWaveSampler
from valid_spaces import get_parameter_spaces


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
        self.m_types = ['constant', 'periodic_boxes', 'periodic_gaussians', 'grf', 'wavelet_grf']
        
        self.setup_directories()
        self.configure_grid()
        
        self.sampler = RealWaveSampler(args.nx, args.ny, args.Lx)
        
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        
        self.reference_ic = None
        self.reference_ic_params = None
        self.parameter_grid = self.generate_parameter_grid()
        self.results = {}
        self.trajectory_features = {}

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
        
        self.phase_space_dir = self.output_dir / "phase_space"
        self.phase_space_dir.mkdir(exist_ok=True)
        
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.heatmap_dir.mkdir(exist_ok=True)

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
            (0.0, 1, 2.), (0.0, 5, 2.), (0.0, 20, 2.),
            (0.0, 1, .4), (0.0, 5, .4), (0.0, 20, .4), 
        ]
        
        m_periodic_params = [     
            (2.0, 2, 0.05), (2.0, 2, 0.1), (2.0, 2, 0.2),
            (2.0, 4, 0.05), (2.0, 4, 0.1), (2.0, 4, 0.2),
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
        plt.savefig(self.plots_dir / f"reference_ic_{self.run_id}.png", dpi=300)
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
    
    def compute_metrics_from_trajectory(self, traj_file, vel_file, m, chunk_size=10):
        traj_data = np.load(traj_file, mmap_mode='r')
        vel_data = np.load(vel_file, mmap_mode='r')
        
        T, nx, ny = traj_data.shape
        
        lyapunov_data = np.zeros(T-1)
        entropy_data = np.zeros(T)
        localization_data = np.zeros(T)
        instability_data = np.zeros(T)
        
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
            
            if chunk_idx == 0:
                first_frame = chunk_traj[0].copy()
                
                dx_0 = (first_frame[1:, :] - first_frame[:-1, :])[:, 1:]
                dy_0 = (first_frame[:, 1:] - first_frame[:, :-1])[1:, :]
                delta_0 = np.sqrt(dx_0**2 + dy_0**2)
                delta_avg_0 = np.mean(delta_0)
            
            for t_idx, frame in enumerate(chunk_traj):
                global_t_idx = start_idx + t_idx
                
                if global_t_idx > 0:
                    dx = (frame[1:, :] - frame[:-1, :])[:, 1:]
                    dy = (frame[:, 1:] - frame[:, :-1])[1:, :]
                    delta = np.sqrt(dx**2 + dy**2)
                    delta_avg = np.mean(delta)
                    
                    lyapunov_data[global_t_idx-1] = np.log(delta_avg / delta_avg_0)
                
                u_fft = np.fft.fft2(frame)
                u_fft = np.fft.fftshift(u_fft)
                power_spectrum = np.abs(u_fft)**2 * self.dx * self.dy / (2 * np.pi)**2
                total_energy = np.sum(power_spectrum)
                
                normalized_spectrum = power_spectrum / total_energy
                valid_indices = normalized_spectrum > 1e-10
                entropy_data[global_t_idx] = -np.sum(normalized_spectrum[valid_indices] * 
                                        np.log(normalized_spectrum[valid_indices]))
                
                u_squared = frame**2
                numerator = np.sum(u_squared**2)
                denominator = np.sum(u_squared)**2
                localization_data[global_t_idx] = numerator / denominator * (nx * ny)
                
                grad_u_x = np.zeros_like(frame)
                grad_u_y = np.zeros_like(frame)
                
                grad_u_x[1:-1, :] = (frame[2:, :] - frame[:-2, :]) / (2 * self.dx)
                grad_u_y[:, 1:-1] = (frame[:, 2:] - frame[:, :-2]) / (2 * self.dy)
                
                grad_u_squared = grad_u_x**2 + grad_u_y**2
                integrand = grad_u_squared * grad_m_magnitude
                instability_data[global_t_idx] = np.sum(integrand) * self.dx * self.dy
        
        feature_vector = self.extract_trajectory_features(traj_data, vel_data)
        
        del traj_data
        
        return {
            'lyapunov': lyapunov_data,
            'entropy': entropy_data,
            'localization': localization_data,
            'instability': instability_data,
            'features': feature_vector
        }
    
    def extract_trajectory_features(self, traj_data, vel_data):
        T, nx, ny = traj_data.shape
        
        sample_points = [0, T//4, T//2, 3*T//4, T-1]
        
        selected_frames = []
        for t in sample_points:
            frame = traj_data[t].flatten()
            selected_frames.append(frame)
        
        pca = PCA(n_components=min(len(selected_frames)-1, 4))
        features = pca.fit_transform(np.array(selected_frames))
        
        additional_features = [
            np.max(traj_data),
            np.min(traj_data),
            np.mean(traj_data),
            np.std(traj_data),
            np.max(vel_data),
            np.mean(np.abs(vel_data))
        ]
        
        mean_frame = np.mean(selected_frames, axis=0)
        std_frame = np.std(selected_frames, axis=0)
        
        downsampled_mean = mean_frame.reshape(nx, ny)[::10, ::10].flatten()
        downsampled_std = std_frame.reshape(nx, ny)[::10, ::10].flatten()
        
        return np.concatenate([features.flatten(), additional_features, 
                              downsampled_mean[:10], downsampled_std[:10]])
    
    def save_to_hdf5(self, run_idx, system_type, m_type, m_params, u0, v0, m, traj_file, vel_file, metrics, elapsed_time):
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
        
        time_points = np.linspace(0, self.args.T, len(entropy))
        lyapunov_time = np.linspace(0, self.args.T, len(lyapunov))
        
        metrics_file = self.analysis_dir / f"metrics_{self.run_id}_{run_idx:04d}.npz"
        np.savez(metrics_file, 
                 time_points=time_points,
                 lyapunov_time=lyapunov_time,
                 lyapunov=lyapunov, 
                 entropy=entropy, 
                 localization=localization, 
                 instability=instability)
        
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        plt.plot(lyapunov_time, lyapunov)
        plt.grid(True)
        plt.title("Lyapunov Exponent Approximation")
        plt.xlabel("Time")
        plt.ylabel("log(delta(t)/delta(0))")
        
        plt.subplot(2, 2, 2)
        plt.plot(time_points, entropy)
        plt.grid(True)
        plt.title("Spectral Energy Entropy")
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        
        plt.subplot(2, 2, 3)
        plt.plot(time_points, localization)
        plt.grid(True)
        plt.title("Energy Localization")
        plt.xlabel("Time")
        plt.ylabel("Localization")
        
        plt.subplot(2, 2, 4)
        plt.plot(time_points, instability)
        plt.grid(True)
        plt.title("Instability Growth")
        plt.xlabel("Time")
        plt.ylabel("Instability Metric")
        
        plt.tight_layout()
        plt.suptitle(f"System: {system_type}, m_type: {m_type}")
        plt.savefig(self.plots_dir / f"metrics_{self.run_id}_{run_idx:04d}.png", dpi=300)
        plt.close()
        
        scalar_metrics = {
            'max_lyapunov': np.max(lyapunov),
            'final_lyapunov': lyapunov[-1],
            'mean_entropy': np.mean(entropy),
            'max_entropy': np.max(entropy),
            'mean_localization': np.mean(localization),
            'max_localization': np.max(localization),
            'total_instability': np.sum(instability) * (self.args.T/self.args.snapshots),
            'entropy_slope': np.polyfit(time_points, entropy, 1)[0],
            'localization_stability': np.std(localization[len(localization)//2:]),
            'lyapunov_growth': np.mean(np.diff(lyapunov))
        }
        
        bifurcation_data = self.analyze_bifurcation(run_idx, system_type, m_type, m_params)
        if bifurcation_data:
            scalar_metrics.update(bifurcation_data)
        
        return scalar_metrics
    
    def analyze_bifurcation(self, run_idx, system_type, m_type, m_params):
        traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"
        
        try:
            traj_data = np.load(traj_file, mmap_mode='r')
            T = traj_data.shape[0]
            
            center_x = traj_data.shape[1] // 2
            center_y = traj_data.shape[2] // 2
            
            center_values = traj_data[:, center_x, center_y]
            
            peaks, _ = self._find_peaks(center_values)
            
            if len(peaks) > 1:
                peak_times = np.array(peaks) * self.args.T / T
                peak_values = center_values[peaks]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(peak_times, peak_values, alpha=0.7)
                
                if m_type == 'constant':
                    plt.title(f"Bifurcation - {system_type} (m={m_params['value']})")
                else:
                    plt.title(f"Bifurcation - {system_type} ({m_type})")
                    
                plt.xlabel("Time")
                plt.ylabel("Peak Value")
                plt.grid(True)
                plt.savefig(self.plots_dir / f"bifurcation_{self.run_id}_{run_idx:04d}.png", dpi=300)
                plt.close()
                
                return {
                    'num_peaks': len(peaks),
                    'peak_value_std': np.std(peak_values),
                    'peak_spacing_std': np.std(np.diff(peak_times)) if len(peaks) > 1 else 0,
                    'peak_value_mean': np.mean(peak_values)
                }
                
        except Exception as e:
            print(f"Error in bifurcation analysis for run {run_idx}: {e}")
            
        return None
    
    def _find_peaks(self, values, threshold=0.5):
        peaks = []
        valleys = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                if values[i] > threshold:
                    peaks.append(i)
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                if values[i] < -threshold:
                    valleys.append(i)
                    
        return peaks, valleys
    
    def execute_sweep(self):
        u0_file, v0_file = self.generate_reference_ic()
        u0, v0 = self.reference_ic
        
        results_data = []
        
        for i, params in enumerate(self.parameter_grid):
            system_type = params['system_type']
            m_type = params['m_type']
            m_params = params['m_params']
            
            try:
                print(f"Run {i+1}/{len(self.parameter_grid)}: System={system_type}, m_type={m_type}")
                
                m = self.generate_spatial_amplification(m_type, **m_params)
                m_file = self.focusing_dir / f"m_{self.run_id}_{i:04d}.npy"
                np.save(m_file, m)
                
                traj_file, vel_file, walltime = self.run_simulation(system_type, m_file, u0_file, v0_file, i)
                
                metrics = self.compute_metrics_from_trajectory(traj_file, vel_file, m)
                
                h5_file = self.save_to_hdf5(i, system_type, m_type, m_params, 
                                           u0, v0, m, traj_file, vel_file, metrics, walltime)
                
                scalar_metrics = self.analyze_metrics(metrics, i, system_type, m_type, m_params)
                
                self.trajectory_features[i] = {
                    'run_idx': i,
                    'system_type': system_type,
                    'm_type': m_type,
                    'features': metrics['features'],
                    'h5_file': str(h5_file)
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
        self.create_sensitivity_heatmaps(results_data)
        self.create_phase_space_analysis()
        return results_data
    
    def create_summary_results(self, results_data):
        import pandas as pd
        
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
        df.to_csv(self.output_dir / f"sweep_results_{self.run_id}.csv", index=False)
        
        self.create_summary_plots(df)
        
    def create_summary_plots(self, results_df):
        system_types = results_df['system_type'].unique()
        m_types = results_df['m_type'].unique()
        
        for metric in ['max_lyapunov', 'mean_entropy', 'max_localization', 'total_instability', 
                      'entropy_slope', 'localization_stability', 'lyapunov_growth']:
            plt.figure(figsize=(16, 12))
            
            for i, system_type in enumerate(system_types):
                plt.subplot(2, 2, i+1)
                
                for m_type in m_types:
                    subset = results_df[(results_df['system_type'] == system_type) & 
                                        (results_df['m_type'] == m_type)]
                    
                    if subset.empty:
                        continue
                        
                    if m_type == 'constant':
                        plt.scatter(subset['m_value'], subset[metric], label=m_type, s=50)
                        
                    elif m_type in ['grf', 'wavelet_grf']:
                        if m_type == 'grf':
                            plt.scatter(subset['m_mean'], subset[metric], label=f"{m_type} (mean)", 
                                      alpha=0.7, s=50)
                        else:
                            plt.scatter(subset['m_mean'], subset[metric], label=f"{m_type} (mean)", 
                                      marker='s', alpha=0.7, s=50)
                        
                    elif m_type in ['periodic_boxes', 'periodic_gaussians']:
                        if m_type == 'periodic_boxes':
                            plt.scatter(subset['m_factor'], subset[metric], label=f"{m_type} (factor)", 
                                      alpha=0.7, s=50)
                        else:
                            plt.scatter(subset['m_factor'], subset[metric], label=f"{m_type} (factor)", 
                                      marker='s', alpha=0.7, s=50)
                
                plt.title(f"{system_type}: {metric}")
                plt.xlabel("Parameter Value")
                plt.ylabel(metric)
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{metric}_summary_{self.run_id}.png", dpi=300)
            plt.close()
    
    def create_sensitivity_heatmaps(self, results_data):
        for system_type in self.system_types:
            for m_type in ['grf', 'wavelet_grf', 'periodic_boxes', 'periodic_gaussians']:
                type_results = [r for r in results_data if r['system_type'] == system_type and r['m_type'] == m_type]
                
                if not type_results:
                    continue
                
                if m_type in ['grf', 'wavelet_grf']:
                    self._create_grf_heatmaps(system_type, m_type, type_results)
                elif m_type in ['periodic_boxes', 'periodic_gaussians']:
                    self._create_periodic_heatmaps(system_type, m_type, type_results)
    
    def _create_grf_heatmaps(self, system_type, m_type, type_results):
        for metric in ['max_lyapunov', 'mean_entropy', 'max_localization', 'total_instability']:
            means = []
            stds = []
            values = []
            
            for result in type_results:
                if 'metrics' not in result or metric not in result['metrics']:
                    continue
                    
                means.append(result['m_params']['mean'])
                stds.append(result['m_params']['std'])
                values.append(result['metrics'][metric])
            
            if not values:
                continue
                
            plt.figure(figsize=(10, 8))
            
            if len(set(means)) > 1 and len(set(stds)) > 1:
                mean_unique = sorted(list(set(means)))
                std_unique = sorted(list(set(stds)))
                
                grid_x, grid_y = np.meshgrid(mean_unique, std_unique)
                
                grid_z = griddata((means, stds), values, (grid_x, grid_y), method='linear')
                
                plt.pcolormesh(grid_x, grid_y, grid_z, cmap='viridis', shading='auto')
                plt.colorbar(label=metric)
                
                plt.scatter(means, stds, c=values, cmap='viridis', edgecolor='k', s=80)
                
                plt.xlabel('Mean')
                plt.ylabel('Std')
                plt.title(f"{system_type} - {m_type}: {metric}")
                
                plt.savefig(self.heatmap_dir / f"heatmap_{system_type}_{m_type}_{metric}_{self.run_id}.png", dpi=300)
                plt.close()
    
    def _create_periodic_heatmaps(self, system_type, m_type, type_results):
        for metric in ['max_lyapunov', 'mean_entropy', 'max_localization', 'total_instability']:
            factors = []
            boxes = []
            values = []
            
            param_key = 'num_gaussians_per_dim' if m_type == 'periodic_gaussians' else 'num_boxes_per_dim'
            
            for result in type_results:
                if 'metrics' not in result or metric not in result['metrics']:
                    continue
                    
                factors.append(result['m_params']['factor'])
                boxes.append(result['m_params'][param_key])
                values.append(result['metrics'][metric])
            
            if not values:
                continue
                
            plt.figure(figsize=(10, 8))
            
            if len(set(factors)) > 1 and len(set(boxes)) > 1:
                factor_unique = sorted(list(set(factors)))
                boxes_unique = sorted(list(set(boxes)))
                
                grid_x, grid_y = np.meshgrid(factor_unique, boxes_unique)
                
                grid_z = griddata((factors, boxes), values, (grid_x, grid_y), method='linear')
                
                plt.pcolormesh(grid_x, grid_y, grid_z, cmap='viridis', shading='auto')
                plt.colorbar(label=metric)
                
                plt.scatter(factors, boxes, c=values, cmap='viridis', edgecolor='k', s=80)
                
                plt.xlabel('Factor')
                plt.ylabel('Boxes per dim')
                plt.title(f"{system_type} - {m_type}: {metric}")
                
                plt.savefig(self.heatmap_dir / f"heatmap_{system_type}_{m_type}_{metric}_{self.run_id}.png", dpi=300)
                plt.close()
    
    def create_phase_space_analysis(self):
        if not self.trajectory_features:
            return
            
        features_list = []
        system_types = []
        m_types = []
        run_indices = []
        
        for run_idx, data in self.trajectory_features.items():
            features_list.append(data['features'])
            system_types.append(data['system_type'])
            m_types.append(data['m_type'])
            run_indices.append(run_idx)
        
        features_array = np.vstack(features_list)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_array)
        
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        tsne_result = tsne.fit_transform(features_array)
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        for st in self.system_types:
            mask = [s == st for s in system_types]
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=st, alpha=0.7)
        
        plt.title("PCA of Trajectory Features (by System Type)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for mt in self.m_types:
            mask = [m == mt for m in m_types]
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=mt, alpha=0.7)
        
        plt.title("PCA of Trajectory Features (by m_type)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.phase_space_dir / f"pca_analysis_{self.run_id}.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        for st in self.system_types:
            mask = [s == st for s in system_types]
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=st, alpha=0.7)
        
        plt.title("t-SNE of Trajectory Features (by System Type)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for mt in self.m_types:
            mask = [m == mt for m in m_types]
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=mt, alpha=0.7)
        
        plt.title("t-SNE of Trajectory Features (by m_type)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.phase_space_dir / f"tsne_analysis_{self.run_id}.png", dpi=300)
        plt.close()
        
        features_df = pd.DataFrame({
            'run_idx': run_indices,
            'system_type': system_types,
            'm_type': m_types,
            'pca_1': pca_result[:, 0],
            'pca_2': pca_result[:, 1],
            'tsne_1': tsne_result[:, 0],
            'tsne_2': tsne_result[:, 1]
        })
        
        features_df.to_csv(self.phase_space_dir / f"phase_space_data_{self.run_id}.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter sweep for nonlinear wave equations")
    
    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y")
    parser.add_argument("--T", type=float, default=10.0, help="Simulation time")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=100, help="Number of snapshots to save")
    
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
    parser.add_argument("--hyperbolic-sine-gordon-exe", type=str, required=True)
    
    parser.add_argument("--output-dir", type=str, default="parameter_sweep_results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    sweep = ParameterSweep(args)
    sweep.execute_sweep()


if __name__ == "__main__":
    main()


