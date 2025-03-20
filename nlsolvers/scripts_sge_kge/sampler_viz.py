import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os
import itertools
import uuid
import time
import warnings
from real_sampler import RealWaveEquationSampler

class RealWaveSpaceMapper:
    def __init__(self, sampler, output_dir="real_wave_mapping_results"):
        self.sampler = sampler
        self.output_dir = output_dir
        self.samples = []
        self.metadata = []
        self.distance_matrix = None
        self.embeddings = {}

        self.X, self.Y = sampler.X, sampler.Y
        self.dx, self.dy = sampler.dx, sampler.dy
        self.cell_area = sampler.dx * sampler.dy
        
        os.makedirs(output_dir, exist_ok=True)
        
    def _prepare_parameter_space(self):
        parameter_spaces = {}
        
        parameter_spaces["kink"] = {
            "system_type": ["sine_gordon", "phi4", "double_sine_gordon"],
            "amplitude": [1.0],
            "width": [0.5, 1.0, 1.5],
            "position": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
            "orientation": [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            "velocity": [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2), (0.2, 0.2)],
            "kink_type": ["standard", "anti", "double"],
            "velocity_type": ["fitting", "zero"]
        }
        
        parameter_spaces["breather"] = {
            "system_type": ["sine_gordon", "phi4", "double_sine_gordon"],
            "amplitude": [0.3, 0.5, 0.8],
            "frequency": [0.7, 0.8, 0.9],
            "width": [0.5, 1.0, 1.5],
            "position": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
            "phase": [0, np.pi/2, np.pi, 3*np.pi/2],
            "orientation": [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            "breather_type": ["standard", "radial"],
            "time_param": [0, 0.5, 1.0],
            "velocity_type": ["fitting", "zero", "grf"]
        }
        
        parameter_spaces["oscillon"] = {
            "system_type": ["phi4", "sine_gordon"],
            "amplitude": [0.3, 0.5, 0.8],
            "frequency": [0.7, 0.8, 0.9],
            "width": [0.5, 1.0, 1.5],
            "position": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
            "phase": [0, np.pi/2, np.pi, 3*np.pi/2],
            "profile": ["gaussian", "sech", "polynomial"],
            "time_param": [0, 0.5, 1.0]
        }
        
        parameter_spaces["multi_oscillon"] = {
            "system_type": ["phi4", "sine_gordon"],
            "n_oscillons": [3, 5, 8],
            "amplitude_range": [(0.3, 0.7), (0.5, 1.0)],
            "width_range": [(0.5, 1.0), (1.0, 2.0)],
            "frequency_range": [(0.7, 0.9), (0.8, 0.95)],
            "position_variance": [0.5, 1.0],
            "arrangement": ["random", "circular", "lattice", "linear"],
            "interaction_strength": [0.5, 0.7, 0.9],
            "time_param": [0, 0.5, 1.0]
        }
        
        parameter_spaces["ring"] = {
            "system_type": ["sine_gordon", "phi4"],
            "amplitude": [0.8, 1.0, 1.2],
            "radius": [1.0, 1.5, 2.0],
            "width": [0.3, 0.5, 0.8],
            "position": [(0, 0), (-0.5, -0.5), (0.5, 0.5)],
            "velocity": [0.0, 0.1, 0.2],
            "ring_type": ["expanding", "kink_antikink"],
            "modulation_strength": [0.0, 0.2, 0.4],
            "modulation_mode": [0, 2, 4]
        }
        
        parameter_spaces["multi_ring"] = {
            "system_type": ["sine_gordon", "phi4"],
            "n_rings": [2, 3, 4],
            "radius_range": [(1.0, 2.0), (1.5, 3.0)],
            "width_range": [(0.3, 0.6), (0.5, 0.8)],
            "position_variance": [0.3, 0.5],
            "arrangement": ["concentric", "random", "circular"],
            "interaction_strength": [0.5, 0.7, 0.9],
            "modulation_strength": [0.0, 0.2, 0.4],
            "modulation_mode_range": [(1, 3), (2, 5)]
        }
        
        parameter_spaces["skyrmion"] = {
            "system_type": ["sine_gordon", "phi4"],
            "amplitude": [0.8, 1.0, 1.2],
            "radius": [0.8, 1.0, 1.5],
            "position": [(0, 0), (-0.5, -0.5), (0.5, 0.5)],
            "charge": [1, -1],
            "profile": ["standard", "compact", "exponential"]
        }
        
        parameter_spaces["skyrmion_lattice"] = {
            "system_type": ["sine_gordon", "phi4"],
            "n_skyrmions": [4, 7, 12],
            "radius_range": [(0.5, 1.0), (0.8, 1.5)],
            "amplitude": [0.8, 1.0, 1.2],
            "arrangement": ["triangular", "square", "random"],
            "separation": [2.0, 2.5, 3.0],
            "charge_distribution": ["alternating", "random", "same"]
        }
        
        parameter_spaces["spiral_wave"] = {
            "num_arms": [1, 2, 3, 4],
            "decay_rate": [0.3, 0.5, 0.7],
            "amplitude": [0.8, 1.0, 1.2],
            "phase": [0, np.pi/2, np.pi, 3*np.pi/2],
            "k_factor": [1.0, 1.5, 2.0]
        }
        
        parameter_spaces["multi_spiral"] = {
            "n_spirals": [2, 3, 4],
            "amplitude_range": [(0.5, 1.0), (0.8, 1.5)],
            "num_arms_range": [(1, 3), (2, 4)],
            "decay_rate_range": [(0.3, 0.5), (0.5, 0.8)],
            "position_variance": [0.5, 1.0],
            "interaction_strength": [0.5, 0.7, 0.9]
        }
        
        parameter_spaces["rogue_wave"] = {
            "system_type": ["sine_gordon", "phi4"],
            "amplitude": [1.5, 2.0, 2.5],
            "background_level": [0.1, 0.2, 0.3],
            "width": [0.5, 0.8, 1.0]
        }
        
        parameter_spaces["multi_rogue"] = {
            "system_type": ["sine_gordon", "phi4"],
            "n_rogues": [2, 3, 4],
            "amplitude_range": [(1.5, 2.5), (2.0, 3.0)],
            "width_range": [(0.5, 1.0), (0.8, 1.5)],
            "background_level": [0.1, 0.2, 0.3],
            "position_variance": [0.3, 0.5, 0.7]
        }
        
        parameter_spaces["fractal_kink"] = {
            "system_type": ["sine_gordon", "phi4"],
            "levels": [2, 3, 4],
            "base_width": [0.8, 1.2, 1.5],
            "scale_factor": [1.5, 2.0, 2.5],
            "amplitude": [0.8, 1.0, 1.2],
            "orientation": [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        }
        
        parameter_spaces["domain_wall_network"] = {
            "system_type": ["sine_gordon", "phi4"],
            "n_walls": [4, 6, 8],
            "width_range": [(0.5, 1.0), (0.8, 1.5)],
            "orientation_variance": [0.3, 0.5, 0.8],
            "interaction_strength": [0.5, 0.7, 0.9]
        }
        
        parameter_spaces["soliton_gas"] = {
            "system_type": ["sine_gordon", "phi4"],
            "n_solitons": [8, 12, 16],
            "width_range": [(0.5, 1.0), (0.8, 1.5)],
            "velocity_scale": [0.1, 0.2, 0.3],
            "interaction_strength": [0.5, 0.7, 0.9]
        }
        
        parameter_spaces["q_ball"] = {
            "system_type": ["phi4"],
            "amplitude": [0.8, 1.0, 1.2],
            "radius": [0.8, 1.0, 1.5],
            "phase": [0, np.pi/2, np.pi, 3*np.pi/2],
            "frequency": [0.6, 0.7, 0.8],
            "charge": [1, 2],
            "time_param": [0, 0.5, 1.0]
        }
        
        parameter_spaces["multi_q_ball"] = {
            "system_type": ["phi4"],
            "n_qballs": [2, 3, 4],
            "amplitude_range": [(0.5, 1.0), (0.8, 1.5)],
            "radius_range": [(0.5, 1.0), (0.8, 1.5)],
            "frequency_range": [(0.6, 0.8), (0.7, 0.9)],
            "position_variance": [0.5, 1.0],
            "interaction_strength": [0.5, 0.7, 0.9],
            "time_param": [0, 0.5, 1.0]
        }
        
        parameter_spaces["vibrational_kink"] = {
            "system_type": ["sine_gordon", "phi4"],
            "amplitude": [0.8, 1.0, 1.2],
            "width": [0.8, 1.0, 1.5],
            "mode_amplitude": [0.2, 0.3, 0.4],
            "mode_frequency": [0.3, 0.5, 0.7],
            "phase": [0, np.pi/2, np.pi, 3*np.pi/2],
            "orientation": [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            "time_param": [0, 0.5, 1.0]
        }
        
        parameter_spaces["radiation_soliton"] = {
            "system_type": ["sine_gordon", "phi4"],
            "soliton_width": [0.8, 1.0, 1.5],
            "radiation_amplitude": [0.2, 0.3, 0.4],
            "radiation_wavelength": [0.3, 0.5, 0.8],
            "radiation_direction": [0, np.pi/4, np.pi/2, 3*np.pi/4]
        }
        
        parameter_spaces["combined"] = {
            "system_type": ["sine_gordon", "phi4"],
            "solution_types": [
                ["kink", "breather", "ring"],
                ["kink", "spiral", "rogue"],
                ["breather", "ring", "spiral"],
                ["oscillon", "spiral", "fractal"],
                ["kink", "skyrmion", "qball"]
            ],
            "weights": [
                [0.4, 0.3, 0.3],
                [0.33, 0.33, 0.34],
                [0.5, 0.3, 0.2]
            ]
        }
        
        return parameter_spaces
    
    def _sample_phenomenon(self, phenomenon_type, system_type=None, **params):
        try:
            u, v = self.sampler.generate_sample(
                system_type=system_type, 
                phenomenon_type=phenomenon_type,
                **params
            )
            
            return (u, v), {
                "phenomenon_type": phenomenon_type,
                "system_type": system_type,
                "params": params
            }
            
        except Exception as e:
            warnings.warn(f"Error generating {phenomenon_type} with system_type={system_type}, params={params}: {e}")
            return None, None
    
    def map_solution_space(self, phenomena_types=None, n_samples_per_config=2, max_configs_per_type=10, 
                           n_neighbors=15):
        if phenomena_types is None:
            phenomena_types = [
                "kink", "breather", "oscillon", "multi_oscillon", 
                "ring_soliton", "multi_ring", "skyrmion", "skyrmion_lattice",
                "spiral_wave", "multi_spiral", "rogue_wave", "multi_rogue",
                "fractal_kink", "domain_wall_network", "soliton_gas", 
                "q_ball", "multi_q_ball", "vibrational_kink", 
                "radiation_soliton", "combined"
            ]
        
        self.samples = []
        self.metadata = []
            
        parameter_spaces = self._prepare_parameter_space()
        
        for phenomenon_type in tqdm(phenomena_types, desc="Generating samples"):
            if phenomenon_type not in parameter_spaces:
                warnings.warn(f"No parameter space defined for {phenomenon_type}. Skipping.")
                continue
            
            param_space = parameter_spaces[phenomenon_type]
            
            system_types = param_space.pop("system_type", [None])
            
            param_keys = list(param_space.keys())
            param_values = [param_space[k] for k in param_keys]
            
            all_combinations = list(itertools.product(*param_values))
            
            if len(all_combinations) > max_configs_per_type:
                step = max(1, len(all_combinations) // max_configs_per_type)
                selected_combinations = all_combinations[::step][:max_configs_per_type]
            else:
                selected_combinations = all_combinations
            
            for system_type in system_types:
                for combination in tqdm(selected_combinations, 
                                    desc=f"Params for {phenomenon_type} (system_type={system_type})",
                                    leave=False):
                    params = {k: v for k, v in zip(param_keys, combination)}
                    
                    for _ in range(n_samples_per_config):
                        sample, metadata = self._sample_phenomenon(
                            phenomenon_type, system_type=system_type, **params
                        )
                        
                        if sample is not None:
                            self.samples.append(sample)
                            self.metadata.append(metadata)
        
        print(f"Generated {len(self.samples)} samples across {len(phenomena_types)} phenomenon types")
            
        print("Computing distance matrix...")
        
        features = []
        for sample in tqdm(self.samples, desc="Extracting features"):
            u, v = sample
            feature_vector = np.concatenate([u.flatten(), v.flatten()])
            features.append(feature_vector)
        
        features_array = np.array(features)
        self.distance_matrix = squareform(pdist(features_array, metric='euclidean'))
        
        print("Computing embeddings...")
        
        self.embeddings['tsne'] = TSNE(
            n_components=2, 
            perplexity=min(n_neighbors, len(self.samples) - 1),
            max_iter=2000,
            random_state=42
        ).fit_transform(self.distance_matrix)
        
        self.embeddings['umap'] = umap.UMAP(
            n_components=2,
            min_dist=0.1,
            n_neighbors=min(n_neighbors, len(self.samples) - 1),
            random_state=42
        ).fit_transform(self.distance_matrix)
        
        return self.samples, self.metadata, self.distance_matrix, self.embeddings
    
    def visualize_solution_space(self, extra_title="", output_prefix=None):
        if 'tsne' not in self.embeddings:
            raise ValueError("Must run map_solution_space first")
            
        if output_prefix is None:
            output_prefix = f"real_wave_map_{len(self.samples)}"
        
        phenomenon_types = sorted(set(m["phenomenon_type"] for m in self.metadata))
        n_phenomena = len(phenomenon_types)
        
        phenomenon_colors = plt.cm.tab20(np.linspace(0, 1, n_phenomena))
        phenomenon_cmap = ListedColormap(phenomenon_colors)
        
        plt.figure(figsize=(14, 10))
        plt.scatter(
            self.embeddings['tsne'][:, 0], 
            self.embeddings['tsne'][:, 1],
            c=[phenomenon_types.index(m["phenomenon_type"]) for m in self.metadata],
            cmap=phenomenon_cmap,
            s=30,
            alpha=0.7
        )
        plt.title(f"{extra_title} t-SNE Visualization of Real Wave Solution Space", fontsize=16)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=phenomenon_colors[i], 
                               markersize=10, label=p) for i, p in enumerate(phenomenon_types)],
            loc='upper right',
            title='Phenomenon Type',
            fontsize=10
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_tsne_by_phenomenon.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        system_types = sorted(set(m["system_type"] for m in self.metadata if m["system_type"] is not None))
        if system_types:
            n_systems = len(system_types)
            system_colors = plt.cm.viridis(np.linspace(0, 1, n_systems))
            system_cmap = ListedColormap(system_colors)
            
            plt.figure(figsize=(14, 10))
            
            system_indices = []
            for m in self.metadata:
                if m["system_type"] is not None:
                    system_indices.append(system_types.index(m["system_type"]))
                else:
                    system_indices.append(-1)
            
            scatter = plt.scatter(
                self.embeddings['tsne'][:, 0], 
                self.embeddings['tsne'][:, 1],
                c=[idx if idx >= 0 else 0 for idx in system_indices],
                cmap=system_cmap,
                s=30,
                alpha=0.7
            )
            
            plt.title(f"{extra_title} t-SNE Visualization by System Type", fontsize=16)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            plt.legend(
                handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=system_colors[i], 
                                   markersize=10, label=s) for i, s in enumerate(system_types)],
                loc='upper right',
                title='System Type',
                fontsize=10
            )
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{output_prefix}_tsne_by_system.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        plt.figure(figsize=(14, 10))
        plt.scatter(
            self.embeddings['umap'][:, 0], 
            self.embeddings['umap'][:, 1],
            c=[phenomenon_types.index(m["phenomenon_type"]) for m in self.metadata],
            cmap=phenomenon_cmap,
            s=30,
            alpha=0.7
        )
        plt.title(f"{extra_title} UMAP Visualization of Real Wave Solution Space", fontsize=16)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=phenomenon_colors[i], 
                               markersize=10, label=p) for i, p in enumerate(phenomenon_types)],
            loc='upper right',
            title='Phenomenon Type',
            fontsize=10
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_umap_by_phenomenon.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        n_rows = min(len(phenomenon_types), 10)
        n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        if n_rows == 1:
            axes = np.array([axes])

        cmaps = ['viridis', 'RdBu_r', 'RdBu_r']
        labels = ['u field', 'v field', 'u+v overlay']

        for row, phenomenon in enumerate(phenomenon_types[:n_rows]):
            indices = [i for i, m in enumerate(self.metadata) if m["phenomenon_type"] == phenomenon]
            if not indices:
                continue
            idx = np.random.choice(indices)
            u, v = self.samples[idx]

            axes[row][0].imshow(u, cmap=cmaps[0])
            plt.colorbar(axes[row][0].imshow(u, cmap=cmaps[0]), ax=axes[row][0])

            axes[row][1].imshow(v, cmap=cmaps[1])
            plt.colorbar(axes[row][1].imshow(v, cmap=cmaps[1]), ax=axes[row][1])
            
            overlay = np.zeros((u.shape[0], u.shape[1], 4))
            overlay[..., 0] = np.clip((u + 1) / 2, 0, 1)
            overlay[..., 2] = np.clip((v + 1) / 2, 0, 1)
            overlay[..., 3] = 0.7
            axes[row][2].imshow(overlay)

            meta = self.metadata[idx]
            system_info = ""
            if meta["system_type"] is not None and row == 0:
                system_info = f"System: {meta['system_type']}"

            axes[row][0].set_ylabel(meta['phenomenon_type'], fontsize=12, rotation=90, va='center', ha='right')

            for col in range(n_cols):
                axes[row][col].axis('off')

        for col, label in enumerate(labels):
            axes[0][col].set_title(label, fontsize=12)

        fig.suptitle(f"{system_info}\n") 
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_detailed_view.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'phenomenon_types': phenomenon_types,
            'system_types': system_types
        }

    def characterize_solution(self, field_pair):
        u, v = field_pair
        properties = {}
        
        properties['u_energy'] = np.sum(u**2) * self.cell_area
        properties['v_energy'] = np.sum(v**2) * self.cell_area
        properties['total_energy'] = properties['u_energy'] + properties['v_energy']
        
        properties['u_peak'] = np.max(np.abs(u))
        properties['v_peak'] = np.max(np.abs(v))
        
        u_centroid = np.array([
            np.sum(self.X * u**2) / np.sum(u**2) if np.sum(u**2) > 0 else 0,
            np.sum(self.Y * u**2) / np.sum(u**2) if np.sum(u**2) > 0 else 0
        ])
        
        v_centroid = np.array([
            np.sum(self.X * v**2) / np.sum(v**2) if np.sum(v**2) > 0 else 0,
            np.sum(self.Y * v**2) / np.sum(v**2) if np.sum(v**2) > 0 else 0
        ])
        
        properties['u_centroid'] = u_centroid
        properties['v_centroid'] = v_centroid
        
        u_radius_of_gyration = np.sqrt(
            np.sum(((self.X - u_centroid[0])**2 +
                    (self.Y - u_centroid[1])**2) * u**2) / np.sum(u**2)
        ) if np.sum(u**2) > 0 else 0
        
        v_radius_of_gyration = np.sqrt(
            np.sum(((self.X - v_centroid[0])**2 +
                    (self.Y - v_centroid[1])**2) * v**2) / np.sum(v**2)
        ) if np.sum(v**2) > 0 else 0
        
        properties['u_radius_of_gyration'] = u_radius_of_gyration
        properties['v_radius_of_gyration'] = v_radius_of_gyration
        
        u_laplacian = ((np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) - 2*u) / self.dx**2 +
                     (np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 2*u) / self.dy**2)
                     
        v_laplacian = ((np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) - 2*v) / self.dx**2 +
                     (np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 2*v) / self.dy**2)
                     
        properties['u_laplacian_energy'] = np.sum(u_laplacian**2) * self.cell_area
        properties['v_laplacian_energy'] = np.sum(v_laplacian**2) * self.cell_area
        
        u_fft = np.fft.fftshift(np.abs(np.fft.fft2(u)))
        v_fft = np.fft.fftshift(np.abs(np.fft.fft2(v)))
        
        kx, ky = self.sampler.k_x, self.sampler.k_y
        KX, KY = self.sampler.KX, self.sampler.KY
        K_mag = self.sampler.K_mag
        
        properties['u_spectral_centroid'] = np.array([
            np.sum(KX * u_fft) / np.sum(u_fft) if np.sum(u_fft) > 0 else 0,
            np.sum(KY * u_fft) / np.sum(u_fft) if np.sum(u_fft) > 0 else 0
        ])
        
        properties['v_spectral_centroid'] = np.array([
            np.sum(KX * v_fft) / np.sum(v_fft) if np.sum(v_fft) > 0 else 0,
            np.sum(KY * v_fft) / np.sum(v_fft) if np.sum(v_fft) > 0 else 0
        ])
        
        properties['u_spectral_radius'] = np.sqrt(
            np.sum(K_mag**2 * u_fft) / np.sum(u_fft)
        ) if np.sum(u_fft) > 0 else 0
        
        properties['v_spectral_radius'] = np.sqrt(
            np.sum(K_mag**2 * v_fft) / np.sum(v_fft)
        ) if np.sum(v_fft) > 0 else 0
        
        properties['u_entropy'] = -np.sum((u**2 / properties['u_energy']) *
                                      np.log(u**2 / properties['u_energy'] + 1e-10)) * self.cell_area if properties['u_energy'] > 0 else 0
                                      
        properties['v_entropy'] = -np.sum((v**2 / properties['v_energy']) *
                                      np.log(v**2 / properties['v_energy'] + 1e-10)) * self.cell_area if properties['v_energy'] > 0 else 0
        
        properties['uv_correlation'] = np.sum(u * v) / (np.sqrt(np.sum(u**2) * np.sum(v**2)) + 1e-10)
        
        feature_vector = np.array([
            properties['u_energy'],
            properties['v_energy'],
            properties['total_energy'],
            properties['u_peak'],
            properties['v_peak'],
            properties['u_centroid'][0],
            properties['u_centroid'][1],
            properties['v_centroid'][0],
            properties['v_centroid'][1],
            properties['u_radius_of_gyration'],
            properties['v_radius_of_gyration'],
            properties['u_laplacian_energy'],
            properties['v_laplacian_energy'],
            properties['u_spectral_centroid'][0],
            properties['u_spectral_centroid'][1],
            properties['v_spectral_centroid'][0],
            properties['v_spectral_centroid'][1],
            properties['u_spectral_radius'],
            properties['v_spectral_radius'],
            properties['u_entropy'],
            properties['v_entropy'],
            properties['uv_correlation']
        ])
        
        return properties, feature_vector
    
    def map_solution_proxies(self, phenomena_types=None, n_samples_per_config=2, max_configs_per_type=10, 
                           n_neighbors=15):
        if phenomena_types is None:
            phenomena_types = [
                "kink", "breather", "oscillon", "multi_oscillon", 
                "ring_soliton", "multi_ring", "skyrmion", "skyrmion_lattice",
                "spiral_wave", "multi_spiral", "rogue_wave", "multi_rogue",
                "fractal_kink", "domain_wall_network", "soliton_gas", 
                "q_ball", "multi_q_ball", "vibrational_kink", 
                "radiation_soliton", "combined"
            ]
        
        self.samples = []
        self.metadata = []
        
        parameter_spaces = self._prepare_parameter_space()
        
        for phenomenon_type in tqdm(phenomena_types, desc="Generating samples"):
            if phenomenon_type not in parameter_spaces:
                warnings.warn(f"No parameter space defined for {phenomenon_type}. Skipping.")
                continue
            
            param_space = parameter_spaces[phenomenon_type]
            
            system_types = param_space.pop("system_type", [None])
            
            param_keys = list(param_space.keys())
            param_values = [param_space[k] for k in param_keys]
            
            all_combinations = list(itertools.product(*param_values))
            
            if len(all_combinations) > max_configs_per_type:
                step = max(1, len(all_combinations) // max_configs_per_type)
                selected_combinations = all_combinations[::step][:max_configs_per_type]
            else:
                selected_combinations = all_combinations
            
            for system_type in system_types:
                for combination in tqdm(selected_combinations, 
                                    desc=f"Params for {phenomenon_type} (system_type={system_type})",
                                    leave=False):
                    params = {k: v for k, v in zip(param_keys, combination)}
                    
                    for _ in range(n_samples_per_config):
                        sample, metadata = self._sample_phenomenon(
                            phenomenon_type, system_type=system_type, **params
                        )
                        
                        if sample is not None:
                            self.samples.append(sample)
                            self.metadata.append(metadata)
        
        print(f"Generated {len(self.samples)} samples across {len(phenomena_types)} phenomenon types")
        
        features = []
        for sample in tqdm(self.samples, desc="Calculating physical proxies"):
            _, feature_vector = self.characterize_solution(sample)
            features.append(feature_vector)
            
        features_array = np.array(features)
        import pdb; pdb.set_trace() 
        self.distance_matrix = squareform(pdist(features_array, metric='euclidean'))
        
        print("Computing embeddings...")
        
        self.embeddings['tsne'] = TSNE(
            n_components=2, 
            perplexity=min(n_neighbors, len(self.samples) - 1),
            max_iter=2000,
            random_state=42
        ).fit_transform(self.distance_matrix)
        
        self.embeddings['umap'] = umap.UMAP(
            n_components=2,
            min_dist=0.1,
            n_neighbors=min(n_neighbors, len(self.samples) - 1),
            random_state=42
        ).fit_transform(self.distance_matrix)
        
        return self.samples, self.metadata, self.distance_matrix, self.embeddings


def main():
    nx = ny = 128
    L = 10.0

    p_waves = [["ring"]]
    p_names = ["ring_soliton"]
    for i, p_list in enumerate(p_waves):
        curr_id = str(uuid.uuid4())[:4]
        output_dir = f"real_wave_mapping_{p_names[i]}_{curr_id}"
        
        sampler = RealWaveEquationSampler(nx, ny, L)
        mapper = RealWaveSpaceMapper(sampler, output_dir=output_dir)
        
        samples, metadata, distance_matrix, embeddings = mapper.map_solution_proxies(
            phenomena_types=p_list,
            n_samples_per_config=1,
            max_configs_per_type=100
        )
        mapper.visualize_solution_space(output_prefix=f"proxies", extra_title="Physical Proxies")
        
        samples, metadata, distance_matrix, embeddings = mapper.map_solution_space(
            phenomena_types=p_list,
            n_samples_per_config=5,
            max_configs_per_type=4
        )
        mapper.visualize_solution_space()
        
        print(f"Mapped {len(samples)} samples across {len(p_list)} phenomenon types")
        print(f"Results saved to {output_dir}")


    
    #phenomena_waves = [
    #    "ring_soliton", 
    #    "skyrmion", "spiral_wave", "rogue_wave", "fractal_kink",
    #    "q_ball", "vibrational_kink", "radiation_soliton"
    #]
    #
    #phenomena_complex = [
    #    "multi_oscillon", "multi_ring", "skyrmion_lattice",
    #    "multi_spiral", "multi_rogue", "domain_wall_network", 
    #    "soliton_gas", "multi_q_ball", "combined"
    #]
    #
    #global_phenomena = phenomena_waves + phenomena_complex
    #
    #p_names = ["waves", "complex", "global"]
    #for i, p_list in enumerate([phenomena_waves, phenomena_complex, global_phenomena]):
    #    curr_id = str(uuid.uuid4())[:4]
    #    output_dir = f"real_wave_mapping_{p_names[i]}_{curr_id}"
    #    
    #    sampler = RealWaveEquationSampler(nx, ny, L)
    #    mapper = RealWaveSpaceMapper(sampler, output_dir=output_dir)
    #    
    #    samples, metadata, distance_matrix, embeddings = mapper.map_solution_proxies(
    #        phenomena_types=p_list,
    #        n_samples_per_config=5,
    #        max_configs_per_type=4
    #    )
    #    mapper.visualize_solution_space(output_prefix=f"proxies", extra_title="Physical Proxies")
    #    
    #    samples, metadata, distance_matrix, embeddings = mapper.map_solution_space(
    #        phenomena_types=p_list,
    #        n_samples_per_config=5,
    #        max_configs_per_type=4
    #    )
    #    mapper.visualize_solution_space()
    #    
    #    print(f"Mapped {len(samples)} samples across {len(p_list)} phenomenon types")
    #    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
