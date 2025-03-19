import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
from scipy import special, stats, spatial
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import itertools
import pickle
import os
import warnings
import time
import uuid

class NLSESolutionSpaceMapper:
    def __init__(self, sampler, output_dir="nlse_mapping_results"):
        self.sampler = sampler
        self.output_dir = output_dir
        self.samples = []
        self.metadata = []
        self.distance_matrix = None
        self.embeddings = {}

        self.X, self.Y = sampler.X, sampler.Y
        self.KX, self.KY = sampler.KX, sampler.KY
        self.dx, self.dy = sampler.dx, sampler.dy
        self.cell_area = sampler.dx * sampler.dy
            
        os.makedirs(output_dir, exist_ok=True)
        
    def _prepare_parameter_space(self):
        parameter_spaces = {}
        
        parameter_spaces["multi_soliton"] = {
            "system_type": ["cubic", "cubic_quintic", "saturable", "glasner_allen_flowers"],
            "width_range": [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)],
            "amplitude_range": [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)],
            "phase_pattern": ["random", "alternating", "synchronized", "vortex"],
            "arrangement": ["linear", "circular", "random", "lattice"],
            "coherence": [0.2, 0.5, 0.8],
            "velocity_scale": [0.0, 0.5, 1.0],
            "chirp_range": [(-0.5, 0.0), (0.0, 0.5)],
            "aspect_ratio_range": [(1.0, 1.0), (1.0, 1.5)]
        }
        
        parameter_spaces["spectral"] = {
            "spectrum_type": ["kolmogorov", "ring", "gaussian_spots", "fractal"],
            "amplitude": [0.5, 1.0, 1.5],
            "k_min": [0.2, 0.5, 1.0],
            "k_max": [3.0, 5.0, 8.0],
            "spectrum_slope": [-5/3, -1.0, -2.0],
            "apply_envelope": [True]
        }
        
        parameter_spaces["chaotic"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "spectral_exponent": [-0.5, -1.0, -1.5],
            "coherent_structures": [True, False],
            "n_structures": [1, 3, 5],
            "apply_envelope": [True]
        }
        
        parameter_spaces["vortex_lattice"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "n_vortices": [3, 5, 7, 9],
            "arrangement": ["square", "triangular", "circular", "random"],
            "charge_distribution": ["alternating", "same", "random"],
            "apply_envelope": [True]
        }
        
        parameter_spaces["ring_soliton"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "radius": [1.0, 2.0, 3.0],
            "width": [0.3, 0.5, 0.8],
            "modulation_type": ["none", "azimuthal", "radial"],
            "modulation_strength": [0.0, 0.2, 0.4],
            "modulation_mode": [0, 1, 2],
            "apply_envelope": [True]
        }
        
        parameter_spaces["multi_ring"] = {
            "amplitude_range": [(0.5, 1.0), (1.0, 1.5)],
            "radius_range": [(1.0, 3.0), (2.0, 5.0)],
            "width_range": [(0.3, 0.6), (0.5, 0.8)],
            "phase_pattern": ["random", "alternating", "synchronized", "vortex"],
            "arrangement": ["linear", "circular", "random", "lattice", "concentric"],
            "modulation_type": ["none", "azimuthal", "radial"],
            "modulation_strength": [0.0, 0.2, 0.4],
            "apply_envelope": [True]
        }
        
        parameter_spaces["dark_soliton"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "width": [0.5, 1.0, 1.5],
            "eccentricity": [1.0, 1.5, 2.0],
            "order": [1, 2],
            "chirp_factor": [-0.2, 0.0, 0.2],
            "apply_envelope": [True]
        }
        
        parameter_spaces["solitary_wave_with_ambient"] = {
            "system_type": ["cubic"],
            "solitary_amplitude": [0.5, 1.0, 1.5],
            "solitary_width": [0.5, 1.0, 1.5],
            "ambient_amplitude": [0.1, 0.3, 0.5],
            "ambient_wavenumber": [1.0, 2.0, 3.0],
            "ambient_direction": [0, np.pi/4, np.pi/2, 3*np.pi/4],
            "ambient_modulation": ["none", "amplitude", "phase"],
            "order": [1, 2],
            "chirp_factor": [-0.2, 0.0, 0.2]
        }
        
        parameter_spaces["free_singularity_adapted"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "Lambda": [0.04, 0.077, 0.092, 0.13],
            "epsilon": [0.01, 0.02, 0.05],
            "background_type": ["random", "gaussian"],
            "background_amplitude": [0.1, 0.3, 0.5],
            "multi_scale": [False, True],
            "n_singularities": [1, 2, 3]
        }
        
        parameter_spaces["logarithmic_singularity_adapted"] = parameter_spaces["free_singularity_adapted"].copy()
        
        parameter_spaces["turbulent_condensate"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "condensate_fraction": [0.3, 0.5, 0.7],
            "temperature": [0.5, 1.0, 1.5],
            "n_modes": [50, 100, 200],
            "spectrum_slope": [-1.0, -1.5, -2.0],
            "modulation_type": ["none", "spatial", "phase"],
            "modulation_strength": [0.0, 0.2, 0.4],
            "apply_envelope": [True]
        }
        
        parameter_spaces["topological_defect_network"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "n_defects": [5, 10, 15],
            "defect_types": [["vortex", "antivortex"]],
            "spatial_distribution": ["poisson", "inhibition", "cluster"],
            "temperature": [0.5, 1.0, 1.5],
            "core_size": [0.3, 0.5, 0.8],
            "interaction_strength": [0.5, 0.7, 1.0],
            "apply_envelope": [True]
        }


        parameter_spaces["akhmediev_breather"] = {
            "amplitude": [.5, 1., 1.5],
            "modulation_frequency": [1., np.pi],
            "growth_rate": [1e-2, .1, 0.49],
            "breather_phase": ['compressed', 'growing', 'decaying'],
            "apply_envelope": [True, False],
            "t_param": [None, 1e-1, 2/3]

        }
       
        parameter_spaces["self_similar_pattern"] = {
                "amplitude": [0.5, 1.],
                "scale_factor": [1.5, 2.],
                "intensity_scaling": [.33, .6, .9],
                "num_iterations": [1, 5, 10],
                "base_pattern": ["ring", "vortex"],
                #"rotation_per_iteration": [.1, .1j, 1j]
        }

        
        return parameter_spaces
    
    def _sample_phenomenon(self, phenomenon_type, system_type=None, **params):
        try:
            if phenomenon_type == "multi_soliton":
                sample = self.sampler.multi_soliton_state(system_type=system_type, **params)
            elif phenomenon_type == "spectral":
                sample = self.sampler.spectral_method(**params)
            elif phenomenon_type == "chaotic":
                sample = self.sampler.chaotic_field(**params)
            elif phenomenon_type == "vortex_lattice":
                sample = self.sampler.vortex_lattice(**params)
            elif phenomenon_type == "ring_soliton":
                sample = self.sampler.ring_soliton(**params)
            elif phenomenon_type == "multi_ring":
                sample = self.sampler.multi_ring(**params)
            elif phenomenon_type == "dark_soliton":
                sample = self.sampler.dark_soliton(**params)
            elif phenomenon_type == "solitary_wave_with_ambient":
                sample = self.sampler.solitary_wave_with_ambient_field(system_type=system_type, **params)
            elif phenomenon_type == "free_singularity_adapted":
                sample = self.sampler.free_singularity_adapted(**params)
            elif phenomenon_type == "logarithmic_singularity_adapted":
                sample = self.sampler.logarithmic_singularity_adapted(**params)
            elif phenomenon_type == "turbulent_condensate":
                sample = self.sampler.turbulent_condensate(**params)
            elif phenomenon_type == "topological_defect_network":
                sample = self.sampler.topological_defect_network(**params)
            elif phenomenon_type == "akhmediev_breather":
                sample = self.sampler.akhmediev_breather(**params)
            elif phenomenon_type == "self_similar_pattern":
                sample = self.sampler.self_similar_pattern(**params)
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            max_abs = np.max(np.abs(sample))
            if max_abs > 0:
                sample = sample / max_abs
                
            return sample, {
                "phenomenon_type": phenomenon_type,
                "system_type": system_type,
                "params": params
            }
            
        except Exception as e:
            warnings.warn(f"Error generating {phenomenon_type} with system_type={system_type}, params={params}: {e}")
            return None, None
    
    def map_solution_space(self, phenomena_types=None, n_samples_per_config=2, max_configs_per_type=10, 
                           n_neighbors=15, use_existing=False, save_intermediate=False):
        if phenomena_types is None:
            phenomena_types = [
                "multi_soliton", "spectral", "chaotic", "vortex_lattice", 
                "ring_soliton", "multi_ring", "dark_soliton", "solitary_wave_with_ambient",
                "free_singularity_adapted", "logarithmic_singularity_adapted", 
                "turbulent_condensate", "topological_defect_network", "akhmediev_breather",
                "self_similar_pattern"
            ]

        # TODO update infra to re-use data!
        
        if use_existing and os.path.exists(f"{self.output_dir}/samples.pkl"):
            #with open(f"{self.output_dir}/samples.pkl", "rb") as f:
            #    self.samples = pickle.load(f)
            #with open(f"{self.output_dir}/metadata.pkl", "rb") as f:
            #    self.metadata = pickle.load(f)
            #    
            #if os.path.exists(f"{self.output_dir}/distance_matrix.pkl"):
            #    with open(f"{self.output_dir}/distance_matrix.pkl", "rb") as f:
            #        self.distance_matrix = pickle.load(f)
                    
            print(f"Loaded {len(self.samples)} existing samples")
        else:
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
                        
                        #if save_intermediate and len(self.samples) % 100 == 0:
                        #    with open(f"{self.output_dir}/samples_intermediate_{len(self.samples)}.pkl", "wb") as f:
                        #        pickle.dump(self.samples, f)
                        #    with open(f"{self.output_dir}/metadata_intermediate_{len(self.samples)}.pkl", "wb") as f:
                        #        pickle.dump(self.metadata, f)
            
            print(f"Generated {len(self.samples)} samples across {len(phenomena_types)} phenomenon types")
            
            #with open(f"{self.output_dir}/samples.pkl", "wb") as f:
            #    pickle.dump(self.samples, f)
            #with open(f"{self.output_dir}/metadata.pkl", "wb") as f:
            #    pickle.dump(self.metadata, f)
        
        if self.distance_matrix is None:
            print("Computing distance matrix...")
            
            features = []
            for sample in tqdm(self.samples, desc="Extracting features"):
                flat = sample.flatten()
                feature_vector = np.concatenate([np.abs(flat), np.angle(flat)])
                features.append(feature_vector)
            
            features_array = np.array(features)
            self.distance_matrix = squareform(pdist(features_array, metric='euclidean'))
            
            #with open(f"{self.output_dir}/distance_matrix.pkl", "wb") as f:
            #    pickle.dump(self.distance_matrix, f)
        
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
        
        #with open(f"{self.output_dir}/embeddings.pkl", "wb") as f:
        #    pickle.dump(self.embeddings, f)
        
        return self.samples, self.metadata, self.distance_matrix, self.embeddings
    
    def visualize_solution_space(self, extra_title="", output_prefix=None):
        if 'tsne' not in self.embeddings:
            raise ValueError("Must run map_solution_space first")
            
        if output_prefix is None:
            output_prefix = f"nlse_map_{len(self.samples)}"
        
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
        plt.title(f"{extra_title} t-SNE Visualization of NLSE Solution Space by Phenomenon Type", fontsize=16)
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
        plt.title(f"{extra_title} UMAP Visualization of NLSE Solution Space by Phenomenon Type", fontsize=16)
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
        phenomenon_types = sorted(set(meta["phenomenon_type"] for meta in self.metadata))

        if len(phenomenon_types) > 7:
            print(f"Sample comparison may be too large for ({len(phenomenon_types)}) phenomena!")

        n_rows = len(phenomenon_types)
        n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        if n_rows == 1:
            axes = np.array([axes])

        cmaps = ['viridis', 'RdBu_r', 'twilight']
        labels = ['amplitude', 'real', 'phase']

        for row, phenomenon in enumerate(phenomenon_types):
            indices = [i for i, m in enumerate(self.metadata) if m["phenomenon_type"] == phenomenon]
            if not indices:
                continue
            idx = np.random.choice(indices)
            sample = self.samples[idx]

            axes[row][0].imshow(np.abs(sample), cmap=cmaps[0])
            plt.colorbar(axes[row][0].imshow(np.abs(sample), cmap=cmaps[0]), ax=axes[row][0])

            axes[row][1].imshow(np.real(sample), cmap=cmaps[1])
            plt.colorbar(axes[row][1].imshow(np.real(sample), cmap=cmaps[1]), ax=axes[row][1])

            axes[row][2].imshow(np.angle(sample), cmap=cmaps[2], vmin=-np.pi, vmax=np.pi)
            plt.colorbar(axes[row][2].imshow(np.angle(sample), cmap=cmaps[2], vmin=-np.pi, vmax=np.pi), ax=axes[row][2])

            meta = self.metadata[idx]
            system_info = ""
            if meta["system_type"] is not None and row == 0:
                system_info = f"System: {meta['system_type']}"

            axes[row][0].set_ylabel(meta['phenomenon_type'], fontsize=12, rotation=90, va='center', ha='right')

            for col in range(n_cols):
                axes[row][col].axis('off')

        for col, label in enumerate(labels):
            axes[0][col].set_title(label, fontsize=12)

        # somehow doesn't really work...
        for row, phenomenon in enumerate(phenomenon_types):
            axes[row][0].set_ylabel(phenomenon, fontsize=12, rotation=90, va='center', ha='right')

        fig.suptitle(f"{system_info}\n") 
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{output_prefix}_detailed_view.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'phenomenon_types': phenomenon_types,
            'system_types': system_types
        }

    def characterize_solution(self, field, compute_all=False):
        properties = {}

        amp = np.abs(field)
        phase = np.angle(field)

        properties['total_power'] = np.sum(amp**2) * self.cell_area
        properties['peak_amplitude'] = np.max(amp)
        properties['centroid'] = np.array([
            np.sum(self.X * amp**2) / np.sum(amp**2),
            np.sum(self.Y * amp**2) / np.sum(amp**2)
        ])

        radius_of_gyration = np.sqrt(
            np.sum(((self.X - properties['centroid'][0])**2 +
                    (self.Y - properties['centroid'][1])**2) * amp**2) / np.sum(amp**2)
        )
        properties['radius_of_gyration'] = radius_of_gyration

        mean_peak_distance = 0
        threshold = 0.5 * properties['peak_amplitude']
        peaks = amp > threshold
        if np.sum(peaks) > 1:
            peak_coords = np.array(np.where(peaks)).T
            distances = spatial.distance.pdist(peak_coords)
            mean_peak_distance = np.mean(distances) * self.dx
        properties['mean_peak_distance'] = mean_peak_distance

        momentum_x = -0.5j * np.sum(field.conj() * (np.roll(field, -1, axis=0) -
                                                 np.roll(field, 1, axis=0))) / (2 * self.dx)
        momentum_y = -0.5j * np.sum(field.conj() * (np.roll(field, -1, axis=1) -
                                                 np.roll(field, 1, axis=1))) / (2 * self.dy)
        properties['momentum'] = np.array([np.real(momentum_x), np.real(momentum_y)])

        angular_momentum = np.sum(
            (self.X * np.real(momentum_y) - self.Y * np.real(momentum_x)) * amp**2
        ) * self.cell_area
        properties['angular_momentum'] = angular_momentum

        properties['entropy'] = -np.sum((amp**2 / properties['total_power']) *
                                        np.log(amp**2 / properties['total_power'] + 1e-10)) * self.cell_area

        if compute_all:
            spectrum = np.fft.fftshift(np.abs(np.fft.fft2(field))**2)
            properties['spectral_centroid'] = np.array([
                np.sum(self.KX * spectrum) / np.sum(spectrum),
                np.sum(self.KY * spectrum) / np.sum(spectrum)
            ])

            spectral_bandwidth = np.sqrt(
                np.sum(((self.KX - properties['spectral_centroid'][0])**2 +
                        (self.KY - properties['spectral_centroid'][1])**2) * spectrum) / np.sum(spectrum)
            )
            properties['spectral_bandwidth'] = spectral_bandwidth

            laplacian = ((np.roll(field, -1, axis=0) + np.roll(field, 1, axis=0) - 2*field) / self.dx**2 +
                         (np.roll(field, -1, axis=1) + np.roll(field, 1, axis=1) - 2*field) / self.dy**2)
            kinetic_energy = np.sum(np.abs(laplacian)**2) * self.cell_area
            properties['kinetic_energy'] = kinetic_energy

    

        p_array        = np.zeros((14,), dtype=np.float64)
        p_array[0]     = properties['total_power']
        p_array[1]     = properties['peak_amplitude']
        p_array[2:4]   = properties['centroid']
        p_array[4]     = properties['radius_of_gyration']
        p_array[5:7]   = properties['momentum']
        p_array[7]     = properties['mean_peak_distance']
        p_array[8]     = properties['angular_momentum']
        p_array[9]     = properties['entropy']
        p_array[10:12] = properties['spectral_centroid'] 
        p_array[12]    = properties['spectral_bandwidth']
        p_array[13]    = properties['kinetic_energy']
        
        return properties, p_array

    def map_solution_proxies(self, phenomena_types=None, n_samples_per_config=2, max_configs_per_type=10, 
                           n_neighbors=15):
        if phenomena_types is None:
            phenomena_types = [
                "multi_soliton", "spectral", "chaotic", "vortex_lattice", 
                "ring_soliton", "multi_ring", "dark_soliton", "solitary_wave_with_ambient",
                "free_singularity_adapted", "logarithmic_singularity_adapted", 
                "turbulent_condensate", "topological_defect_network", "akhemediev_breather",
                "self_similar_pattern"
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

        samples_proxied = []
        for sample in tqdm(self.samples, desc="Calculating physical proxies"):
            _, data = self.characterize_solution(sample, compute_all=True)
            samples_proxied.append(data)
        features_array = np.array(samples_proxied)
        #del self.samples # left out for now
    
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
    from nlse_sampler import NLSEPhenomenonSampler
    import time
    
    nx = ny = 128
    L = 10.

    
    phenomena_to_map_chaotic = [
        "turbulent_condensate", "spectral", "chaotic",
        "solitary_wave_with_ambient", "topological_defect_network",
        "logarithmic_singularity_adapted",
            ] 

    phenomena_to_map_defined = [
        "multi_soliton", "vortex_lattice", "dark_soliton",
        "multi_ring", "akhmediev_breather", "self_similar_pattern"
            ]
    global_phenomena = phenomena_to_map_chaotic + phenomena_to_map_defined 

    p_names = ["chaotic", "defined", "global"]
    for i, p_list in enumerate([phenomena_to_map_chaotic, phenomena_to_map_defined, global_phenomena]):  
        curr_id = str(uuid.uuid4())[:4]
        output_dir = f"comparing_sample_diversity_{p_names[i]}_{curr_id}" 

        sampler_proxies = NLSEPhenomenonSampler(nx, ny, L)
        mapper_proxies  = NLSESolutionSpaceMapper(sampler_proxies, output_dir=output_dir)
        samples, metadata, distance_matrix, embeddings = mapper_proxies.map_solution_proxies(
            phenomena_types=p_list,
            n_samples_per_config=10,
            max_configs_per_type=5
        ) 
        mapper_proxies.visualize_solution_space(output_prefix= f"proxies", extra_title="proxies") 

        # a little braindead but saves us refactoring time
        del sampler_proxies
        del mapper_proxies

        sampler = NLSEPhenomenonSampler(nx, ny, L)
        mapper  = NLSESolutionSpaceMapper(sampler, output_dir=output_dir)
        samples, metadata, distance_matrix, embeddings = mapper.map_solution_space(
            phenomena_types=p_list,
            n_samples_per_config=10,
            max_configs_per_type=5
        ) 
        mapper.visualize_solution_space() 
        print(f"Mapped {len(samples)} samples across {len(p_list)} phenomenon types")
        print(f"Results saved to {output_dir}")
 
        del sampler
        del mapper
 
if __name__ == "__main__":
    main()
