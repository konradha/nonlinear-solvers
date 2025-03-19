import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import os
import itertools
import time
from collections import defaultdict

class NLSEPhenomenonDiversityAnalyzer:
    def __init__(self, sampler, output_dir="phenomenon_diversity"):
        self.sampler = sampler
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.samples = []
        self.metadata = []
        self.feature_embeddings = {}
        self.proxy_embeddings = {}
        
        self.X, self.Y = sampler.X, sampler.Y
        self.KX, self.KY = sampler.KX, sampler.KY
        self.dx, self.dy = sampler.dx, sampler.dy
        self.cell_area = sampler.dx * sampler.dy
        
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
            "chirp_range": [(-0.1, 0.1), (-1.0, 1.0)]
        }
        
        parameter_spaces["spectral"] = {
            "spectrum_type": ["kolmogorov", "ring", "gaussian_spots", "fractal"],
            "amplitude": [0.5, 1.0, 1.5],
            "k_min": [0.2, 0.5, 1.0],
            "k_max": [3.0, 5.0, 8.0],
            "spectrum_slope": [-5/3, -1.0, -2.0],
            "apply_envelope": [True, False]
        }
        
        parameter_spaces["chaotic"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "spectral_exponent": [-0.5, -1.0, -1.5],
            "coherent_structures": [True, False],
            "n_structures": [1, 3, 5],
            "apply_envelope": [True, False]
        }
        
        parameter_spaces["vortex_lattice"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "n_vortices": [3, 5, 7, 9],
            "arrangement": ["square", "triangular", "circular", "random"],
            "charge_distribution": ["alternating", "same", "random"],
            "apply_envelope": [True, False]
        }
        
        parameter_spaces["dark_soliton"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "width": [0.5, 1.0, 1.5],
            "eccentricity": [1.0, 1.5, 2.0],
            "order": [1, 2],
            "chirp_factor": [-0.2, 0.0, 0.2],
            "apply_envelope": [True, False]
        }
        
        parameter_spaces["multi_ring"] = {
            "amplitude_range": [(0.5, 1.0), (1.0, 1.5)],
            "radius_range": [(1.0, 3.0), (2.0, 5.0)],
            "width_range": [(0.3, 0.6), (0.5, 0.8)],
            "phase_pattern": ["random", "alternating", "synchronized", "vortex"],
            "arrangement": ["concentric", "lattice", "random"],
            "modulation_type": ["none", "azimuthal", "radial"],
            "apply_envelope": [True, False]
        }
        
        parameter_spaces["solitary_wave_with_ambient"] = {
            "system_type": ["cubic"],
            "solitary_amplitude": [0.5, 1.0, 1.5],
            "solitary_width": [0.5, 1.0, 1.5],
            "ambient_amplitude": [0.1, 0.3, 0.5],
            "ambient_wavenumber": [1.0, 2.0, 3.0],
            "ambient_direction": [0, np.pi/4, np.pi/2, 3*np.pi/4],
            "ambient_modulation": ["none", "amplitude", "phase"]
        }
        
        parameter_spaces["logarithmic_singularity_adapted"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "Lambda": [0.04, 0.077, 0.092, 0.13],
            "epsilon": [0.01, 0.02, 0.05],
            "background_type": ["random", "gaussian"],
            "background_amplitude": [0.1, 0.3, 0.5],
            "multi_scale": [False, True],
            "n_singularities": [1, 2, 3]
        }
        
        parameter_spaces["turbulent_condensate"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "condensate_fraction": [0.3, 0.5, 0.7],
            "temperature": [0.5, 1.0, 1.5],
            "n_modes": [50, 100, 200],
            "spectrum_slope": [-1.0, -1.5, -2.0],
            "apply_envelope": [True, False]
        }
        
        parameter_spaces["topological_defect_network"] = {
            "amplitude": [0.5, 1.0, 1.5],
            "n_defects": [5, 10, 15, 1000],
            "defect_types": [["vortex", "antivortex"], ["domain_wall"]],
            "spatial_distribution": ["poisson", "inhibition", "cluster"],
            "temperature": [0.5, 1.0, 1.5],
            "core_size": [0.3, 0.5, 0.8],
            "apply_envelope": [True, False]
        }
        
        return parameter_spaces
    
    def _generate_sample(self, phenomenon_type, system_type=None, **params):
        try:
            if phenomenon_type == "multi_soliton":
                sample = self.sampler.multi_soliton_state(system_type=system_type, **params)
            elif phenomenon_type == "spectral":
                sample = self.sampler.spectral_method(**params)
            elif phenomenon_type == "chaotic":
                sample = self.sampler.chaotic_field(**params)
            elif phenomenon_type == "vortex_lattice":
                sample = self.sampler.vortex_lattice(**params)
            elif phenomenon_type == "dark_soliton":
                sample = self.sampler.dark_soliton(**params)
            elif phenomenon_type == "multi_ring":
                sample = self.sampler.multi_ring(**params)
            elif phenomenon_type == "solitary_wave_with_ambient":
                sample = self.sampler.solitary_wave_with_ambient_field(system_type=system_type, **params)
            elif phenomenon_type == "logarithmic_singularity_adapted":
                sample = self.sampler.logarithmic_singularity_adapted(**params)
            elif phenomenon_type == "turbulent_condensate":
                sample = self.sampler.turbulent_condensate(**params)
            elif phenomenon_type == "topological_defect_network":
                sample = self.sampler.topological_defect_network(**params)
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            max_abs = np.max(np.abs(sample))
            if max_abs > 0:
                sample = sample / max_abs
                
            return sample
            
        except Exception as e:
            print(f"Error generating {phenomenon_type}: {e}")
            return None
    
    def analyze_phenomenon_diversity(self, phenomenon_type, max_samples=50, random_seed=42):
        np.random.seed(random_seed)
        parameter_spaces = self._prepare_parameter_space()
        
        if phenomenon_type not in parameter_spaces:
            raise ValueError(f"No parameter space defined for {phenomenon_type}")
        
        param_space = parameter_spaces[phenomenon_type].copy()
        system_types = param_space.pop("system_type", [None])
        
        param_keys = list(param_space.keys())
        param_values = [param_space[k] for k in param_keys]
        
        all_combinations = list(itertools.product(*param_values))
        np.random.shuffle(all_combinations)
        
        if len(all_combinations) > max_samples:
            selected_combinations = all_combinations[:max_samples]
        else:
            selected_combinations = all_combinations
        
        self.samples = []
        self.metadata = []
        self.param_values = defaultdict(set)
        
        print(f"Generating {len(selected_combinations)} samples for {phenomenon_type}")
        
        for system_type in system_types:
            for combination in tqdm(selected_combinations):
                params = {k: v for k, v in zip(param_keys, combination)}
                
                sample = self._generate_sample(phenomenon_type, system_type, **params)
                
                if sample is not None:
                    properties, proxy_data = self._calculate_properties(sample)
                    
                    self.samples.append(sample)
                    metadata_entry = {
                        "phenomenon_type": phenomenon_type,
                        "system_type": system_type,
                        "params": params,
                        "properties": properties,
                        "proxy_data": proxy_data
                    }
                    self.metadata.append(metadata_entry)
                    
                    self.param_values["system_type"].add(str(system_type) if system_type is not None else "None")
                    for k, v in params.items():
                        if isinstance(v, (list, tuple)):
                            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                self.param_values[k].add(str(v))
                            else:
                                self.param_values[k].add(str(v[0]) if len(v) > 0 else "empty")
                        else:
                            self.param_values[k].add(str(v))
        
        n_samples = len(self.samples)
        print(f"Successfully generated {n_samples} samples")
        
        if n_samples < 4:
            print("Too few samples to compute embeddings")
            return self.samples, self.metadata
        
        self._compute_embeddings(n_samples)
        self._visualize_samples(phenomenon_type)
        self._visualize_embeddings(phenomenon_type)
        self._visualize_physical_properties(phenomenon_type)
        
        return self.samples, self.metadata
        
    def _calculate_properties(self, field):
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
            distances = np.sqrt(np.sum((peak_coords[:, None, :] - peak_coords[None, :, :]) ** 2, axis=2))
            mean_peak_distance = np.mean(distances[distances > 0]) * self.dx
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
        
        p_array = np.zeros(14, dtype=np.float64)
        p_array[0] = properties['total_power']
        p_array[1] = properties['peak_amplitude']
        p_array[2:4] = properties['centroid']
        p_array[4] = properties['radius_of_gyration']
        p_array[5:7] = properties['momentum']
        p_array[7] = properties['mean_peak_distance']
        p_array[8] = properties['angular_momentum']
        p_array[9] = properties['entropy']
        p_array[10:12] = properties['spectral_centroid']
        p_array[12] = properties['spectral_bandwidth']
        p_array[13] = properties['kinetic_energy']
        
        return properties, p_array
    
    def _compute_embeddings(self, n_samples):
        feature_vectors = []
        for sample in tqdm(self.samples, desc="Extracting features"):
            flat = sample.flatten()
            feature_vector = np.concatenate([np.abs(flat), np.angle(flat)])
            feature_vectors.append(feature_vector)
        
        feature_array = np.array(feature_vectors)
        distance_matrix = squareform(pdist(feature_array, metric='euclidean'))
        
        proxy_array = np.array([m["proxy_data"] for m in self.metadata])
        proxy_distance_matrix = squareform(pdist(proxy_array, metric='euclidean'))
        
        perplexity = min(30, n_samples - 1)
        if perplexity < 5:
            perplexity = 5
        
        print("Computing t-SNE embeddings...")
        self.feature_embeddings['tsne'] = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42
        ).fit_transform(distance_matrix)
        
        self.proxy_embeddings['tsne'] = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42
        ).fit_transform(proxy_distance_matrix)
        
        n_neighbors = min(15, n_samples - 1)
        if n_neighbors < 2:
            n_neighbors = 2
            
        print("Computing UMAP embeddings...")
        self.feature_embeddings['umap'] = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=42
        ).fit_transform(distance_matrix)
        
        self.proxy_embeddings['umap'] = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=42
        ).fit_transform(proxy_distance_matrix)
    
    def _visualize_samples(self, phenomenon_type):
        n_samples = min(8, len(self.samples))
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))

        if n_samples == 1:
            axes = np.array([axes])

        cmaps = ['viridis', 'RdBu_r', 'twilight']
        labels = ['Amplitude', 'Real', 'Phase']

        for row, (sample, meta) in enumerate(zip(self.samples[:n_samples], self.metadata[:n_samples])):
            axes[row][0].imshow(np.abs(sample), cmap=cmaps[0])
            plt.colorbar(axes[row][0].imshow(np.abs(sample), cmap=cmaps[0]), ax=axes[row][0])

            axes[row][1].imshow(np.real(sample), cmap=cmaps[1])
            plt.colorbar(axes[row][1].imshow(np.real(sample), cmap=cmaps[1]), ax=axes[row][1])

            axes[row][2].imshow(np.angle(sample), cmap=cmaps[2], vmin=-np.pi, vmax=np.pi)
            plt.colorbar(axes[row][2].imshow(np.angle(sample), cmap=cmaps[2], vmin=-np.pi, vmax=np.pi), ax=axes[row][2])

            param_text = []
            for k, v in meta["params"].items():
                if isinstance(v, (list, tuple)):
                    v_str = f"{k}: {v}"
                else:
                    v_str = f"{k}: {v}"
                param_text.append(v_str)
            
            param_str = ", ".join(param_text[:3])
            if len(param_text) > 3:
                param_str += "..."
            
            system_info = ""
            if meta["system_type"] is not None:
                system_info = f" ({meta['system_type']})"

            axes[row][0].set_ylabel(f"Sample {row+1}{system_info}\n{param_str}", fontsize=10, rotation=0, labelpad=80, va='center', ha='right')

            for col in range(3):
                axes[row][col].axis('off')

        for col, label in enumerate(labels):
            axes[0][col].set_title(label, fontsize=12)

        fig.suptitle(f"{phenomenon_type} Parameter Space Samples")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{phenomenon_type}_sample_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_embeddings(self, phenomenon_type):
        from matplotlib.colors import Normalize, ListedColormap
        
        for embedding_name, embedding_data in [('tsne', self.feature_embeddings['tsne']), 
                                            ('umap', self.feature_embeddings['umap'])]:
            plt.figure(figsize=(12, 10))
            
            system_types = sorted(set(m["system_type"] for m in self.metadata if m["system_type"] is not None))
            if system_types:
                colors = plt.cm.tab10(np.linspace(0, 1, len(system_types)))
                cmap = ListedColormap(colors)
                
                system_indices = []
                for meta in self.metadata:
                    if meta["system_type"] is not None:
                        system_indices.append(system_types.index(meta["system_type"]))
                    else:
                        system_indices.append(-1)
                
                scatter = plt.scatter(
                    embedding_data[:, 0],
                    embedding_data[:, 1],
                    c=[idx if idx >= 0 else len(system_types) for idx in system_indices],
                    cmap=cmap,
                    s=80,
                    alpha=0.8
                )
                
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=10, 
                                        label=system_type) 
                                for i, system_type in enumerate(system_types)]
                
                plt.legend(handles=legend_elements, title="System Type")
            else:
                plt.scatter(
                    embedding_data[:, 0],
                    embedding_data[:, 1],
                    s=80,
                    alpha=0.8
                )
            
            plt.title(f"{phenomenon_type} - {embedding_name.upper()} Visualization", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{phenomenon_type}_{embedding_name}_system_types.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            for param_name, param_values in self.param_values.items():
                if len(param_values) <= 1 or param_name == "system_type":
                    continue
                    
                plt.figure(figsize=(12, 10))
                
                if len(param_values) <= 10:
                    param_values_list = sorted(list(param_values))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(param_values_list)))
                    cmap = ListedColormap(colors)
                    
                    param_indices = []
                    for meta in self.metadata:
                        if param_name in meta["params"]:
                            param_value = meta["params"][param_name]
                            if isinstance(param_value, (list, tuple)):
                                param_str = str(param_value)
                            else:
                                param_str = str(param_value)
                            
                            if param_str in param_values_list:
                                param_indices.append(param_values_list.index(param_str))
                            else:
                                param_indices.append(-1)
                        else:
                            param_indices.append(-1)
                    
                    scatter = plt.scatter(
                        embedding_data[:, 0],
                        embedding_data[:, 1],
                        c=[idx if idx >= 0 else len(param_values_list) for idx in param_indices],
                        cmap=cmap,
                        s=80,
                        alpha=0.8
                    )
                    
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=colors[i], markersize=10, 
                                            label=val) 
                                    for i, val in enumerate(param_values_list)]
                    
                    plt.legend(handles=legend_elements, title=param_name)
                else:
                    try:
                        numeric_values = []
                        for meta in self.metadata:
                            param_value = meta["params"].get(param_name)
                            if param_value is not None:
                                if isinstance(param_value, (list, tuple)):
                                    if len(param_value) == 2 and all(isinstance(x, (int, float)) for x in param_value):
                                        numeric_values.append(float(param_value[0]))
                                    else:
                                        numeric_values.append(0)
                                elif isinstance(param_value, (int, float)):
                                    numeric_values.append(float(param_value))
                                elif param_value in ['True', 'False']:
                                    numeric_values.append(1 if param_value == 'True' else 0)
                                else:
                                    numeric_values.append(0)
                            else:
                                numeric_values.append(0)
                        
                        norm = Normalize(vmin=min(numeric_values), vmax=max(numeric_values))
                        scatter = plt.scatter(
                            embedding_data[:, 0],
                            embedding_data[:, 1],
                            c=numeric_values,
                            cmap='viridis',
                            norm=norm,
                            s=80,
                            alpha=0.8
                        )
                        
                        plt.colorbar(scatter, label=param_name)
                    except ValueError:
                        param_indices = []
                        for meta in self.metadata:
                            param_value = meta["params"].get(param_name)
                            if param_value is not None:
                                if isinstance(param_value, (list, tuple)):
                                    param_str = str(param_value[0])
                                else:
                                    param_str = str(param_value)
                                
                                param_indices.append(hash(param_str) % 256)
                            else:
                                param_indices.append(0)
                        
                        scatter = plt.scatter(
                            embedding_data[:, 0],
                            embedding_data[:, 1],
                            c=param_indices,
                            cmap='tab20',
                            s=80,
                            alpha=0.8
                        )
                
                plt.title(f"{phenomenon_type} - {embedding_name.upper()} by {param_name}", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/{phenomenon_type}_{embedding_name}_{param_name}.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def _visualize_physical_properties(self, phenomenon_type):
        from matplotlib.colors import Normalize
        
        key_properties = [
            'total_power', 'peak_amplitude', 'radius_of_gyration', 
            'entropy', 'angular_momentum', 'spectral_bandwidth'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, prop in enumerate(key_properties):
            property_values = [meta["properties"][prop] for meta in self.metadata]
            
            norm = Normalize(vmin=min(property_values), vmax=max(property_values))
            scatter = axes[i].scatter(
                self.feature_embeddings['tsne'][:, 0],
                self.feature_embeddings['tsne'][:, 1],
                c=property_values,
                cmap='plasma',
                norm=norm,
                s=80,
                alpha=0.8
            )
            
            axes[i].set_title(f"{prop.replace('_', ' ').title()}")
            plt.colorbar(scatter, ax=axes[i], label=prop)
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f"{phenomenon_type} - Physical Properties in t-SNE Space", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{phenomenon_type}_physical_properties.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        correlation_matrix = np.zeros((len(key_properties), len(key_properties)))
        for i, prop1 in enumerate(key_properties):
            for j, prop2 in enumerate(key_properties):
                prop1_values = [meta["properties"][prop1] for meta in self.metadata]
                prop2_values = [meta["properties"][prop2] for meta in self.metadata]
                
                correlation = np.corrcoef(prop1_values, prop2_values)[0, 1]
                correlation_matrix[i, j] = correlation
        
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation Coefficient')
        
        plt.xticks(np.arange(len(key_properties)), [p.replace('_', ' ').title() for p in key_properties], rotation=45, ha='right')
        plt.yticks(np.arange(len(key_properties)), [p.replace('_', ' ').title() for p in key_properties])
        
        for i in range(len(key_properties)):
            for j in range(len(key_properties)):
                plt.text(j, i, f"{correlation_matrix[i, j]:.2f}", ha='center', va='center', 
                        color='black' if abs(correlation_matrix[i, j]) < 0.7 else 'white')
        
        plt.title(f"{phenomenon_type} - Physical Property Correlations")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{phenomenon_type}_property_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_parameter_influence(self, phenomenon_type, parameter_name):
        if not self.samples or not self.metadata:
            raise ValueError("Must run analyze_phenomenon_diversity first")
            
        param_values = set()
        if parameter_name == "system_type":
            for meta in self.metadata:
                if meta["system_type"] is not None:
                    param_values.add(meta["system_type"])
        else:
            for meta in self.metadata:
                param_value = meta["params"].get(parameter_name)
                if param_value is not None:
                    if isinstance(param_value, (list, tuple)):
                        param_values.add(str(param_value))
                    else:
                        param_values.add(str(param_value))
        
        param_values = sorted(list(param_values))
        if not param_values:
            print(f"Parameter {parameter_name} not found in any samples")
            return
        
        if len(param_values) <= 1:
            print(f"Parameter {parameter_name} has only one value: {param_values[0]}")
            return
            
        fig, axes = plt.subplots(len(param_values), 3, figsize=(15, 5 * len(param_values)))
        if len(param_values) == 1:
            axes = np.array([axes])
            
        cmaps = ['viridis', 'RdBu_r', 'twilight']
        labels = ['Amplitude', 'Real', 'Phase']
        
        for i, param_value in enumerate(param_values):
            matching_samples = []
            for sample, meta in zip(self.samples, self.metadata):
                if parameter_name == "system_type":
                    if meta["system_type"] == param_value:
                        matching_samples.append(sample)
                else:
                    meta_value = meta["params"].get(parameter_name)
                    if meta_value is not None:
                        if isinstance(meta_value, (list, tuple)):
                            meta_str = str(meta_value)
                        else:
                            meta_str = str(meta_value)
                            
                        if meta_str == param_value:
                            matching_samples.append(sample)
            
            if matching_samples:
                sample = matching_samples[0]
                
                axes[i][0].imshow(np.abs(sample), cmap=cmaps[0])
                plt.colorbar(axes[i][0].imshow(np.abs(sample), cmap=cmaps[0]), ax=axes[i][0])
                
                axes[i][1].imshow(np.real(sample), cmap=cmaps[1])
                plt.colorbar(axes[i][1].imshow(np.real(sample), cmap=cmaps[1]), ax=axes[i][1])
                
                axes[i][2].imshow(np.angle(sample), cmap=cmaps[2], vmin=-np.pi, vmax=np.pi)
                plt.colorbar(axes[i][2].imshow(np.angle(sample), cmap=cmaps[2], vmin=-np.pi, vmax=np.pi), ax=axes[i][2])
                
                axes[i][0].set_ylabel(f"{parameter_name} = {param_value}", fontsize=12, rotation=0, labelpad=80, va='center', ha='right')
                
                for col in range(3):
                    axes[i][col].axis('off')
        
        for col, label in enumerate(labels):
            axes[0][col].set_title(label, fontsize=12)
            
        plt.suptitle(f"{phenomenon_type} - Effect of {parameter_name} Parameter", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{phenomenon_type}_param_influence_{parameter_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        key_properties = ['total_power', 'radius_of_gyration', 'entropy', 'spectral_bandwidth']
        fig, axes = plt.subplots(len(key_properties), 1, figsize=(12, 4 * len(key_properties)))
        
        for i, prop in enumerate(key_properties):
            param_to_values = {}
            
            for meta in self.metadata:
                param_match = False
                
                if parameter_name == "system_type":
                    if meta["system_type"] == param_value:
                        param_match = True
                        param_str = str(meta["system_type"])
                else:
                    meta_value = meta["params"].get(parameter_name)
                    if meta_value is not None:
                        if isinstance(meta_value, (list, tuple)):
                            param_str = str(meta_value)
                        else:
                            param_str = str(meta_value)
                        
                        param_match = True
                
                if param_match:
                    if param_str not in param_to_values:
                        param_to_values[param_str] = []
                    
                    param_to_values[param_str].append(meta["properties"][prop])
            
            for param_value, values in param_to_values.items():
                if len(values) > 0:
                    pos = list(param_to_values.keys()).index(param_value)
                    axes[i].boxplot(values, positions=[pos], 
                                    widths=0.6, labels=[param_value])
                
            axes[i].set_title(f"{prop.replace('_', ' ').title()}")
            axes[i].grid(True, alpha=0.3)
            
            if i == len(key_properties) - 1:
                axes[i].set_xlabel(parameter_name)
                
        plt.suptitle(f"{phenomenon_type} - {parameter_name} Parameter Effect on Properties", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{phenomenon_type}_param_property_effect_{parameter_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_phenomena(self, phenomenon_types, max_samples_per_type=20, random_seed=42):
        output_dir = f"{self.output_dir}/phenomenon_comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        np.random.seed(random_seed)
        all_samples = []
        all_metadata = []
        
        for phenomenon_type in phenomenon_types:
            self.samples = []
            self.metadata = []
            self.analyze_phenomenon_diversity(phenomenon_type, max_samples=max_samples_per_type, random_seed=random_seed)
            
            all_samples.extend(self.samples)
            all_metadata.extend(self.metadata)
        
        self.samples = all_samples
        self.metadata = all_metadata
        
        feature_vectors = []
        for sample in tqdm(self.samples, desc="Extracting features"):
            flat = sample.flatten()
            feature_vector = np.concatenate([np.abs(flat), np.angle(flat)])
            feature_vectors.append(feature_vector)
        
        feature_array = np.array(feature_vectors)
        distance_matrix = squareform(pdist(feature_array, metric='euclidean'))
        
        proxy_array = np.array([m["proxy_data"] for m in self.metadata])
        proxy_distance_matrix = squareform(pdist(proxy_array, metric='euclidean'))
        
        perplexity = min(30, len(self.samples) - 1)
        if perplexity < 5:
            perplexity = 5
        
        print("Computing t-SNE embeddings...")
        tsne_embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42
        ).fit_transform(distance_matrix)
        
        tsne_proxy_embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42
        ).fit_transform(proxy_distance_matrix)
        
        from matplotlib.colors import ListedColormap
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(phenomenon_types)))
        cmap = ListedColormap(colors)
        
        phenomenon_indices = [phenomenon_types.index(m["phenomenon_type"]) for m in self.metadata]
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            tsne_embedding[:, 0],
            tsne_embedding[:, 1],
            c=phenomenon_indices,
            cmap=cmap,
            s=80,
            alpha=0.8
        )
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=colors[i], markersize=10, 
                                  label=phenomenon) 
                        for i, phenomenon in enumerate(phenomenon_types)]
        
        plt.legend(handles=legend_elements, title="Phenomenon Type")
        plt.title("t-SNE Visualization of NLSE Phenomenon Types - Direct Features", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_phenomenon_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            tsne_proxy_embedding[:, 0],
            tsne_proxy_embedding[:, 1],
            c=phenomenon_indices,
            cmap=cmap,
            s=80,
            alpha=0.8
        )
        
        plt.legend(handles=legend_elements, title="Phenomenon Type")
        plt.title("t-SNE Visualization of NLSE Phenomenon Types - Physical Proxies", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_proxy_phenomenon_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        pairwise_distances = np.zeros((len(phenomenon_types), len(phenomenon_types)))
        
        for i, type1 in enumerate(phenomenon_types):
            for j, type2 in enumerate(phenomenon_types):
                indices1 = [k for k, m in enumerate(self.metadata) if m["phenomenon_type"] == type1]
                indices2 = [k for k, m in enumerate(self.metadata) if m["phenomenon_type"] == type2]
                
                distances = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        distances.append(distance_matrix[idx1, idx2])
                
                pairwise_distances[i, j] = np.mean(distances)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(pairwise_distances, cmap='viridis')
        plt.colorbar(label='Mean Distance')
        
        plt.xticks(np.arange(len(phenomenon_types)), phenomenon_types, rotation=45, ha='right')
        plt.yticks(np.arange(len(phenomenon_types)), phenomenon_types)
        
        for i in range(len(phenomenon_types)):
            for j in range(len(phenomenon_types)):
                plt.text(j, i, f"{pairwise_distances[i, j]:.2f}", ha='center', va='center', 
                        color='white' if pairwise_distances[i, j] > np.mean(pairwise_distances) else 'black')
        
        plt.title("Pairwise Distances Between Phenomenon Types")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pairwise_distances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return all_samples, all_metadata, tsne_embedding, tsne_proxy_embedding


def analyze_nlse_phenomenon(phenomenon_type, output_dir=None, max_samples=50):
    from the_final_nlse_sampler import NLSEPhenomenonSampler
    
    nx = ny = 128
    L = 10.0
    
    if output_dir is None:
        output_dir = f"{phenomenon_type}_analysis"
    
    sampler = NLSEPhenomenonSampler(nx, ny, L)
    analyzer = NLSEPhenomenonDiversityAnalyzer(sampler, output_dir=output_dir)
    
    samples, metadata = analyzer.analyze_phenomenon_diversity(
        phenomenon_type=phenomenon_type,
        max_samples=max_samples
    )
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return analyzer, samples, metadata


if __name__ == "__main__":
    from the_final_nlse_sampler import NLSEPhenomenonSampler
    
    nx = ny = 128
    L = 10.0
    
    #mock_sampler = NLSEPhenomenonSampler(nx, ny, L)
    #mock_analyzer = NLSEPhenomenonDiversityAnalyzer(mock_sampler, output_dir="mock_dir") 
    #ps = mock_analyzer._prepare_parameter_space()
    #del mock_sampler
    #del mock_analyzer

    #for phenomenon in ps.keys():
    for phenomenon in ["topological_defect_network"]:
        sampler = NLSEPhenomenonSampler(nx, ny, L)  
        output_dir = f"{phenomenon}_diversity"
        analyzer = NLSEPhenomenonDiversityAnalyzer(sampler, output_dir=output_dir)
        ps = analyzer._prepare_parameter_space()
        analyzer.analyze_phenomenon_diversity(
            phenomenon_type=phenomenon,
            max_samples=100
        ) 
        for param in ps[phenomenon].keys(): 
            analyzer.analyze_parameter_influence(phenomenon, param) 

        del sampler
        del analyzer
