import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import uuid
from tqdm import tqdm
import itertools
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

from valid_spaces import get_parameter_spaces

class WaveSolutionMapper:
    def __init__(self, sampler, output_dir=None):
        self.sampler = sampler
        if output_dir is None:
            self.output_dir = f"samples_{str(uuid.uuid4())[:4]}"
        else:
            self.output_dir = output_dir
        self.samples = []
        self.metadata = []
        self.distance_matrix = None
        self.embeddings = {}

        os.makedirs(output_dir, exist_ok=True)

    def generate_sample(self, phenomenon_type, system_type=None, **params):
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
            warnings.warn(
                f"Error generating {phenomenon_type} with {system_type}: {e}")
            return None, None

    def map_solution_space(self, phenomena_types=None, n_samples_per_config=2,
                           max_configs_per_type=15, n_neighbors=15):
        if phenomena_types is None:
            phenomena_types = [
                "kink_solution", "kink_field", "breather_solution", "multi_breather_field",
                "ring_soliton", "multi_ring_state", "spiral_wave_field",
                "skyrmion_solution", "skyrmion_lattice", "grf_modulated_soliton_field",
            ]

        self.samples = []
        self.metadata = []

        parameter_spaces = get_parameter_spaces(self.sampler.L)

        for phenomenon_type in tqdm(
                phenomena_types, desc="Processing phenomenon types"):
            if phenomenon_type not in parameter_spaces:
                warnings.warn(
                    f"No parameter space defined for {phenomenon_type}. Skipping.")
                continue

            param_space = parameter_spaces[phenomenon_type].copy()

            system_types = param_space.pop("system_type", [None])

            param_keys = list(param_space.keys())
            param_values = [param_space[k] for k in param_keys]

            all_combinations = list(itertools.product(*param_values))

            if len(all_combinations) > max_configs_per_type:
                selected_indices = np.random.choice(
                    len(all_combinations),
                    size=max_configs_per_type,
                    replace=False
                )
                selected_combinations = [all_combinations[i]
                                         for i in selected_indices]
            else:
                selected_combinations = all_combinations

            for system_type in system_types:
                for combination in tqdm(selected_combinations,
                                        desc=f"{phenomenon_type} ({system_type})",
                                        leave=False):
                    params = {k: v for k, v in zip(param_keys, combination)}

                    for _ in range(n_samples_per_config):
                        sample, metadata = self.generate_sample(
                            phenomenon_type, system_type=system_type, **params
                        )

                        if sample is not None:
                            self.samples.append(sample)
                            self.metadata.append(metadata)

        print(
            f"Generated {len(self.samples)} samples across {len(phenomena_types)} phenomenon types")

        features = []
        for sample in tqdm(self.samples, desc="Extracting features"):
            u, v = sample
            feature_vector = np.concatenate([u.flatten(), v.flatten()])
            features.append(feature_vector)

        features_array = np.array(features)
        self.distance_matrix = squareform(
            pdist(features_array, metric='euclidean'))

        print("Computing embeddings...")

        n_neighbors = min(n_neighbors, len(self.samples) - 1)

        self.embeddings['tsne'] = TSNE(
            n_components=2,
            perplexity=n_neighbors,
            n_iter=2000,
            random_state=42
        ).fit_transform(self.distance_matrix)

        self.embeddings['umap'] = umap.UMAP(
            n_components=2,
            min_dist=0.1,
            n_neighbors=n_neighbors,
            random_state=42
        ).fit_transform(self.distance_matrix)

        return self.samples, self.metadata, self.distance_matrix, self.embeddings

    def calculate_physical_properties(self, field_pair):
        u, v = field_pair
        properties = {}

        dx = self.sampler.dx
        dy = self.sampler.dy
        cell_area = dx * dy

        properties['u_energy'] = np.sum(u**2) * cell_area
        properties['v_energy'] = np.sum(v**2) * cell_area
        properties['total_energy'] = properties['u_energy'] + \
            properties['v_energy']

        properties['u_peak'] = np.max(np.abs(u))
        properties['v_peak'] = np.max(np.abs(v))

        X, Y = self.sampler.X, self.sampler.Y

        u2_sum = np.sum(u**2)
        v2_sum = np.sum(v**2)

        if u2_sum > 0:
            u_centroid = np.array(
                [np.sum(X * u**2) / u2_sum, np.sum(Y * u**2) / u2_sum])
        else:
            u_centroid = np.array([0, 0])

        if v2_sum > 0:
            v_centroid = np.array(
                [np.sum(X * v**2) / v2_sum, np.sum(Y * v**2) / v2_sum])
        else:
            v_centroid = np.array([0, 0])

        properties['u_centroid'] = u_centroid
        properties['v_centroid'] = v_centroid

        if u2_sum > 0:
            u_radius_of_gyration = np.sqrt(
                np.sum(((X - u_centroid[0])**2 +
                       (Y - u_centroid[1])**2) * u**2) / u2_sum
            )
        else:
            u_radius_of_gyration = 0

        if v2_sum > 0:
            v_radius_of_gyration = np.sqrt(
                np.sum(((X - v_centroid[0])**2 +
                       (Y - v_centroid[1])**2) * v**2) / v2_sum
            )
        else:
            v_radius_of_gyration = 0

        properties['u_radius_of_gyration'] = u_radius_of_gyration
        properties['v_radius_of_gyration'] = v_radius_of_gyration

        u_laplacian = (
            (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) - 2 * u) / dx**2 +
            (np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 2 * u) / dy**2
        )

        v_laplacian = (
            (np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) - 2 * v) / dx**2 +
            (np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 2 * v) / dy**2
        )

        properties['u_laplacian_energy'] = np.sum(u_laplacian**2) * cell_area
        properties['v_laplacian_energy'] = np.sum(v_laplacian**2) * cell_area

        u_gradient_x = (np.roll(u, -1, axis=0) -
                        np.roll(u, 1, axis=0)) / (2 * dx)
        u_gradient_y = (np.roll(u, -1, axis=1) -
                        np.roll(u, 1, axis=1)) / (2 * dy)
        v_gradient_x = (np.roll(v, -1, axis=0) -
                        np.roll(v, 1, axis=0)) / (2 * dx)
        v_gradient_y = (np.roll(v, -1, axis=1) -
                        np.roll(v, 1, axis=1)) / (2 * dy)

        properties['u_gradient_energy'] = np.sum(
            u_gradient_x**2 + u_gradient_y**2) * cell_area
        properties['v_gradient_energy'] = np.sum(
            v_gradient_x**2 + v_gradient_y**2) * cell_area

        properties['topological_charge'] = np.sum(
            (u_gradient_x * v_gradient_y - u_gradient_y * v_gradient_x) /
            (2 * np.pi * (u**2 + v**2 + 1e-10))
        ) * cell_area

        u_fft = np.fft.fftshift(np.abs(np.fft.fft2(u)))
        v_fft = np.fft.fftshift(np.abs(np.fft.fft2(v)))

        kx = np.fft.fftshift(np.fft.fftfreq(u.shape[0], dx))
        ky = np.fft.fftshift(np.fft.fftfreq(u.shape[1], dy))
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K_mag = np.sqrt(KX**2 + KY**2)

        u_fft_sum = np.sum(u_fft)
        v_fft_sum = np.sum(v_fft)

        if u_fft_sum > 0:
            properties['u_spectral_radius'] = np.sqrt(
                np.sum(K_mag**2 * u_fft) / u_fft_sum)
        else:
            properties['u_spectral_radius'] = 0

        if v_fft_sum > 0:
            properties['v_spectral_radius'] = np.sqrt(
                np.sum(K_mag**2 * v_fft) / v_fft_sum)
        else:
            properties['v_spectral_radius'] = 0

        u_norm = np.sqrt(np.sum(u**2))
        v_norm = np.sqrt(np.sum(v**2))
        if u_norm > 0 and v_norm > 0:
            properties['uv_correlation'] = np.sum(u * v) / (u_norm * v_norm)
        else:
            properties['uv_correlation'] = 0

        feature_vector = np.array([
            properties['total_energy'],
            properties['u_peak'],
            properties['v_peak'],
            properties['u_radius_of_gyration'],
            properties['v_radius_of_gyration'],
            properties['u_laplacian_energy'],
            properties['v_laplacian_energy'],
            properties['u_gradient_energy'],
            properties['v_gradient_energy'],
            properties['u_spectral_radius'],
            properties['v_spectral_radius'],
            properties['uv_correlation'],
            properties['topological_charge']
        ])

        return properties, feature_vector

    def map_physical_properties(self, phenomena_types=None, n_samples_per_config=2,
                                max_configs_per_type=15, n_neighbors=15):
        if phenomena_types is None:
            phenomena_types = [
                "kink_solution", "kink_field", "breather_solution", "multi_breather_field",
                "ring_soliton", "multi_ring_state", "spiral_wave_field",
                "skyrmion_solution", "skyrmion_lattice", "grf_modulated_soliton_field",
            ]

        self.samples = []
        self.metadata = []

        parameter_spaces = get_parameter_spaces(self.sampler.L)

        for phenomenon_type in tqdm(
                phenomena_types, desc="Processing phenomenon types"):
            if phenomenon_type not in parameter_spaces:
                warnings.warn(
                    f"No parameter space defined for {phenomenon_type}. Skipping.")
                continue

            param_space = parameter_spaces[phenomenon_type].copy()

            system_types = param_space.pop("system_type", [None])

            param_keys = list(param_space.keys())
            param_values = [param_space[k] for k in param_keys]

            all_combinations = list(itertools.product(*param_values))

            if len(all_combinations) > max_configs_per_type:
                selected_indices = np.random.choice(
                    len(all_combinations),
                    size=max_configs_per_type,
                    replace=False
                )
                selected_combinations = [all_combinations[i]
                                         for i in selected_indices]
            else:
                selected_combinations = all_combinations

            for system_type in system_types:
                for combination in tqdm(selected_combinations,
                                        desc=f"{phenomenon_type} ({system_type})",
                                        leave=False):
                    params = {k: v for k, v in zip(param_keys, combination)}

                    for _ in range(n_samples_per_config):
                        sample, metadata = self.generate_sample(
                            phenomenon_type, system_type=system_type, **params
                        )

                        if sample is not None:
                            self.samples.append(sample)
                            self.metadata.append(metadata)

        print(
            f"Generated {len(self.samples)} samples across {len(phenomena_types)} phenomenon types")

        features = []
        for sample in tqdm(self.samples,
                           desc="Calculating physical properties"):
            _, feature_vector = self.calculate_physical_properties(sample)
            features.append(feature_vector)

        features_array = np.array(features)

        features_array = (features_array - np.mean(features_array,
                          axis=0)) / (np.std(features_array, axis=0) + 1e-10)

        self.distance_matrix = squareform(
            pdist(features_array, metric='euclidean'))

        print("Computing embeddings...")

        n_neighbors = min(n_neighbors, len(self.samples) - 1)

        self.embeddings['tsne'] = TSNE(
            n_components=2,
            perplexity=n_neighbors,
            n_iter=2000,
            random_state=42
        ).fit_transform(features_array)

        self.embeddings['umap'] = umap.UMAP(
            n_components=2,
            min_dist=0.1,
            n_neighbors=n_neighbors,
            random_state=42
        ).fit_transform(features_array)

        return self.samples, self.metadata, self.distance_matrix, self.embeddings

    def visualize_solution_space(self, embedding_type='tsne', color_by='phenomenon',
                                 title_prefix="", output_prefix=None):
        if embedding_type not in self.embeddings:
            raise ValueError(f"Embedding {embedding_type} not found")

        if output_prefix is None:
            output_prefix = f"wave_map_{len(self.samples)}"

        embedding = self.embeddings[embedding_type]

        phenomenon_types = sorted(
            set(m["phenomenon_type"] for m in self.metadata))
        n_phenomena = len(phenomenon_types)

        phenomenon_colors = plt.cm.tab20(np.linspace(0, 1, n_phenomena))
        phenomenon_cmap = ListedColormap(phenomenon_colors)

        system_types = sorted(
            set(m["system_type"] for m in self.metadata if m["system_type"] is not None))
        n_systems = len(system_types)
        system_colors = plt.cm.viridis(np.linspace(0, 1, n_systems))
        system_cmap = ListedColormap(system_colors)

        fig, ax = plt.subplots(figsize=(14, 10))

        if color_by == 'phenomenon':
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=[phenomenon_types.index(m["phenomenon_type"])
                   for m in self.metadata],
                cmap=phenomenon_cmap,
                s=30,
                alpha=0.7
            )

            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=phenomenon_colors[i],
                           markersize=10, label=p) for i, p in enumerate(phenomenon_types)
            ]

            legend_title = 'Phenomenon Type'

        elif color_by == 'system':
            system_indices = []
            for m in self.metadata:
                if m["system_type"] is not None:
                    system_indices.append(system_types.index(m["system_type"]))
                else:
                    system_indices.append(-1)

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=[idx if idx >= 0 else 0 for idx in system_indices],
                cmap=system_cmap,
                s=30,
                alpha=0.7
            )

            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=system_colors[i],
                           markersize=10, label=s) for i, s in enumerate(system_types)
            ]

            legend_title = 'System Type'

        embedding_name = embedding_type.upper()
        ax.set_title(
            f"{title_prefix}{embedding_name} Visualization of Wave Solutions",
            fontsize=16)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        ax.legend(
            handles=legend_handles,
            loc='upper right',
            title=legend_title,
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{output_prefix}_{embedding_type}_{color_by}.png",
            dpi=300,
            bbox_inches='tight')
        plt.close()

        return fig, ax

    def create_sample_gallery(self, output_prefix=None, samples_per_type=2):
        if output_prefix is None:
            output_prefix = f"wave_gallery_{len(self.samples)}"

        phenomenon_types = sorted(
            set(m["phenomenon_type"] for m in self.metadata))
        n_rows = len(phenomenon_types)
        n_cols = samples_per_type * 3

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(
                n_cols * 2, n_rows * 2))
        if n_rows == 1:
            axes = np.array([axes])

        cmaps = ['viridis', 'RdBu_r', 'RdBu_r']

        for row, phenomenon in enumerate(phenomenon_types):
            indices = [
                i for i, m in enumerate(
                    self.metadata) if m["phenomenon_type"] == phenomenon]

            if not indices:
                continue

            selected_indices = np.random.choice(indices, size=min(
                samples_per_type, len(indices)), replace=False)

            for i, idx in enumerate(selected_indices):
                u, v = self.samples[idx]

                col_offset = i * 3

                axes[row, col_offset].imshow(u, cmap=cmaps[0])
                axes[row, col_offset].set_title("u field")

                axes[row, col_offset + 1].imshow(v, cmap=cmaps[1])
                axes[row, col_offset + 1].set_title("v field")

                overlay = np.zeros((u.shape[0], u.shape[1], 4))
                overlay[..., 0] = np.clip((u + 1) / 2, 0, 1)
                overlay[..., 2] = np.clip((v + 1) / 2, 0, 1)
                overlay[..., 3] = 0.7

                axes[row, col_offset + 2].imshow(overlay)
                axes[row, col_offset + 2].set_title("overlay")

            for j in range(len(selected_indices) * 3, n_cols):
                axes[row, j].axis('off')

            axes[row, 0].set_ylabel(
                phenomenon, fontsize=12, rotation=90, va='center', ha='right')

        for row in range(n_rows):
            for col in range(n_cols):
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{output_prefix}_gallery.png",
            dpi=300,
            bbox_inches='tight')
        plt.close()

        return fig, axes

    def create_system_comparison(self, phenomenon_type, output_prefix=None):
        if output_prefix is None:
            output_prefix = f"wave_system_comparison_{phenomenon_type}"

        system_types = sorted(set(m["system_type"] for m in self.metadata
                              if m["phenomenon_type"] == phenomenon_type and m["system_type"] is not None))

        if len(system_types) < 2:
            print(f"Not enough system types for {phenomenon_type}")
            return None, None

        n_systems = len(system_types)
        n_cols = 3

        fig, axes = plt.subplots(
            n_systems, n_cols, figsize=(
                n_cols * 3, n_systems * 3))

        if n_systems == 1:
            axes = np.array([axes])

        cmaps = ['viridis', 'RdBu_r', 'RdBu_r']

        for i, system_type in enumerate(system_types):
            indices = [j for j, m in enumerate(self.metadata)
                       if m["phenomenon_type"] == phenomenon_type and m["system_type"] == system_type]

            if not indices:
                continue

            idx = indices[0]
            u, v = self.samples[idx]

            axes[i, 0].imshow(u, cmap=cmaps[0])
            axes[i, 0].set_title(f"{system_type} - u field")

            axes[i, 1].imshow(v, cmap=cmaps[1])
            axes[i, 1].set_title(f"{system_type} - v field")

            overlay = np.zeros((u.shape[0], u.shape[1], 4))
            overlay[..., 0] = np.clip((u + 1) / 2, 0, 1)
            overlay[..., 2] = np.clip((v + 1) / 2, 0, 1)
            overlay[..., 3] = 0.7

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"{system_type} - overlay")

        for i in range(n_systems):
            for j in range(n_cols):
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{output_prefix}.png",
            dpi=300,
            bbox_inches='tight')
        plt.close()

        return fig, axes

    def analyze_physical_properties(self, output_prefix=None):
        if not self.samples:
            raise ValueError("No samples available")

        if output_prefix is None:
            output_prefix = f"wave_physics_{len(self.samples)}"

        physical_properties = []

        for sample in tqdm(self.samples, desc="Analyzing physical properties"):
            props, _ = self.calculate_physical_properties(sample)
            physical_properties.append(props)

        phenomenon_types = sorted(
            set(m["phenomenon_type"] for m in self.metadata))

        features_to_plot = [
            ('total_energy', 'Energy'),
            ('topological_charge', 'Topological Charge'),
            ('u_radius_of_gyration', 'Spatial Extent'),
            ('u_spectral_radius', 'Spectral Width'),
            ('uv_correlation', 'u-v Correlation'),
            ('u_gradient_energy', 'Gradient Energy')
        ]

        n_plots = len(features_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 5))

        for i, (feature_name, feature_label) in enumerate(features_to_plot):
            boxplot_data = []
            boxplot_labels = []

            for phenomenon in phenomenon_types:
                phenomenon_indices = [
                    j for j, m in enumerate(
                        self.metadata) if m["phenomenon_type"] == phenomenon]

                if phenomenon_indices:
                    feature_values = [
                        physical_properties[j][feature_name] for j in phenomenon_indices]
                    boxplot_data.append(feature_values)
                    boxplot_labels.append(phenomenon)

            axes[i].boxplot(boxplot_data, vert=True, patch_artist=True)
            axes[i].set_xticklabels(boxplot_labels, rotation=90)
            axes[i].set_title(feature_label)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{output_prefix}_analysis.png",
            dpi=300,
            bbox_inches='tight')
        plt.close()

        return fig, axes

    def run_comprehensive_analysis(self, phenomena_types=None, n_samples_per_config=1,
                                   max_configs_per_type=10, map_direct=True, map_physical=True,
                                   samples_per_type=2):
        run_id = str(uuid.uuid4())[:4]
        output_prefix = f"wave_analysis_{run_id}"

        if map_direct:
            print(f"\n{'='*80}\nMapping direct field values...\n{'='*80}\n")
            self.map_solution_space(
                phenomena_types=phenomena_types,
                n_samples_per_config=n_samples_per_config,
                max_configs_per_type=max_configs_per_type
            )

            print("Creating visualizations for direct mapping...")
            self.visualize_solution_space(embedding_type='tsne', color_by='phenomenon',
                                          output_prefix=f"{output_prefix}_direct")
            self.visualize_solution_space(embedding_type='tsne', color_by='system',
                                          output_prefix=f"{output_prefix}_direct")
            self.visualize_solution_space(embedding_type='umap', color_by='phenomenon',
                                          output_prefix=f"{output_prefix}_direct")
            self.visualize_solution_space(embedding_type='umap', color_by='system',
                                          output_prefix=f"{output_prefix}_direct")

            self.create_sample_gallery(output_prefix=f"{output_prefix}_direct",
                                       samples_per_type=samples_per_type)

        if map_physical:
            print(f"\n{'='*80}\nMapping physical properties...\n{'='*80}\n")
            self.map_physical_properties(
                phenomena_types=phenomena_types,
                n_samples_per_config=n_samples_per_config,
                max_configs_per_type=max_configs_per_type
            )

            print("Creating visualizations for physical property mapping...")
            self.visualize_solution_space(embedding_type='tsne', color_by='phenomenon',
                                          title_prefix="Physical Properties - ",
                                          output_prefix=f"{output_prefix}_physical")
            self.visualize_solution_space(embedding_type='tsne', color_by='system',
                                          title_prefix="Physical Properties - ",
                                          output_prefix=f"{output_prefix}_physical")
            self.visualize_solution_space(embedding_type='umap', color_by='phenomenon',
                                          title_prefix="Physical Properties - ",
                                          output_prefix=f"{output_prefix}_physical")
            self.visualize_solution_space(embedding_type='umap', color_by='system',
                                          title_prefix="Physical Properties - ",
                                          output_prefix=f"{output_prefix}_physical")

            self.analyze_physical_properties(
                output_prefix=f"{output_prefix}_physical")

        phenomena_to_compare = set(m["phenomenon_type"] for m in self.metadata)
        for phenomenon_type in phenomena_to_compare:
            self.create_system_comparison(phenomenon_type,
                                          output_prefix=f"{output_prefix}_compare_{phenomenon_type}")

        return output_prefix


def run_wave_mapping(
    sampler,
    output_dir=None,
    phenomena_types=None,
    n_samples_per_config=1,
    max_configs_per_type=10,
    map_direct=True,
    map_physical=True,
    samples_per_type=2,
    seed=42
):
    np.random.seed(seed)

    if output_dir is None:
        run_id = str(uuid.uuid4())[:4]
        output_dir = f"wave_solution_mapping_{run_id}"

    if phenomena_types is None:
        phenomena_types = [
            "kink_solution", "kink_array_field",
            "kink_field", "breather_solution", "multi_breather_field",
            "ring_soliton", "spiral_wave_field", "skyrmion_solution",
            "skyrmion_lattice", "skyrmion_like_field",
            "multi_ring_state", "q_ball_solution",
            "multi_q_ball", "colliding_rings", "grf_modulated_soliton_field"
        ]

    mapper = WaveSolutionMapper(sampler, output_dir=output_dir)

    output_prefix = mapper.run_comprehensive_analysis(
        phenomena_types=phenomena_types,
        n_samples_per_config=n_samples_per_config,
        max_configs_per_type=max_configs_per_type,
        map_direct=map_direct,
        map_physical=map_physical,
        samples_per_type=samples_per_type
    )

    return mapper


if __name__ == '__main__':
    from real_sampler import RealWaveSampler
    nx = ny = 128
    L = 10.0

    seed = np.random.randint(1 << 10)

    sampler = RealWaveSampler(nx=nx, ny=ny, L=L)
    run_wave_mapping(
        sampler=sampler,
        n_samples_per_config=5,
        max_configs_per_type=10,
        map_direct=True,
        map_physical=True,
        samples_per_type=4,
        seed=seed
    )
