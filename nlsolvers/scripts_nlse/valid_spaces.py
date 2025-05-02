# This file describes exemplary parameter spaces for different phenomena.
# These may very well be adapted, transformed and changed. Currently though,
# it's suggested to not diverge too heavily from them as only those have been
# tested so far.

import numpy as np


def get_parameter_spaces():
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
        "spectrum_slope": [-5 / 3, -1.0, -2.0],
        "apply_envelope": [False, True]
    }

    parameter_spaces["chaotic"] = {
        "amplitude": [0.5, 1.0, 1.5],
        "spectral_exponent": [-0.5, -1.0, -1.5],
        "coherent_structures": [True, False],
        "n_structures": [1, 3, 5],
        "apply_envelope": [False, True]
    }

    parameter_spaces["vortex_lattice"] = {
        "amplitude": [0.5, 1.0, 1.5],
        "n_vortices": [3, 5, 7, 9],
        "arrangement": ["square", "triangular", "circular", "random"],
        "charge_distribution": ["alternating", "same", "random"],
        "apply_envelope": [False, True]
    }

    parameter_spaces["ring_soliton"] = {
        "amplitude": [0.5, 1.0, 1.5],
        "radius": [1.0, 2.0, 3.0],
        "width": [0.3, 0.5, 0.8],
        "modulation_type": ["none", "azimuthal", "radial"],
        "modulation_strength": [0.0, 0.2, 0.4],
        "modulation_mode": [0, 1, 2],
        "apply_envelope": [False, True]
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
        "ambient_direction": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
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
        "apply_envelope": [False, True]
    }

    parameter_spaces["topological_defect_network"] = {
        "amplitude": [0.5, 1.0, 1.5],
        "n_defects": [5, 10, 15],
        "defect_types": [["vortex", "antivortex"]],
        "spatial_distribution": ["poisson", "inhibition", "cluster"],
        "temperature": [0.5, 1.0, 1.5],
        "core_size": [0.3, 0.5, 0.8],
        "interaction_strength": [0.5, 0.7, 1.0],
        "apply_envelope": [False, True]
    }

    parameter_spaces["akhmediev_breather"] = {
        "amplitude": [.5, 1., 1.5],
        "modulation_frequency": [1., np.pi],
        "growth_rate": [1e-2, .1, 0.49],
        "breather_phase": ['compressed', 'growing', 'decaying'],
        "apply_envelope": [True, False],
        "t_param": [None, 1e-1, 2 / 3]

    }

    parameter_spaces["self_similar_pattern"] = {
        "amplitude": [0.5, 1.],
        "scale_factor": [1.5, 2.],
        "intensity_scaling": [.33, .6, .9],
        "num_iterations": [1, 5, 10],
        "base_pattern": ["ring", "vortex"],
    }

    return parameter_spaces


def get_parameter_spaces_3d():
    parameter_spaces = {}

    parameter_spaces["multi_soliton_state"] = {
        "system_type": ["cubic"],
        "amplitude_range": [(0.5, 1.0), (0.8, 1.2), (1.0, 1.5), (1.5, 2.0)],
        "width_range": [(0.5, 1.0), (0.8, 1.2), (1.0, 1.5), (1.5, 2.0)],
        "position_variance": [0.5, 1.0, 1.5, 2.0],
        "velocity_scale": [0.0, 0.2, 0.5, 1.0, 1.5],
        "phase_pattern": ["random",
            "alternating", "synchronized", "vortex", "3d_vortex",
            "radial", "spiral", "z_dependent", "partial_coherence"],
        "arrangement": ["linear", "planar_grid", "circular",
            "spherical", "random", "lattice", "hierarchical"],
        "separation": [3.0, 5.0, 7.0, 10.0],
        "apply_envelope": [False],
        "envelope_width": [0.5, 0.7, 0.9],
        "Lambda_range": [(0.02, 0.08), (0.04, 0.14), (0.1, 0.2)],
        "coherence": [0.2, 0.5, 0.8, 1.0],
        "interaction_strength": [0.3, 0.5, 0.7, 1.0],
        "cluster_levels": [1, 2, 3, 4],
        "order_range": [(1, 2), (1, 3), (2, 3)],
        "chirp_range": [(-0.2, -0.1), (-0.1, 0.1), (0.0, 0.1), (0.1, 0.2)],
        "aspect_ratio_x_range": [(1.0, 1.0), (1.0, 1.5), (1.5, 2.0)],
        "aspect_ratio_y_range": [(1.0, 1.0), (1.0, 1.5), (1.5, 2.0)],
        "phase_value": [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
    }

    # parameter_spaces["multi_soliton_state"] = {
    #     "system_type": ["cubic"],
    #     "amplitude_range": [(0.5, 0.8)],
    #     "width_range": [(1.5, 2.5)],    
    #     "arrangement": ["planar_grid", "linear", "random"],
    #     "position_variance": [0.5], 
    #     "separation": [7.0, 10.0],
    #     "velocity_scale": [0.0],
    #     "phase_pattern": ["synchronized", "random"],
    #     "phase_value": [0.0],
    #     "order_range": [(1, 2)],
    #     "chirp_range": [(0.0, 0.0)],
    #     "aspect_ratio_x_range": [(1.0, 1.0)],
    #     "aspect_ratio_y_range": [(1.0, 1.0)]
    # }



    parameter_spaces["skyrmion_tube"] = {
        "amplitude_range": [(0.5, 1.0), (0.8, 1.5), (1.0, 2.0), (1.5, 2.5)],
        "radius_range": [(0.5, 1.5), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)],
        "width_range": [(0.3, 0.8), (0.5, 1.5), (1.0, 2.0), (1.5, 2.5)],
        "position_variance": [0.3, 0.5, 1.0, 1.5],
        "phase_range": [(0, np.pi), (0, 2 * np.pi), (np.pi / 2, 3 * np.pi / 2)],
        "winding_range": [(1, 2), (1, 3), (2, 4)],
        "k_z_range": [(0.1, 0.5), (0.3, 0.8), (0.5, 1.0), (0.8, 1.5)],
        "velocity_scale": [0.0, 0.1, 0.3, 0.5, 0.8],
        "chirp_range": [(-0.2, -0.1), (-0.1, 0.1), (0.0, 0.1), (0.1, 0.2)],
        "tube_count_range": [(1, 3), (2, 5), (3, 8)],
        "apply_envelope": [False],
        "envelope_width": [0.5, 0.7, 0.9],
        "tube_arrangement": ["random", "circular", "linear", "lattice"],
        "interaction_strength": [0.3, 0.5, 0.7, 1.0],
        "deformation_factor": [0.0, 0.1, 0.2, 0.3, 0.5]
    }

    return parameter_spaces
