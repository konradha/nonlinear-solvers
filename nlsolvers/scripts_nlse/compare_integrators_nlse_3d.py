import argparse
import sys

from compare_utils_nlse_3d import NlseComparer3d, SYSTEM_CONFIG_NLSE
from spatial_amplification_3d import (
    make_grid,
    create_constant_m
)

from nonlinearity_profiles_3d import highlight_profiles

def main():
    parser = argparse.ArgumentParser(description="Compare two different NLSE integrators (3D case)")
    nlse_choices = list(SYSTEM_CONFIG_NLSE.keys())
    default_system = 'NLSE_cubic' if 'NLSE_cubic' in nlse_choices else nlse_choices[0]

    parser.add_argument("--system-type", type=str, default=default_system, choices=nlse_choices,
                        help="Type of the NLSE system.")
    parser.add_argument("--exe1", type=str, required=True, help="Path to the first integrator executable.")
    parser.add_argument("--name1", type=str, required=True, help="Name for the first integrator (e.g., SS2, Host).")
    parser.add_argument("--exe2", type=str, required=True, help="Path to the second integrator executable.")
    parser.add_argument("--name2", type=str, required=True, help="Name for the second integrator (e.g., Gautschi, Device).")

    parser.add_argument("--nx", type=int, default=80, help="Number of grid points in x.")
    parser.add_argument("--ny", type=int, default=80, help="Number of grid points in y.")
    parser.add_argument("--nz", type=int, default=80, help="Number of grid points in z.")
   
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x.")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y.")
    parser.add_argument("--Lz", type=float, default=10.0, help="Domain half-width in z.")

    parser.add_argument("--T", type=float, default=1.0, help="Total simulation time.")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time steps.")
    parser.add_argument("--num-snapshots", type=int, default=100, help="Number of snapshots.")
    parser.add_argument("--ic-type", type=str, default="multi_soliton_state", help="Initial condition type (passed to sampler).")

    parser.add_argument("--c-m-pair", type=str, choices=['optimal',
        'resonant_cavity', 'focusing_soliton', 'sharp_interfaces',
        'multi_scale', 'fractal_nonlinear', 'waveguide', 'grf_threshold',
        'anisotropic', 'maybe_blowup'], default=None,
        help="Choose pair between c(x,y,z) and m(x,y,z) (if None will yield constant fields")

    parser.add_argument("--m-value", type=float, default=1.0, help="Coefficient 'm'.")
    parser.add_argument("--output-dir", type=str, required=True, help="Dir for outputs.")
    parser.add_argument("--keep-temps", action="store_true", help="Keep temporary .npy files.")

    parsed_args = parser.parse_args()
    selected_config = SYSTEM_CONFIG_NLSE.get(default_system, {})

    final_selected_config = SYSTEM_CONFIG_NLSE.get(parsed_args.system_type)
    if final_selected_config:
        for param_name in final_selected_config.get('params', []):
            if param_name != 'm_value' and param_name in parsed_args and getattr(parsed_args, param_name, None) is None:
                parser.error(f"--{param_name} is required for system-type '{parsed_args.system_type}'")

    comparer = NlseComparer3d(parsed_args)
    comparer.execute()

if __name__ == "__main__":
    main()
