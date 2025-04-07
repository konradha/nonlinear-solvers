import argparse
import sys
from compare_utils_nlse import NlseComparer, SYSTEM_CONFIG_NLSE

def main():
    parser = argparse.ArgumentParser(description="Compare two different NLSE integrators.")
    nlse_choices = list(SYSTEM_CONFIG_NLSE.keys())
    default_system = 'NLSE_cubic' if 'NLSE_cubic' in nlse_choices else nlse_choices[0]

    parser.add_argument("--system-type", type=str, default=default_system, choices=nlse_choices,
                        help="Type of the NLSE system.")
    parser.add_argument("--exe1", type=str, required=True, help="Path to the first integrator executable.")
    parser.add_argument("--name1", type=str, required=True, help="Name for the first integrator (e.g., SS2, Host).")
    parser.add_argument("--exe2", type=str, required=True, help="Path to the second integrator executable.")
    parser.add_argument("--name2", type=str, required=True, help="Name for the second integrator (e.g., Gautschi, Device).")

    parser.add_argument("--nx", type=int, default=128, help="Number of grid points in x.")
    parser.add_argument("--ny", type=int, default=128, help="Number of grid points in y.")
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x.")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y.")
    parser.add_argument("--T", type=float, default=1.0, help="Total simulation time.")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time steps.")
    parser.add_argument("--num-snapshots", type=int, default=100, help="Number of snapshots.")
    parser.add_argument("--ic-type", type=str, default="spectral", help="Initial condition type (passed to sampler).")

    temp_args, _ = parser.parse_known_args()
    selected_config = SYSTEM_CONFIG_NLSE.get(temp_args.system_type, {})

    parser.add_argument("--m-value", type=float, default=1.0, help="Coefficient 'm'.")
    if 'sigma1' in selected_config.get('params', []):
        parser.add_argument("--sigma1", type=float, help="Coefficient sigma1 for NLSE-CQ.")
    if 'sigma2' in selected_config.get('params', []):
        parser.add_argument("--sigma2", type=float, help="Coefficient sigma2 for NLSE-CQ.")
    if 'kappa' in selected_config.get('params', []):
         parser.add_argument("--kappa", type=float, help="Coefficient kappa for NLSE-Sat.")

    parser.add_argument("--output-dir", type=str, default="compare_nlse_results", help="Dir for outputs.")
    parser.add_argument("--keep-temps", action="store_true", help="Keep temporary .npy files.")

    parsed_args = parser.parse_args()

    final_selected_config = SYSTEM_CONFIG_NLSE.get(parsed_args.system_type)
    if final_selected_config:
        for param_name in final_selected_config.get('params', []):
            if param_name != 'm_value' and param_name in parsed_args and getattr(parsed_args, param_name, None) is None:
                parser.error(f"--{param_name} is required for system-type '{parsed_args.system_type}'")

    comparer = NlseComparer(parsed_args)
    comparer.execute()

if __name__ == "__main__":
    main()
