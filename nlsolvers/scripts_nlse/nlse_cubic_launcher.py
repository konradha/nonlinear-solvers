import argparse
import os
import time
import uuid
import datetime
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from spatial_amplification import (
    create_constant_m,
    create_periodic_boxes,
    create_periodic_gaussians,
    create_grf,
    create_wavelet_modulated_grf,
    scale_m_to_range
)


from downsampling import downsample_fft, downsample_interpolation
from nlse_sampler import NLSEPhenomenonSampler
from visualization import animate_simulation
from valid_spaces import get_parameter_spaces


class NLSELauncher:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.setup_directories()
        self.configure_grid()
        # restriction to play nice with hand-rolled C++ integrators
        assert args.nx == args.ny and np.abs(args.Lx - args.Ly) < 1e-8
        self.sampler = NLSEPhenomenonSampler(args.nx, args.ny, args.Lx)

    def setup_directories(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.ic_dir = self.output_dir / "initial_conditions"
        self.ic_dir.mkdir(exist_ok=True)

        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(exist_ok=True)

        self.focusing_dir = self.output_dir / "focusing"
        self.focusing_dir.mkdir(exist_ok=True)

        self.anisotropy_dir = self.output_dir / "anisotropy"
        self.anisotropy_dir.mkdir(exist_ok=True)

        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.h5_dir = self.output_dir / "hdf5"
        self.h5_dir.mkdir(exist_ok=True)

        self.save_parameter_file()

    def save_parameter_file(self):
        params_file = self.output_dir / f"params_{self.run_id}.txt"
        with open(params_file, "w") as f:
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Grid: {self.args.nx}x{self.args.ny}\n")
            f.write(f"Domain: {self.args.Lx}x{self.args.Ly}\n")
            f.write(
                f"Time: T={self.args.T}, steps={self.args.nt}, snapshots={self.args.snapshots}\n")
            f.write(f"Phenomenon: {self.args.phenomenon}\n")
            f.write(f"Amplification: {self.args.m_type}\n")
            f.write(f"Executable: {self.args.exe}\n")

    def configure_grid(self):
        self.x = np.linspace(-self.args.Lx, self.args.Lx, self.args.nx)
        self.y = np.linspace(-self.args.Ly, self.args.Ly, self.args.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.dx = 2 * self.args.Lx / (self.args.nx - 1)
        self.dy = 2 * self.args.Ly / (self.args.ny - 1)
        self.dV = self.dx * self.dy

    def generate_initial_condition(self, run_idx, phenomenon_params=None):
        if phenomenon_params is None:
            phenomenon_params = self.sample_phenomenon_params()

        u0 = self.sampler.generate_ensemble(
            self.args.phenomenon,
            n_samples=1,
            **phenomenon_params
        )

        print(np.min(np.abs(u0 ** 2)))

        if isinstance(u0, torch.Tensor):
            u0 = u0.detach().numpy()

        # scale to max amplitude 1 -- better than setting norm(u0) = 1
        # which makes for more difficult scaling of samplers (both u0 and m)
        u0 = u0 / np.max(np.abs(u0))

        return u0, phenomenon_params

    def sample_phenomenon_params(self):
        # TODO: enable setting hard params via args
        parameter_spaces = get_parameter_spaces()
        if self.args.phenomenon in parameter_spaces:
            space = parameter_spaces[self.args.phenomenon]
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
            raise ValueError(
                f"Unknown phenomenon type: {self.args.phenomenon}")

    def generate_spatial_amplification(self, run_idx):
        if self.args.m_type == "constant":
            m = create_constant_m(self.X, self.Y, value=self.args.m_mean)
        elif self.args.m_type == "periodic_gaussians":
            m = create_periodic_gaussians(
                self.args.nx,
                self.args.Lx,
                factor=self.args.m_factor,
                num_gaussians_per_dim=self.args.m_boxes_per_dim,
                sigma=self.args.m_box_length,
                wall_dist=self.args.m_wall_dist
            )
        elif self.args.m_type == "periodic_boxes":
            m = create_periodic_boxes(
                self.args.nx,
                self.args.Lx,
                factor=self.args.m_factor,
                num_boxes_per_dim=self.args.m_boxes_per_dim,
                box_length=self.args.m_box_length,
                wall_dist=self.args.m_wall_dist
            )
        elif self.args.m_type == "grf":
            m = create_grf(
                self.args.nx,
                self.args.ny,
                self.args.Lx,
                self.args.Ly,
                mean=self.args.m_mean,
                std=self.args.m_std,
                scale=self.args.m_scale
            )

        elif self.args.m_type == "wavelet_grf":
            m = create_wavelet_modulated_grf(
                self.args.nx,
                self.args.ny,
                self.args.Lx,
                self.args.Ly,
                wavelet_scale=self.args.m_wavelet_scale,
                grf_scale=self.args.m_scale,
                mean=self.args.m_mean,
                std=self.args.m_std
            )
        else:
            raise ValueError(f"Unknown m_type: {self.args.m_type}")

        return m.astype(np.float64) # safety first

    def generate_anisotropy(self, run_idx):
        c = np.ones((self.args.nx, self.args.ny), dtype=np.float64)
        return c

    def run_simulation(self, run_idx, u0, m, c):
        ic_file = self.ic_dir / f"ic_{self.run_id}_{run_idx:04d}.npy"
        np.save(ic_file, u0)

        m_file = self.focusing_dir / f"m_{self.run_id}_{run_idx:04d}.npy"
        np.save(m_file, m)

        c_file = self.anisotropy_dir / f"c_{self.run_id}_{run_idx:04d}.npy"
        np.save(c_file, m)

        traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"

        exe_path = Path(self.args.exe)
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable {self.args.exe} not found")

        cmd = [
            str(exe_path),
            str(self.args.nx),
            str(self.args.ny),
            str(self.args.Lx),
            str(self.args.Ly),
            str(ic_file),
            str(traj_file),
            str(self.args.T),
            str(self.args.nt),
            str(self.args.snapshots),
            str(m_file),
            str(c_file)
        ]

        start_time = time.time()
        import subprocess
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")

        end_time = time.time()
        walltime = end_time - start_time

        traj_data = np.load(traj_file)
        return traj_data, walltime

    def downsample_trajectory(self, traj_data):
        if self.args.dr_strategy == 'none':
            return traj_data
        elif self.args.dr_strategy == 'FFT':
            return downsample_fft(traj_data, target_shape=(
                self.args.dr_x, self.args.dr_y))
        elif self.args.dr_strategy == 'interpolation':
            return downsample_interpolation(
                traj_data,
                target_shape=(self.args.dr_x, self.args.dr_y),
                Lx=self.args.Lx,
                Ly=self.args.Ly
            )
        else:
            raise ValueError(
                f"Unknown downsampling strategy: {self.args.dr_strategy}")

    def save_to_hdf5(self, run_idx, u0, m, c, traj_data,
                     phenomenon_params, elapsed_time):
        h5_file = self.h5_dir / f"run_{self.run_id}_{run_idx:04d}.h5"
        with h5py.File(h5_file, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['problem_type'] = 'nlse_cubic'
            meta.attrs['boundary_condition'] = 'noflux'
            meta.attrs['run_id'] = self.run_id
            meta.attrs['run_index'] = run_idx
            meta.attrs['timestamp'] = str(datetime.datetime.now())
            meta.attrs['elapsed_time'] = elapsed_time
            meta.attrs['phenomenon'] = self.args.phenomenon

            for key, value in phenomenon_params.items():
                if isinstance(value, (list, tuple)) and len(value) <= 10:
                    meta.attrs[f'phenomenon_{key}'] = str(value)
                else:
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

            m_grp = f.create_group('focusing')
            m_grp.attrs['type'] = self.args.m_type
            if self.args.m_type in ["grf", "wavelet_grf"]:
                m_grp.attrs['mean'] = self.args.m_mean
                m_grp.attrs['std'] = self.args.m_std
                m_grp.attrs['scale'] = self.args.m_scale
            m_grp.create_dataset('m', data=m)
            f.create_dataset('c', data=c)
            f.create_dataset('u', data=traj_data)
            f.create_dataset('X', data=self.X)
            f.create_dataset('Y', data=self.Y)

        return h5_file

    def create_visualization(self, run_idx, traj_data,
                             m, phenomenon_params, walltime):
        animation_title = self.generate_animation_title(
            phenomenon_params,
            walltime
        )

        animation_output = self.traj_dir / f"{self.run_id}_{run_idx:04d}.mp4"
        animate_simulation(
            self.X,
            self.Y,
            traj_data,
            self.args.snapshots,
            animation_output,
            is_complex=True,
            title=animation_title
        )

    def generate_animation_title(self, phenomenon_params, walltime):
        cuda_avail = torch.cuda.is_available()
        cuda_string = "device (presumably)" if cuda_avail else "host"

        m_string = "Constant m=1" if self.args.m_type == "constant" else \
            "Periodic Gaussians" if self.args.m_type == "periodic_gaussians" else \
            f"Periodic Boxes" if self.args.m_type == "periodic_boxes" else \
            f"GRF: s={self.args.m_scale:.2f}, μ={self.args.m_mean:.2f}, σ={self.args.m_std:.2f}" if self.args.m_type == "grf" else \
            f"Wavelet GRF: s={self.args.m_scale:.2f}, μ={self.args.m_mean:.2f}, σ={self.args.m_std:.2f}, wavelet_scale={self.args.m_wavelet_scale:.2f}"

        params_str = ", ".join([f"{k}={v}" for k, v in phenomenon_params.items(
        ) if not isinstance(v, (list, tuple))])
        if len(params_str) > 100:
            params_str = params_str[:97] + "..."

        return (
            f"NLSE iu_t + Δu + m(x,y)|u|²u = 0\n"
            f"{self.args.phenomenon}, m: {m_string}\n"
            f"domain: [0, T={self.args.T}] x [-{self.args.Lx:.2f}, {self.args.Lx:.2f}] x [-{self.args.Ly:.2f}, {self.args.Ly:.2f}]\n"
            f"resolution n_t={self.args.nt}, n_x={self.args.nx}, n_y={self.args.ny}\n"
            f"downsampled to: {self.args.dr_x}x{self.args.dr_y} using '{self.args.dr_strategy}'\n"
            f"samples collected: {self.args.snapshots}, walltime={walltime:.2f} seconds on {cuda_string}\n\n\n"
        )

    def cleanup(self, run_idx):
        if self.args.delete_intermediates:
            ic_file = self.ic_dir / f"ic_{self.run_id}_{run_idx:04d}.npy"
            m_file = self.focusing_dir / f"m_{self.run_id}_{run_idx:04d}.npy"
            traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"

            for file in [ic_file, m_file, traj_file]:
                if file.exists():
                    os.unlink(file)

    def run(self):
        for i in range(self.args.num_runs):
            try:
                u0, phenomenon_params = self.generate_initial_condition(i)
                print(f"Run {i+1}/{self.args.num_runs}", phenomenon_params)
                m = self.generate_spatial_amplification(i)
                c = self.generate_anisotropy(i)
                traj_data, walltime = self.run_simulation(i, u0, m, c)
                traj_data = self.downsample_trajectory(traj_data)
                self.save_to_hdf5(
                    i, u0, m, c, traj_data, phenomenon_params, walltime)

                if self.args.visualize:
                    self.create_visualization(
                        i, traj_data, m, phenomenon_params, walltime)
                self.cleanup(i)
                print(f"Walltime: {walltime:.4f} seconds")

            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                continue

        if self.args.delete_intermediates:
            params_file = self.output_dir / f"params_{self.run_id}.txt"
            if params_file.exists():
                os.unlink(params_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced NLSE cubic nonlinearity solver launcher")

    parser.add_argument("--phenomenon", type=str, default="multi_soliton",
                        choices=["multi_soliton", "spectral", "chaotic", "vortex_lattice",
                                 "dark_soliton", "multi_ring", "solitary_wave_with_ambient",
                                 "logarithmic_singularity_adapted", "turbulent_condensate",
                                 "topological_defect_network", "akhmediev_breather",
                                 "self_similar_pattern"],
                        required=True,
                        help="Phenomenon type to simulate")

    parser.add_argument("--m_type", type=str, default="constant",
                        choices=["constant", "periodic_boxes", "periodic_gaussians",
                                 "grf", "wavelet_grf"],
                        help="Type of spatial amplification function m(x,y)")

    parser.add_argument("--m_mean", type=float, default=1.0,
                        help="Mean value for GRF-based m(x,y)")
    parser.add_argument("--m_std", type=float, default=0.5,
                        help="Standard deviation for GRF-based m(x,y)")
    parser.add_argument("--m_scale", type=float, default=2.0,
                        help="Correlation scale for GRF-based m(x,y)")
    parser.add_argument("--m_wavelet_scale", type=float, default=1.0,
                        help="Wavelet scale for wavelet modulated GRF")
    parser.add_argument("--m_factor", type=float, default=1.5,
                        help="Factor for periodic boxes amplification")
    parser.add_argument("--m_boxes_per_dim", type=int, default=3,
                        help="Number of boxes per dimension for periodic boxes")
    parser.add_argument("--m_box_length", type=float, default=0.1,
                        help="Relative box length for periodic boxes (fraction of L)")
    parser.add_argument("--m_wall_dist", type=float, default=0.1,
                        help="Wall distance for periodic boxes (fraction of L)")
    parser.add_argument("--anisotropy", type=str, default="one", choices=["one"])

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
    parser.add_argument("--T", type=float, default=1.5, help="Simulation time")
    parser.add_argument(
        "--nt",
        type=int,
        default=500,
        help="Number of time steps")
    parser.add_argument(
        "--snapshots",
        type=int,
        default=100,
        help="Number of snapshots")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory")
    parser.add_argument(
        "--exe",
        required=True,
        type=str,
        help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs to perform")

    parser.add_argument("--dr-x", type=int, default=128,
                        help="Number of gridpoints to sample down for in x-direction")
    parser.add_argument("--dr-y", type=int, default=128,
                        help="Number of gridpoints to sample down for in y-direction")
    parser.add_argument("--dr-strategy", choices=["FFT", "interpolation", "none"],
                        default="interpolation",
                        help="Downsampling strategy: Default is interpolation due to " +
                             "non-periodic boundary conditions. Choose 'none' to keep the resolution")

    parser.add_argument("--delete-intermediates", action="store_true", default=True,
                        help="Removing intermediate files (mostly npy to be recovered from hdf5)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Create an animation from downsampled trajectory for all runs")

    parser.add_argument("--parameter-sweep", action="store_true",
                        help="Perform a parameter sweep across phenomenon parameters")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))

    return args


def main():
    args = parse_args()
    launcher = NLSELauncher(args)
    launcher.run()


if __name__ == "__main__":
    sys.exit(main())
