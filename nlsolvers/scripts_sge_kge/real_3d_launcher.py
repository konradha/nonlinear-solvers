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

from spatial_amplification_3d import (
    make_grid,
    create_constant_m
)

from nonlinearity_profiles_3d import highlight_profiles

from downsampling import (
        downsample_fft_3d, downsample_interpolation_3d
        )


from visualization import (
        animate_isosurface, extrema_tracking,
        comparative_animation
        )


from real_sampler import RealWaveSampler3d as RealWaveSampler
from valid_spaces import get_parameter_spaces_3d

from classify_trajectory import batch_process_solutions
from global_analysis import analyze_all_runs

sys.stdout.reconfigure(line_buffering=True)

 
class Launcher:
    def __init__(self, args):
        self.args = args
        self.run_id = str(uuid.uuid4())[:8]
        self.setup_directories()
        self.configure_grid()
        assert args.nx == args.ny and np.abs(args.Lx - args.Ly) < 1e-8
        assert args.ny == args.nz and np.abs(args.Ly - args.Lz) < 1e-8
        self.sampler = RealWaveSampler(args.nx, args.ny, args.nz, args.Lx)
        self.system_type = args.system_type

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
            f.write(f"Grid: {self.args.nx}x{self.args.ny}x{self.args.nz}\n")
            f.write(f"Domain: {self.args.Lx}x{self.args.Ly}x{self.args.Lz}\n")
            f.write(
                f"Time: T={self.args.T}, steps={self.args.nt}, snapshots={self.args.snapshots}\n")
            f.write(f"Phenomenon: {self.args.phenomenon}\n")
            f.write(f"Amplification: {self.args.m_type}\n")
            f.write(f"Anisotropy: {self.args.c_type}\n")
            f.write(f"Executable: {self.args.exe}\n")

    def configure_grid(self):
        self.X, self.Y, self.Z = make_grid(self.args.nx, self.args.Lx)
        self.X_dr, self.Y_dr, self.Z_dr = make_grid(self.args.dr_x, self.args.Lx)

        self.dx = 2 * self.args.Lx / (self.args.nx - 1)
        self.dy = 2 * self.args.Ly / (self.args.ny - 1)
        self.dz = 2 * self.args.Lz / (self.args.nz - 1)
        self.dV = self.dx * self.dy * self.dz

    def generate_initial_condition(self, run_idx, phenomenon_params=None):
        if phenomenon_params is None:
            phenomenon_params = self.sample_phenomenon_params()
        print(f"run {run_idx + 1} parameters:", phenomenon_params)
        u0, v0 = self.sampler.generate_initial_condition(
            phenomenon_type=self.args.phenomenon, **phenomenon_params)

        if isinstance(u0, torch.Tensor):
            u0 = u0.detach().numpy()
        if isinstance(v0, torch.Tensor):
            v0 = v0.detach().numpy()
        #import pdb; pdb.set_trace()
        return (u0, v0), phenomenon_params

    def sample_phenomenon_params(self):
        parameter_spaces = get_parameter_spaces_3d(self.args.Lx)
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

            params["system_type"] = self.system_type
            params["velocity_type"] = self.args.velocity_type
            return params
        else:
            raise ValueError(
                f"Unknown phenomenon type: {self.args.phenomenon}")

    def generate_spatial_amplification(self, run_idx):
        if self.args.m_type == "constant":
            m = create_constant_m(
                self.X, value=self.args.m_mean)
        else:
            raise ValueError(f"Unknown m_type: {self.args.m_type}")
        #import pdb; pdb.set_trace()
        return m.astype(np.float64)  # safety first

    def generate_anisotropy(self, run_idx):
        if self.args.c_type == "constant":
            c = create_constant_m(
                self.X, value=self.args.c_mean)
        else:
            raise ValueError(f"Unknown m_type: {self.args.c_type}")
        #import pdb; pdb.set_trace()
        return c.astype(np.float64)  # safety first

    def run_simulation(self, run_idx, u0, v0, m, c):
        u0_file = self.ic_dir / f"u0_{self.run_id}_{run_idx:04d}.npy"
        np.save(u0_file, u0)

        v0_file = self.ic_dir / f"v0_{self.run_id}_{run_idx:04d}.npy"
        np.save(v0_file, v0)

        m_file = self.focusing_dir / f"m_{self.run_id}_{run_idx:04d}.npy"
        np.save(m_file, m)

        c_file = self.anisotropy_dir / f"c_{self.run_id}_{run_idx:04d}.npy"
        np.save(c_file, c)

        traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"
        vel_file = self.traj_dir / f"vel_{self.run_id}_{run_idx:04d}.npy"

        exe_path = Path(self.args.exe)
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable {self.args.exe} not found")

        cmd = [
            str(exe_path),
            str(self.args.nx),
            str(self.args.ny),
            str(self.args.nz),
            str(self.args.Lx),
            str(self.args.Ly),
            str(self.args.Lz),
            str(u0_file),
            str(v0_file),
            str(traj_file),
            str(vel_file),
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
            print("Command used:", " ".join(cmd))
            print(f"Warnings/Errors: {result.stderr}")
            raise Exception

        end_time = time.time()
        walltime = end_time - start_time

        traj_data = np.load(traj_file)
        vel_data = np.load(vel_file)
        return (traj_data, vel_data), walltime

    def downsample_trajectory(self, traj_data):
        if self.args.dr_strategy == 'none':
            return traj_data

        elif self.args.dr_strategy == 'FFT':
            return downsample_fft_3d(
                traj_data,
                target_shape=(self.args.dr_x, self.args.dr_y, self.args.dr_z),
            )

        elif self.args.dr_strategy == 'interpolation':
            return downsample_interpolation_3d(
                traj_data,
                target_shape=(self.args.dr_x, self.args.dr_y, self.args.dr_z),
                Lx=self.args.Lx,
                Ly=self.args.Ly,
                Lz=self.args.Lz
            )
        else:
            raise ValueError(
                f"Unknown downsampling strategy: {self.args.dr_strategy}")

    def save_to_hdf5(self, run_idx, u0, v0, m, c, traj_data, vel_data,
                     phenomenon_params, elapsed_time):
        h5_file = self.h5_dir / f"run_{self.run_id}_{run_idx:04d}.h5"
        with h5py.File(h5_file, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['problem_type'] = self.system_type
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
            grid.attrs['nz'] = self.args.nz
            grid.attrs['Lx'] = self.args.Lx
            grid.attrs['Ly'] = self.args.Ly
            grid.attrs['Lz'] = self.args.Ly

            time_grp = f.create_group('time')
            time_grp.attrs['T'] = self.args.T
            time_grp.attrs['nt'] = self.args.nt
            time_grp.attrs['num_snapshots'] = self.args.snapshots

            ic_grp = f.create_group('initial_condition')
            ic_grp.create_dataset('u0', data=u0)
            ic_grp.create_dataset('v0', data=v0)

            m_grp = f.create_group('focusing')
            m_grp.attrs['type'] = self.args.m_type
            m_grp.create_dataset('m', data=m)

            c_grp = f.create_group('anisotropy')
            c_grp.attrs['type'] = self.args.c_type
            c_grp.create_dataset('c', data=c)

            f.create_dataset('u', data=traj_data)
            f.create_dataset('v', data=vel_data)
            f.create_dataset('X', data=self.X)
            f.create_dataset('Y', data=self.Y)
            f.create_dataset('Z', data=self.Y)

        return h5_file

    def create_visualization(self, run_idx, traj_data,
                             m, c, phenomenon_params, walltime):
        # animation_title = self.generate_animation_title(
        #     phenomenon_params,
        #     walltime
        # )
        animation_title = ""
        extrema_output = self.traj_dir / f"extrema_{self.run_id}_{run_idx:04d}.mp4"
        comparative_output = self.traj_dir / f"compare_{self.run_id}_{run_idx:04d}.mp4" 
        print(extrema_output)
        #print(comparative_output) 

        # small hack even though we originally disallowed ...
        # m_dr = self.downsample_trajectory(m.reshape((1, self.args.nx, self.args.ny, self.args.nz)))
        # c_dr = self.downsample_trajectory(c.reshape((1, self.args.nx, self.args.ny, self.args.nz)))

        extrema_tracking(self.X_dr, self.Y_dr, self.Z_dr, traj_data, animation_title, extrema_output, T=self.args.T)
        #comparative_animation(self.X_dr, self.Y_dr, self.Z_dr, traj_data, self.args.T, c_dr, m_dr, animation_title, comparative_output)


    def generate_animation_title(self, phenomenon_params, walltime):
        cuda_avail = torch.cuda.is_available()
        cuda_string = "device (presumably)" if cuda_avail else "host"
        m_string = f"Constant m={self.args.m_mean}" if self.args.m_type == "constant" else None
        c_string = f"Constant c={self.args.c_mean}" if self.args.c_type == "constant" else None
        if m_string is None or c_string is None:
            raise NotImplemented

        params_str = ", ".join([f"{k}={v}" for k, v in phenomenon_params.items(
        ) if not isinstance(v, (list, tuple))])
        if len(params_str) > 100:
            params_str = params_str[:97] + "..."
        if "klein_gordon":
            str_start = "Klein-Gordon: $u_{tt} = div(c(x,y,z) grad(u)) - m(x,y,z) u$\n"
        else:
            raise Exception("Invalid system type")

        return (
            f"{str_start}"
            f"{self.args.phenomenon}, m: {m_string}, c: {c_string}\n"
            f"domain: [0, T={self.args.T}] x [-{self.args.Lx:.2f}, {self.args.Lx:.2f}] x [-{self.args.Ly:.2f}, {self.args.Ly:.2f}]\n"
            f"resolution n_t={self.args.nt}, n_x={self.args.nx}, n_y={self.args.ny}\n"
            f"downsampled to: {self.args.dr_x}x{self.args.dr_y} using '{self.args.dr_strategy}'\n"
            f"samples collected: {self.args.snapshots}, walltime={walltime:.2f} seconds on {cuda_string}\n\n\n"
        )

    def cleanup(self, run_idx):
        if self.args.delete_intermediates:
            u0_file = self.ic_dir / f"u0_{self.run_id}_{run_idx:04d}.npy"
            v0_file = self.ic_dir / f"v0_{self.run_id}_{run_idx:04d}.npy"
            m_file = self.focusing_dir / f"m_{self.run_id}_{run_idx:04d}.npy"
            traj_file = self.traj_dir / f"traj_{self.run_id}_{run_idx:04d}.npy"
            vel_file = self.traj_dir / f"vel_{self.run_id}_{run_idx:04d}.npy"

            for file in [u0_file, v0_file, m_file, traj_file, vel_file]:
                if file.exists():
                    os.unlink(file)

    def run(self):
        dt = self.args.T / self.args.nt
        dx = 2 * self.args.Lx / (self.args.nx - 1)
        dy = 2 * self.args.Ly / (self.args.ny - 1)

        sols = dict()
        for i in range(self.args.num_runs):
            try:
                pre_start = time.time()
                (u0, v0), phenomenon_params = self.generate_initial_condition(i)
                if self.args.c_m_pair is None:
                    m = self.generate_spatial_amplification(i)
                    c = self.generate_anisotropy(i)
                else:
                    profiles = highlight_profiles(self.args.nx, self.args.Lx) 
                    c, m = profiles[self.args.c_m_pair]
                
                pre_end = time.time()
                (traj_data, vel_data), walltime = self.run_simulation(i, u0, v0, m, c)
                post_start = time.time()

                # animate using high-res data
                # TODO implement viz in 3d
                
                traj_data = self.downsample_trajectory(traj_data.reshape(
                                self.args.snapshots, self.args.nx, self.args.ny, self.args.nz)
                            )
                vel_data  = self.downsample_trajectory(vel_data.reshape(
                                self.args.snapshots, self.args.nx, self.args.ny, self.args.nz)
                            )
                sols[f"{self.run_id}_{i}"] = (traj_data, vel_data)
                # save downsampled (u, v) trajectory only to perform analysis
                self.save_to_hdf5(
                    i, u0, v0, m, c, traj_data, vel_data,
                    phenomenon_params, walltime)
                # signification diff to 2d: visualize downsampled data
                if self.args.visualize:
                    self.create_visualization(
                        i, traj_data, m, c, phenomenon_params, walltime)
                if self.args.delete_intermediates:
                    self.cleanup(i)
                post_end = time.time()
                total_t = walltime + \
                    abs(pre_end - pre_start) + abs(post_end - post_start)
                part_pre = abs(pre_end - pre_start) / total_t
                part_run = walltime / total_t
                part_pst = abs(post_end - post_start) / total_t
                print(
                    f"walltime: {total_t:.2f}s, pre: {part_pre * 100:.1f}%, run: {part_run * 100:.1f}%, post: {part_pst*100:.1f}%")

            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                # import traceback as t
                # import pdb
                # t.print_exc()
                # pdb.set_trace()
                continue
        # TODO implement analyses in 3d
        #analysis_start = time.time()
        # batch_process_solutions(sols, self.args.dr_x, self.args.dr_y, self.args.Lx, self.args.Lx,
        # self.args.T / self.args.snapshots, self.args.T,
        # save_dir=self.analysis_dir, system_type=self.system_type)

        # analyze_all_runs(self.h5_dir, output_dir=self.analysis_dir, pattern=f"run_{self.run_id}_*.h5",
        #                 run_id=self.run_id, system_type=self.system_type)
        #analysis_end = time.time()
        #print(f"analysis took {(analysis_end - analysis_start):.2f}s")

        if self.args.delete_intermediates:
            params_file = self.output_dir / f"params_{self.run_id}.txt"
            if params_file.exists():
                os.unlink(params_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real solver launcher (3D): u_tt = div(c(x,y,z) grad(u)) - m(x,y,z) f(u)")

    parser.add_argument("--system-type", type=str, default="klein_gordon",
                        choices=["klein_gordon"])

    parser.add_argument("--phenomenon", type=str, default="q_ball_soliton",
                        choices=["kink_field", "q_ball_soliton"],
                        help="Phenomenon type to simulate")
    parser.add_argument("--c-m-pair", type=str, choices=['optimal',
        'resonant_cavity', 'focusing_soliton', 'sharp_interfaces',
        'multi_scale', 'fractal_nonlinear', 'waveguide', 'grf_threshold',
        'anisotropic', 'maybe_blowup'], default=None, help="Choose pair between c(x,y,z) and m(x,y,z)")

    parser.add_argument("--velocity-type", type=str, default="zero",
                        choices=["zero", "fitting", "random"])

    parser.add_argument("--m_type", type=str, default="constant",
                        choices=["constant"],
                        help="Type of spatial amplification function m(x,y,z)")
    parser.add_argument(
        "--m-mean",
        type=float,
        default=1.,
        help="Scalar multiplier for focussing field m(x,y,z)")
    parser.add_argument("--c-type", type=str, default="constant",
                        choices=["constant"],
                        help="Type of anisotropy c(x,y,z)")
    parser.add_argument(
        "--c-mean",
        type=float,
        default=1.,
        help="Scalar multiplier for anisotropy field c(x,y,z)")

    parser.add_argument("--nx", type=int, default=128, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in y")
    parser.add_argument("--nz", type=int, default=128, help="Grid points in z")
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
        "--Lz",
        type=float,
        default=10.0,
        help="Domain half-width in z")

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
        default="results-3d",
        help="Output directory")
    parser.add_argument(
        "--exe",
        type=str,
        required=True,
        help="Path to executable")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs to perform")

    parser.add_argument("--dr-x", type=int, default=128,
                        help="Number of gridpoints to sample down for in x-direction")
    parser.add_argument("--dr-y", type=int, default=128,
                        help="Number of gridpoints to sample down for in y-direction")
    parser.add_argument("--dr-z", type=int, default=128,
                        help="Number of gridpoints to sample down for in z-direction")
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
    launcher = Launcher(args)
    launcher.run()


if __name__ == "__main__":
    sys.exit(main())
