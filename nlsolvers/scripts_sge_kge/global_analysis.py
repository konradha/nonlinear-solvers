import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
import glob
from pathlib import Path


class GlobalAnalyzer:
    def __init__(self, hdf5_dir, output_dir=None,
                 run_id=None, system_type=None):
        assert hdf5_dir is not None
        assert output_dir is not None
        assert run_id is not None
        assert system_type is not None
        self.hdf5_dir = Path(hdf5_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.run_id = run_id
        self.system_type = system_type

    def load_all_runs(self, pattern="run_*.h5"):
        files = glob.glob(str(self.hdf5_dir / pattern))
        runs = {}

        for file_path in files:
            run_id = Path(file_path).stem
            runs[run_id] = self._load_run(file_path)

        return runs

    def _load_run(self, file_path):
        with h5py.File(file_path, 'r') as f:
            nx = f['grid'].attrs['nx']
            ny = f['grid'].attrs['ny']
            Lx = f['grid'].attrs['Lx']
            Ly = f['grid'].attrs['Ly']
            T = f['time'].attrs['T']
            nt = f['time'].attrs['num_snapshots']

            u = f['u'][:]
            v = f['v'][:] if 'v' in f else None
            X = f['X'][:]
            Y = f['Y'][:]

            m = f['focusing']['m'][:] if 'focusing' in f and 'm' in f['focusing'] else None

            ic_grp = f['initial_condition']
            u0 = ic_grp['u0'][:] if 'u0' in ic_grp else None
            v0 = ic_grp['v0'][:] if 'v0' in ic_grp else None

            dx, dy = 2 * Lx / (nx - 1), 2 * Ly / (ny - 1)

            metadata = {}
            metadata['dx'] = dx
            metadata['dy'] = dy
            metadata['Lx'] = Lx
            metadata['Ly'] = Ly
            metadata['T'] = T
            metadata['nt'] = f['time'].attrs['nt']
            metadata['num_snapshots'] = f['time'].attrs['num_snapshots']
            metadata['downsampling_strategy'] = metadata['dr_x'] = metadata['dr_y'] = None

            if 'metadata' in f:
                for key, value in f['metadata'].attrs.items():
                    metadata[key] = value

            focusing_params = {}
            if 'focusing' in f:
                for key, value in f['focusing'].attrs.items():
                    focusing_params[key] = value

            run_data = {
                'nx': nx, 'ny': ny,
                'Lx': Lx, 'Ly': Ly,
                'dx': dx, 'dy': dy,
                'T': T, 'nt': nt,
                'u': u, 'v': v,
                'u0': u0, 'v0': v0,
                'm': m,
                'X': X, 'Y': Y,
                'metadata': metadata,
                'focusing_params': focusing_params
            }

            return run_data

    def compute_metrics(self, runs):
        metrics = {}

        for run_id, run_data in runs.items():
            u = run_data['u']
            v = run_data['v']
            dx, dy = run_data['dx'], run_data['dy']
            X, Y = run_data['X'], run_data['Y']
            nt = run_data['nt']  # snapshots here!

            T = run_data['T']

            time_array = np.linspace(0, T, nt)

            kinetic = []
            gradient_energy = []
            potential = []

            if v is not None:
                kinetic = 0.5 * np.sum(v**2, axis=(1, 2)) * dx * dy

            for i, ut in enumerate(u):
                grad_x = np.gradient(ut, dx, axis=0)
                grad_y = np.gradient(ut, dy, axis=1)
                gradient_e = 0.5 * np.sum(grad_x**2 + grad_y**2) * dx * dy
                gradient_energy.append(gradient_e)

            if self.system_type == "sine_gordon":
                potential = np.sum(1 - np.cos(u), axis=(1, 2)) * dx * dy
            elif self.system_type == "double_sine_gordon":
                potential = np.sum((1 - np.cos(u)) + (.6 / 2)
                                   * (1 - np.cos(2 * u)), axis=(1, 2)) * dx * dy
            elif self.system_type == "hyperbolic_sine_gordon":
                potential = np.sum(np.cosh(u) - 1, axis=(1, 2)) * dx * dy
            elif self.system_type == "klein_gordon":
                potential = np.sum(.5 * u ** 2, axis=(1, 2)) * dx * dy
            elif self.system_type == "phi4":
                potential = np.sum((u**2 - 1)**2 / 4, axis=(1, 2)) * dx * dy
            else:
                raise Exception("Invalid system type")

            total_energy = np.array(
                kinetic) + np.array(gradient_energy) + np.array(potential)
            energy_logdiff = [
                np.nan] + list(np.log10(np.abs(total_energy[1:] - total_energy[0])))

            norm = np.sum(u**2, axis=(1, 2)) / np.sum(u[0]**2)
            max_amplitude = np.max(np.abs(u), axis=(1, 2)
                                   ) / np.max(np.abs(u[0]))

            metrics[run_id] = {
                'time': time_array,
                'kinetic': np.array(kinetic),
                'gradient': np.array(gradient_energy),
                'potential': potential,
                'total_energy': total_energy,
                'energy_logdiff': energy_logdiff,
                'norm': norm,
                'max_amplitude': max_amplitude,
                'metadata': run_data['metadata'],
                'focusing_params': run_data['focusing_params'],
                'u0': run_data['u0'],
                'v0': run_data['v0'],
                'm': run_data['m'],
                'X': X,
                'Y': Y,
                'snapshots': nt,
                'T': T,
                'dx': dx,
                'dy': dy
            }

        return metrics

    def create_global_dashboard(
            self, metrics, figsize=(16, 18), filename=None):
        assert filename is not None
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])

        self._plot_description(fig, gs[0, 0], metrics)
        self._plot_initial_velocity(fig, gs[0, 1], metrics)
        self._plot_m_field(fig, gs[0, 2], metrics)

        self._plot_energy_components(fig, gs[1, 0:2], metrics)
        self._plot_energy_conservation(fig, gs[1, 2], metrics)
        self._plot_norm_conservation(fig, gs[2, 0:2], metrics)
        self._plot_max_amplitude(fig, gs[2, 2], metrics)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_description(self, fig, gs, metrics):
        ax = fig.add_subplot(gs)
        ax.axis('off')

        first_run_id = list(metrics.keys())[0]
        metadata = metrics[first_run_id]['metadata']

        if self.system_type == "sine_gordon":
            text = f"SGE trajectory\n\n"
        elif self.system_type == "double_sine_gordon":
            text = f"Double SGE trajectory\n\n"
        elif self.system_type == "hyperbolic_sine_gordon":
            text = f"Hyperbolic SGE trajectory\n\n"
        elif self.system_type == "klein_gordon":
            text = f"KGE trajectory\n\n"
        elif self.system_type == "phi4":
            text = f"$\\phi-4$ trajectory\n\n"
        else:
            raise Exception("Invalid system type")

        if 'problem_type' in metadata:
            text += f"Problem type: {metadata['problem_type']}\n"

        if 'boundary_condition' in metadata:
            text += f"BCs: {metadata['boundary_condition']}\n"

        if 'phenomenon' in metadata:
            text += f"Phenomenon: {metadata['phenomenon']}\n"

        for key, value in metadata.items():
            if key.startswith('phenomenon_'):
                param_name = key.replace('phenomenon_', '')
                text += f"{param_name}: {value}\n"

        text += "simulation params:\n"
        text += f"$L_x = L_y = {metadata['Lx']:.2f}$ $\\delta_x = \\delta_y = {metadata['dx']:.3e}$, $T={metadata['T']:.2f}$\n"
        text += f"for ${metadata['nt']}$ steps, collected ${metadata['num_snapshots']}$"
        # TODO downsampling
        text += f"\ncomparing {len(metrics)} runs with ID: {self.run_id}\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def _plot_initial_velocity(self, fig, gs, metrics):
        ax = fig.add_subplot(gs)
        for run_id, data in metrics.items():
            if data['v0'] is not None:
                v0 = data['v0']
                X, Y = data['X'], data['Y']

                im = ax.pcolormesh(X, Y, v0, cmap='coolwarm', shading='auto')
                fig.colorbar(im, ax=ax,)
                ax.set_title(f"v0 (exemplary)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect('equal')
                break
        else:
            ax.text(0.5, 0.5, "No initial velocity data available",
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_m_field(self, fig, gs, metrics):
        ax = fig.add_subplot(gs, projection='3d')
        for run_id, data in metrics.items():
            if data['m'] is not None:
                m = data['m']
                X, Y = data['X'], data['Y']
                stride = 1
                surf = ax.plot_surface(X[::stride, ::stride],
                                       Y[::stride, ::stride],
                                       m[::stride, ::stride],
                                       cmap='viridis',
                                       linewidth=0,
                                       antialiased=True)

                #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5,)
                ax.set_title("$m(x,y)$ (exemplary)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                break
        else:
            ax = fig.add_subplot(gs)
            ax.text(0.5, 0.5, "No spatial amplification field available",
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_energy_components(self, fig, gs, metrics):
        ax = fig.add_subplot(gs)
        energy_types = ['kinetic', 'gradient', 'potential']
        line_styles = ['-', '-', '-']
        markers = ['+', '1', 10]
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        for i, (run_id, data) in enumerate(metrics.items()):
            label = run_id
            ax.plot(data['time'], data['kinetic'] + data['gradient'] + data['potential'],
                    color=colors[i],
                    alpha=0.2, label=label)
            ax.scatter(data['time'], data['kinetic'] + data['gradient'] + data['potential'],
                       color=colors[i],
                       marker=10,)

        ax.set_title("Energy evolution")
        ax.set_xlabel("T / [1]")
        ax.set_ylabel("E / [1]")
        ax.grid(True)
        ax.legend(loc='best')

    def _plot_energy_conservation(self, fig, gs, metrics):
        ax = fig.add_subplot(gs)
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        for i, (run_id, data) in enumerate(metrics.items()):
            ax.plot(data['time'], data['energy_logdiff'],
                    color=colors[i], label=run_id)

        ax.set_title("Energy logdiff")
        ax.set_xlabel("T / [1]")
        ax.set_ylabel("$log|E - E_0|$")
        ax.grid(True)
        ax.legend(loc='best')

    def _plot_norm_conservation(self, fig, gs, metrics):
        ax = fig.add_subplot(gs)
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        for i, (run_id, data) in enumerate(metrics.items()):
            ax.plot(data['time'], data['norm'],
                    color=colors[i], label=run_id)

        ax.set_title("Norm Conservation ($L^2$ ratio)")
        ax.set_xlabel("T / [1]")
        ax.set_ylabel("$||u||^2 / ||u_0 ||^2$")
        ax.grid(True)
        ax.legend(loc='best')

    def _plot_max_amplitude(self, fig, gs, metrics):
        ax = fig.add_subplot(gs)
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        for i, (run_id, data) in enumerate(metrics.items()):
            ax.plot(data['time'], data['max_amplitude'],
                    color=colors[i], label=run_id)

        ax.set_title("Max Amplitude (ratio)")
        ax.set_xlabel("T/[1]")
        ax.set_ylabel("max|u|/max|u₀|")
        ax.grid(True)
        ax.legend(loc='best')


def analyze_all_runs(hdf5_dir, output_dir=None,
                     pattern="run_*.h5", run_id=None, system_type=None):
    analyzer = GlobalAnalyzer(hdf5_dir, output_dir, run_id, system_type)
    runs = analyzer.load_all_runs(pattern)
    metrics = analyzer.compute_metrics(runs)
    analyzer.create_global_dashboard(
        metrics, filename=f"{run_id}_comparative_analysis.png")
    return metrics
