import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft2, fftshift
import os


class SolitonDashboard:
    def __init__(self, nx, ny, Lx, Ly, dt, T, save_dir, system_type):
        assert save_dir is not None
        assert system_type is not None
        self.system_type = system_type
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = 2 * Lx / (nx - 1), 2 * Ly / (ny - 1)
        self.dt, self.T = dt, T
        self.nt = int(self.T / self.dt)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.x = np.linspace(-Lx, Lx, nx)
        self.y = np.linspace(-Ly, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def create_dashboard(self, u, v=None, name=None):
        features = self._analyze(u, v)
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        vmin, vmax = np.min(u), np.max(u)
        self._plot_states(fig, gs[0, :], u, vmin, vmax)
        self._plot_dynamics(fig, gs[1, :], features)
        self._plot_analysis(fig, gs[2, :], features)

        plt.savefig(
            f"{self.save_dir}/{name or 'dashboard'}.png",
            bbox_inches='tight',
            dpi=300)
        plt.close()

    def _plot_states(self, fig, gs, u, vmin, vmax):
        subgs = gs.subgridspec(1, 3)
        tn = np.linspace(0, self.T, self.nt)
        times_idx = [0, self.nt // 2, -1]
        axes = []
        for i, t in enumerate(times_idx):
            ax = fig.add_subplot(subgs[0, i])
            im = ax.imshow(u[t], cmap='RdBu_r', vmin=vmin, vmax=vmax,
                           extent=[-self.Lx, self.Lx, -self.Ly, self.Ly])
            ax.set_title(f"t={tn[t]:.1f}")
            axes.append(ax)

        cax = fig.add_axes([0.91, 0.72, 0.01, 0.15])
        fig.colorbar(im, cax=cax).set_label("Amplitude")

    def _plot_dynamics(self, fig, gs, features):
        subgs = gs.subgridspec(1, 3)
        t = np.arange(len(features['energy'])) * self.dt
        tn = np.linspace(0, self.T, self.nt)

        ax1 = fig.add_subplot(subgs[0, 0])
        norms = features['conservation']['norm']
        kinetic, gradient_energy, potential = features['energy']
        energies = kinetic + gradient_energy + potential
        energies_logdiff = [
            np.nan] + [np.log(np.abs(energies[i] - energies[0])) for i in range(1, self.nt)]
        ax1.plot(tn, norms, 'k-', label="$ u^2 / u_0^2 $")
        ax1.set_ylabel("Norm")
        ax1.legend()
        ax1.grid(True)

        ax2 = fig.add_subplot(subgs[0, 1])
        ax2.semilogy(
            tn,
            features['terms']['laplacian'],
            'b-',
            label='$\\Delta u$')
        ax2.semilogy(
            tn,
            features['terms']['nonlinear'],
            'r--',
            label='$sin u$')
        ax2.set_ylabel("Term Magnitudes")
        ax2.legend()
        ax2.grid(True)

        ax3 = ax2.twinx()
        ax3.semilogy(tn, features['terms']['ratio'], 'g:', alpha=0.7)
        #ax3.set_ylabel("Ratio", rotation=270, labelpad=15)
        ax3.legend(['$\\frac{\\Delta u}{sin(u)}$'], loc='upper right')  # TODO
        ax3.grid(True)

        ax4 = fig.add_subplot(subgs[0, 2])
        kinetic, gradient_energy, potential = features['energy']
        energies = list(features['energy'])

        # TODO complete correct names
        if self.system_type == "sine_gordon":
            potential_name = f"$\\int (1 - cos(u)) dx dy$"
        elif self.system_type == "double_sine_gordon":
            potential_name = f"$\\int \left((1 - cos(u)) + (.6/2)(1 - cos(u))  \\right)dx dy$"
        elif self.system_type == "hyperbolic_sine_gordon":
            potential_name = f"$\\int (cosh(u) - 1) dx dy$"
        elif self.system_type == "klein_gordon":
            potential_name = f"$\\int 1/2 u^2 dx dy$"
        elif self.system_type == "phi4":
            potential_name = f"$\\int 1/4 (u^2 - 1)^2 dx dy$"
        else:
            raise Exception("Invalid system type")

        e_names = ["$\\int 1/2 u_t^2 dx dy$",
                   "$1/2 \\int \\nabla u dx dy$", potential_name]
        for i, name in enumerate(e_names):
            ax4.plot(tn, energies[i], linestyle='-.', label=name)
        ax4.legend()
        ax4.grid(True)

    def _plot_analysis(self, fig, gs, features):
        subgs = gs.subgridspec(1, 3)

        ax1 = fig.add_subplot(subgs[0, 0])
        traj = features['trajectory']
        ax1.plot(traj[:, 0], traj[:, 1], 'b-')
        ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        ax1.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8)
        ax1.legend(['Path', 'Start', 'End'], loc='best')
        ax1.set_title(f"Velocity: {features['velocity']:.4f}")

        ax2 = fig.add_subplot(subgs[0, 1])
        ax2.imshow(
            np.log10(
                features['power_spectrum'] +
                1e-10),
            cmap='viridis')
        ax2.set_title("Power Spectrum")

        ax3 = fig.add_subplot(subgs[0, 2])
        ax3.bar([1, 2, 3], [features['localization'],
                features['symmetry'], features['stability']])
        ax3.set_xticks([1, 2, 3])
        ax3.set_xticklabels(['Localization', 'Symmetry', 'Stability'])

    def _analyze(self, u, v):
        return {
            'energy': self._compute_energy(u, v),
            'conservation': {'norm': np.sum(u**2, axis=(1, 2)) / np.sum(u[0] ** 2)},
            'terms': self._compute_terms(u),
            'trajectory': self._compute_trajectory(u),
            'power_spectrum': np.abs(fftshift(fft2(u[-1])))**2,
            'velocity': self._compute_velocity(u),
            'localization': np.max(u[-1]) / np.mean(np.abs(u[-1])),
            'symmetry': self._compute_symmetry(u[-1]),
            'stability': np.std(u[-1] / u[0])
        }

    def _compute_velocity(self, u):
        com = [np.sum(self.X * u[t]) / np.sum(u[t]) for t in range(u.shape[0])]
        return (com[-1] - com[0]) / (u.shape[0] * self.dt)

    def _compute_symmetry(self, u):
        rotated = np.rot90(u)
        return np.corrcoef(u.flatten(), rotated.flatten())[0, 1]

    def _compute_terms(self, u):
        laplacian = [np.sum(np.gradient(np.gradient(ut, self.dx, axis=0), self.dx, axis=0) +
                            np.gradient(np.gradient(ut, self.dy, axis=1), self.dy, axis=1))**2
                     * self.dx * self.dy for ut in u]
        # TODO adapt to system_type
        # especially correct computation in "nonlinear"
        # maybe just keep it to have one thing comparable for all plots 
        nonlinear = [np.sum(np.sin(ut)) * self.dx * self.dy for ut in u]
        return {'laplacian': laplacian, 'nonlinear': nonlinear,
                'ratio': [l / n if n > 0 else 0 for l, n in zip(laplacian, nonlinear)]}

    def _compute_energy(self, u, v):
        # IMPORTANT: As we mostly don't save v we only crudely approximate energy conservation!
        # Kinetic energy is "neglected"
        kinetic = 0.5 * np.sum(v**2, axis=(1, 2)) * self.dx * \
            self.dy if v is not None else [np.nan] * self.nt
        gradient_energy = np.zeros(u.shape[0])
        for i, ut in enumerate(u):
            grad_x = np.gradient(ut, self.dx, axis=0)
            grad_y = np.gradient(ut, self.dy, axis=1)
            gradient_energy[i] = 0.5 * \
                np.sum(grad_x**2 + grad_y**2) * self.dx * self.dy

        # TODO actually implement correct lambda depending on system type
        if self.system_type == "sine_gordon":
            def potential_lambda(u): return np.sum(
                1 - np.cos(u), axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "double_sine_gordon":
            def potential_lambda(u): return np.sum(
                1 - np.cos(u), axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "hyperbolic_sine_gordon":
            def potential_lambda(u): return np.sum(
                1 - np.cos(u), axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "klein_gordon":
            def potential_lambda(u): return np.sum(
                1 - np.cos(u), axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "phi4":
            def potential_lambda(u): return np.sum(
                1 - np.cos(u), axis=(1, 2)) * self.dx * self.dy
        else:
            raise Exception("Invalid system type")

        if self.system_type == "sine_gordon":
            def potential_lambda(u): return np.sum(1 - np.cos(u), axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "double_sine_gordon":
            def potential_lambda(u): return np.sum((1 - np.cos(u)) + (.6/2)*(1 - np.cos(2*u)), axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "hyperbolic_sine_gordon":
            def potential_lambda(u): return np.sum(np.cosh(u) - 1, axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "klein_gordon":
            def potential_lambda(u): return np.sum(.5 * u ** 2, axis=(1, 2)) * self.dx * self.dy
        elif self.system_type == "phi4":
            def potential_lambda(u): return np.sum((u**2 - 1)**2 / 4, axis=(1, 2)) * self.dx * self.dy
        else:
            raise Exception("Invalid system type")
        potential = potential_lambda(u)
        return kinetic, gradient_energy, potential

    def _compute_trajectory(self, u):
        return np.array([(np.sum(self.X * ut) / np.sum(ut), np.sum(self.Y * ut) / np.sum(ut))
                         for ut in u])


def batch_process_solutions(solution_dict, nx, ny,
                            Lx, Ly, dt, T, save_dir=None, system_type=None):
    dashboard = SolitonDashboard(nx, ny, Lx, Ly, dt, T, save_dir, system_type)
    results = {}

    for name, solution in solution_dict.items():
        if isinstance(solution, tuple) and len(solution) == 2:
            u, v = solution
            dashboard.create_dashboard(u, v, name=name)
        else:
            u = solution
            dashboard.create_dashboard(u, name=name)
