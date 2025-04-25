#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

from pathlib import Path
import argparse
import glob
import os

import warnings
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
warnings.filterwarnings('ignore')

from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import ndimage
from scipy.signal import cwt, ricker
import matplotlib.animation as animation
from tqdm import tqdm


from downsampling import downsample_interpolation_3d



class NLSEDashboardGenerator:
    def __init__(self, directory, output_dir=None, summary_only=False, detailed_panels=False):
        self.directory = Path(directory)
        self.output_dir = Path(output_dir) if output_dir else self.directory / "dashboards" 
        self.output_dir.mkdir(exist_ok=True, parents=True)
        if detailed_panels:
            self.detailed_panels = True
            self.detailed_dir = self.output_dir / "detailed_panels"
            self.detailed_dir.mkdir(exist_ok=True, parents=True)

        self.summary_only = summary_only
        
        self.h5_files = sorted(glob.glob(str(self.directory / "*.h5")))
        if not self.h5_files:
            raise ValueError(f"No HDF5 files found in {directory}")
        
        print(f"Found {len(self.h5_files)} HDF5 files for analysis")
    
    def process_all_files(self):
        for file_path in tqdm(self.h5_files, desc="Generating dashboards"):
            file_id = Path(file_path).stem
            if not self.summary_only:
                self.generate_dashboards(file_path, file_id)
            if self.detailed_panels:
                self.generate_detailed_viz(file_path, file_id)
 
        self.generate_summary()

    def generate_detailed_viz(self, file_path, file_id):
        with h5py.File(file_path, 'r') as f:
            meta = dict(f['metadata'].attrs)
            grid_data = dict(f['grid'].attrs)
            time_data = dict(f['time'].attrs)
            u_data = f['u'][:]
            
            T = time_data['T']
            nt = u_data.shape[0]
            time_values = np.linspace(0, T, nt)
            
            nx = grid_data['nx']
            ny = grid_data['ny']
            nz = grid_data['nz']
            
            Lx = grid_data['Lx']
            Ly = grid_data['Ly']
            Lz = grid_data['Lz']

            self.L = Lx
            self.n = nx
            
            self.dx = 2 * self.L / (self.n - 1)

            
            detailed_file_dir = self.detailed_dir / file_id 
            detailed_file_dir.mkdir(exist_ok=True, parents=True)
            
            fname_multi = detailed_file_dir / (
                    f"multi_scale_" + file_id + ".png" 
                    )
            fname_critical = detailed_file_dir / (
                    f"critical_" + file_id + ".png" 
                    )
            fname_temp_phase = detailed_file_dir / (
                    f"temp_phase_" + file_id + ".png" 
                    )
            fname_phase = detailed_file_dir / (
                    f"complex_phase_" + file_id + ".png" 
                    )
            fname_spectral = detailed_file_dir / (
                    f"spectral_" + file_id + ".png" 
                    )

            fname_tp_3d = detailed_file_dir / (
                    f"tp_rendering_" + file_id + ".png" 
                    )
            fname_critical_3d = detailed_file_dir / (
                    f"critical_3d_" + file_id + ".png" 
                    )
            fname_iso_3d = detailed_file_dir / (
                    f"iso_3d_" + file_id + ".png" 
                    )

            temporal_phase_portrait(u_data, T=T, fname=fname_temp_phase)
            multi_scale_signature_display(u_data,
                    timepoints=[0, nt // 2, -1],
                    timepoint_names=[0, T / 2, T],
                    fname=str(fname_multi))
            critical_phenomena_tracker(u_data,
                    timepoints=[0, nt // 2, -1],
                    timepoint_names=[0, T / 2, T],
                    fname=fname_critical) 
            complex_phase_evolution_display(u_data,
                    timepoints=[0, nt // 2, -1],
                    timepoints_names=[0, T / 2, T],
                    fname=fname_phase)
            spectral_signature_evolution(u_data,
                    timepoints=[0, nt // 2, -1],
                    timepoint_names=[0, T / 2, T],
                    fname=fname_spectral)
            phase_colored_isosurfaces(u_data,
                    timestep=nt // 2, timestep_name=T/2,
                    iso_value=0.5, fname=fname_tp_3d)
            critical_points_3d(u_data,
                    timestep=nt // 2, timestep_name=T/2,
                    fname=fname_critical_3d)
            compare_timepoints_isosurfaces(u_data,
                    timepoints=[0, nt // 2, -1],
                    timepoints_names=[0, T / 2, T],
                    iso_value=0.5, fname=fname_iso_3d)
            
           
    
    def generate_dashboards(self, file_path, file_id):
        with h5py.File(file_path, 'r') as f:
            meta = dict(f['metadata'].attrs)
            grid_data = dict(f['grid'].attrs)
            time_data = dict(f['time'].attrs)
            
            u0 = f['initial_condition/u0'][:]
            u_data = f['u'][:]
            m = f['focusing/m'][:]
            c = f['anisotropy/c'][:]
            
            T = time_data['T']
            nt = u_data.shape[0]
            time_values = np.linspace(0, T, nt)
            
            nx = grid_data['nx']
            ny = grid_data['ny']
            nz = grid_data['nz']
            
            Lx = grid_data['Lx']
            Ly = grid_data['Ly']
            Lz = grid_data['Lz']

            self.L = Lx
            
            dx = 2 * Lx / (nx - 1)
            dy = 2 * Ly / (ny - 1)
            dz = 2 * Lz / (nz - 1)
            
            self.generate_initial_condition_dashboard(file_id, u0, meta)
            self.generate_fields_dashboard(file_id, m, c, meta)
            self.generate_energy_dashboard(file_id, u_data, m, c, dx, dy, dz, time_values, meta)
            self.generate_evolution_dashboard(file_id, u_data, time_values, meta)
    
    def generate_initial_condition_dashboard(self, file_id, u0, meta):
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig)
        
        problem_type = meta.get('problem_type', 'Unknown')
        phenomenon = meta.get('phenomenon', 'Unknown')
        
        fig.suptitle(f"Initial Condition Dashboard - {file_id}\nType: {problem_type} - Phenomenon: {phenomenon}", 
                    fontsize=16, y=0.98)
        
        amplitude = np.abs(u0)
        phase = np.angle(u0)
        
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_isosurface(ax1, amplitude, value_name="Amplitude", cmap='viridis')
        
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        self._plot_volume_slices(ax2, amplitude, value_name="Amplitude", cmap='viridis')
        
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        self._plot_phase_visualization(ax3, u0, value_name="Phase")
        
        ax4 = fig.add_subplot(gs[1, 1])
        info_text = f"""
        Initial Condition Summary:
        
        Max Amplitude: {amplitude.max():.4f}
        Min Amplitude: {amplitude.min():.4f}
        Mean Amplitude: {amplitude.mean():.4f}
        
        Total Mass (L² Norm): {np.sum(amplitude**2):.4f}
        
        Shape: {u0.shape}
        
        Phenomenon Parameters:
        """
        
        for key, value in meta.items():
            if key.startswith('phenomenon_'):
                param_name = key.replace('phenomenon_', '')
                info_text += f"\n{param_name}: {value}"
        
        ax4.text(0.05, 0.95, info_text, verticalalignment='top', fontsize=10)
        ax4.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / f"{file_id}_initial_condition.png", dpi=150)
        plt.close(fig)
    
    def generate_fields_dashboard(self, file_id, m, c, meta):
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig)
        
        problem_type = meta.get('problem_type', 'Unknown')
        phenomenon = meta.get('phenomenon', 'Unknown')
        
        fig.suptitle(f"Field Coefficients Dashboard - {file_id}\nType: {problem_type} - Phenomenon: {phenomenon}", 
                    fontsize=16, y=0.98)
        
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_isosurface(ax1, m, value_name="Focusing (m)", cmap='plasma')
        
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        self._plot_isosurface(ax2, c, value_name="Anisotropy (c)", cmap='cividis')
        
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        self._plot_volume_slices(ax3, m, value_name="Focusing (m)", cmap='plasma')
        
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        self._plot_volume_slices(ax4, c, value_name="Anisotropy (c)", cmap='cividis')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / f"{file_id}_fields.png", dpi=150)
        plt.close(fig)
    
    def generate_energy_dashboard(self, file_id, u_data, m, c, dx, dy, dz, time_values, meta):
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 2, figure=fig)
        
        problem_type = meta.get('problem_type', 'Unknown')
        phenomenon = meta.get('phenomenon', 'Unknown')
        
        fig.suptitle(f"Energy Dashboard - {file_id}\nType: {problem_type} - Phenomenon: {phenomenon}", 
                    fontsize=16, y=0.98)
        
        energy_metrics = self._calculate_energy_metrics(u_data, m, c, dx, dy, dz)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_values, energy_metrics['kinetic_energy'], label='$\\Delta u$', linewidth=2)
        ax1.plot(time_values, energy_metrics['potential_energy'], label='$|u|^2$', linewidth=2)
        ax1.plot(time_values, energy_metrics['total_energy'], label='$H$', linewidth=3, color='black')
        ax1.set_xlabel('T / [1]')
        ax1.set_ylabel('E / [1]')
        ax1.set_title('Energy Evolution')
        ax1.legend()
        ax1.grid(True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_values, energy_metrics['mass'], linewidth=2, color='blue')
        ax2.set_xlabel('T / [1]')
        ax2.set_ylabel('$L_2$')
        ax2.set_title('Mass Conservation')
        ax2.grid(True)
        
        ax3 = fig.add_subplot(gs[1, 0])
        logscale = [np.nan] + list(np.log(np.abs(energy_metrics['total_energy'][0] -
            energy_metrics['total_energy'][1:])))
        ax3.plot(time_values, logscale, 
                linewidth=2, color='red')
        ax3.set_xlabel('T / [1]')
        ax3.set_ylabel('$\log |E - E_0|$')
        ax3.set_title('Energy Conservation')
        ax3.grid(True)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_values, energy_metrics['mass'] / energy_metrics['mass'][0], 
                linewidth=2, color='green')
        ax4.set_xlabel('T / [1]')
        ax4.set_ylabel('relative $L_2$')
        ax4.set_title('Mass Conservation')
        ax4.grid(True)
        
        ax5 = fig.add_subplot(gs[2, 0])
        max_amplitude = np.array([np.max(np.abs(u_data[t])) for t in range(u_data.shape[0])])
        ax5.plot(time_values, max_amplitude, linewidth=2, color='purple')
        ax5.set_xlabel('T / [1]')
        ax5.set_ylabel('$\max |u|$')
        ax5.set_title('Peak Amplitude Evolution')
        ax5.grid(True)
        
        ax6 = fig.add_subplot(gs[2, 1])
        max_gradient = self._calculate_max_gradient(u_data, dx, dy, dz)
        ax6.plot(time_values, max_gradient, linewidth=2, color='orange')
        ax6.set_xlabel('T / [1]')
        ax6.set_ylabel('$\max \\nabla u$')
        ax6.set_title('Peak Gradient Evolution')
        ax6.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / f"{file_id}_energy.png", dpi=150)
        plt.close(fig)
    
    def generate_evolution_dashboard(self, file_id, u_data, time_values, meta):
        if u_data.shape[0] < 3:
            return
            
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig)
        
        problem_type = meta.get('problem_type', 'Unknown')
        phenomenon = meta.get('phenomenon', 'Unknown')
        
        fig.suptitle(f"Evolution Dashboard - {file_id}\nType: {problem_type} - Phenomenon: {phenomenon}", 
                    fontsize=16, y=0.98)
        
        time_indices = [0, u_data.shape[0]//2, u_data.shape[0]-1]
        titles = ['Initial State', 'Middle State', 'Final State']
        
        for i, (t_idx, title) in enumerate(zip(time_indices, titles)):
            u = u_data[t_idx]
            amplitude = np.abs(u)
            
            ax = fig.add_subplot(gs[i, 0], projection='3d')
            self._plot_isosurface(ax, amplitude, value_name=f"Amplitude at t={time_values[t_idx]:.2f}", cmap='viridis')
            
            ax = fig.add_subplot(gs[i, 1], projection='3d')
            self._plot_volume_slices(ax, amplitude, value_name=f"Amplitude at t={time_values[t_idx]:.2f}", cmap='viridis')
            
            ax = fig.add_subplot(gs[i, 2], projection='3d')
            self._plot_phase_visualization(ax, u, value_name=f"Phase at t={time_values[t_idx]:.2f}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / f"{file_id}_evolution.png", dpi=150)
        plt.close(fig)
    
    def generate_summary(self):
        blowup_data = []
        
        for file_path in self.h5_files:
            file_id = Path(file_path).stem
            
            with h5py.File(file_path, 'r') as f:
                meta = dict(f['metadata'].attrs)
                grid_data = dict(f['grid'].attrs)
                time_data = dict(f['time'].attrs)
                
                u_data = f['u'][:]
                m = f['focusing/m'][:]
                c = f['anisotropy/c'][:]
 
                self.L = grid_data['Lx']    

                dx = 2 * grid_data['Lx'] / (grid_data['nx'] - 1)
                dy = 2 * grid_data['Ly'] / (grid_data['ny'] - 1)
                dz = 2 * grid_data['Lz'] / (grid_data['nz'] - 1)
                
                max_amplitude = np.array([np.max(np.abs(u_data[t])) for t in range(u_data.shape[0])])
                max_gradient = self._calculate_max_gradient(u_data, dx, dy, dz)
                
                energy = self._calculate_energy_metrics(u_data, m, c, dx, dy, dz)
                
                #is_blowup = (max_amplitude[-1] / max_amplitude[0] > 10 or 
                #           max_gradient[-1] / max_gradient[0] > 50 or
                #           np.abs((energy['total_energy'][-1] - energy['total_energy'][0]) / 
                #                 energy['total_energy'][0]) > 0.2)

                # let's try less complicated
                factor = 45
                is_blowup = max_amplitude[-1] / max_amplitude[0] > factor
                
                blowup_data.append({
                    'file_id': file_id,
                    'phenomenon': meta.get('phenomenon', 'Unknown'),
                    'problem_type': meta.get('problem_type', 'Unknown'),
                    'max_amplitude_growth': max_amplitude[-1] / max_amplitude[0],
                    'max_gradient_growth': max_gradient[-1] / max_gradient[0],
                    'energy_variation': np.abs((energy['total_energy'][-1] - energy['total_energy'][0]) / 
                                              energy['total_energy'][0]),
                    'is_blowup': is_blowup
                })
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig)
        
        fig.suptitle(f"Summary Dashboard - All Simulations", fontsize=16, y=0.98)
        
        ax1 = fig.add_subplot(gs[0, 0])
        file_ids = [d['file_id'] for d in blowup_data]
        amp_growth = [d['max_amplitude_growth'] for d in blowup_data]
        colors = ['red' if d['is_blowup'] else 'blue' for d in blowup_data]
        ax1.bar(file_ids, amp_growth, color=colors)
        ax1.set_title('Maximum Amplitude Growth')
        ax1.set_xlabel('Simulation')
        ax1.set_ylabel('growth Ratio (Final/Initial)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', rotation=90)
        
        ax2 = fig.add_subplot(gs[0, 1])
        grad_growth = [d['max_gradient_growth'] for d in blowup_data]
        ax2.bar(file_ids, grad_growth, color=colors)
        ax2.set_title('Maximum Gradient Growth')
        ax2.set_xlabel('Simulation')
        ax2.set_ylabel('Growth Ratio (Final/Initial)')
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=90)
        
        ax3 = fig.add_subplot(gs[1, 0])
        energy_var = [d['energy_variation'] for d in blowup_data]
        ax3.bar(file_ids, energy_var, color=colors)
        ax3.set_title('Relative Energy Variation')
        ax3.set_xlabel('Simulation')
        ax3.set_ylabel('$|E_T - E_0|/|E_0|$')
        ax3.set_yscale('log')
        ax3.tick_params(axis='x', rotation=90)
        
        ax4 = fig.add_subplot(gs[1, 1])
        blowup_count = sum(1 for d in blowup_data if d['is_blowup'])
        stable_count = len(blowup_data) - blowup_count
        ax4.pie([blowup_count, stable_count], 
               labels=['Blowup Detected', 'Stable Evolution'], 
               autopct='%1.1f%%', 
               colors=['red', 'blue'])
        ax4_title_str = f'{blowup_count}/{len(blowup_data)} simulations show blowup' +\
                '\n$\\frac{|u_T|}{|u_0|} > f$\n' + f'$f=${factor}'   
        ax4.set_title(ax4_title_str)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / "summary_dashboard.png", dpi=150)
        plt.close(fig)
    
    def _plot_isosurface(self, ax, data, value_name="Value", cmap='viridis', levels=3):
        try:
            data_cpy = np.abs(data)
            vmin, vmax = data_cpy.min(), data_cpy.max()
            
            if vmin == vmax:
                ax.text(0.5, 0.5, "Constant field - no isosurface to display", 
                       horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                return
            
            step = (vmax - vmin) / (levels + 1)
            isovalues = [vmin + step * (i + 1) for i in range(levels)]
            
            cmap_obj = plt.get_cmap(cmap)
            colors = [cmap_obj((val - vmin) / (vmax - vmin)) for val in isovalues]
            
            nx, ny, nz = data.shape
            
            for level, color in zip(isovalues, colors):
                verts, faces, _, _ = measure.marching_cubes(data, level=level)
                
                mesh = Poly3DCollection(verts[faces], alpha=0.3)
                mesh.set_edgecolor('none')
                mesh.set_facecolor(color)
                ax.add_collection3d(mesh)
            
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_zlim(0, nz)
            
            ax.set_title(f"{value_name} - Isosurface")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        except Exception as e:
            print(f"Could not generate isosurface {str(e)}")
            #ax.text(0.5, 0.5, f"Could not generate isosurface: {str(e)}", 
            #       horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    def _plot_volume_slices(self, ax, data, value_name="Value", cmap='viridis'):
        try:
            nx, ny, nz = data.shape
            
            x_mid, y_mid, z_mid = nx // 2, ny // 2, nz // 2
            
            x, y, z = np.mgrid[:nx, :ny, :nz]
            
            xy_slice = np.zeros((nx, ny, nz))
            xy_slice[:, :, z_mid] = 1
            
            xz_slice = np.zeros((nx, ny, nz))
            xz_slice[:, y_mid, :] = 1
            
            yz_slice = np.zeros((nx, ny, nz))
            yz_slice[x_mid, :, :] = 1
            
            vmin, vmax = data.min(), data.max()
            
            ax.scatter(x, y, z, c=data, cmap=cmap, alpha=xy_slice * 0.7, marker='.', vmin=vmin, vmax=vmax)
            ax.scatter(x, y, z, c=data, cmap=cmap, alpha=xz_slice * 0.7, marker='.', vmin=vmin, vmax=vmax)
            ax.scatter(x, y, z, c=data, cmap=cmap, alpha=yz_slice * 0.7, marker='.', vmin=vmin, vmax=vmax)
            
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_zlim(0, nz)
            
            ax.set_title(f"{value_name} - Volume Slices")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not generate volume slices: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    def _plot_phase_visualization(self, ax, data, value_name="Phase"):
        try:
            amplitude = np.abs(data)
            phase = np.angle(data)
            
            nx, ny, nz = data.shape
            x_mid, y_mid, z_mid = nx // 2, ny // 2, nz // 2
            
            threshold = amplitude.max() * 0.2
            mask = amplitude > threshold
            
            x_sample = np.arange(0, nx, max(1, nx // 20))
            y_sample = np.arange(0, ny, max(1, ny // 20))
            z_sample = np.arange(0, nz, max(1, nz // 20))
            
            X, Y, Z = np.meshgrid(x_sample, y_sample, z_sample, indexing='ij')
            
            U = np.cos(phase[X, Y, Z]) * amplitude[X, Y, Z] / amplitude.max()
            V = np.sin(phase[X, Y, Z]) * amplitude[X, Y, Z] / amplitude.max()
            W = np.zeros_like(U)
            
            ax.quiver(X, Y, Z, U, V, W, length=3.0, normalize=True, cmap='hsv', 
                     linewidth=2.0, arrow_length_ratio=0.3)
            
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_zlim(0, nz)
            
            ax.set_title(f"{value_name} - Vector Visualization")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not generate phase visualization: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    def _calculate_energy_metrics(self, u_data, m, c, dx, dy, dz):
        nt = u_data.shape[0]
        dV = dx * dy * dz
        
        kinetic_energy = np.zeros(nt)
        potential_energy = np.zeros(nt)
        total_energy = np.zeros(nt)
        mass = np.zeros(nt)

        # downsampling rate
        dr = u_data[0].shape[0]
         
        # we want u_i and m, c to work nicely together
        ms = m.shape[0]
        m = downsample_interpolation_3d(
                m.reshape(1, ms, ms, ms),
                target_shape=(dr, dr, dr),
                Lx=self.L,
                Ly=self.L,
                Lz=self.L
            )
        m = m.reshape(dr, dr, dr)

        cs = c.shape[0]
        c = downsample_interpolation_3d(
                c.reshape(1, cs, cs, cs),
                target_shape=(dr, dr, dr),
                Lx=self.L,
                Ly=self.L,
                Lz=self.L
            )
        c = c.reshape(dr, dr, dr)
 
        for t in range(nt):
            u = u_data[t]
            
            grad_u_x = np.gradient(u, dx, axis=0)
            grad_u_y = np.gradient(u, dy, axis=1)
            grad_u_z = np.gradient(u, dz, axis=2)
            
            kinetic_energy[t] = 0.5 * dV * np.sum(c * (np.abs(grad_u_x)**2 + 
                                                      np.abs(grad_u_y)**2 + 
                                                      np.abs(grad_u_z)**2))
            
            potential_energy[t] = -0.5 * dV * np.sum(m * np.abs(u)**4)
            total_energy[t] = kinetic_energy[t] + potential_energy[t]
            mass[t] = dV * np.sum(np.abs(u)**2)
        
        return {
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'total_energy': total_energy,
            'mass': mass
        }
    
    def _calculate_max_gradient(self, u_data, dx, dy, dz):
        nt = u_data.shape[0]
        max_gradient = np.zeros(nt)
        
        for t in range(nt):
            u = u_data[t]
            grad_u_x = np.gradient(u, dx, axis=0)
            grad_u_y = np.gradient(u, dy, axis=1)
            grad_u_z = np.gradient(u, dz, axis=2)
            
            gradient_norm = np.sqrt(np.abs(grad_u_x)**2 + np.abs(grad_u_y)**2 + np.abs(grad_u_z)**2)
            max_gradient[t] = np.max(gradient_norm)
            
        return max_gradient

def complex_phase_evolution_display(u, timepoints=[0, -1], timepoints_names=[0, 1], fname=None):
    nt, nx, ny, nz = u.shape

    if len(timepoints) == 1:
        timepoints = [timepoints[0]]
    elif len(timepoints) == 2:
        step = max(1, (timepoints[1] - timepoints[0]) // 4)
        timepoints = range(timepoints[0], timepoints[1] + 1, step)

    fig, axes = plt.subplots(1, len(timepoints), figsize=(4*len(timepoints), 4))
    if len(timepoints) == 1:
        axes = [axes]

    assert len(timepoints)  == len(timepoints_names)
    for i, t in enumerate(timepoints):
        z_mid = nz // 2
        data = u[t, :, :, z_mid]

        phase = np.angle(data)
        magnitude = np.abs(data)

        hsv_data = np.zeros((nx, ny, 3))
        hsv_data[:, :, 0] = (phase + np.pi) / (2 * np.pi)
        hsv_data[:, :, 1] = 1.0
        hsv_data[:, :, 2] = 1.0 - np.exp(-magnitude / np.max(magnitude) * 3)

        rgb_data = hsv_to_rgb(hsv_data)

        im = axes[i].imshow(rgb_data, origin='lower')
        axes[i].set_title(f't = {timepoints_names[i]:.2f}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    return fig, axes

def critical_phenomena_tracker(u, timepoints=[0, -1], timepoint_names=[0, -1], fname=None):
    nt, nx, ny, nz = u.shape

    if len(timepoints) == 1:
        timepoints = [timepoints[0]]
    elif len(timepoints) == 2:
        step = max(1, (timepoints[1] - timepoints[0]) // 4)
        timepoints = range(timepoints[0], timepoints[1] + 1, step)

    fig, axes = plt.subplots(2, len(timepoints), figsize=(4*len(timepoints), 8))
    if len(timepoints) == 1:
        axes = axes.reshape(2, 1)

    for i, t in enumerate(timepoints):
        z_mid = nz // 2
        data = u[t, :, :, z_mid]

        magnitude = np.abs(data)
        max_mag = np.max(magnitude)

        phase = np.angle(data)

        phase_gradient_x, phase_gradient_y = np.gradient(phase)
        phase_curl = np.abs(np.gradient(phase_gradient_x)[1] - np.gradient(phase_gradient_y)[0])

        zeros_mask = magnitude < max_mag * 0.05

        axes[0, i].imshow(magnitude, origin='lower', cmap='viridis')
        axes[0, i].contour(zeros_mask, levels=[0.5], colors='red', linewidths=2)
        axes[0, i].set_title(f'Zeros (t = {timepoint_names[i]:.2f})')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        im = axes[1, i].imshow(phase_curl, origin='lower', cmap='hot')
        axes[1, i].set_title(f'Phase Singularities (t = {timepoint_names[i]:.2f})')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    return fig, axes

def multi_scale_signature_display(u, timepoints=[0, -1], timepoint_names=[0., 1.],
        scales=[2, 8, 32], fname=None):
    nt, nx, ny, nz = u.shape

    assert len(timepoints) == len(timepoint_names)

    if len(timepoints) == 1:
        timepoints = [timepoints[0]]
    elif len(timepoints) == 2:
        step = max(1, (timepoints[1] - timepoints[0]) // 4)
        timepoints = range(timepoints[0], timepoints[1] + 1, step)

    fig, axes = plt.subplots(len(scales), len(timepoints), figsize=(4*len(timepoints), 3*len(scales)))

    if len(timepoints) == 1 and len(scales) == 1:
        axes = np.array([[axes]])
    elif len(timepoints) == 1:
        axes = axes.reshape(-1, 1)
    elif len(scales) == 1:
        axes = axes.reshape(1, -1)

    z_mid = nz // 2
    max_val = 0

    wavelet_results = {}
    for i, t in enumerate(timepoints):
        data = np.abs(u[t, :, :, z_mid])

        for j, scale in enumerate(scales):
            wavelet = ndimage.gaussian_filter(data, sigma=scale)
            detail = data - wavelet

            wavelet_results[(i, j)] = detail
            max_val = max(max_val, np.max(np.abs(detail)))

    for (i, j), detail in wavelet_results.items():
        im = axes[j, i].imshow(detail, origin='lower', cmap='seismic',
                             vmin=-max_val, vmax=max_val)
        axes[j, i].set_title(f't = {timepoint_names[i]:.2f}, scale = {scales[j]}')
        axes[j, i].set_xticks([])
        axes[j, i].set_yticks([])

        divider = make_axes_locatable(axes[j, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig) 
    return fig, axes

def temporal_phase_portrait(u, T=1., spatial_points=None, fname=None):
    nt, nx, ny, nz = u.shape

    if spatial_points is None:
        spatial_points = [(nx//2, ny//2, nz//2),
                        (nx//4, ny//2, nz//2),
                        (3*nx//4, ny//2, nz//2)]

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, (x, y, z) in enumerate(spatial_points):
        trajectory = u[:, x, y, z]

        points = np.array([trajectory.real, trajectory.imag]).T
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

        norm = plt.Normalize(0, T)
        linescale = np.linspace(0, T, nt-1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(linescale)
        line = ax.add_collection(lc)

        ax.plot(trajectory.real[0], trajectory.imag[0], 'o',
             color=cm.viridis(0), markersize=8)
        ax.plot(trajectory.real[-1], trajectory.imag[-1], 's',
             color=cm.viridis(1.0), markersize=8)

    ax.set_xlabel('$\Re(u)$')
    ax.set_ylabel('$\Im(u)$')
    ax.grid(True)
    ax.set_aspect('equal')

    fig.colorbar(line, ax=ax, label='Time')

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    return fig, ax

def spectral_signature_evolution(u, timepoints=[0, -1], timepoint_names=[0, 1.], fname=None):
    nt, nx, ny, nz = u.shape
    assert len(timepoints) == len(timepoint_names)

    if len(timepoints) == 1:
        timepoints = [timepoints[0]]
    elif len(timepoints) == 2:
        step = max(1, (timepoints[1] - timepoints[0]) // 4)
        timepoints = range(timepoints[0], timepoints[1] + 1, step)

    fig, axes = plt.subplots(1, len(timepoints), figsize=(4*len(timepoints), 4))
    if len(timepoints) == 1:
        axes = [axes]

    z_mid = nz // 2
    max_val = 0

    for i, t in enumerate(timepoints):
        data = u[t, :, :, z_mid]

        fft_data = np.fft.fftshift(np.fft.fft2(data))
        power_spectrum = np.log(np.abs(fft_data) + 1)

        max_val = max(max_val, np.max(power_spectrum))

        im = axes[i].imshow(power_spectrum, origin='lower', cmap='inferno')
        axes[i].set_title(f't = {timepoint_names[i]:.2f}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for i in range(len(timepoints)):
        im = axes[i].images[0]
        im.set_clim(0, max_val)

        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    return fig, axes


def complex_isosurface_3d(u, timestep=0, iso_value=0.5, fname=None):
    nt, nx, ny, nz = u.shape
    
    magnitude = np.abs(u[timestep])
    phase = np.angle(u[timestep])
    
    norm_magnitude = magnitude / np.max(magnitude)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    verts, faces, normals, values = measure.marching_cubes(norm_magnitude, iso_value)
    
    mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                          triangles=faces, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Isosurface at |u| = {iso_value:.2f} (t = {timestep})')
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    
    return fig, ax

def compare_timepoints_isosurfaces(u, timepoints=[0, -1], timepoints_names=[0., 1.],
        iso_value=0.5, fname=None):
    nt, nx, ny, nz = u.shape
    
    if len(timepoints) == 2 and timepoints[1] == -1:
        timepoints = [timepoints[0], nt-1]
    
    if len(timepoints) == 2:
        timepoints = np.linspace(timepoints[0], timepoints[1], 4, dtype=int)
    
    fig = plt.figure(figsize=(16, 12))
    
    for i, timestep in enumerate(timepoints):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        magnitude = np.abs(u[timestep])
        norm_magnitude = magnitude / np.max(magnitude)
        
        verts, faces, normals, values = measure.marching_cubes(norm_magnitude, iso_value)
        
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                       triangles=faces, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f't = {timepoints_names[i]:.2f}')
    
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    
    return fig

def phase_colored_isosurfaces(u, timestep=0, timestep_name=0., iso_value=0.5, fname=None):
    nt, nx, ny, nz = u.shape
    
    magnitude = np.abs(u[timestep])
    phase = np.angle(u[timestep])
    
    norm_magnitude = magnitude / np.max(magnitude)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    verts, faces, normals, values = measure.marching_cubes(norm_magnitude, iso_value)
    
    x, y, z = verts.T
    i, j, k = np.floor(verts).astype(int).T
    i = np.clip(i, 0, nx-1)
    j = np.clip(j, 0, ny-1)
    k = np.clip(k, 0, nz-1)
    
    phase_values = np.array([phase[i[f[0]], j[f[0]], k[f[0]]] for f in faces])
    norm_phase = (phase_values + np.pi) / (2 * np.pi)
    
    ax.plot_trisurf(x, y, z, triangles=faces, alpha=0.8)
    surf = ax.plot_trisurf(x, y, z, triangles=faces, alpha=0.8)
    
    colors = cm.hsv(norm_phase)
    surf.set_facecolor(colors)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Phase-colored isosurface (|u| = {iso_value:.2f}, t = {timestep_name:.2f})')
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    
    return fig, ax

def critical_points_3d(u, timestep=0, timestep_name=0., threshold=1e-2, fname=None):
    nt, nx, ny, nz = u.shape
    
    magnitude = np.abs(u[timestep])
    phase = np.angle(u[timestep])
    
    max_mag = np.max(magnitude)
    
    grad_x, grad_y, grad_z = np.gradient(u[timestep])
    grad_magnitude = np.sqrt(
            grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    
    zeros = (magnitude < threshold * max_mag) & (grad_magnitude < threshold * np.max(grad_magnitude))
    
    x, y, z = np.where(zeros)
    
    if len(x) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        norm_magnitude = magnitude / np.max(magnitude)
        verts, faces, _, _ = measure.marching_cubes(norm_magnitude, threshold)
        
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                       triangles=faces, color='lightgray', alpha=0.3)
        
        ax.scatter(x, y, z, c='red', s=50, alpha=1.0)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Critical points (t = {timestep_name:.2f})')
        
        if fname is not None:
            plt.savefig(fname, dpi=200)
            plt.close(fig)
        
        return fig, ax
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.text(nx/2, ny/2, nz/2, "No critical points found", 
               ha='center', va='center', fontsize=14)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Critical points (t = {timestep})')
        
        if fname is not None:
            plt.savefig(fname, dpi=200)
            plt.close(fig)
        
        return fig, ax

def scatter_3d_magnitude_phase(u, timestep=0, sampling=4, fname=None):
    nt, nx, ny, nz = u.shape
    
    magnitude = np.abs(u[timestep])
    phase = np.angle(u[timestep])
    
    max_mag = np.max(magnitude)
    
    x, y, z = np.meshgrid(
        np.arange(0, nx, sampling),
        np.arange(0, ny, sampling),
        np.arange(0, nz, sampling)
    )
    
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    magnitudes = np.array([magnitude[i, j, k] for i, j, k in zip(x, y, z)])
    phases = np.array([phase[i, j, k] for i, j, k in zip(x, y, z)])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, 
                        c=phases, 
                        s=magnitudes/max_mag*100,
                        cmap='hsv', 
                        alpha=0.6,
                        vmin=-np.pi, vmax=np.pi)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Scatter (t = {timestep}, sampling = {sampling})')
    
    cbar = plt.colorbar(scatter, ax=ax, label='Phase')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    
    if fname is not None:
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    
    return fig, ax


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate NLSE analysis dashboards')
    parser.add_argument('directory', help='Directory containing HDF5 files')
    parser.add_argument('--output', '-o', help='Output directory for dashboards')
    parser.add_argument('--summary-only', action='store_true', default=False,
            help='Only perform (conservative) analysis to detect blowup')
    parser.add_argument('--detailed-panels', action='store_true', default=False,
            help='Detailed visualizations of different trajectories') 
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generator = NLSEDashboardGenerator(args.directory, args.output, args.summary_only, args.detailed_panels)
    generator.process_all_files()
    print(f"Dashboards generated in {generator.output_dir}")
