import os
import argparse
import subprocess
import time
from pathlib import Path
import uuid
import sys
import datetime
import logging
import shutil

import h5py
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
#from tqdm import tqdm

from devise_structured_samplers import (
    sampler_system_specific_normalized,
    sampler_interface_study,
    sampler_amplitude_sensitivity
)

from precise_nlse_phenomena import NLSEPhenomenonSampler
from periodic_trap import make_periodic_boxes
from real_samplers import generate_grf

def setup_logging(log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format
    )
    return logging.getLogger(__name__)
logger = setup_logging()
#########


def analyze_conservation_laws(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    nx = args.get('nx', 200)
    ny = args.get('ny', 200)
    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)

    dx = 2 * Lx / (nx - 1)
    dy = 2 * Ly / (ny - 1)
    time_values = np.linspace(0, T, num_snapshots)

    system_names = []
    trajectories = []
    system_params = []

    if 'cubic' in combined_results and 'traj' in combined_results['cubic']:
        system_names.append("Cubic")
        trajectories.append(combined_results['cubic']['traj'])
        system_params.append(('cubic', None))

    for key in combined_results:
        if key.startswith('cubic_quintic_') and 'traj' in combined_results[key]:
            s1 = combined_results[key]['s1']
            s2 = combined_results[key]['s2']
            system_names.append(f"CQ: s1={s1}, s2={s2}")
            trajectories.append(combined_results[key]['traj'])
            system_params.append(('cubic_quintic', (s1, s2)))

    for key in combined_results:
        if key.startswith('saturating_') and 'traj' in combined_results[key]:
            kappa = combined_results[key]['kappa']
            system_names.append(f"Sat: κ={kappa}")
            trajectories.append(combined_results[key]['traj'])
            system_params.append(('saturating', kappa))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for i, (system_name, traj, (system_type, params)) in enumerate(zip(system_names, trajectories, system_params)):
        norm = calculate_norm(traj, dx, dy)

        if system_type == 'cubic':
            energy = calculate_energy_cubic(traj, dx, dy)
        elif system_type == 'cubic_quintic':
            s1, s2 = params
            energy = calculate_energy_cubic_quintic(traj, dx, dy, s1, s2)
        elif system_type == 'saturating':
            kappa = params
            energy = calculate_energy_saturating(traj, dx, dy, kappa)
        norm_rel = norm / norm[0]
        energy_rel = energy / energy[0]

        axes[0].plot(time_values, norm_rel, label=system_name, linewidth=2)
        axes[1].plot(time_values, energy_rel, label=system_name, linewidth=2)

    axes[0].set_title("Norm Conservation")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Relative Norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Energy Conservation")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Relative Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"Conservation Laws: {sampler_type}, {m_type}, m_scale={m_scaling:.2f}")
    output_file = analysis_dir / f"conservation_{sampler_type}_{m_type}_m{m_scaling:.2f}_{run_id}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def analyze_intensity_evolution(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)
    time_values = np.linspace(0, T, num_snapshots)

    system_names = []
    trajectories = []

    if 'cubic' in combined_results and 'traj' in combined_results['cubic']:
        system_names.append("Cubic")
        trajectories.append(combined_results['cubic']['traj'])

    for key in combined_results:
        if key.startswith('cubic_quintic_') and 'traj' in combined_results[key]:
            s1 = combined_results[key]['s1']
            s2 = combined_results[key]['s2']
            system_names.append(f"CQ: s1={s1}, s2={s2}")
            trajectories.append(combined_results[key]['traj'])

    for key in combined_results:
        if key.startswith('saturating_') and 'traj' in combined_results[key]:
            kappa = combined_results[key]['kappa']
            system_names.append(f"Sat: κ={kappa}")
            trajectories.append(combined_results[key]['traj'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (system_name, traj) in enumerate(zip(system_names, trajectories)):
        max_intensity = np.array([np.max(np.abs(traj[t])**2) for t in range(num_snapshots)])
        total_intensity = np.array([np.sum(np.abs(traj[t])**2) for t in range(num_snapshots)])

        ax1.plot(time_values, max_intensity, label=system_name, linewidth=2)
        ax2.plot(time_values, total_intensity / total_intensity[0], label=system_name, linewidth=2)

    ax1.set_title("Peak Intensity Evolution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Max |u|²")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Normalized Total Intensity")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("∫|u|² / ∫|u₀|²")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"Intensity Evolution: {sampler_type}, {m_type}, m_scale={m_scaling:.2f}")

    output_file = analysis_dir / f"intensity_{sampler_type}_{m_type}_m{m_scaling:.2f}_{run_id}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def create_snapshot_comparison(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    comparison_dir = Path(output_dir) / "comparisons"
    comparison_dir.mkdir(exist_ok=True, parents=True)

    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)

    system_names = []
    trajectories = []

    if 'cubic' in combined_results and 'traj' in combined_results['cubic']:
        system_names.append("Cubic")
        trajectories.append(combined_results['cubic']['traj'])

    for key in combined_results:
        if key.startswith('cubic_quintic_') and 'traj' in combined_results[key]:
            s1 = combined_results[key]['s1']
            s2 = combined_results[key]['s2']
            system_names.append(f"CQ: s1={s1}, s2={s2}")
            trajectories.append(combined_results[key]['traj'])

    for key in combined_results:
        if key.startswith('saturating_') and 'traj' in combined_results[key]:
            kappa = combined_results[key]['kappa']
            system_names.append(f"Sat: κ={kappa}")
            trajectories.append(combined_results[key]['traj'])

    time_indices = [0, num_snapshots // 2, num_snapshots - 1]
    time_labels = [f"t=0", f"t={T/2:.2f}", f"t={T:.2f}"]

    fig, axes = plt.subplots(len(system_names), 3, figsize=(15, 3 * len(system_names)))

    extent = [-Lx, Lx, -Ly, Ly]

    for i, (system_name, traj) in enumerate(zip(system_names, trajectories)):
        for j, (time_idx, time_label) in enumerate(zip(time_indices, time_labels)):
            if len(system_names) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            intensity = np.abs(traj[time_idx])**2
            vmax = np.max(intensity) 
            im = ax.imshow(intensity, extent=extent, vmin=0, vmax=vmax, cmap='viridis')
            ax.set_title(f"{system_name}, {time_label}\nMax: {vmax:.3f}")
            ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Comparison at Different Times: {sampler_type}, {m_type}, m_scale={m_scaling:.2f}")

    output_file = comparison_dir / f"snapshots_{sampler_type}_{m_type}_m{m_scaling:.2f}_{run_id}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def create_difference_heatmaps(combined_results,
        output_dir,
        run_id,
        m_scaling,
        sampler_type,
        m_type,
        args):
    diff_dir = Path(output_dir) / "analysis"
    diff_dir.mkdir(exist_ok=True, parents=True)

    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    nx = args.get('nx', 200)
    ny = args.get('ny', 200)
    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(x, y)

    if 'cubic' not in combined_results or 'traj' not in combined_results['cubic']:
        return "Error: Cubic trajectory not found for reference"

    cubic_traj = combined_results['cubic']['traj']
    system_names = []
    trajectories = []

    for key in combined_results:
        if key != 'cubic' and 'traj' in combined_results[key]:
            trajectories.append(combined_results[key]['traj'])
            if key.startswith('cubic_quintic_'):
                s1 = combined_results[key]['s1']
                s2 = combined_results[key]['s2']
                system_names.append(f"CQ: s1={s1}, s2={s2}")
            elif key.startswith('saturating_'):
                kappa = combined_results[key]['kappa']
                system_names.append(f"Sat: κ={kappa}")
            else:
                system_names.append(key)

    time_indices = [0, num_snapshots // 4, num_snapshots // 2, 3 * num_snapshots // 4, num_snapshots - 1]
    time_labels = [f"t=0", f"t={T/4:.2f}", f"t={T/2:.2f}", f"t={3*T/4:.2f}", f"t={T:.2f}"]

    for i, (system_name, traj) in enumerate(zip(system_names, trajectories)):
        fig, axes = plt.subplots(1, len(time_indices), figsize=(5*len(time_indices), 5))

        max_diff = 0
        for j, (time_idx, time_label) in enumerate(zip(time_indices, time_labels)):
            cubic_intensity = np.abs(cubic_traj[time_idx])**2
            system_intensity = np.abs(traj[time_idx])**2

            max_cubic = np.max(cubic_intensity)
            max_system = np.max(system_intensity)

            norm_cubic = cubic_intensity / max_cubic
            norm_system = system_intensity / max_system

            diff = norm_system - norm_cubic
            max_diff = max(max_diff, np.max(np.abs(diff)))

        for j, (time_idx, time_label) in enumerate(zip(time_indices, time_labels)):
            cubic_intensity = np.abs(cubic_traj[time_idx])**2
            system_intensity = np.abs(traj[time_idx])**2

            max_cubic = np.max(cubic_intensity)
            max_system = np.max(system_intensity)

            norm_cubic = cubic_intensity / max_cubic
            norm_system = system_intensity / max_system

            diff = norm_system - norm_cubic

            im = axes[j].imshow(diff, extent=[-Lx, Lx, -Ly, Ly], origin='lower',
                               cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
            axes[j].set_title(f"{time_label}")
            axes[j].set_xlabel("x")
            if j == 0:
                axes[j].set_ylabel("y")

        fig.colorbar(im, ax=axes, label="Normalized Intensity Difference")
        plt.suptitle(f"Difference from Cubic: {system_name}")

        output_file = diff_dir / f"diff_{system_name.replace(':', '_').replace('=', '_').replace(' ', '_').replace(',', '')}_{run_id}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
    return str(diff_dir)

def analyze_instability_growth_rates(combined_results,
        output_dir, run_id, m_scaling, sampler_type,
        m_type, args):
    from scipy.ndimage import gaussian_filter
    instab_dir = Path(output_dir) / "analysis"
    instab_dir.mkdir(exist_ok=True, parents=True)

    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)
    dt = T / (num_snapshots - 1)
    time_points = np.linspace(0, T, num_snapshots)

    system_names = []
    growth_rates = []
    perturbation_norms = {}

    for key, data in combined_results.items():
        if 'traj' in data:
            if key == 'cubic':
                system_name = "Cubic"
            elif key.startswith('cubic_quintic_'):
                s1 = data['s1']
                s2 = data['s2']
                system_name = f"CQ: s1={s1}, s2={s2}"
            elif key.startswith('saturating_'):
                kappa = data['kappa']
                system_name = f"Sat: κ={kappa}"
            else:
                system_name = key

            system_names.append(system_name)
            traj = data['traj']

            initial_profile = np.abs(traj[0])**2
            max_idx = np.unravel_index(np.argmax(initial_profile), initial_profile.shape)

            fluctuations = []
            for t in range(num_snapshots):
                profile = np.abs(traj[t])**2
                smoothed = gaussian_filter(profile, sigma=2)
                fluctuation = profile - smoothed
                fluctuation_norm = np.linalg.norm(fluctuation)
                fluctuations.append(fluctuation_norm)

            perturbation_norms[system_name] = fluctuations

            non_zero_indices = np.where(np.array(fluctuations) > 1e-10)[0]
            if len(non_zero_indices) > 2:
                time_values = time_points[non_zero_indices]
                log_fluct = np.log(np.array(fluctuations)[non_zero_indices])
                slope, _ = np.polyfit(time_values, log_fluct, 1)
                growth_rates.append(slope)
            else:
                growth_rates.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(system_names, growth_rates)
    ax1.set_ylabel("Instability Growth Rate")
    ax1.set_xlabel("System Type")
    ax1.set_title("Instability Growth Rates for Different Potentials")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    for system_name in system_names:
        ax2.semilogy(time_points, perturbation_norms[system_name], label=system_name)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Perturbation Norm (log scale)")
    ax2.set_title("Evolution of Instabilities Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = instab_dir / f"instability_growth_{run_id}.png"
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def create_phase_comparison(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    comparison_dir = Path(output_dir) / "phase_comparisons"
    comparison_dir.mkdir(exist_ok=True, parents=True)

    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)

    system_names = []
    trajectories = []

    if 'cubic' in combined_results and 'traj' in combined_results['cubic']:
        system_names.append("Cubic")
        trajectories.append(combined_results['cubic']['traj'])

    for key in combined_results:
        if key.startswith('cubic_quintic_') and 'traj' in combined_results[key]:
            s1 = combined_results[key]['s1']
            s2 = combined_results[key]['s2']
            system_names.append(f"CQ: s1={s1}, s2={s2}")
            trajectories.append(combined_results[key]['traj'])

    for key in combined_results:
        if key.startswith('saturating_') and 'traj' in combined_results[key]:
            kappa = combined_results[key]['kappa']
            system_names.append(f"Sat: κ={kappa}")
            trajectories.append(combined_results[key]['traj'])
    time_indices = [0, num_snapshots // 2, num_snapshots - 1]
    time_labels = [f"t=0", f"t={T/2:.2f}", f"t={T:.2f}"]

    fig, axes = plt.subplots(len(system_names), 3, figsize=(15, 3 * len(system_names)))

    extent = [-Lx, Lx, -Ly, Ly]

    for i, (system_name, traj) in enumerate(zip(system_names, trajectories)):
        for j, (time_idx, time_label) in enumerate(zip(time_indices, time_labels)):
            if len(system_names) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            phase = np.angle(traj[time_idx])
            intensity = np.abs(traj[time_idx])**2
            max_intensity = np.max(intensity)
            mask = intensity < max_intensity * 0.05
            phase_masked = np.ma.array(phase, mask=mask)
            im = ax.imshow(phase_masked, extent=extent, cmap='hsv', vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"{system_name}, {time_label}")

            ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Phase')

    plt.suptitle(f"Phase Evolution: {sampler_type}, {m_type}, m_scale={m_scaling:.2f}")

    output_file = comparison_dir / f"phase_{sampler_type}_{m_type}_m{m_scaling:.2f}_{run_id}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def create_profile_shape_comparison(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    comparison_dir = Path(output_dir) / "profile_comparisons"
    comparison_dir.mkdir(exist_ok=True, parents=True)

    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    nx = args.get('nx', 200)
    ny = args.get('ny', 200)
    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)

    system_names = []
    trajectories = []

    if 'cubic' in combined_results and 'traj' in combined_results['cubic']:
        system_names.append("Cubic")
        trajectories.append(combined_results['cubic']['traj'])

    for key in combined_results:
        if key.startswith('cubic_quintic_') and 'traj' in combined_results[key]:
            s1 = combined_results[key]['s1']
            s2 = combined_results[key]['s2']
            system_names.append(f"CQ: s1={s1}, s2={s2}")
            trajectories.append(combined_results[key]['traj'])

    for key in combined_results:
        if key.startswith('saturating_') and 'traj' in combined_results[key]:
            kappa = combined_results[key]['kappa']
            system_names.append(f"Sat: κ={kappa}")
            trajectories.append(combined_results[key]['traj'])
    time_indices = [0, num_snapshots // 2, num_snapshots - 1]
    time_labels = [f"t=0", f"t={T/2:.2f}", f"t={T:.2f}"]

    fig, axes = plt.subplots(len(system_names), 3, figsize=(15, 3 * len(system_names)))

    for i, (system_name, traj) in enumerate(zip(system_names, trajectories)):
        for j, (time_idx, time_label) in enumerate(zip(time_indices, time_labels)):
            if len(system_names) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            intensity = np.abs(traj[time_idx])**2
            max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
            center_x, center_y = max_idx[0], max_idx[1]
            h_profile = intensity[center_x, :]
            v_profile = intensity[:, center_y]
            h_profile = h_profile / np.max(h_profile)
            v_profile = v_profile / np.max(v_profile)
            ax.plot(x, h_profile, 'b-', label='Horizontal')
            ax.plot(y, v_profile, 'r--', label='Vertical')

            ax.set_title(f"{system_name}, {time_label}")
            ax.set_xlabel("Position")
            ax.set_ylabel("Normalized Intensity")
            ax.grid(True, alpha=0.3)

            if i == 0 and j == 0:
                ax.legend()

    plt.suptitle(f"Profile Shape Evolution: {sampler_type}, {m_type}, m_scale={m_scaling:.2f}")
    output_file = comparison_dir / f"profiles_{sampler_type}_{m_type}_m{m_scaling:.2f}_{run_id}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def create_shape_metrics_comparison(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    comparison_dir = Path(output_dir) / "shape_metrics"
    comparison_dir.mkdir(exist_ok=True, parents=True)


    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    T = args.get('T', 2.0)
    num_snapshots = args.get('snapshots', 30)

    system_names = []
    trajectories = []

    if 'cubic' in combined_results and 'traj' in combined_results['cubic']:
        system_names.append("Cubic")
        trajectories.append(combined_results['cubic']['traj'])

    for key in combined_results:
        if key.startswith('cubic_quintic_') and 'traj' in combined_results[key]:
            s1 = combined_results[key]['s1']
            s2 = combined_results[key]['s2']
            system_names.append(f"CQ: s1={s1}, s2={s2}")
            trajectories.append(combined_results[key]['traj'])

    for key in combined_results:
        if key.startswith('saturating_') and 'traj' in combined_results[key]:
            kappa = combined_results[key]['kappa']
            system_names.append(f"Sat: κ={kappa}")
            trajectories.append(combined_results[key]['traj'])

    time_values = np.linspace(0, T, num_snapshots)

    peak_intensities = []
    effective_widths = []

    for traj in trajectories:
        peaks = []
        widths = []

        for t in range(num_snapshots):
            intensity = np.abs(traj[t])**2
            peak = np.max(intensity)
            peaks.append(peak)

            nx, ny = intensity.shape
            x = np.linspace(-Lx, Lx, nx)
            y = np.linspace(-Ly, Ly, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            total = np.sum(intensity)
            if total > 0:
                com_x = np.sum(X * intensity) / total
                com_y = np.sum(Y * intensity) / total
                r_squared = (X - com_x)**2 + (Y - com_y)**2
                moment2 = np.sum(r_squared * intensity) / total
                width = np.sqrt(moment2)
                widths.append(width)
            else:
                widths.append(0)

        peak_intensities.append(peaks)
        effective_widths.append(widths)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (system_name, widths) in enumerate(zip(system_names, effective_widths)):
        axes[0].plot(time_values, widths, label=system_name, linewidth=2)

    axes[0].set_title("Width Evolution")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Effective Width")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for i, (system_name, peaks, widths) in enumerate(zip(system_names, peak_intensities, effective_widths)):
        axes[1].plot(peaks, widths, 'o-', label=system_name, alpha=0.7)

    axes[1].set_title("Width vs Peak Intensity")
    axes[1].set_xlabel("Peak Intensity")
    axes[1].set_ylabel("Effective Width")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"Shape Metrics: {sampler_type}, {m_type}, m_scale={m_scaling:.2f}")

    output_file = comparison_dir / f"shape_metrics_{sampler_type}_{m_type}_m{m_scaling:.2f}_{run_id}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return str(output_file)

def calculate_norm(traj, dx, dy):
    nt = traj.shape[0]
    norm = np.zeros(nt)

    for t in range(nt):
        norm[t] = np.sum(np.abs(traj[t])**2) * dx * dy

    return norm

def calculate_energy_cubic(traj, dx, dy):
    nt, nx, ny = traj.shape
    energy = np.zeros(nt)

    for t in range(nt):
        ux = np.zeros((nx-2, ny-2), dtype=complex)
        uy = np.zeros((nx-2, ny-2), dtype=complex)
        ux = (traj[t, 2:nx, 1:ny-1] - traj[t, 0:nx-2, 1:ny-1]) / (2 * dx)
        uy = (traj[t, 1:nx-1, 2:ny] - traj[t, 1:nx-1, 0:ny-2]) / (2 * dy)
        u_interior = traj[t, 1:nx-1, 1:ny-1]
        gradient_squared = np.abs(ux)**2 + np.abs(uy)**2
        u_squared = np.abs(u_interior)**2
        nonlinear_term = u_squared**2 / 2
        integrand = gradient_squared - nonlinear_term
        energy[t] = np.sum(integrand) * dx * dy

    return energy

def calculate_energy_cubic_quintic(traj, dx, dy, s1, s2):
    nt, nx, ny = traj.shape
    energy = np.zeros(nt)

    for t in range(nt):
        ux = np.zeros((nx-2, ny-2), dtype=complex)
        uy = np.zeros((nx-2, ny-2), dtype=complex)
        ux = (traj[t, 2:nx, 1:ny-1] - traj[t, 0:nx-2, 1:ny-1]) / (2 * dx)
        uy = (traj[t, 1:nx-1, 2:ny] - traj[t, 1:nx-1, 0:ny-2]) / (2 * dy)
        u_interior = traj[t, 1:nx-1, 1:ny-1]
        gradient_squared = np.abs(ux)**2 + np.abs(uy)**2
        u_squared = np.abs(u_interior)**2
        cubic_term = s1 * u_squared**2 / 2
        quintic_term = s2 * u_squared**3 / 3
        integrand = gradient_squared - cubic_term - quintic_term
        energy[t] = np.sum(integrand) * dx * dy

    return energy

def calculate_energy_saturating(traj, dx, dy, kappa):
    nt, nx, ny = traj.shape
    energy = np.zeros(nt)

    for t in range(nt):
        ux = np.zeros((nx-2, ny-2), dtype=complex)
        uy = np.zeros((nx-2, ny-2), dtype=complex)
        ux = (traj[t, 2:nx, 1:ny-1] - traj[t, 0:nx-2, 1:ny-1]) / (2 * dx)
        uy = (traj[t, 1:nx-1, 2:ny] - traj[t, 1:nx-1, 0:ny-2]) / (2 * dy)
        u_interior = traj[t, 1:nx-1, 1:ny-1]
        gradient_squared = np.abs(ux)**2 + np.abs(uy)**2
        u_squared = np.abs(u_interior)**2
        nonlinear_term = np.log(1 + kappa * u_squared) / kappa
        integrand = gradient_squared - nonlinear_term
        energy[t] = np.sum(integrand) * dx * dy

    return energy

def perform_comprehensive_analysis(combined_results, output_dir, run_id, m_scaling, sampler_type, m_type, args):
    output_files = []

    try:
        conservation_file = analyze_conservation_laws(combined_results, output_dir, run_id,
                                                     m_scaling, sampler_type, m_type, args)
        output_files.append(conservation_file)
    except Exception as e:
        logging.error(f"Error analyzing conservation laws: {e}")

    try:
        intensity_file = analyze_intensity_evolution(combined_results, output_dir, run_id,
                                                    m_scaling, sampler_type, m_type, args)
        output_files.append(intensity_file)
    except Exception as e:
        logging.error(f"Error analyzing intensity evolution: {e}")

    try:
        snapshot_file = create_snapshot_comparison(combined_results, output_dir, run_id,
                                                  m_scaling, sampler_type, m_type, args)
        output_files.append(snapshot_file)
    except Exception as e:
        logging.error(f"Error creating snapshot comparison: {e}")

    try:
        phase_file = create_phase_comparison(combined_results, output_dir, run_id,
                                                  m_scaling, sampler_type, m_type, args)
        output_files.append(phase_file)
    except Exception as e:
        logging.error(f"Error creating phase comparison: {e}")

    try:
        profile_shape_file = create_profile_shape_comparison(combined_results, output_dir, run_id,
                                                  m_scaling, sampler_type, m_type, args)
        output_files.append(profile_shape_file)
    except Exception as e:
        logging.error(f"Error creating profile shape comparison: {e}")

    try:
        shape_metrics_file = create_shape_metrics_comparison(combined_results, output_dir, run_id,
                                                  m_scaling, sampler_type, m_type, args)
        output_files.append(shape_metrics_file)
    except Exception as e:
        logging.error(f"Error creating shape metrics comparison: {e}")
     

    try:
        instability_file = analyze_instability_growth_rates(combined_results,
                            output_dir, run_id, m_scaling, sampler_type,
                            m_type, args)
    except Exception as e:
        logging.error(f"Error creating shape instability analysis: {e}")
    try: 
        difference_heatmaps_file = create_difference_heatmaps(combined_results,
            output_dir, run_id, m_scaling, sampler_type, m_type, args)
    except Exception as e:
        logging.error(f"Error creating difference heatmaps: {e}")


    return output_files

def cleanup_intermediate_files(combined_results):
    files_to_remove = []
    for key in combined_results:
        if 'traj_file' in combined_results[key]:
            files_to_remove.append(combined_results[key]['traj_file'])

    for file_path in files_to_remove:
        try:
            os.unlink(file_path)
            logging.info(f"Removed intermediate file: {file_path}")
        except Exception as e:
            logging.error(f"Error removing {file_path}: {e}")

    return files_to_remove


##########
"""
def calculate_physical_scaling(u0, m, nonlinearity_type=None, params=None):
    nx, ny = u0.shape
    dx = 2 * 10.0 / (nx - 1)
    dy = 2 * 10.0 / (ny - 1)
    
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K_SQ = KX**2 + KY**2
    
    u0_ft = np.fft.fft2(u0)
    laplacian_ft = -K_SQ * u0_ft
    laplacian_u0 = np.fft.ifft2(laplacian_ft).real
    
    max_laplacian = np.max(np.abs(laplacian_u0))
    max_m = np.max(np.abs(m))
    
    if nonlinearity_type == 'cubic' or nonlinearity_type is None:
        nonlinear_term = np.abs(u0)**2 * u0
    elif nonlinearity_type == 'cubic_quintic':
        s1 = params.get('s1', 1.0)
        s2 = params.get('s2', -0.1)
        nonlinear_term = (s1 * np.abs(u0)**2 + s2 * np.abs(u0)**4) * u0
    elif nonlinearity_type == 'saturating':
        kappa = params.get('kappa', 1.0)
        nonlinear_term = (np.abs(u0)**2 / (1 + kappa * np.abs(u0)**2)) * u0
    else:
        nonlinear_term = np.abs(u0)**2 * u0
    
    max_nonlinear = np.max(np.abs(nonlinear_term))
    
    if max_m <= 1e-10 or max_nonlinear <= 1e-10:
        return 1.0
    
    scale = max_laplacian / (max_m * max_nonlinear)
    
    if nonlinearity_type == 'cubic':
        scale = np.clip(scale, 0.1, 100.0)
    elif nonlinearity_type == 'cubic_quintic':
        scale = np.clip(scale, 0.01, 10.0)
    if nonlinearity_type == 'saturating':
        kappa = params.get('kappa', 1.0)
        max_amplitude = np.max(np.abs(u0)**2)
        saturation_adjustment = 1.0 / (1.0 + 0.5 * kappa * max_amplitude)
        scale = scale * saturation_adjustment
        scale = np.clip(scale, 0.1, 15.0)
    else:
        scale = np.clip(scale, 0.01, 100.0)
    
    if not np.isfinite(scale):
        return 1.0     
    return scale

def calculate_physical_scaling(u0, m, L, nonlinearity_type=None, params=None):
    n, _ = u0.shape
    dx = 2 * L / (n - 1)
    dy = 2 * L / (n - 1)
    
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K_SQ = KX**2 + KY**2
    
    u0_ft = np.fft.fft2(u0)
    laplacian_ft = -K_SQ * u0_ft
    laplacian_u0 = np.fft.ifft2(laplacian_ft).real 
    max_laplacian = np.max(np.abs(laplacian_u0))

    mask_m_positive = m >= 0
    mask_m_negative = m  < 0

    has_pos = np.sum(mask_m_positive) > 0
    has_neg = np.sum(mask_m_negative) > 0

    m_max_pos = np.max(np.abs(m[mask_m_positive])) if has_pos else 1. 
    m_max_neg = np.max(np.abs(m[mask_m_negative])) if has_neg else 1. 

    if nonlinearity_type == 'cubic' or nonlinearity_type is None:
        nonlinear_term_pos = np.abs(u0[mask_m_positive])**2 * u0[mask_m_positive] if has_pos else 1.
        nonlinear_term_neg = np.abs(u0[mask_m_negative])**2 * u0[mask_m_negative] if has_neg else 1.
    elif nonlinearity_type == 'cubic_quintic':
        s1 = params.get('s1', 1.0)
        s2 = params.get('s2', -0.1)
        nonlinear_term_pos = (s1 * np.abs(u0[mask_m_positive])**2 + s2 * np.abs(u0[mask_m_positive])**4) * u0[mask_m_positive] if has_pos else 1.
        nonlinear_term_neg = (s1 * np.abs(u0[mask_m_negative])**2 + s2 * np.abs(u0[mask_m_negative])**4) * u0[mask_m_negative] if has_neg else 1.
    elif nonlinearity_type == 'saturating':
        kappa = params.get('kappa', 1.0)
        nonlinear_term_pos = (np.abs(u0[mask_m_positive])**2 / (1 + kappa * np.abs(u0[mask_m_positive])**2)) * u0[mask_m_positive] if has_pos else 1.
        nonlinear_term_neg = (np.abs(u0[mask_m_negative])**2 / (1 + kappa * np.abs(u0[mask_m_negative])**2)) * u0[mask_m_negative] if has_neg else 1.
    else:
        nonlinear_term_pos = np.abs(u0[mask_m_positive])**2 * u0[mask_m_positive] if has_pos else 1.
        nonlinear_term_neg = np.abs(u0[mask_m_negative])**2 * u0[mask_m_negative] if has_neg else 1.

    max_nonlinear_pos = np.max(np.abs(nonlinear_term_pos)) 
    max_nonlinear_neg = np.max(np.abs(nonlinear_term_neg))
    
    scale_pos = max_laplacian / (m_max_pos * max_nonlinear_pos)
    scale_neg = max_laplacian / (m_max_neg * max_nonlinear_neg)

    if has_pos:
        m[mask_m_positive] *= scale_pos
    if has_neg:
        m[mask_m_negative] *= scale_neg

    return m, (scale_pos, scale_neg) 
"""
def calculate_physical_scaling(u0, m, L, nonlinearity_type=None, params=None):
    # virial-based
    n, _ = u0.shape
    dx = 2 * L / (n - 1)
    dy = 2 * L / (n - 1)
    area = dx * dy

    m_focusing = m > 0
    m_defocusing = m < 0

    mass = np.sum(np.abs(u0)**2) * area
    grad_x = np.gradient(u0, dx, axis=1)
    grad_y = np.gradient(u0, dy, axis=0)
    gradient_energy = np.sum((np.abs(grad_x)**2 + np.abs(grad_y)**2) * area)

    if nonlinearity_type == 'cubic' or nonlinearity_type is None:
        V_func = lambda u: 0.5 * np.abs(u)**4
        p = 4  
    elif nonlinearity_type == 'cubic_quintic':
        s1 = params.get('s1', 1.0)
        s2 = params.get('s2', -0.1)
        V_func = lambda u: 0.5 * s1 * np.abs(u)**4 + (s2/3.0) * np.abs(u)**6
        if abs(s1) > abs(s2) * np.max(np.abs(u0)**2):
            p = 4
        else:
            p = 6
    elif nonlinearity_type == 'saturating':
        kappa = params.get('kappa', 1.0)

        V_func = lambda u: np.log(1 + kappa * np.abs(u)**2) / kappa
        u_rms = np.sqrt(np.mean(np.abs(u0)**2))
        p = 4 * (1 + kappa * u_rms**2) / (1 + 2 * kappa * u_rms**2)
    
    V_focusing = np.zeros_like(u0)
    V_defocusing = np.zeros_like(u0)
    
    V_focusing[m_focusing] = V_func(u0[m_focusing])
    V_defocusing[m_defocusing] = V_func(u0[m_defocusing])
    
    E_focusing = np.sum(m[m_focusing] * V_focusing[m_focusing]) * area
    E_defocusing = np.sum(m[m_defocusing] * V_defocusing[m_defocusing]) * area
   
    critical_mass = 11.7
    if E_focusing > 0 and abs(E_defocusing) > 0:
        competition_ratio = abs(E_defocusing) / E_focusing
    else:
        competition_ratio = 1.0
    if mass > 0.7 * critical_mass:
        focus_scale = 0.6 / (1 + 0.2 * competition_ratio)
    else:
        focus_scale = 0.8 / (1 + 0.1 * competition_ratio)
    defocus_scale = 1.2 * focus_scale
    if nonlinearity_type == 'cubic_quintic':
        s1 = params.get('s1', 1.0)
        s2 = params.get('s2', -0.1)
        
        if s1 > 0 and s2 > 0:
            focus_scale *= 0.6
            defocus_scale *= 0.8
        elif s1 > 0 and s2 < 0:
            focus_scale *= 0.8
            defocus_scale *= 1.1
    elif nonlinearity_type == 'saturating':
        kappa = params.get('kappa', 1.0)
        focus_scale *= 0.5 / (1 + 0.3 * kappa)
        defocus_scale *= 0.7 / (1 + 0.2 * kappa)

    E_total = abs(E_focusing) + abs(E_defocusing)
    
    if E_total > 0:
        focusing_weight = abs(E_focusing) / E_total
        defocusing_weight = abs(E_defocusing) / E_total
        
        unified_scale = focusing_weight * focus_scale + defocusing_weight * defocus_scale
        virial_ratio = 4 * gradient_energy / (p * abs(E_focusing + E_defocusing))
        
        if virial_ratio < 1.0:    
            unified_scale *= 0.7  
        elif virial_ratio > 4.0:  
            unified_scale *= 1.2  
            

        unified_scale = np.clip(unified_scale, 0.1, 5.0)
    else:
        unified_scale = 1.0

    unified_scale *= 3. # this is tested for systems which have been normalized as u0 = u0 / np.max(np.abs(u0))
    unified_scale = float(unified_scale) 
    m_scaled = m * unified_scale
    
    return np.real(m_scaled).astype(np.float64), (unified_scale/2, unified_scale/2)

     
def generate_and_save_field(m_scaling, sampler_type, m_type, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ic_dir = output_dir / "initial_conditions"
    ic_dir.mkdir(exist_ok=True)
    m_dir = output_dir / "m_fields"
    m_dir.mkdir(exist_ok=True)
    scaled_m_dir = output_dir / "scaled_m_fields"
    scaled_m_dir.mkdir(exist_ok=True)
    
    nx = args.nx
    ny = args.ny
    Lx = args.Lx
    Ly = args.Ly
    dx = dy = 2 * Lx / (nx - 1)
    
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    run_id = str(uuid.uuid4())[:8]
    
    sampler = NLSEPhenomenonSampler(nx, ny, Lx)
    if sampler_type == "normalized":
        u, m_base = sampler_system_specific_normalized(X, Y, 'cubic', alpha=1.0, width=1.0)
    elif sampler_type == "interface":
        u, m_base = sampler_interface_study(X, Y, 'cubic', alpha=1.0, width=1.0)
    elif sampler_type == "amplitude":
        u, m_base = sampler_amplitude_sensitivity(X, Y, 'cubic', alpha=1.0, width=1.0)
    elif sampler_type == "multi_soliton":
        u = sampler.generate_ensemble("multi_soliton", n_samples=1)[0] 
    elif sampler_type == "spectral":
        u = sampler.generate_ensemble("spectral", n_samples=1)[0]
    elif sampler_type == "chaotic":
        u = sampler.generate_ensemble("chaotic", n_samples=1)[0]
    elif sampler_type == 'logarithmic_singularity':
        u = sampler.generate_ensemble('logarithmic_singularity_adapted', n_samples=1)[0]
    elif sampler_type == 'free_singularity':
        u = sampler.generate_ensemble('free_singularity_adapted', n_samples=1)[0]
    elif sampler_type == 'multi_ring':
        u = sampler.generate_ensemble('multi_ring', n_samples=1)[0]
    elif sampler_type == 'vortex_lattice':
        u = sampler.generate_ensemble('vortex_lattice', n_samples=1)[0]
    elif sampler_type == 'one':
        u = np.ones((nx, ny), dtype=np.complex128) * (1 + 1j)
    elif sampler_type == 'neg':
        u = -1 * np.ones((nx, ny), dtype=np.complex128) * (1 + 1j)
    else:
        logger.error(sampler_type)
        raise ValueError(f"Unknown sampler type: {sampler_type}") 

    norm = np.sum(np.abs(u)**2) * dx * dy
    # not actually normalizing but it's a scale we can nicely adapt to!
    if np.abs(norm - 1.) > 1.:
        u = u / np.max(np.abs(u))
    
    if m_type == "uniform":
        m_base = np.ones_like(X)
    elif m_type == "step":
        m_base = np.ones_like(X)
        m_base[:, nx//2:] = -1.0
    elif m_type == "radial":
        r = np.sqrt((X)**2 + (Y)**2)
        m_base = np.tanh(r - Lx/2)
    elif m_type == "periodic_boxes":
        num_boxes_per_dim = 4#np.random.randint(3, 8)
        m_base = make_periodic_boxes(nx, Lx, factor=2.,
                box_length=.45 / num_boxes_per_dim, num_boxes_per_dim=num_boxes_per_dim, wall_dist=.1) 
    elif m_type == "grf":
        m_base = generate_grf(nx, ny, Lx, Ly)
    else:
        raise ValueError(f"Unknown m type: {m_type}")
    
    m = m_base * m_scaling
    
    u0_path = ic_dir / f"u0_{sampler_type}_{run_id}.npy"
    m_path = m_dir / f"m_{m_type}_{m_scaling:.4f}_{run_id}.npy"
    
    np.save(u0_path, u)
    np.save(m_path, m)
    
    return u0_path, m_path, run_id, u, m

def create_scaled_m_file(m, scale, nonlinearity_type, params, output_dir, m_type, m_scaling, run_id):
    scaled_m_dir = Path(output_dir) / "scaled_m_fields"
    scaled_m_dir.mkdir(exist_ok=True, parents=True)

    # we now rely on the physical scaling?
    scaled_m = m
    
    # we scale with a new approach, see above
    #scaled_m = m * scale
 
    if nonlinearity_type == 'cubic':
        scaled_m_path = scaled_m_dir / f"m_{m_type}_{m_scaling:.4f}_cubic_{scale:.4f}_{run_id}.npy"
    elif nonlinearity_type == 'cubic_quintic':
        s1 = params.get('s1', 1.0)
        s2 = params.get('s2', -0.1)
        scaled_m_path = scaled_m_dir / f"m_{m_type}_{m_scaling:.4f}_cq_s1_{s1:.4f}_s2_{s2:.4f}_{scale:.4f}_{run_id}.npy"
    elif nonlinearity_type == 'saturating':
        kappa = params.get('kappa', 1.0)
        scaled_m_path = scaled_m_dir / f"m_{m_type}_{m_scaling:.4f}_sat_kappa_{kappa:.4f}_{scale:.4f}_{run_id}.npy"
    else:
        scaled_m_path = scaled_m_dir / f"m_{m_type}_{m_scaling:.4f}_{nonlinearity_type}_{scale:.4f}_{run_id}.npy"
   
    np.save(scaled_m_path, scaled_m)
    return scaled_m_path

def run_parameter_sweep(args):
    global_args = {
        'output_dir': args.output_dir,
        'nx': args.nx,
        'ny': args.ny,
        'Lx': args.Lx,
        'Ly': args.Ly,
        'T': args.T,
        'nt': args.nt,
        'snapshots': args.snapshots
    }
    
    cubic_args = {
        'exe': args.cubic_exe,
        **global_args
    }
    
    cubic_quintic_args = {
        'exe': args.cubic_quintic_exe,
        's1_s2_pairs': [
            (args.cq_s1_1, args.cq_s2_1),
            (args.cq_s1_2, args.cq_s2_2),
            (args.cq_s1_3, args.cq_s2_3)
        ],
        **global_args
    }
    
    saturating_args = {
        'exe': args.saturating_exe,
        'kappa_values': [
            args.sat_kappa_1,
            args.sat_kappa_2,
            args.sat_kappa_3
        ],
        **global_args
    }
    
    sampler_types = args.sampler_types
    m_types = args.m_types
    m_scalings = args.m_scalings
    
    result_files = []
    
    for sampler_type in sampler_types:
        for m_type in m_types:
            for m_scaling in m_scalings:
                logger.info(f"Running sweep: sampler={sampler_type}, m_type={m_type}, m_scaling={m_scaling}")
                u0_path, m_path, run_id, u0, m_base = generate_and_save_field(m_scaling, sampler_type, m_type, args)
                
                # globally adapted now for all! 
                m_base = m_scaling * m_base

                # m, (scale_pos, scale_neg) = calculate_physical_scaling(u0, m_base, args.Lx, 'cubic', params=args)
                # cubic_scale = scale_pos + scale_neg

                cubic_scale = cq_scale = sat_scale = m_scaling

                logger.info(f"Calculated physical scaling for cubic: {cubic_scale:.4f}")
                scaled_m_path_cubic = create_scaled_m_file(m, cubic_scale, 'cubic', {}, 
                                                         args.output_dir, m_type, m_scaling, run_id)
                results_cubic = run_trajectory_cubic(u0_path, scaled_m_path_cubic, cubic_args, run_id, cubic_scale)
                
                results_cubic_quintic = {}
                for i, (s1, s2) in enumerate(cubic_quintic_args['s1_s2_pairs']):
                    cq_params = {'s1': s1, 's2': s2}
                    # m, (scale_pos, scale_neg) = calculate_physical_scaling(u0, m_base, args.Lx, 'cubic_quintic', cq_params)
                    # cq_scale = scale_pos + scale_neg 
                    logger.info(f"Calculated physical scaling for cubic-quintic (s1={s1}, s2={s2}): {cq_scale:.4f}")
                    scaled_m_path_cq = create_scaled_m_file(m, cq_scale, 'cubic_quintic', cq_params, 
                                                          args.output_dir, m_type, m_scaling, run_id)
                    cq_results = run_trajectory_cubic_quintic(u0_path, scaled_m_path_cq, cubic_quintic_args, 
                                                            run_id, i, s1, s2, cq_scale)
                    results_cubic_quintic.update(cq_results)
                
                results_saturating = {}
                for i, kappa in enumerate(saturating_args['kappa_values']):
                    sat_params = {'kappa': kappa}
                    # m, (scale_pos, scale_neg) = calculate_physical_scaling(u0, m_base, args.Lx, 'saturating', sat_params)
                    # sat_scale = scale_pos + scale_neg 
                    logger.info(f"Calculated physical scaling for saturating (kappa={kappa}): {sat_scale:.4f}")
                    scaled_m_path_sat = create_scaled_m_file(m, sat_scale, 'saturating', sat_params, 
                                                          args.output_dir, m_type, m_scaling, run_id)
                    sat_results = run_trajectory_saturating(u0_path, scaled_m_path_sat, saturating_args, 
                                                         run_id, i, kappa, sat_scale)
                    results_saturating.update(sat_results)
                
                combined_results = {
                    'cubic': results_cubic,
                    **results_cubic_quintic,
                    **results_saturating
                }
                
                output_files = perform_comprehensive_analysis(combined_results, args.output_dir, run_id, 
                                                            m_scaling, sampler_type, m_type, global_args)
                
                cleanup_intermediate_files(combined_results)
    
    return result_files

def run_parameter_sweep_no_physical_scaling(args):
    global_args = {
        'output_dir': args.output_dir,
        'nx': args.nx,
        'ny': args.ny,
        'Lx': args.Lx,
        'Ly': args.Ly,
        'T': args.T,
        'nt': args.nt,
        'snapshots': args.snapshots
    }
    
    cubic_args = {
        'exe': args.cubic_exe,
        **global_args
    }
    
    cubic_quintic_args = {
        'exe': args.cubic_quintic_exe,
        's1_s2_pairs': [
            (args.cq_s1_1, args.cq_s2_1),
            (args.cq_s1_2, args.cq_s2_2),
            (args.cq_s1_3, args.cq_s2_3)
        ],
        **global_args
    }
    
    saturating_args = {
        'exe': args.saturating_exe,
        'kappa_values': [
            args.sat_kappa_1,
            args.sat_kappa_2,
            args.sat_kappa_3
        ],
        **global_args
    }
    
    sampler_types = args.sampler_types
    m_types = args.m_types
    m_scalings = args.m_scalings
    
    result_files = []
    
    for sampler_type in sampler_types:
        for m_type in m_types:
            for m_scaling in m_scalings:
                logger.info(f"Running sweep: sampler={sampler_type}, m_type={m_type}, m_scaling={m_scaling}")
                u0_path, m_path, run_id, u0, m = generate_and_save_field(m_scaling, sampler_type, m_type, args)      
                cubic_scale = cq_scale = sat_scale = 1.

                results_cubic = run_trajectory_cubic(u0_path, m_path, cubic_args, run_id, cubic_scale) 
                results_cubic_quintic = {}
                for i, (s1, s2) in enumerate(cubic_quintic_args['s1_s2_pairs']):
                    cq_params = {'s1': s1, 's2': s2} 
                    cq_results = run_trajectory_cubic_quintic(u0_path, m_path, cubic_quintic_args, 
                                                            run_id, i, s1, s2, cq_scale)
                    results_cubic_quintic.update(cq_results)
                
                results_saturating = {}
                for i, kappa in enumerate(saturating_args['kappa_values']):
                    sat_params = {'kappa': kappa}
                    sat_results = run_trajectory_saturating(u0_path, m_path, saturating_args, 
                                                         run_id, i, kappa, sat_scale)
                    results_saturating.update(sat_results)
                
                combined_results = {
                    'cubic': results_cubic,
                    **results_cubic_quintic,
                    **results_saturating
                }
                
                output_files = perform_comprehensive_analysis(combined_results, args.output_dir, run_id, 
                                                            m_scaling, sampler_type, m_type, global_args)
                
                cleanup_intermediate_files(combined_results)
    
    return result_files

def run_parameter_sweep_no_physical_scaling_mpi(args):
    from mpi4py import MPI
    import time
    import os
    import shutil
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    global_args = {
        'output_dir': args.output_dir,
        'nx': args.nx,
        'ny': args.ny,
        'Lx': args.Lx,
        'Ly': args.Ly,
        'T': args.T,
        'nt': args.nt,
        'snapshots': args.snapshots
    }
    
    cubic_args = {
        'exe': args.cubic_exe,
        **global_args
    }
    
    cubic_quintic_args = {
        'exe': args.cubic_quintic_exe,
        's1_s2_pairs': [
            (args.cq_s1_1, args.cq_s2_1),
            (args.cq_s1_2, args.cq_s2_2),
            (args.cq_s1_3, args.cq_s2_3)
        ],
        **global_args
    }
    
    saturating_args = {
        'exe': args.saturating_exe,
        'kappa_values': [
            args.sat_kappa_1,
            args.sat_kappa_2,
            args.sat_kappa_3
        ],
        **global_args
    }
    
    sampler_types = args.sampler_types
    m_types = args.m_types
    m_scalings = args.m_scalings
    
    if rank == 0:
        result_files = []
    
    for sampler_type in sampler_types:
        for m_type in m_types:
            for m_scaling in m_scalings:
                if rank == 0:
                    logger.info(f"Running sweep: sampler={sampler_type}, m_type={m_type}, m_scaling={m_scaling}")
                    u0_path, m_path, run_id, u0, m = generate_and_save_field(m_scaling, sampler_type, m_type, args)
                    
                    run_data = {
                        'u0_path': u0_path,
                        'm_path': m_path,
                        'run_id': run_id
                    }
                else:
                    run_data = None
                
                run_data = comm.bcast(run_data, root=0)
                
                u0_path = run_data['u0_path']
                m_path = run_data['m_path']
                run_id = run_data['run_id']
                
                rank_dir = f"{args.output_dir}/rank_{rank}"
                if not os.path.exists(rank_dir):
                    os.makedirs(rank_dir, exist_ok=True)
                
                rank_u0_path = f"{rank_dir}/u0_{run_id}.npy"
                rank_m_path = f"{rank_dir}/m_{run_id}.npy"
                
                if os.path.exists(u0_path) and os.path.exists(m_path):
                    shutil.copy2(u0_path, rank_u0_path)
                    shutil.copy2(m_path, rank_m_path)
                
                cubic_scale = cq_scale = sat_scale = 1.
                
                results_cubic = None
                results_cubic_quintic = {}
                results_saturating = {}
                
                if rank == 0:
                    results_cubic = run_trajectory_cubic(rank_u0_path, rank_m_path, cubic_args, run_id, cubic_scale)
                
                s1_s2_pairs = cubic_quintic_args['s1_s2_pairs']
                if rank == 1 and len(s1_s2_pairs) > 0:
                    s1, s2 = s1_s2_pairs[0]
                    cq_results = run_trajectory_cubic_quintic(rank_u0_path, rank_m_path, cubic_quintic_args, 
                                                           run_id, 0, s1, s2, cq_scale)
                    comm.send(cq_results, dest=0)
                
                if rank == 2 and len(s1_s2_pairs) > 1:
                    s1, s2 = s1_s2_pairs[1]
                    cq_results = run_trajectory_cubic_quintic(rank_u0_path, rank_m_path, cubic_quintic_args, 
                                                           run_id, 1, s1, s2, cq_scale)
                    comm.send(cq_results, dest=0)
                
                if rank == 3 and len(s1_s2_pairs) > 2:
                    s1, s2 = s1_s2_pairs[2]
                    cq_results = run_trajectory_cubic_quintic(rank_u0_path, rank_m_path, cubic_quintic_args, 
                                                           run_id, 2, s1, s2, cq_scale)
                    comm.send(cq_results, dest=0)
                
                kappa_values = saturating_args['kappa_values']
                if rank == 4 and len(kappa_values) > 0:
                    kappa = kappa_values[0]
                    sat_results = run_trajectory_saturating(rank_u0_path, rank_m_path, saturating_args, 
                                                         run_id, 0, kappa, sat_scale)
                    comm.send(sat_results, dest=0)
                
                if rank == 5 and len(kappa_values) > 1:
                    kappa = kappa_values[1]
                    sat_results = run_trajectory_saturating(rank_u0_path, rank_m_path, saturating_args, 
                                                         run_id, 1, kappa, sat_scale)
                    comm.send(sat_results, dest=0)
                
                if rank == 6 and len(kappa_values) > 2:
                    kappa = kappa_values[2]
                    sat_results = run_trajectory_saturating(rank_u0_path, rank_m_path, saturating_args, 
                                                         run_id, 2, kappa, sat_scale)
                    comm.send(sat_results, dest=0)
                
                if rank == 0:
                    if 1 < size and len(s1_s2_pairs) > 0:
                        cq_results = comm.recv(source=1)
                        results_cubic_quintic.update(cq_results)
                    
                    if 2 < size and len(s1_s2_pairs) > 1:
                        cq_results = comm.recv(source=2)
                        results_cubic_quintic.update(cq_results)
                    
                    if 3 < size and len(s1_s2_pairs) > 2:
                        cq_results = comm.recv(source=3)
                        results_cubic_quintic.update(cq_results)
                    
                    if 4 < size and len(kappa_values) > 0:
                        sat_results = comm.recv(source=4)
                        results_saturating.update(sat_results)
                    
                    if 5 < size and len(kappa_values) > 1:
                        sat_results = comm.recv(source=5)
                        results_saturating.update(sat_results)
                    
                    if 6 < size and len(kappa_values) > 2:
                        sat_results = comm.recv(source=6)
                        results_saturating.update(sat_results)
                    
                    combined_results = {
                        'cubic': results_cubic,
                        **results_cubic_quintic,
                        **results_saturating
                    }
                    
                    output_files = perform_comprehensive_analysis(combined_results, args.output_dir, run_id, 
                                                                m_scaling, sampler_type, m_type, global_args)
                    
                    cleanup_intermediate_files(combined_results)
                    result_files.extend(output_files)
                
                for r in range(size):
                    if rank == r:
                        try:
                            if os.path.exists(rank_u0_path):
                                os.remove(rank_u0_path)
                            if os.path.exists(rank_m_path):
                                os.remove(rank_m_path)
                        except:
                            pass
                    comm.Barrier()
    
    if rank == 0:
        return result_files
    else:
        return []

def run_trajectory_cubic(u0_path, m_path, args, run_id, scale):
    output_dir = Path(args.get('output_dir', 'nlse_sweep_results'))
    output_dir.mkdir(exist_ok=True, parents=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    nx = args.get('nx', 256)
    ny = args.get('ny', 256)
    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    T = args.get('T', 1.5)
    nt = args.get('nt', 500)
    num_snapshots = args.get('snapshots', 100)

    cubic_exe = args.get('exe')
    cubic_traj_file = traj_dir / f"cubic_{run_id}.npy"
    cubic_cmd = [
        str(cubic_exe),
        str(nx),
        str(ny),
        str(Lx),
        str(Ly),
        str(u0_path),
        str(cubic_traj_file),
        str(T),
        str(nt),
        str(num_snapshots),
        str(m_path)
    ]

    start_time = time.time()
    logger.info(f"NLSE run with {' '.join(cubic_cmd)}")
    try:
        subprocess.run(cubic_cmd, check=True, capture_output=True, text=True)
        cubic_traj = np.load(cubic_traj_file)
        cubic_walltime = time.time() - start_time
        logger.info(f"walltime: {cubic_walltime:.2f}")

        results = {
            'traj': cubic_traj,
            'traj_file': cubic_traj_file,
            'walltime': cubic_walltime,
            'scale': scale
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running cubic NLSE: {e}")
        logger.error(f"{' '.join(cubic_cmd)}")
        results = {'error': str(e)}

    return results

def run_trajectory_cubic_quintic(u0_path, m_path, args, run_id, index, s1, s2, scale):
    output_dir = Path(args.get('output_dir', 'nlse_sweep_results'))
    output_dir.mkdir(exist_ok=True, parents=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    nx = args.get('nx', 256)
    ny = args.get('ny', 256)
    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    T = args.get('T', 1.5)
    nt = args.get('nt', 500)
    num_snapshots = args.get('snapshots', 100)

    cubic_quintic_exe = args.get('exe')
    results = {}

    cq_traj_file = traj_dir / f"cubic_quintic_s1_{s1:.4f}_s2_{s2:.4f}_{run_id}.npy"
    cq_cmd = [
        str(cubic_quintic_exe),
        str(nx),
        str(ny),
        str(Lx),
        str(Ly),
        str(s1),
        str(s2),
        str(u0_path),
        str(cq_traj_file),
        str(T),
        str(nt),
        str(num_snapshots),
        str(m_path)
    ]
    
    logger.info(f"NLSE cubic-quintic run with {' '.join(cq_cmd)}")
    start_time = time.time()
    try:
        subprocess.run(cq_cmd, check=True, capture_output=True, text=True)
        cq_traj = np.load(cq_traj_file)
        cq_walltime = time.time() - start_time
        logger.info(f"walltime {cq_walltime:.2f}")

        results[f'cubic_quintic_{index}'] = {
            'traj': cq_traj,
            'traj_file': cq_traj_file,
            'walltime': cq_walltime,
            's1': s1,
            's2': s2,
            'scale': scale
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running cubic-quintic NLSE with s1={s1}, s2={s2}: {e}")
        logger.error(f"{' '.join(cq_cmd)}")
        results[f'cubic_quintic_{index}'] = {'error': str(e), 's1': s1, 's2': s2}
    
    return results

def run_trajectory_saturating(u0_path, m_path, args, run_id, index, kappa, scale):
    output_dir = Path(args.get('output_dir', 'nlse_sweep_results'))
    output_dir.mkdir(exist_ok=True, parents=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    nx = args.get('nx', 256)
    ny = args.get('ny', 256)
    Lx = args.get('Lx', 10.0)
    Ly = args.get('Ly', 10.0)
    T = args.get('T', 1.5)
    nt = args.get('nt', 500)
    num_snapshots = args.get('snapshots', 100)

    saturating_exe = args.get('exe')
    results = {}

    sat_traj_file = traj_dir / f"saturating_kappa_{kappa:.4f}_{run_id}.npy"
    sat_cmd = [
        str(saturating_exe),
        str(nx),
        str(ny),
        str(Lx),
        str(Ly),
        str(kappa),
        str(u0_path),
        str(sat_traj_file),
        str(T),
        str(nt),
        str(num_snapshots),
        str(m_path)
    ]

    logger.info(f"NLSE saturating run with {' '.join(sat_cmd)}")
    start_time = time.time()
    try:
        subprocess.run(sat_cmd, check=True, capture_output=True, text=True)
        sat_traj = np.load(sat_traj_file)
        sat_walltime = time.time() - start_time
        logger.info(f"walltime {sat_walltime:.2f}")

        results[f'saturating_{index}'] = {
            'traj': sat_traj,
            'traj_file': sat_traj_file,
            'walltime': sat_walltime,
            'kappa': kappa,
            'scale': scale
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running saturating NLSE with kappa={kappa}: {e}")
        logger.error(f"{' '.join(sat_cmd)}")
        results[f'saturating_{index}'] = {'error': str(e), 'kappa': kappa}

    return results

def main():
    parser = argparse.ArgumentParser(description="NLSE Parameter Sweep Study")
    
    parser.add_argument("--cubic-exe", type=str, required=True, help="Path to cubic NLSE executable")
    parser.add_argument("--cubic-quintic-exe", type=str, required=True, help="Path to cubic-quintic NLSE executable")
    parser.add_argument("--saturating-exe", type=str, required=True, help="Path to saturating NLSE executable")
    
    parser.add_argument("--nx", type=int, default=200, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=200, help="Grid points in y")
    parser.add_argument("--Lx", type=float, default=10.0, help="Domain half-width in x")
    parser.add_argument("--Ly", type=float, default=10.0, help="Domain half-width in y")
    parser.add_argument("--T", type=float, default=2.0, help="Simulation time")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--snapshots", type=int, default=30, help="Number of snapshots")
    
    parser.add_argument("--cq-s1-1", type=float, default=0.6, help="First s1 value for cubic-quintic NLSE")
    parser.add_argument("--cq-s2-1", type=float, default=-0.06, help="First s2 value for cubic-quintic NLSE")
    parser.add_argument("--cq-s1-2", type=float, default=1.0, help="Second s1 value for cubic-quintic NLSE")
    parser.add_argument("--cq-s2-2", type=float, default=-0.1, help="Second s2 value for cubic-quintic NLSE")
    parser.add_argument("--cq-s1-3", type=float, default=3.0, help="Third s1 value for cubic-quintic NLSE")
    parser.add_argument("--cq-s2-3", type=float, default=-1.0, help="Third s2 value for cubic-quintic NLSE")
    
    parser.add_argument("--sat-kappa-1", type=float, default=0.5, help="First kappa value for saturating NLSE")
    parser.add_argument("--sat-kappa-2", type=float, default=1.0, help="Second kappa value for saturating NLSE")
    parser.add_argument("--sat-kappa-3", type=float, default=5.0, help="Third kappa value for saturating NLSE")

    """
    all_types = [
        'fundamental_soliton',
        !!!!!!!!! 'multi_soliton',
        !!!!!!!!! 'spectral',
        !!!!!!!!! 'chaotic',
        'vortex',
        !!!!!!!!! 'vortex_lattice',
        'dark_soliton',
        !!!!!!!!! 'solitary_wave_with_ambient',
        !!!!!!!!! 'logarithmic_singularity',
        !!!!!!!!! 'free_singularity',
        'transparent_solitary_wave',
        !!!!!!!!! 'colliding_solitary_waves',
        'oscillating_breather',
        'ring_soliton',
        !!!!!!!!!  'multi_ring'
    ]
    """

    
    parser.add_argument("--sampler-types", nargs="+", default=['neg'], help="Types of samplers to use")
    parser.add_argument("--m-types", nargs="+", default=["periodic_boxes"], #["uniform", "step", "radial", "periodic_boxes"], 
                      help="Types of m field to use")
    parser.add_argument("--m-scalings", nargs="+", type=float, default=[.99, -.99],#default=[-.9, .9], 
                      help="Scaling factors for m field")
    
    parser.add_argument("--output-dir", type=str, default="nlse_sweep_results", help="Output directory")
    parser.add_argument("--clean-intermediate", action="store_true", help="Clean up intermediate files")
    
    args = parser.parse_args()
    
    # run_parameter_sweep(args) # TODO fix
    # run_parameter_sweep_no_physical_scaling(args)
    run_parameter_sweep_no_physical_scaling_mpi(args)

if __name__ == "__main__":
    main()
