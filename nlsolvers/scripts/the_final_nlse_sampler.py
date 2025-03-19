import numpy as np
from scipy import special, stats, spatial
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from tqdm import tqdm

class NLSEPhenomenonSampler:
    def __init__(self, nx, ny, L):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.x = np.linspace(-L, L, nx)
        self.y = np.linspace(-L, L, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.r = np.sqrt(self.X**2 + self.Y**2)
        self.theta = np.arctan2(self.Y, self.X)
        self.dx = 2*L/(nx-1)
        self.dy = 2*L/(ny-1)
        self.cell_area = self.dx * self.dy
        self.k_max = np.pi / self.dx
        
        self.k_x = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        self.k_y = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        self.KX, self.KY = np.meshgrid(self.k_x, self.k_y, indexing='ij')
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2)
    
    def _envelope(self, field, width_factor=0.7):
        envelope_width = width_factor * self.L
        envelope = np.exp(-self.r**2/(2*envelope_width**2))
        return field * envelope
    
    def _normalize_power(self, field, target_power):
        current_power = np.sum(np.abs(field)**2) * self.cell_area
        if current_power > 0:
            return field * np.sqrt(target_power / current_power)
        return field
    
    def _sech(self, x):
        return 1.0 / np.cosh(x)

    def fundamental_soliton(self, system_type, amplitude=1.0, width=1.0, position=(0, 0),
                           phase=0.2, velocity=(0.0, 0.0), sigma1=1.0, sigma2=-0.1,
                           kappa=1.0, apply_envelope=True, envelope_width=0.7, Lambda=0.1,
                           chirp_factor=0.0, aspect_ratio=1.0, orientation=0.0, order=1):
        x0, y0 = position
        vx, vy = velocity
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        r_local = np.sqrt((X_rot/aspect_ratio)**2 + Y_rot**2)
        
        momentum_phase = vx * (self.X - x0) + vy * (self.Y - y0)
        chirp_phase = chirp_factor * r_local**2
        total_phase = momentum_phase + phase + chirp_phase

        if system_type == 'cubic':
            if order == 1:
                profile = amplitude * self._sech(r_local/width)
            else:
                profile = amplitude * self._sech(r_local/width)**order
        elif system_type == 'cubic_quintic':
            beta = -sigma2 * amplitude**2 / sigma1
            if beta > 0:
                if order == 1:
                    profile = amplitude * self._sech(r_local/width) / np.sqrt(1 + beta * self._sech(r_local/width)**2)
                else:
                    profile = amplitude * self._sech(r_local/width)**order / np.sqrt(1 + beta * self._sech(r_local/width)**(2*order))
            else:
                if order == 1:
                    profile = amplitude * self._sech(r_local/width)
                else:
                    profile = amplitude * self._sech(r_local/width)**order
        elif system_type == 'saturable':
            if order == 1:
                sech_term = self._sech(r_local/width)
                denom = np.sqrt(1 + kappa * amplitude**2 * sech_term**2)
                profile = amplitude * sech_term / denom
            else:
                sech_term = self._sech(r_local/width)**order
                denom = np.sqrt(1 + kappa * amplitude**2 * sech_term**2)
                profile = amplitude * sech_term / denom
        elif system_type == 'glasner_allen_flowers':
            if order == 1:
                sech_term = self._sech(np.sqrt(Lambda) * r_local)
                profile = amplitude * sech_term / np.sqrt(9 - 48 * Lambda * sech_term**2 + 31)
            else:
                sech_term = self._sech(np.sqrt(Lambda) * r_local)**order
                profile = amplitude * sech_term / np.sqrt(9 - 48 * Lambda * sech_term**(2/order) + 31)
        else:
            raise ValueError(f"Unknown system type: {system_type}")

        u = profile * np.exp(1j * total_phase)

        if apply_envelope:
            u = self._envelope(u, envelope_width)

        return u
    
    def multi_soliton_state(self, system_type, amplitude_range=(0.8, 1.2), 
                          width_range=(0.8, 1.2), position_variance=1.0, velocity_scale=1.0, 
                          phase_pattern='vortex', arrangement='random', separation=5.0, 
                          sigma1=1.0, sigma2=-0.1, kappa=1.0, apply_envelope=False, 
                          envelope_width=0.7, Lambda_range=(0.04, 0.14), coherence=0.8,
                          interaction_strength=0.5, cluster_levels=1, order_range=(1, 2),
                          chirp_range=(-0.1, 0.1), aspect_ratio_range=(1.0, 1.5)):
        u = np.zeros_like(self.X, dtype=complex)
        n_solitons = np.random.randint(3, 12) 
        
        if arrangement == 'linear':
            base_positions = [(i - (n_solitons-1)/2) * separation for i in range(n_solitons)]
            positions = [(pos, 0) for pos in base_positions]
        elif arrangement == 'circular':
            positions = []
            for i in range(n_solitons):
                angle = 2 * np.pi * i / n_solitons
                x = separation * np.cos(angle)
                y = separation * np.sin(angle)
                positions.append((x, y))
        elif arrangement == 'random':
            positions = []
            for _ in range(n_solitons):
                x = np.random.normal(0, position_variance * self.L/4)
                y = np.random.normal(0, position_variance * self.L/4)
                positions.append((x, y))
        elif arrangement == 'lattice':
            side = int(np.ceil(np.sqrt(n_solitons)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_solitons:
                        x = (i - (side-1)/2) * separation
                        y = (j - (side-1)/2) * separation
                        positions.append((x, y))
        elif arrangement == 'hierarchical':
            positions = []
            if cluster_levels <= 1:
                cluster_centers = [(0, 0)]
            else:
                cluster_centers = []
                for i in range(cluster_levels):
                    angle = 2 * np.pi * i / cluster_levels
                    x = separation * 2 * np.cos(angle)
                    y = separation * 2 * np.sin(angle)
                    cluster_centers.append((x, y))
            
            solitons_per_cluster = n_solitons // len(cluster_centers)
            remainder = n_solitons % len(cluster_centers)
            
            for i, (cx, cy) in enumerate(cluster_centers):
                cluster_size = solitons_per_cluster + (1 if i < remainder else 0)
                for j in range(cluster_size):
                    if j == 0 and cluster_levels > 1:
                        positions.append((cx, cy))
                    else:
                        angle = 2 * np.pi * j / cluster_size
                        x = cx + separation * 0.5 * np.cos(angle)
                        y = cy + separation * 0.5 * np.sin(angle)
                        positions.append((x, y))
        
        if phase_pattern == 'random':
            phases = np.random.uniform(0, 2*np.pi, n_solitons)
        elif phase_pattern == 'alternating':
            phases = [i * np.pi for i in range(n_solitons)]
        elif phase_pattern == 'synchronized':
            phases = np.zeros(n_solitons)
        elif phase_pattern == 'vortex':
            center_x = np.mean([p[0] for p in positions])
            center_y = np.mean([p[1] for p in positions])
            phases = [np.arctan2(p[1] - center_y, p[0] - center_x) for p in positions]
        elif phase_pattern == 'partial_coherence':
            base_phase = np.random.uniform(0, 2*np.pi)
            phases = []
            for _ in range(n_solitons):
                if np.random.random() < coherence:
                    phases.append(base_phase)
                else:
                    phases.append(np.random.uniform(0, 2*np.pi))
        
        for i, ((x0, y0), phase) in enumerate(zip(positions, phases)):
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            Lambda = np.random.uniform(*Lambda_range)
            order = np.random.randint(*order_range)
            chirp = np.random.uniform(*chirp_range)
            aspect_ratio = np.random.uniform(*aspect_ratio_range)
            orientation = np.random.uniform(0, 2*np.pi)
            
            if velocity_scale > 0:
                if arrangement == 'circular':
                    angle = 2 * np.pi * i / n_solitons
                    vx = -velocity_scale * np.cos(angle)
                    vy = -velocity_scale * np.sin(angle)
                else:
                    vx = np.random.normal(0, velocity_scale)
                    vy = np.random.normal(0, velocity_scale)
            else:
                vx, vy = 0, 0
            
            component = self.fundamental_soliton(
                system_type, amplitude=amplitude, width=width, 
                position=(x0, y0), phase=phase, velocity=(vx, vy),
                sigma1=sigma1, sigma2=sigma2, kappa=kappa,
                apply_envelope=False, Lambda=Lambda, order=order,
                chirp_factor=chirp, aspect_ratio=aspect_ratio,
                orientation=orientation
            )
            
            if interaction_strength < 1.0 and i > 0:
                u = u + component * interaction_strength
            else:
                u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u

    def akhmediev_breather(self, amplitude=1.0, modulation_frequency=1.0,
            growth_rate=0.5, position=None, phase=None, orientation=None,
            breather_phase='compressed', apply_envelope=True, envelope_width=0.7,
            aspect_ratio=1.0, t_param=None):
        position_variance = 1.
        if position is None:
                x0, y0 = np.random.normal(0, position_variance * self.L/4, 2)
        if phase is None:
            phase = np.random.random(1) * 1.j
        if orientation is None:
            orientation = np.random.random(1) * np.pi
        if t_param is None:
            t_param = np.random.random(1)

        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        X_scaled = X_rot / aspect_ratio
        
        a = growth_rate
        omega_mod = modulation_frequency
        
        if not 0 < a < 0.5:
            a = np.clip(a, 0.001, 0.499)
        
        b = np.sqrt(8*a*(1-2*a))
        
        if breather_phase == 'compressed':
            z = 0
        elif breather_phase == 'growing':
            z = -1.0
        elif breather_phase == 'decaying':
            z = 1.0
        else:
            z = float(breather_phase)
        
        P0 = amplitude * amplitude
        
        numerator = (1 - 4*a) * np.cosh(b * z) + np.sqrt(2*a) * np.cos(omega_mod * X_scaled) + 1j * b * np.sinh(b * z)
        denominator = 2 * a * np.cos(omega_mod * X_scaled) - np.cosh(b * z)
        phase_evolution = np.exp(1j * (t_param + phase))
        u = np.sqrt(P0) * numerator / denominator * phase_evolution
        u = u.astype(complex) 
        if apply_envelope:
            envelope = np.exp(-Y_rot**2/(2*envelope_width**2))
            u = u * envelope
        
        return u
    
    def spectral_method(self, spectrum_type='fractal', n_modes=50, 
                      amplitude=1.0, k_min=0.5, k_max=5.0, spectrum_slope=-5/3,
                      randomize_phases=True, apply_envelope=False, envelope_width=0.7,
                      hurst_exponent=0.7, correlation_length=1.0, spectral_scales=1,
                      intermittency_index=0.0, long_range_corr=False):
        spectrum = np.zeros((self.nx, self.ny), dtype=complex)
        
        if spectrum_type == 'kolmogorov':
            if spectral_scales == 1:
                for _ in range(n_modes):
                    k_mag = np.random.uniform(k_min, k_max)
                    k_angle = np.random.uniform(0, 2*np.pi)
                    
                    kx = k_mag * np.cos(k_angle)
                    ky = k_mag * np.sin(k_angle)
                    
                    ikx = int(kx * self.nx / (2 * self.k_max)) % self.nx
                    iky = int(ky * self.ny / (2 * self.k_max)) % self.ny
                    
                    if randomize_phases:
                        phase = np.random.uniform(0, 2*np.pi)
                    else:
                        phase = 0
                    
                    amp = k_mag**spectrum_slope
                    spectrum[ikx, iky] = amp * np.exp(1j * phase)
            else:
                scales = np.logspace(np.log10(k_min), np.log10(k_max), spectral_scales)
                scale_width = (k_max - k_min) / spectral_scales
                
                for scale_idx in range(spectral_scales):
                    k_center = scales[scale_idx]
                    k_low = max(k_min, k_center - scale_width/2)
                    k_high = min(k_max, k_center + scale_width/2)
                    
                    scale_slope = spectrum_slope * (1 + (np.random.random() - 0.5) * intermittency_index)
                    
                    for _ in range(n_modes // spectral_scales):
                        k_mag = np.random.uniform(k_low, k_high)
                        k_angle = np.random.uniform(0, 2*np.pi)
                        
                        kx = k_mag * np.cos(k_angle)
                        ky = k_mag * np.sin(k_angle)
                        
                        ikx = int(kx * self.nx / (2 * self.k_max)) % self.nx
                        iky = int(ky * self.ny / (2 * self.k_max)) % self.ny
                        
                        if randomize_phases:
                            if long_range_corr and scale_idx > 0:
                                phase = np.random.normal(spectrum[ikx, iky].imag, 1-hurst_exponent)
                            else:
                                phase = np.random.uniform(0, 2*np.pi)
                        else:
                            phase = 0
                        
                        amp = k_mag**scale_slope
                        spectrum[ikx, iky] = amp * np.exp(1j * phase)
        
        elif spectrum_type == 'ring':
            k_ring = (k_min + k_max) / 2
            ring_width = (k_max - k_min) / 2
            
            for i in range(self.nx):
                for j in range(self.ny):
                    kx = 2 * self.k_max * (i / self.nx - 0.5)
                    ky = 2 * self.k_max * (j / self.ny - 0.5)
                    k_mag = np.sqrt(kx**2 + ky**2)
                    
                    if k_min <= k_mag <= k_max:
                        if randomize_phases:
                            phase = np.random.uniform(0, 2*np.pi)
                        else:
                            phase = 0
                        
                        amp = np.exp(-(k_mag - k_ring)**2 / (2 * ring_width**2))
                        spectrum[i, j] = amp * np.exp(1j * phase)
        
        elif spectrum_type == 'gaussian_spots':
            for _ in range(n_modes):
                k_mag = np.random.uniform(k_min, k_max)
                k_angle = np.random.uniform(0, 2*np.pi)
                
                kx_center = k_mag * np.cos(k_angle)
                ky_center = k_mag * np.sin(k_angle)
                
                spot_width = np.random.uniform(0.1, 0.5) * correlation_length
                
                for i in range(self.nx):
                    for j in range(self.ny):
                        kx = 2 * self.k_max * (i / self.nx - 0.5)
                        ky = 2 * self.k_max * (j / self.ny - 0.5)
                        
                        dist = np.sqrt((kx - kx_center)**2 + (ky - ky_center)**2)
                        if dist < 3 * spot_width:
                            if randomize_phases:
                                phase = np.random.uniform(0, 2*np.pi)
                            else:
                                phase = 0
                            
                            amp = np.exp(-dist**2 / (2 * spot_width**2))
                            spectrum[i, j] += amp * np.exp(1j * phase)
        
        elif spectrum_type == 'fractal':
            beta = 2 * hurst_exponent + 1  
            
            for i in range(self.nx):
                for j in range(self.ny):
                    if i == 0 and j == 0:
                        continue
                        
                    kx = 2 * self.k_max * (i / self.nx - 0.5)
                    ky = 2 * self.k_max * (j / self.ny - 0.5)
                    k_mag = np.sqrt(kx**2 + ky**2)
                    
                    if k_min <= k_mag <= k_max:
                        if randomize_phases:
                            phase = np.random.uniform(0, 2*np.pi)
                        else:
                            phase = 0
                        
                        amp = k_mag**(-beta/2)
                        spectrum[i, j] = amp * np.exp(1j * phase)
        
        u = np.fft.ifft2(np.fft.ifftshift(spectrum))
        u = u / np.std(np.abs(u)) * amplitude
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
   
    def chaotic_field(self, amplitude=1.0, spectral_exponent=-1.5, coherent_structures=True,
                    n_structures=3, apply_envelope=True, envelope_width=0.7,
                    intermittency_index=0.0, structure_types=None):
        u = self.spectral_method(
            spectrum_type='kolmogorov', amplitude=amplitude,
            spectrum_slope=spectral_exponent, apply_envelope=False,
            intermittency_index=intermittency_index
        )
        
        if coherent_structures:
            if structure_types is None:
                structure_types = ['vortex', 'soliton']
                
            for _ in range(n_structures):
                structure_type = np.random.choice(structure_types)
                x0 = np.random.uniform(-self.L/3, self.L/3)
                y0 = np.random.uniform(-self.L/3, self.L/3)
                
                if structure_type == 'vortex':
                    charge = np.random.choice([-1, 1])
                    r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                    theta_local = np.arctan2(self.Y - y0, self.X - x0)
                    structure = np.tanh(r_local) * np.exp(1j * charge * theta_local)
                    
                elif structure_type == 'soliton':
                    width = np.random.uniform(0.5, 1.5)
                    r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                    structure = 2.0 * np.exp(-r_local**2/(2*width**2)) * np.exp(1j * np.random.uniform(0, 2*np.pi))
                
                elif structure_type == 'ring':
                    radius = np.random.uniform(1.0, 3.0)
                    width = np.random.uniform(0.3, 0.7)
                    r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
                    structure = 2.0 * np.exp(-(r_local - radius)**2/(2*width**2)) * np.exp(1j * np.random.uniform(0, 2*np.pi))
                
                weight = np.random.uniform(0.2, 0.5)
                u += weight * structure
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def vortex_solution(self, amplitude=1.0, position=(0, 0), charge=1, 
                       core_size=1.0, apply_envelope=True, envelope_width=0.7,
                       eccentricity=1.0, orientation=0.0, radial_mode=0):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        r_local = np.sqrt((X_rot/eccentricity)**2 + Y_rot**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)
        
        if radial_mode == 0:
            profile = amplitude * np.tanh(r_local / core_size)
        else:
            profile = amplitude * np.tanh(r_local / core_size) * (1 - np.exp(-(r_local / (radial_mode * core_size))**2))
            for i in range(1, radial_mode + 1):
                profile *= (r_local / core_size - i * np.pi)**2
            profile = np.abs(profile) / np.max(np.abs(profile)) * amplitude
            
        phase = charge * theta_local
        
        u = profile * np.exp(1j * phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def vortex_lattice(self, amplitude=1.0, n_vortices=5, arrangement='random',
                      separation=2.0, charge_distribution='alternating', apply_envelope=True, 
                      envelope_width=0.8, eccentricity=1.0, core_size_range=(0.5, 1.5),
                      radial_mode=0):
        u = np.ones_like(self.X, dtype=complex)
        
        if arrangement == 'square':
            side = int(np.ceil(np.sqrt(n_vortices)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_vortices:
                        x = (i - (side-1)/2) * separation
                        y = (j - (side-1)/2) * separation
                        positions.append((x, y))
        elif arrangement == 'triangular':
            positions = []
            rows = int(np.ceil(np.sqrt(n_vortices * 2 / np.sqrt(3))))
            for i in range(rows):
                offset = (i % 2) * 0.5 * separation
                for j in range(int(np.ceil(n_vortices / rows))):
                    if len(positions) < n_vortices:
                        x = (j - int(np.ceil(n_vortices / rows) - 1)/2) * separation + offset
                        y = (i - (rows-1)/2) * separation * np.sqrt(3)/2
                        positions.append((x, y))
        elif arrangement == 'circular':
            positions = []
            for i in range(n_vortices):
                angle = 2 * np.pi * i / n_vortices
                x = separation * np.cos(angle)
                y = separation * np.sin(angle)
                positions.append((x, y))
        elif arrangement == 'quasicrystal':
            positions = []
            symmetry = np.random.choice([5, 7, 8, 9, 11])
            for i in range(n_vortices):
                shell = i // symmetry
                if shell >= 3:
                    break
                angle_idx = i % symmetry
                angle = 2 * np.pi * angle_idx / symmetry
                x = separation * (shell + 1) * np.cos(angle)
                y = separation * (shell + 1) * np.sin(angle)
                positions.append((x, y))
        else:  # random
            positions = []
            for _ in range(n_vortices):
                x = np.random.uniform(-self.L/3, self.L/3)
                y = np.random.uniform(-self.L/3, self.L/3)
                positions.append((x, y))
        
        if charge_distribution == 'alternating':
            charges = [(i % 2) * 2 - 1 for i in range(n_vortices)]
        elif charge_distribution == 'same':
            charge_value = np.random.choice([-1, 1])
            charges = [charge_value] * n_vortices
        elif charge_distribution == 'random':
            charges = np.random.choice([-1, 1], n_vortices)
        elif charge_distribution == 'fractional':
            base_charge = np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
            charges = [base_charge] * n_vortices
        
        for (x0, y0), charge in zip(positions, charges):
            r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            theta_local = np.arctan2(self.Y - y0, self.X - x0)
            
            core_size = np.random.uniform(*core_size_range)
            
            if radial_mode == 0:
                profile = (r_local / core_size) / np.sqrt(r_local**2 + core_size**2)
            else:
                profile = (r_local / core_size) / np.sqrt(r_local**2 + core_size**2)
                for i in range(1, radial_mode + 1):
                    profile *= (r_local / core_size - i * np.pi)**2
            
            u *= profile * np.exp(1j * charge * theta_local)
        
        u = amplitude * u / np.max(np.abs(u))
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def dark_soliton(self, amplitude=1.0, width=1.0, position=None, orientation=None,
                    velocity=None, apply_envelope=True, envelope_width=0.7,
                    eccentricity=1.0, order=1, chirp_factor=0.0):
        if position is None:
            position = np.random.rand(2) * self.L / 3

        if velocity is None:
            velocity = np.random.rand(2) * np.random.randn(2)

        if orientation is None:
            orientation = np.random.choice(np.linspace(-np.pi, np.pi, 100))

        x0, y0 = position
        vx, vy = velocity
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        rotated_X = X_rot / eccentricity
        rotated_Y = Y_rot
        
        r_local = np.sqrt(rotated_X**2 + rotated_Y**2)
        
        if order == 1:
            profile = amplitude * np.tanh(rotated_X / width)
        else:
            profile = amplitude * np.tanh(rotated_X / width)**order
            
        momentum_phase = vx * (self.X - x0) + vy * (self.Y - y0)
        chirp_phase = chirp_factor * r_local**2
        
        u = profile * np.exp(1j * (momentum_phase + chirp_phase))
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def solitary_wave_with_ambient_field(self, system_type='cubic', solitary_amplitude=1.0, 
                                        solitary_width=1.0, solitary_position=None, 
                                        solitary_phase=0.0, solitary_velocity=None,
                                        ambient_amplitude=0.3, ambient_wavenumber=2.0,
                                        ambient_direction=-1., ambient_phase=0.1,
                                        ambient_width=3.0, sigma1=1.0, sigma2=-0.1,
                                        kappa=1.0, Lambda=0.1, epsilon=0.025,
                                        order=1, chirp_factor=0.1, aspect_ratio=1.0,
                                        ambient_modulation='phase'):
        if solitary_position is None:
            solitary_position = np.random.rand(2) * self.L / 3
        if solitary_velocity is None:
            solitary_velocity = np.random.rand(2)

        soliton = self.fundamental_soliton(
            system_type, amplitude=solitary_amplitude/epsilon, width=solitary_width*epsilon,
            position=solitary_position, phase=solitary_phase, 
            velocity=solitary_velocity, sigma1=sigma1, sigma2=sigma2, 
            kappa=kappa, Lambda=Lambda, order=order, chirp_factor=chirp_factor,
            aspect_ratio=aspect_ratio
        )
        
        x0, y0 = solitary_position
        kx = ambient_wavenumber * np.cos(ambient_direction)
        ky = ambient_wavenumber * np.sin(ambient_direction)
        
        if ambient_modulation == 'none':
            gaussian_envelope = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * ambient_width**2))
            ambient_wave = ambient_amplitude * np.exp(1j * (kx * self.X + ky * self.Y + ambient_phase)) * gaussian_envelope
        elif ambient_modulation == 'amplitude':
            modulation_freq = np.random.uniform(0.5, 2.0)
            modulation = 1 + 0.5 * np.cos(modulation_freq * (self.X + self.Y))
            gaussian_envelope = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * ambient_width**2))
            ambient_wave = ambient_amplitude * modulation * np.exp(1j * (kx * self.X + ky * self.Y + ambient_phase)) * gaussian_envelope
        elif ambient_modulation == 'phase':
            modulation_freq = np.random.uniform(0.5, 2.0)
            phase_mod = 0.5 * np.sin(modulation_freq * (self.X + self.Y))
            gaussian_envelope = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * ambient_width**2))
            ambient_wave = ambient_amplitude * np.exp(1j * (kx * self.X + ky * self.Y + ambient_phase + phase_mod)) * gaussian_envelope
        
        return soliton + ambient_wave
        
    def logarithmic_singularity(self, position=(0, 0), amplitude=1.0, m_lambda=0.5,
                               background_type='random', background_amplitude=0.3,
                               cutoff_radius=0.1, regularization='smooth',
                               background_phase=None):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2 + 1e-10)  # avoid division by zero
        
        singular_part = amplitude * np.log(r_local)
        
        if background_phase is None:
            phase = np.random.uniform(0, 2*np.pi)
        else:
            phase = background_phase
        
        if background_type == 'random':
            background = self.spectral_method(amplitude=background_amplitude, apply_envelope=False)
        elif background_type == 'gaussian':
            background = background_amplitude * np.exp(-r_local**2 / (2 * (self.L/4)**2))
        else:
            background = np.zeros_like(self.X)
        
        u = (singular_part * m_lambda + background) * np.exp(1j * phase)
        
        if regularization == 'smooth':
            mask = (r_local < cutoff_radius)
            smooth_factor = np.ones_like(self.X)
            smooth_factor[mask] = r_local[mask] / cutoff_radius
            u = u * smooth_factor
        elif regularization == 'cutoff':
            mask = (r_local < cutoff_radius)
            u[mask] = u[mask[0], mask[1]] if np.any(mask) else 0
        
        return u
        
    def free_singularity_solution(self, position=(0, 0), amplitude=1.0, m_lambda=0.5,
                                 epsilon=0.01, chi=0.5, background_type='random', 
                                 background_amplitude=0.3, background_phase=None,
                                 regularization='smooth'):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2 + 1e-10)
        
        A = amplitude * np.exp(1j * (background_phase if background_phase is not None else np.random.uniform(0, 2*np.pi)))
        log_term = chi * np.log(r_local/epsilon) * m_lambda
        
        if background_type == 'random':
            background = self.spectral_method(amplitude=background_amplitude, apply_envelope=False)
        elif background_type == 'gaussian':
            background = background_amplitude * np.exp(-r_local**2 / (2 * (self.L/4)**2))
        else:
            background = np.zeros_like(self.X)
        
        u = A + A * log_term + background
        
        if regularization == 'smooth':
            mask = (r_local < 0.5 * epsilon)
            smooth_factor = np.ones_like(self.X)
            smooth_factor[mask] = r_local[mask] / (0.5 * epsilon)
            u = u * smooth_factor
        
        return u
        
    def transparent_solitary_wave(self, amplitude=1.0, width=1.0, position=(0, 0),
                                phase=0.0, velocity=(0.0, 0.0), apply_envelope=True,
                                envelope_width=0.7, order=1, aspect_ratio=1.0,
                                orientation=0.0, chirp_factor=0.0):
        Lambda = 0.077  # The special value where m(λ) ≈ 0 according to the paper
        
        return self.fundamental_soliton(
            'glasner_allen_flowers', amplitude=amplitude, width=width,
            position=position, phase=phase, velocity=velocity,
            apply_envelope=apply_envelope, envelope_width=envelope_width,
            Lambda=Lambda, order=order, aspect_ratio=aspect_ratio,
            orientation=orientation, chirp_factor=chirp_factor
        )
        
    def colliding_solitary_waves(self, system_type='cubic', n_waves=2, angle_range=(0, 2*np.pi),
                               amplitude_range=(0.8, 1.2), width_range=(0.8, 1.2),
                               velocity_magnitude=2.0, separation=5.0, impact_parameter=0,
                               sigma1=1.0, sigma2=-0.1, kappa=1.0, Lambda_range=(0.04, 0.14),
                               order_range=(1, 2), aspect_ratio_range=(1.0, 1.5),
                               chirp_range=(-0.1, 0.1)):
        u = np.zeros_like(self.X, dtype=complex)
        
        center = (0, 0)
        angle_step = 2 * np.pi / n_waves
        
        for i in range(n_waves):
            if angle_range[1] - angle_range[0] < 2*np.pi:
                angle = np.random.uniform(*angle_range)
            else:
                angle = angle_range[0] + i * angle_step
                
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            Lambda = np.random.uniform(*Lambda_range)
            order = np.random.randint(*order_range)
            aspect_ratio = np.random.uniform(*aspect_ratio_range)
            chirp = np.random.uniform(*chirp_range)
            orientation = angle + np.pi/2
            
            impact_shift = impact_parameter * np.random.uniform(-1, 1)
            
            position_angle = angle
            direction_angle = angle + np.pi
            
            x = center[0] + separation * np.cos(position_angle)
            y = center[1] + separation * np.sin(position_angle) + impact_shift
            
            vx = -velocity_magnitude * np.cos(direction_angle)
            vy = -velocity_magnitude * np.sin(direction_angle)
            
            phase = np.random.uniform(0, 2*np.pi)
            
            soliton = self.fundamental_soliton(
                system_type, amplitude=amplitude, width=width,
                position=(x, y), phase=phase, velocity=(vx, vy),
                sigma1=sigma1, sigma2=sigma2, kappa=kappa, Lambda=Lambda,
                order=order, aspect_ratio=aspect_ratio, orientation=orientation,
                chirp_factor=chirp
            )
            
            u += soliton
            
        return u
        
    def oscillating_breather(self, amplitude=1.0, frequency=1.0, width=1.0, position=(0, 0),
                           phase=0.0, apply_envelope=True, envelope_width=0.7,
                           oscillation_type='radial', order=1, aspect_ratio=1.0,
                           orientation=0.0):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        r_local = np.sqrt((X_rot/aspect_ratio)**2 + Y_rot**2)
        
        envelope = np.exp(-r_local**2/(2*width**2))
        
        if oscillation_type == 'radial':
            oscillator = np.cos(frequency * r_local + phase)
        elif oscillation_type == 'azimuthal':
            theta_local = np.arctan2(Y_rot, X_rot)
            oscillator = np.cos(frequency * theta_local + phase)
        elif oscillation_type == 'mixed':
            theta_local = np.arctan2(Y_rot, X_rot)
            oscillator = np.cos(frequency * r_local + phase) * np.cos(order * theta_local)
        else:  # directional
            oscillator = np.cos(frequency * X_rot + phase)
        
        u = amplitude * envelope * oscillator * np.exp(1j * phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def ring_soliton(self, amplitude=1.0, radius=3.0, width=0.5, position=(0, 0),
                   phase=0.0, apply_envelope=True, envelope_width=0.7,
                   modulation_type='none', modulation_strength=0.0, modulation_mode=0,
                   aspect_ratio=1.0, orientation=0.0, radial_nodes=0):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        r_local = np.sqrt((X_rot/aspect_ratio)**2 + Y_rot**2)
        theta_local = np.arctan2(Y_rot, X_rot)
        
        profile = amplitude * np.exp(-(r_local - radius)**2/(2*width**2))
        
        if modulation_type == 'azimuthal':
            profile *= (1 + modulation_strength * np.cos(modulation_mode * theta_local))
        elif modulation_type == 'radial':
            profile *= (1 + modulation_strength * np.cos(modulation_mode * np.pi * r_local / radius))
        
        if radial_nodes > 0:
            for i in range(radial_nodes):
                inner_radius = radius * (i + 1) / (radial_nodes + 1)
                profile *= (r_local - inner_radius)**2
            profile = profile / np.max(profile) * amplitude
        
        u = profile * np.exp(1j * phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width) 
        return u

    def multi_ring(self, amplitude_range=(0.8, 1.2), radius_range=(1.0, 5.0),
                          width_range=(0.3, 0.8), position_variance=1.0, 
                          phase_pattern='random', arrangement='random', separation=5.0, 
                          apply_envelope=False, envelope_width=0.7, 
                          modulation_type='none', modulation_strength=0.0, modulation_mode=0,
                          aspect_ratio_range=(1.0, 1.5), orientation_range=(0, 2*np.pi),
                          radial_nodes_range=(0, 2), n_rings=None):
        u = np.zeros_like(self.X, dtype=complex)
        
        if n_rings is None:
            n_rings = np.random.randint(3, 8)
            
        if arrangement == 'linear':
            base_positions = [(i - (n_rings-1)/2) * separation for i in range(n_rings)]
            positions = [(pos, 0) for pos in base_positions]
        elif arrangement == 'circular':
            positions = []
            for i in range(n_rings):
                angle = 2 * np.pi * i / n_rings
                x = separation * np.cos(angle)
                y = separation * np.sin(angle)
                positions.append((x, y))
        elif arrangement == 'random':
            positions = []
            for _ in range(n_rings):
                x = np.random.normal(0, position_variance * self.L/4)
                y = np.random.normal(0, position_variance * self.L/4)
                positions.append((x, y))
        elif arrangement == 'lattice':
            side = int(np.ceil(np.sqrt(n_rings)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_rings:
                        x = (i - (side-1)/2) * separation
                        y = (j - (side-1)/2) * separation
                        positions.append((x, y))
        elif arrangement == 'concentric':
            positions = [(0, 0)] * n_rings
        
        if phase_pattern == 'random':
            phases = np.random.uniform(0, 2*np.pi, n_rings)
        elif phase_pattern == 'alternating':
            phases = [i * np.pi for i in range(n_rings)]
        elif phase_pattern == 'synchronized':
            phases = np.zeros(n_rings)
        elif phase_pattern == 'vortex':
            center_x = np.mean([p[0] for p in positions])
            center_y = np.mean([p[1] for p in positions])
            phases = [np.arctan2(p[1] - center_y, p[0] - center_x) for p in positions]
        
        for i, ((x0, y0), phase) in enumerate(zip(positions, phases)):
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            
            if arrangement == 'concentric':
                radius = (i + 1) * (radius_range[1] - radius_range[0]) / n_rings + radius_range[0]
            else:
                radius = np.random.uniform(*radius_range)
                
            aspect_ratio = np.random.uniform(*aspect_ratio_range)
            orientation = np.random.uniform(*orientation_range)
            radial_nodes = np.random.randint(*radial_nodes_range)
            
            component = self.ring_soliton(
                amplitude, radius=radius, width=width, position=(x0, y0),
                phase=phase, apply_envelope=False, modulation_type=modulation_type,
                modulation_strength=modulation_strength, modulation_mode=modulation_mode,
                aspect_ratio=aspect_ratio, orientation=orientation, radial_nodes=radial_nodes
            )
            u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def free_singularity_adapted(self, position=None, amplitude=None, Lambda=None,
                              epsilon=None, background_type=None,
                              background_amplitude=None, phase=None,
                              multi_scale=False, n_singularities=1):
        def m_lambda_function(lambda_val):
            if abs(lambda_val - 0.077) < 0.005:
                return 0.0
            elif lambda_val < 0.077:
                return -0.3 + (lambda_val - 0.04) * 4.0
            elif lambda_val < 0.092:
                slope = 1.5 / (0.092 - 0.077)
                return (lambda_val - 0.077) * slope
            else:
                return 1.5 - (lambda_val - 0.092) * 2.0

        def a_lambda_function(lambda_val):
            if abs(lambda_val - 0.092) < 0.005:
                return 0.0
            elif lambda_val < 0.077:
                return 0.5 - (lambda_val - 0.04) * 3.0
            elif lambda_val < 0.092:
                slope = -0.5 / (0.092 - 0.077)
                return (lambda_val - 0.077) * slope
            else:
                return -0.5 + (lambda_val - 0.092) * 1.0

        def create_greens_function(X, Y, epsilon=1e-8):
            G = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    r = np.sqrt(X[i,j]**2 + Y[i,j]**2)
                    if r > epsilon:
                        G[i,j] = -np.log(r) / (2 * np.pi)
                    else:
                        G[i,j] = -np.log(epsilon) / (2 * np.pi)
            return G

        u = np.zeros_like(self.X, dtype=complex)
        
        for k in range(n_singularities):
            if position is None:
                x0 = np.random.uniform(-0.5*self.L, 0.5*self.L)
                y0 = np.random.uniform(-0.5*self.L, 0.5*self.L)
                pos = (x0, y0)
            elif isinstance(position, list):
                pos = position[k % len(position)]
            else:
                pos = position

            if amplitude is None:
                amp = np.random.uniform(0.5, 2.0)
            elif isinstance(amplitude, list):
                amp = amplitude[k % len(amplitude)]
            else:
                amp = amplitude

            if Lambda is None:
                if multi_scale and k > 0:
                    options = [0.04, 0.06, 0.077, 0.085, 0.092, 0.11, 0.13]
                    weights = [0.15, 0.1, 0.3, 0.1, 0.2, 0.1, 0.05]
                    lam = np.random.choice(options, p=weights)
                else:
                    options = [0.04, 0.06, 0.077, 0.085, 0.092, 0.11, 0.13]
                    weights = [0.15, 0.1, 0.3, 0.1, 0.2, 0.1, 0.05]
                    lam = np.random.choice(options, p=weights)
            elif isinstance(Lambda, list):
                lam = Lambda[k % len(Lambda)]
            else:
                lam = Lambda

            if epsilon is None:
                eps = np.random.uniform(0.005, 0.05)
            elif isinstance(epsilon, list):
                eps = epsilon[k % len(epsilon)]
            else:
                eps = epsilon

            if background_type is None:
                bg_type = np.random.choice(['random', 'gaussian'])
            elif isinstance(background_type, list):
                bg_type = background_type[k % len(background_type)]
            else:
                bg_type = background_type

            if background_amplitude is None:
                bg_amp = np.random.uniform(0.1, 0.5)
            elif isinstance(background_amplitude, list):
                bg_amp = background_amplitude[k % len(background_amplitude)]
            else:
                bg_amp = background_amplitude

            if phase is None:
                ph = np.random.uniform(0, 2*np.pi)
            elif isinstance(phase, list):
                ph = phase[k % len(phase)]
            else:
                ph = phase

            x0, y0 = pos
            X_shifted = self.X - x0
            Y_shifted = self.Y - y0
            r_local = np.sqrt(X_shifted**2 + Y_shifted**2 + 1e-12)

            m_lambda = m_lambda_function(lam)
            a_lambda = a_lambda_function(lam)

            chi_epsilon = 1.0 / (max(-m_lambda * np.log(eps) + a_lambda, 0.01))

            A = amp * np.exp(1j * ph)

            if bg_type == 'random':
                psi_R = self.spectral_method(amplitude=bg_amp, apply_envelope=True)
            elif bg_type == 'gaussian':
                psi_R = bg_amp * np.exp(-r_local**2 / (2 * (self.L/4)**2)) * np.exp(1j * np.random.uniform(0, 2*np.pi))
            else:
                raise NotImplemented

            G = create_greens_function(X_shifted, Y_shifted)

            singular_part = 2 * np.pi * A * m_lambda * chi_epsilon * G

            cutoff_radius = 3 * eps
            smooth_factor = np.ones_like(self.X)
            mask = (r_local < cutoff_radius)
            smooth_factor[mask] = (r_local[mask] / cutoff_radius)**2

            psi_R_at_singular_point = A
            psi_R = psi_R - psi_R[self.nx//2, self.ny//2] + psi_R_at_singular_point

            component = psi_R + singular_part * smooth_factor
            
            if multi_scale and k > 0:
                scale_factor = 0.5**k
                component = component * scale_factor
                
            u += component

        return u

    def logarithmic_singularity_adapted(self, position=None, amplitude=None, Lambda=None,
                              epsilon=None, background_type=None,
                              background_amplitude=None, phase=None,
                              multi_scale=False, n_singularities=1):
        return self.free_singularity_adapted(position, amplitude, Lambda,
                                         epsilon, background_type,
                                         background_amplitude, phase,
                                         multi_scale, n_singularities)
    
    def turbulent_condensate(self, amplitude=1.0, condensate_fraction=0.5, 
                           temperature=1.0, n_modes=100, k_min=0.5, k_max=8.0,
                           spectrum_slope=-2.0, apply_envelope=True, envelope_width=0.7,
                           condensate_phase=None, modulation_type='none',
                           modulation_strength=0.2, modulation_scale=2.0):
        u_k = np.zeros((self.nx, self.ny), dtype=complex)
        
        if condensate_phase is None:
            condensate_phase = np.random.uniform(0, 2*np.pi)
            
        condensate_amplitude = amplitude * np.sqrt(condensate_fraction)
        thermal_amplitude = amplitude * np.sqrt(1 - condensate_fraction)
        
        condensate = np.ones_like(self.X) * condensate_amplitude * np.exp(1j * condensate_phase)
        
        if modulation_type == 'spatial':
            modulation = 1 + modulation_strength * np.cos(2*np.pi*self.X/modulation_scale) * np.cos(2*np.pi*self.Y/modulation_scale)
            condensate *= modulation
        elif modulation_type == 'phase':
            phase_mod = modulation_strength * np.sin(2*np.pi*self.X/modulation_scale) * np.sin(2*np.pi*self.Y/modulation_scale)
            condensate *= np.exp(1j * phase_mod)
            
        for i in range(self.nx):
            for j in range(self.ny):
                kx = self.k_x[i]
                ky = self.k_y[j]
                k_mag = np.sqrt(kx**2 + ky**2)
                
                if k_min <= k_mag <= k_max and (i != 0 or j != 0):
                    T_k = temperature / (1 + (k_mag / k_min)**(-spectrum_slope))
                    amplitude_k = np.sqrt(T_k)
                    phase_k = np.random.uniform(0, 2*np.pi)
                    u_k[i, j] = amplitude_k * np.exp(1j * phase_k)
        
        thermal_field = np.fft.ifft2(u_k).real + 1j * np.fft.ifft2(u_k).imag
        thermal_field = thermal_field / np.std(np.abs(thermal_field)) * thermal_amplitude
        
        u = condensate + thermal_field
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def topological_defect_network(self, amplitude=1.0, n_defects=10, defect_types=None,
                                  spatial_distribution=None, domain_size=None,
                                  temperature=1.0, core_size=0.5, apply_envelope=True,
                                  envelope_width=0.8, interaction_strength=0.7):
        if domain_size is None:
            domain_size = 0.7 * self.L
            
        if defect_types is None:
            defect_types = np.random.choice([
                ['vortex', 'antivortex'], ['domain_wall']
                ])

        if spatial_distribution is None:
            spatial_distribution = np.random.choice([
                'poisson', 'inhibition', 'cluster'])
            
        positions = []
        types = []
        charges = []
        
        if spatial_distribution == 'poisson':
            for _ in range(n_defects):
                x = np.random.uniform(-domain_size, domain_size)
                y = np.random.uniform(-domain_size, domain_size)
                positions.append((x, y))
                types.append(np.random.choice(defect_types))
        elif spatial_distribution == 'inhibition':
            min_distance = 2 * core_size
            attempts = 0
            max_attempts = 1000
            while len(positions) < n_defects and attempts < max_attempts:
                x = np.random.uniform(-domain_size, domain_size)
                y = np.random.uniform(-domain_size, domain_size)
                
                too_close = False
                for px, py in positions:
                    dist = np.sqrt((x-px)**2 + (y-py)**2)
                    if dist < min_distance:
                        too_close = True
                        break
                        
                if not too_close:
                    positions.append((x, y))
                    types.append(np.random.choice(defect_types))
                
                attempts += 1
        elif spatial_distribution == 'cluster':
            n_clusters = max(1, n_defects // 5)
            cluster_centers = []
            for _ in range(n_clusters):
                x = np.random.uniform(-domain_size * 0.8, domain_size * 0.8)
                y = np.random.uniform(-domain_size * 0.8, domain_size * 0.8)
                cluster_centers.append((x, y))
                
            for _ in range(n_defects):
                cluster_idx = np.random.randint(0, n_clusters)
                cx, cy = cluster_centers[cluster_idx]
                
                x = cx + np.random.normal(0, domain_size * 0.15)
                y = cy + np.random.normal(0, domain_size * 0.15)
                
                positions.append((x, y))
                types.append(np.random.choice(defect_types))
        
        for defect_type in types:
            if defect_type == 'vortex':
                charges.append(1)
            elif defect_type == 'antivortex':
                charges.append(-1)
            elif defect_type == 'domain_wall':
                charges.append(0)
                
        u = np.ones_like(self.X, dtype=complex)
        
        domain_walls = []
        for (x, y), defect_type, charge in zip(positions, types, charges):
            if defect_type in ['vortex', 'antivortex']:
                r_local = np.sqrt((self.X - x)**2 + (self.Y - y)**2)
                theta_local = np.arctan2(self.Y - y, self.X - x)
                
                profile = np.tanh(r_local / core_size)
                defect = profile * np.exp(1j * charge * theta_local)
                
                if interaction_strength < 1.0:
                    u = u * (1.0 - interaction_strength + interaction_strength * defect)
                else:
                    u = u * defect
            elif defect_type == 'domain_wall':
                angle = np.random.uniform(0, 2*np.pi)
                nx = np.cos(angle)
                ny = np.sin(angle)
                
                distance = (self.X - x) * nx + (self.Y - y) * ny
                wall_profile = np.tanh(distance / core_size)
                
                domain_walls.append(wall_profile)
        
        for wall in domain_walls:
            if interaction_strength < 1.0:
                u = u * (1.0 - interaction_strength + interaction_strength * wall)
            else:
                u = u * wall
                
        thermal_field = self.spectral_method(
            amplitude=temperature * amplitude,
            apply_envelope=False,
            spectrum_slope=-1.5
        )
        
        u = u * amplitude + thermal_field * (1 - np.abs(u)/amplitude)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def quasiperiodic_structure(self, amplitude=1.0, symmetry_order=5, amplitude_decay=1.0,
                              phase_noise=0.0, apply_envelope=True, envelope_width=0.8,
                              k_min=1.0, k_max=5.0, n_shells=3, phase_symmetry=True,
                              central_spot=True, position=(0, 0), orientation=0.0,
                              modulation_wavelength=None):
        x0, y0 = position
        k_vectors = []
        amplitudes = []
        phases = []
        
        for i in range(symmetry_order):
            angle = 2 * np.pi * i / symmetry_order + orientation
            for shell in range(1, n_shells + 1):
                k_mag = k_min + (k_max - k_min) * shell / n_shells
                kx = k_mag * np.cos(angle)
                ky = k_mag * np.sin(angle)
                k_vectors.append((kx, ky))
                
                shell_amplitude = amplitude * np.exp(-amplitude_decay * (shell - 1))
                amplitudes.append(shell_amplitude)
                
                if phase_symmetry:
                    if phase_noise > 0:
                        phases.append(angle + np.random.normal(0, phase_noise))
                    else:
                        phases.append(angle)
                else:
                    phases.append(np.random.uniform(0, 2*np.pi))
        
        u = np.zeros_like(self.X, dtype=complex)
        
        if central_spot:
            u += amplitude * np.exp(1j * orientation)
        
        for (kx, ky), amp, phase in zip(k_vectors, amplitudes, phases):
            wave = amp * np.exp(1j * (kx * (self.X - x0) + ky * (self.Y - y0) + phase))
            u += wave
            
        if modulation_wavelength is not None:
            modulation = np.cos(2 * np.pi * (self.X - x0) / modulation_wavelength) * np.cos(2 * np.pi * (self.Y - y0) / modulation_wavelength)
            u *= (1 + 0.2 * modulation)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def self_similar_pattern(self, amplitude=1.0, scale_factor=2.0, intensity_scaling=0.6,
                           num_iterations=3, base_pattern='soliton', apply_envelope=True,
                           envelope_width=0.8, position=(0, 0), phase=0.0,
                           rotation_per_iteration=0.0, width=1.0, seed_amplitude=None):
        x0, y0 = position
        
        if seed_amplitude is None:
            seed_amplitude = amplitude
            
        if base_pattern == 'soliton':
            base = self.fundamental_soliton(
                'cubic', amplitude=seed_amplitude, width=width,
                position=position, phase=phase, apply_envelope=False
            )
        elif base_pattern == 'vortex':
            base = self.vortex_solution(
                amplitude=seed_amplitude, position=position,
                core_size=width, apply_envelope=False
            )
        elif base_pattern == 'ring':
            base = self.ring_soliton(
                amplitude=seed_amplitude, radius=width*2,
                width=width, position=position, phase=phase,
                apply_envelope=False
            )
        else:
            base = self.spectral_method(
                amplitude=seed_amplitude, apply_envelope=False,
                k_min=1/width, k_max=3/width
            )
            
        u = base.copy()
        
        for i in range(1, num_iterations + 1):
            scaled_amplitude = amplitude * (intensity_scaling ** i)
            current_scale = scale_factor ** i
            current_rotation = rotation_per_iteration * i
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                        
                    new_x = x0 + width * current_scale * dx
                    new_y = y0 + width * current_scale * dy
                    
                    if base_pattern == 'soliton':
                        component = self.fundamental_soliton(
                            'cubic', amplitude=scaled_amplitude, width=width,
                            position=(new_x, new_y), phase=phase + current_rotation,
                            apply_envelope=False
                        )
                    elif base_pattern == 'vortex':
                        component = self.vortex_solution(
                            amplitude=scaled_amplitude, position=(new_x, new_y),
                            core_size=width, apply_envelope=False,
                            orientation=current_rotation
                        )
                    elif base_pattern == 'ring':
                        component = self.ring_soliton(
                            amplitude=scaled_amplitude, radius=width*2,
                            width=width, position=(new_x, new_y),
                            phase=phase + current_rotation, apply_envelope=False
                        )
                    else:
                        component = self.spectral_method(
                            amplitude=scaled_amplitude, apply_envelope=False,
                            k_min=1/width, k_max=3/width
                        )
                        
                    u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def rogue_wave_precursor(self, amplitude=1.0, background_amplitude=0.5, 
                           perturbation_strength=0.2, transverse_wavenumber=2.0,
                           length_scale=2.0, position=(0, 0), phase=0.0,
                           apply_envelope=True, envelope_width=0.8,
                           perturbation_type='lorentzian'):
        x0, y0 = position
        
        background = background_amplitude * np.exp(1j * phase)
        
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        
        if perturbation_type == 'sech':
            envelope = 1 / np.cosh(r_local / length_scale)
        elif perturbation_type == 'gaussian':
            envelope = np.exp(-r_local**2 / (2 * length_scale**2))
        elif perturbation_type == 'lorentzian':
            envelope = 1 / (1 + (r_local / length_scale)**2)
        else:
            envelope = 1 / np.cosh(r_local / length_scale)
            
        kr = transverse_wavenumber * r_local
        perturbation = perturbation_strength * (1 + 1j * transverse_wavenumber) * envelope * np.exp(1j * kr)
        
        u = background * (1 + perturbation)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return amplitude * u
        
    def generate_ensemble(self, phenomenon_type, system_type='cubic', n_samples=10, 
                        parameter_ranges=None, **fixed_params):
        samples = []
        
        if parameter_ranges is None:
            parameter_ranges = {}
            
        for _ in range(n_samples):
            params = fixed_params.copy()
            
            for param, range_values in parameter_ranges.items():
                if isinstance(range_values, list):
                    params[param] = np.random.choice(range_values)
                elif isinstance(range_values, tuple) and len(range_values) == 2:
                    if isinstance(range_values[0], int) and isinstance(range_values[1], int):
                        params[param] = np.random.randint(range_values[0], range_values[1]+1)
                    else:
                        params[param] = np.random.uniform(range_values[0], range_values[1])
                else:
                    raise ValueError(f"Invalid parameter range for {param}: {range_values}")
            
            if phenomenon_type == 'fundamental_soliton':
                sample = self.fundamental_soliton(system_type=system_type, **params)
            elif phenomenon_type == 'multi_soliton':
                sample = self.multi_soliton_state(system_type=system_type, **params)
            elif phenomenon_type == 'spectral':
                sample = self.spectral_method(**params)
            elif phenomenon_type == 'chaotic':
                sample = self.chaotic_field(**params)
            elif phenomenon_type == 'vortex':
                sample = self.vortex_solution(**params)
            elif phenomenon_type == 'vortex_lattice':
                sample = self.vortex_lattice(**params)
            elif phenomenon_type == 'dark_soliton':
                sample = self.dark_soliton(**params)
            elif phenomenon_type == 'solitary_wave_with_ambient':
                sample = self.solitary_wave_with_ambient_field(system_type=system_type, **params)
            elif phenomenon_type == 'logarithmic_singularity':
                sample = self.logarithmic_singularity(**params)
            elif phenomenon_type == 'free_singularity':
                sample = self.free_singularity_solution(**params)
            elif phenomenon_type == 'free_singularity_adapted':
                sample = self.free_singularity_adapted(**params)
            elif phenomenon_type == 'logarithmic_singularity_adapted':
                sample = self.logarithmic_singularity_adapted(**params) 
            elif phenomenon_type == 'transparent_solitary_wave':
                sample = self.transparent_solitary_wave(**params)
            elif phenomenon_type == 'colliding_solitary_waves':
                sample = self.colliding_solitary_waves(system_type=system_type, **params)
            elif phenomenon_type == 'oscillating_breather':
                sample = self.oscillating_breather(**params)
            elif phenomenon_type == 'ring_soliton':
                sample = self.ring_soliton(**params)
            elif phenomenon_type == 'multi_ring':
                sample = self.multi_ring(**params) 
            elif phenomenon_type == 'turbulent_condensate':
                sample = self.turbulent_condensate(**params)
            elif phenomenon_type == 'topological_defect_network':
                sample = self.topological_defect_network(**params)
            elif phenomenon_type == 'quasiperiodic_structure':
                sample = self.quasiperiodic_structure(**params)
            elif phenomenon_type == 'self_similar_pattern':
                sample = self.self_similar_pattern(**params)
            elif phenomenon_type == 'rogue_wave_precursor':
                sample = self.rogue_wave_precursor(**params)
            elif phenomenon_type == 'akhmediev_breather':
                sample = self.akhmediev_breather(**params)
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            samples.append(sample)
        
        return samples
    
    def generate_diverse_ensemble(self, phenomenon_type, system_type='cubic', n_samples=10,
                               parameter_ranges=None, similarity_threshold=0.2,
                               max_attempts=100, diversity_metric='l2', **fixed_params):
        samples = []
        attempts = 0
        
        if parameter_ranges is None:
            parameter_ranges = {}
            
        def diversity_distance(s1, s2):
            if diversity_metric == 'l2':
                s1_norm = np.sqrt(np.sum(np.abs(s1)**2))
                s2_norm = np.sqrt(np.sum(np.abs(s2)**2))
                if s1_norm == 0 or s2_norm == 0:
                    return 1.0
                return np.sqrt(np.sum(np.abs(s1/s1_norm - s2/s2_norm)**2))
            elif diversity_metric == 'spectral':
                s1_fft = np.fft.fft2(s1)
                s2_fft = np.fft.fft2(s2)
                s1_fft_abs = np.abs(s1_fft)
                s2_fft_abs = np.abs(s2_fft)
                s1_fft_norm = np.sqrt(np.sum(s1_fft_abs**2))
                s2_fft_norm = np.sqrt(np.sum(s2_fft_abs**2))
                if s1_fft_norm == 0 or s2_fft_norm == 0:
                    return 1.0
                overlap = np.sum(s1_fft_abs * s2_fft_abs) / (s1_fft_norm * s2_fft_norm)
                return 1.0 - overlap
            else:
                return 0.0  # No diversity check
        discarded = 0 
        while len(samples) < n_samples and attempts < max_attempts:
            params = fixed_params.copy()
            
            for param, range_values in parameter_ranges.items():
                if isinstance(range_values, list):
                    params[param] = np.random.choice(range_values)
                elif isinstance(range_values, tuple) and len(range_values) == 2:
                    if isinstance(range_values[0], int) and isinstance(range_values[1], int):
                        params[param] = np.random.randint(range_values[0], range_values[1]+1)
                    else:
                        params[param] = np.random.uniform(range_values[0], range_values[1])
                else:
                    raise ValueError(f"Invalid parameter range for {param}: {range_values}")
            
            if phenomenon_type == 'fundamental_soliton':
                sample = self.fundamental_soliton(system_type=system_type, **params)
            elif phenomenon_type == 'multi_soliton':
                sample = self.multi_soliton_state(system_type=system_type, **params)
            elif phenomenon_type == 'spectral': # good?
                sample = self.spectral_method(**params)
            elif phenomenon_type == 'chaotic': # good?
                sample = self.chaotic_field(**params)
            elif phenomenon_type == 'vortex': # base for vortices, alright
                sample = self.vortex_solution(**params)
            elif phenomenon_type == 'vortex_lattice': # good
                sample = self.vortex_lattice(**params)
            elif phenomenon_type == 'dark_soliton': # good? (needs runs, parameters)
                sample = self.dark_soliton(**params)
            elif phenomenon_type == 'solitary_wave_with_ambient': # good? (needs runs, parameters)
                sample = self.solitary_wave_with_ambient_field(system_type=system_type, **params)
            elif phenomenon_type == 'free_singularity_adapted': # good? (needs runs)
                sample = self.free_singularity_adapted(**params)
            elif phenomenon_type == 'logarithmic_singularity_adapted': # good? (needs runs)
                sample = self.logarithmic_singularity_adapted(**params) 
            elif phenomenon_type == 'transparent_solitary_wave': # shit sampler -- background though?
                sample = self.transparent_solitary_wave(**params)
            elif phenomenon_type == 'colliding_solitary_waves': # qualitatively fairly similar
                sample = self.colliding_solitary_waves(system_type=system_type, **params)
            elif phenomenon_type == 'oscillating_breather': # shit sampler -- background though?
                sample = self.oscillating_breather(**params)
            elif phenomenon_type == 'ring_soliton': # tested with trajectories, good
                sample = self.ring_soliton(**params)
            elif phenomenon_type == 'multi_ring': # tested with trajectories, good
                sample = self.multi_ring(**params) 
            elif phenomenon_type == 'turbulent_condensate': # good?
                sample = self.turbulent_condensate(**params)
            elif phenomenon_type == 'topological_defect_network': # good?
                sample = self.topological_defect_network(**params)
            elif phenomenon_type == 'quasiperiodic_structure': # shit sampler -- background though?
                sample = self.quasiperiodic_structure(**params)
            elif phenomenon_type == 'self_similar_pattern': # shit sampler -- background though?
                sample = self.self_similar_pattern(**params)
            elif phenomenon_type == 'rogue_wave_precursor': # shit sampler -- background though?
                sample = self.rogue_wave_precursor(**params)
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            if len(samples) == 0:
                max_abs = np.max(np.abs(sample))
                if max_abs > 0:
                    sample = sample / max_abs
                samples.append(sample)
            else:
                is_diverse = True
                for existing_sample in samples:
                    dist = diversity_distance(sample, existing_sample)
                    if dist < similarity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    # need to watch out for solutions which somehow yield nan values!
                    nans = np.sum(sample == np.nan)
                    if nans:
                        attempts += 1
                        continue
                    else:  
                        max_abs = np.max(np.abs(sample))
                        if max_abs > 0:
                            sample = sample / max_abs
                        samples.append(sample)
                else:
                    discarded += 1 
            attempts += 1
        print("trash ratio:", discarded / attempts)            
        return samples

def tsne_complex_fields(samples, perplexity=30, n_iter=1000):
    import numpy as np
    from sklearn.manifold import TSNE
    features = []
    for sample in samples:
        flat = sample.flatten()
        feature_vector = np.concatenate([np.abs(flat), np.angle(flat)])
        features.append(feature_vector)

    features_array = np.array(features)

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embedding = tsne.fit_transform(features_array)
    return embedding

def compare_phenomena_tsne(sampler, n_samples=10, perplexity=30, n_iter=2000):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    phenomenon_groups = [
        {
            'name': 'Soliton-based',
            'types': ['multi_soliton', 'multi_ring'],
            'params': {'system_type': 'glasner_allen_flowers'}
        },
        {
            'name': 'Vortex-based',
            'types': ['vortex_lattice'],
            'params': {}
        },
        {
            'name': 'Spectral/Chaotic',
            'types': ['spectral', 'chaotic', 'turbulent_condensate'],
            'params': {}
        },
        {
            'name': 'Topological', # not really sure
            'types': ['topological_defect_network', 'dark_soliton'],
            'params': {}
        },
        {
            'name': 'Ambient',
            'types': ['solitary_wave_with_ambient'],
            'params': {'system_type': 'glasner_allen_flowers'}
        },
        {
            'name': 'Singularities',
            'types': ['free_singularity_adapted', 'logarithmic_singularity_adapted'],
            'params': {}
        }
    ]

    meta_ensemble = []
    labels_group = []
    labels_type = []
    all_types = []

    for group_idx, group in enumerate(phenomenon_groups):
        for phenomenon_type in group['types']:
            all_types.append(phenomenon_type)
            type_idx = len(all_types) - 1

            params = group['params'].copy()

            for _ in range(n_samples):
                try:
                    if phenomenon_type == 'multi_soliton':
                        sample = sampler.multi_soliton_state(**params)
                    elif phenomenon_type == 'spectral':
                        sample = sampler.spectral_method()
                    elif phenomenon_type == 'chaotic':
                        sample = sampler.chaotic_field()
                    elif phenomenon_type == 'vortex':
                        sample = sampler.vortex_solution()
                    elif phenomenon_type == 'vortex_lattice':
                        sample = sampler.vortex_lattice()
                    elif phenomenon_type == 'dark_soliton':
                        sample = sampler.dark_soliton()
                    elif phenomenon_type == 'solitary_wave_with_ambient':
                        sample = sampler.solitary_wave_with_ambient_field(**params)
                    elif phenomenon_type == 'free_singularity_adapted':
                        sample = sampler.free_singularity_adapted()
                    elif phenomenon_type == 'logarithmic_singularity_adapted':
                        sample = sampler.logarithmic_singularity_adapted()
                    elif phenomenon_type == 'ring_soliton':
                        sample = sampler.ring_soliton()
                    elif phenomenon_type == 'multi_ring':
                        sample = sampler.multi_ring()
                    elif phenomenon_type == 'turbulent_condensate':
                        sample = sampler.turbulent_condensate()
                    elif phenomenon_type == 'topological_defect_network':
                        sample = sampler.topological_defect_network()
                    else:
                        continue

                    max_amp = np.max(np.abs(sample))
                    if max_amp > 0:
                        sample = sample / max_amp

                    meta_ensemble.append(sample)
                    labels_group.append(group_idx)
                    labels_type.append(type_idx)
                except Exception as e:
                    print(f"Error generating {phenomenon_type}: {e}")

    features = []
    for sample in meta_ensemble:
        flat = sample.flatten()
        real_imag = np.concatenate([np.real(flat), np.imag(flat)])
        features.append(real_imag)

    features_array = np.array(features)

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embedding = tsne.fit_transform(features_array)

    group_names = [group['name'] for group in phenomenon_groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    group_colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=[group_colors[i] for i in labels_group], s=70, alpha=0.8)
    ax1.set_title('t-SNE by Phenomenon Group', fontsize=16, fontweight='bold')
    ax1.axis('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])

    legend_elements = [Patch(facecolor=group_colors[i], label=name) for i, name in enumerate(group_names)]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)

    type_colors = plt.cm.tab20(np.linspace(0, 1, len(all_types)))
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=[type_colors[i] for i in labels_type], s=70, alpha=0.8)
    ax2.set_title('t-SNE by Specific Phenomenon Type', fontsize=16, fontweight='bold')
    ax2.axis('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])

    legend_elements = [Patch(facecolor=type_colors[i], label=name) for i, name in enumerate(all_types)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    plt.figure(figsize=(14, 10))
    for i, phenomenon_type in enumerate(all_types):
        type_mask = np.array(labels_type) == i
        if np.any(type_mask):
            plt.scatter(embedding[type_mask, 0], embedding[type_mask, 1],
                       s=80, alpha=0.8, label=phenomenon_type)

    plt.title('t-SNE of NLSE Phenomena Types', fontsize=18, fontweight='bold')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    return fig

def detailed_phenomenon_comparison(sampler, n_samples=15, perplexity=30, n_iter=2000):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    phenomena = [
            {'type': 'akhmediev_breather', 'name': 'Breather', 'system_type': None},
        #{'type': 'multi_soliton', 'name': 'Multi-Soliton', 'system_type': 'cubic'},
        #{'type': 'vortex_lattice', 'name': 'Vortex Lattice', 'system_type': None},
        #{'type': 'ring_soliton', 'name': 'Ring Soliton', 'system_type': None},
        #{'type': 'multi_ring', 'name': 'Multi-Ring', 'system_type': None},
        #{'type': 'dark_soliton', 'name': 'Dark Soliton', 'system_type': None},
        #{'type': 'solitary_wave_with_ambient', 'name': 'Solitary w/ Ambient', 'system_type': 'cubic'},
        #{'type': 'spectral', 'name': 'Spectral Field', 'system_type': None},
        #{'type': 'chaotic', 'name': 'Chaotic Field', 'system_type': None},
        #{'type': 'free_singularity_adapted', 'name': 'Free Singularity', 'system_type': None},
        #{'type': 'turbulent_condensate', 'name': 'Turbulent Condensate', 'system_type': None},
        #{'type': 'topological_defect_network', 'name': 'Topological Defect Network', 'system_type': None}
    ]

    meta_ensemble = []
    labels = []

    for idx, p in enumerate(phenomena):
        for _ in range(n_samples):
            try:
                if p['type'] == 'multi_soliton':
                    sample = sampler.multi_soliton_state(system_type=p['system_type'])
                elif p['type'] == 'vortex_lattice':
                    sample = sampler.vortex_lattice()
                elif p['type'] == 'ring_soliton':
                    sample = sampler.ring_soliton()
                elif p['type'] == 'multi_ring':
                    sample = sampler.multi_ring()
                elif p['type'] == 'dark_soliton':
                    sample = sampler.dark_soliton()
                elif p['type'] == 'solitary_wave_with_ambient':
                    sample = sampler.solitary_wave_with_ambient_field(system_type=p['system_type'])
                elif p['type'] == 'spectral':
                    sample = sampler.spectral_method()
                elif p['type'] == 'chaotic':
                    sample = sampler.chaotic_field()
                elif p['type'] == 'free_singularity_adapted':
                    sample = sampler.free_singularity_adapted()
                elif p['type'] == 'turbulent_condensate':
                    sample = sampler.turbulent_condensate()
                elif p['type'] == 'topological_defect_network':
                    sample = sampler.topological_defect_network()
                elif p['type'] == 'akhmediev_breather':
                    sample = sampler.akhmediev_breather()
                else:
                    continue

                max_amp = np.max(np.abs(sample))
                if max_amp > 0:
                    sample = sample / max_amp

                if np.sum(sample == np.nan) > 1:
                    print("Encountered invalid sample")
                    continue

                meta_ensemble.append(sample)
                labels.append(idx)
            except Exception as e:
                print(f"Error generating {p['type']}: {e}")
    features = []
    for sample in meta_ensemble:
        flat = sample.flatten()
        feature_vector = np.concatenate([np.abs(flat), np.angle(flat)])
        features.append(feature_vector)

    features_array = np.array(features)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embedding = tsne.fit_transform(features_array)

    cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(phenomena)))
    custom_cmap = ListedColormap(colors)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=custom_cmap,
                         s=100, alpha=0.8, edgecolors='k', linewidths=0.5)

    plt.title('t-SNE Visualization of NLSE Phenomena', fontsize=18, fontweight='bold')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])

    patches = [mpatches.Patch(color=colors[i], label=p['name']) for i, p in enumerate(phenomena)]
    plt.legend(handles=patches, loc='upper right', fontsize=12)

    plt.tight_layout()

    return plt.gcf()

def detailed_soliton_parameter_scaling(sampler, n_samples=15, perplexity=30, n_iter=2000):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    phenomena = [
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic', 'arrangement': 'linear', 'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic_quintic', 'arrangement': 'linear',  'chirp_range':(-0.1, 0.1)},  
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'saturable' , 'arrangement': 'linear',  'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'glasner_allen_flowers' , 'arrangement': 'linear',  'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic', 'arrangement': 'circular',  'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic_quintic', 'arrangement': 'circular',  'chirp_range':(-0.1, 0.1)},  
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'saturable' , 'arrangement': 'circular',  'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'glasner_allen_flowers' , 'arrangement': 'circular',  'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic', 'arrangement': 'linear', 'chirp_range':(-1, 1.)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic_quintic', 'arrangement': 'linear',  'chirp_range':(-1, 1)},  
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'saturable' , 'arrangement': 'linear',  'chirp_range':(-1, 1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'glasner_allen_flowers' , 'arrangement': 'linear',  'chirp_range':(-1, 1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic', 'arrangement': 'circular',  'chirp_range':(-0.1, 0.1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'cubic_quintic', 'arrangement': 'circular',  'chirp_range':(-1, 1)},  
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'saturable' , 'arrangement': 'circular',  'chirp_range':(-1, 1)},
        {'type': 'multi_soliton', 'name': 'Multi-Soliton',
            'system_type': 'glasner_allen_flowers' , 'arrangement': 'circular',  'chirp_range':(-1, 1)},

    ]

    meta_ensemble = []
    labels = []

    for idx, p in enumerate(phenomena):
        for _ in range(n_samples):
            try:
                if p['type'] == 'multi_soliton':
                    sample = sampler.multi_soliton_state(
                            system_type=p['system_type'], arrangement=p['arrangement'],
                            chirp_range=p['chirp_range'])
                meta_ensemble.append(sample)
                labels.append(idx)
            except Exception as e:
                print(f"Error generating {p['type']}: {e}")
    features = []
    for sample in meta_ensemble:
        flat = sample.flatten()
        feature_vector = np.concatenate([np.abs(flat), np.angle(flat)])
        features.append(feature_vector)

    features_array = np.array(features)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter)
    embedding = tsne.fit_transform(features_array)

    cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(phenomena)))
    custom_cmap = ListedColormap(colors)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=custom_cmap,
                         s=100, alpha=0.8, edgecolors='k', linewidths=0.5)

    plt.title('t-SNE Visualization of NLSE Phenomena', fontsize=18, fontweight='bold')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)

    patches = [mpatches.Patch(color=colors[i], label=f"{p['system_type']} - {p['arrangement']} - {p['chirp_range']}") for i, p in enumerate(phenomena)]
    plt.legend(handles=patches, fontsize=12)

    plt.tight_layout()
    return plt.gcf()




if __name__ == '__main__':
    n = 100
    L = 10.
    sampler = NLSEPhenomenonSampler(n, n, L)
    detailed_phenomenon_comparison(sampler, n_samples=100, perplexity=30, n_iter=2000)
    #detailed_soliton_parameter_scaling(sampler, n_samples=100, perplexity=30, n_iter=2000)
    plt.show()   

    # good params to get a nice map!
    #compare_phenomena_tsne(sampler, n_samples=500, n_iter=2000)
    #detailed_phenomenon_comparison(sampler, n_samples=100, n_iter=2000)

    #all_types = [
    #    'fundamental_soliton',
    #    'multi_soliton',
    #    'spectral',
    #    'chaotic',
    #    'vortex',
    #    'vortex_lattice',
    #    'dark_soliton',
    #    'solitary_wave_with_ambient',
    #    'logarithmic_singularity',
    #    'free_singularity',
    #    'transparent_solitary_wave',
    #    'colliding_solitary_waves',
    #    'oscillating_breather',
    #    'ring_soliton'
    #]

    #for phenomenon_type in ['multi_soliton']: 
    #    if phenomenon_type != 'multi_soliton':
    #        ensemble = sampler.generate_diverse_ensemble(phenomenon_type,
    #                n_samples=100, max_attempts=1000, )
    #        for i, sample in enumerate(ensemble):
    #            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    #            im0 = axs[0, 0].imshow(np.abs(sample), cmap='viridis')
    #            axs[0, 0].set_title("amplitude")
    #            fig.colorbar(im0, ax=axs[0, 0])
    #            im1 = axs[0, 1].imshow(np.angle(sample), cmap='hsv')
    #            axs[0, 1].set_title("phase")
    #            fig.colorbar(im1, ax=axs[0, 1])
    #            im2 = axs[1, 0].imshow(np.real(sample), cmap='coolwarm')
    #            axs[1, 0].set_title("real")
    #            fig.colorbar(im2, ax=axs[1, 0])
    #            im3 = axs[1, 1].imshow(np.imag(sample), cmap='coolwarm')
    #            axs[1, 1].set_title("imag")
    #            fig.colorbar(im3, ax=axs[1, 1])
    #            plt.suptitle(f"{phenomenon_type} - {i + 1}", fontsize=16)
    #            plt.tight_layout()
    #            plt.show()
    #            

    #    if phenomenon_type == 'multi_soliton': 
    #        arrangements = ['linear', 'circular', 'random', 'lattice', 'hierarchical']
    #        phase_patterns = ['random', 'alternating', 'synchronized', 'vortex', 'partial_coherence', ]
    #        system_types = ['cubic', 'saturable', 'cubic_quintic', 'glasner_allen_flowers']
    #        n_samples = 10
    #        print("Generating", n_samples * len(arrangements) * len(phase_patterns) * len(system_types), "samples")
    #        meta_ensemble = []
    #        labels_system = []
    #        labels_arrangement = []
    #        labels_phase = []

    #        for s, system_type in enumerate(system_types) : 
    #            for a, arrangement in enumerate(arrangements):
    #                for p, phase_pattern in enumerate(phase_patterns):
    #                    ensemble = sampler.generate_diverse_ensemble(phenomenon_type,
    #                            system_type=system_type, arrangement=arrangement,
    #                            phase_pattern=phase_pattern,
    #                            n_samples=n_samples, max_attempts=n_samples * 10, )
    #                    for e in ensemble:
    #                        meta_ensemble.append(e)
    #                        labels_system.append(a)
    #                        labels_arrangement.append(a)
    #                        labels_phase.append(p)

    #        embedding = tsne_complex_fields(meta_ensemble, n_iter=1000) 
    #        plt.figure(figsize=(18, 6))

    #        def plot_with_equal_aspect(position, embedding, labels, cmap, title, label_names):
    #            plt.subplot(1, 3, position)
    #            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, alpha=0.7, s=50)
    #            plt.title(title, fontsize=14, fontweight='bold')
    #            cbar = plt.colorbar(scatter, ticks=range(len(label_names)))
    #            cbar.set_label(title.split(' by ')[1], fontsize=12)
    #            cbar.ax.set_yticklabels(label_names)
    #            plt.clim(-0.5, len(label_names)-0.5)
    #            plt.axis('equal')
    #            ax = plt.gca()
    #            ax.set_yticks([])
    #            ax.set_xticks([])
    #            ax.spines['top'].set_visible(True)
    #            ax.spines['right'].set_visible(True)
    #            ax.spines['bottom'].set_visible(True)
    #            ax.spines['left'].set_visible(True)
    #        
    #        plot_with_equal_aspect(1, embedding, labels_system, 'tab10', 't-SNE by System Type', system_types)
    #        plot_with_equal_aspect(2, embedding, labels_arrangement, 'viridis', 't-SNE by Arrangement', arrangements)
    #        plot_with_equal_aspect(3, embedding, labels_phase, 'plasma', 't-SNE by Phase Pattern', phase_patterns) 
    #        plt.tight_layout()
    #        plt.show()

