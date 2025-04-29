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
            breather_phase='compressed', apply_envelope=False, envelope_width=0.7,
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


    def ring_soliton(self, amplitude=1.0, radius=3.0, width=0.5, position=None,
                   phase=0.0, apply_envelope=False, envelope_width=0.7,
                   modulation_type='none', modulation_strength=0.0, modulation_mode=0,
                   aspect_ratio=1.0, orientation=0.0, radial_nodes=0):
        if position is None:
            x0, y0 = np.random.rand(2) * self.L / 3
        else:
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
            n_rings = np.random.randint(3, 6)
            
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
        
        ring_types = np.random.choice(['standard', 'chirped', 'modulated'], n_rings)
        chirp_factors = np.random.uniform(0.05, 0.4, n_rings)
        global_phase_field = 0
        
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
            ring_type = ring_types[i]
            
            dx = self.X - x0
            dy = self.Y - y0
            r_local = np.sqrt(dx**2 + dy**2)
            theta_local = np.arctan2(dy, dx)
            
            if ring_type == 'standard':
                component = self.ring_soliton(
                    amplitude, radius=radius, width=width, position=(x0, y0),
                    phase=phase, apply_envelope=False, 
                    modulation_type=modulation_type if modulation_type != 'none' else 'azimuthal',
                    modulation_strength=modulation_strength if modulation_strength > 0 else 0.2,
                    modulation_mode=modulation_mode if modulation_mode > 0 else i % 3 + 1,
                    aspect_ratio=aspect_ratio, orientation=orientation, radial_nodes=radial_nodes
                )
            elif ring_type == 'chirped':
                chirp_factor = chirp_factors[i]
                radial_chirp = chirp_factor * (r_local - radius)**2
                component = self.ring_soliton(
                    amplitude, radius=radius, width=width, position=(x0, y0),
                    phase=phase, apply_envelope=False,
                    modulation_type='azimuthal', modulation_strength=0.3,
                    modulation_mode=i % 3 + 1, aspect_ratio=aspect_ratio, 
                    orientation=orientation, radial_nodes=radial_nodes
                )
                component *= np.exp(1j * radial_chirp)
            else:  # modulated
                angular_freq = (i % 4) + 1
                radial_freq = (i % 3) + 1
                phase_modulation = 0.3 * np.sin(angular_freq * theta_local) * np.sin(radial_freq * np.pi * (r_local - radius) / width)
                component = self.ring_soliton(
                    amplitude, radius=radius, width=width, position=(x0, y0),
                    phase=phase, apply_envelope=False,
                    modulation_type='azimuthal', modulation_strength=0.25,
                    modulation_mode=2, aspect_ratio=aspect_ratio,
                    orientation=orientation, radial_nodes=radial_nodes
                )
                component *= np.exp(1j * phase_modulation)
            
            u += component
            
            if i < n_rings - 1:
                dx_next = positions[i+1][0] - x0
                dy_next = positions[i+1][1] - y0
                dist = np.sqrt(dx_next**2 + dy_next**2)
                if dist > 0:
                    interaction_phase = 0.2 * np.exp(
                            -(r_local - radius)**2/(2*width**2)) * np.exp(
                                    -((self.X - positions[i+1][0])**2 + (self.Y - positions[i+1][1])**2)/(4*radius**2))
                    global_phase_field += interaction_phase
        
        if np.abs(np.sum(global_phase_field)) < 1e-2:
            u *= np.exp(1j * global_phase_field)
        
        if arrangement == 'concentric' or arrangement == 'circular':
            center_x = np.mean([p[0] for p in positions])
            center_y = np.mean([p[1] for p in positions])
            vortex_field = np.exp(1j * np.arctan2(self.Y - center_y, self.X - center_x))
            u *= (0.7 + 0.3 * vortex_field)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def free_singularity_adapted(self, position=None, amplitude=None, Lambda=None,
                              epsilon=None, background_type=None,
                              background_amplitude=None, phase=None,
                              multi_scale=False, n_singularities=1):
        # (0.77) -- the special value where m(λ) ≈ 0 according to the paper
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
            # pretty braindead way of doing things but we'll leave it as is
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
        choice1 = ['vortex'] 
        choice2 = ['domain_wall']
        choice3 = ['antivortex']

        if domain_size is None:
            domain_size = 0.7 * self.L
            
        if defect_types is None:
            defect_types = np.random.choice(
                list(range(3)))
            if defect_types == 0:
                defect_types = choice1 + choice3
            elif defect_types == 1:
                defect_types = choice2
            elif defect_types == 2:
                defect_types = choice3
            else:
                raise


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
    def self_similar_pattern(self, amplitude=1.0, scale_factor=2.0, intensity_scaling=0.6,
                          num_iterations=5, base_pattern=None, apply_envelope=True,
                          envelope_width=0.8, position=None, phase=None,
                          rotation_per_iteration=0.1, width=1.0, seed_amplitude=None):
        if base_pattern is None:
            base_pattern = np.random.choice(['soliton', 'vortex', 'ring'])

        if position is None:
            position = (np.random.uniform(-0.3*self.L, 0.3*self.L), 
                       np.random.uniform(-0.3*self.L, 0.3*self.L))
        
        x0, y0 = position

        if phase is None:
            phase = np.random.uniform(0, 2*np.pi)
        
        if seed_amplitude is None:
            seed_amplitude = amplitude
            
        base_width = width
        structure_points = []
        all_components = []
        
        if base_pattern == 'soliton':
            base = self.fundamental_soliton(
                'cubic', amplitude=seed_amplitude, width=base_width,
                position=position, phase=phase, apply_envelope=False,
                chirp_factor=0.3
            )
            all_components.append((position, base_width, 0, 'soliton'))
        elif base_pattern == 'vortex':
            base = self.vortex_solution(
                amplitude=seed_amplitude, position=position,
                core_size=base_width, apply_envelope=False,
                charge=1
            )
            all_components.append((position, base_width, 0, 'vortex'))
        elif base_pattern == 'ring':
            base = self.ring_soliton(
                amplitude=seed_amplitude, radius=base_width*2,
                width=base_width/2, position=position, phase=phase,
                apply_envelope=False
            )
            all_components.append((position, base_width, 0, 'ring'))
        else:
            base = self.spectral_method(
                amplitude=seed_amplitude, apply_envelope=False,
                k_min=1/base_width, k_max=3/base_width
            )
            all_components.append((position, base_width, 0, 'spectral'))
                
        u = base.copy()
        
        for i in range(1, num_iterations + 1):
            scaled_amplitude = amplitude * (intensity_scaling ** i)
            current_scale = scale_factor ** i
            current_rotation = rotation_per_iteration * i
            
            branching = 3 + (i % 3)
            for k in range(branching):
                angle = 2*np.pi*k/branching + current_rotation
                dx = np.cos(angle)
                dy = np.sin(angle)
                
                new_x = x0 + width * current_scale * dx
                new_y = y0 + width * current_scale * dy
                
                local_phase = phase + np.arctan2(dy, dx) + current_rotation
                local_width = base_width * (0.9 + 0.2 * (i % 2))
                
                if base_pattern == 'soliton':
                    component = self.fundamental_soliton(
                        'cubic', amplitude=scaled_amplitude, width=local_width,
                        position=(new_x, new_y), phase=local_phase,
                        apply_envelope=False, chirp_factor=0.2 + 0.1*i,
                        aspect_ratio=1.0 + 0.3*np.sin(i*np.pi/2)
                    )
                    all_components.append(((new_x, new_y), local_width, i, 'soliton'))
                elif base_pattern == 'vortex':
                    charge = 1 if i % 2 == 0 else -1
                    component = self.vortex_solution(
                        amplitude=scaled_amplitude, position=(new_x, new_y),
                        core_size=local_width, apply_envelope=False,
                        orientation=current_rotation,
                        charge=charge
                    )
                    all_components.append(((new_x, new_y), local_width, i, 'vortex'))
                elif base_pattern == 'ring':
                    component = self.ring_soliton(
                        amplitude=scaled_amplitude, radius=local_width*2,
                        width=local_width/2, position=(new_x, new_y),
                        phase=local_phase, apply_envelope=False,
                        modulation_type='azimuthal' if i % 2 == 0 else 'none',
                        modulation_strength=0.3,
                        modulation_mode=1 + (i % 3)
                    )
                    all_components.append(((new_x, new_y), local_width, i, 'ring'))
                else:
                    component = self.spectral_method(
                        amplitude=scaled_amplitude, apply_envelope=False,
                        k_min=1/local_width, k_max=3/local_width
                    )
                    all_components.append(((new_x, new_y), local_width, i, 'spectral'))
                    
                u += component
                structure_points.append((new_x, new_y, local_width, i))
        
        velocity_field = np.zeros_like(u, dtype=complex)
        
        for (cx, cy), cwidth, level, ctype in all_components:
            dx = self.X - cx
            dy = self.Y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            if ctype == 'vortex':
                charge = 1 if level % 2 == 0 else -1
                local_field = charge * np.exp(1j * theta) * np.exp(-r**2/(4*cwidth**2))
            elif ctype == 'ring':
                radius = cwidth * 2
                local_field = np.exp(1j * (theta + np.pi/2)) * np.exp(-(r-radius)**2/(cwidth**2))
            else:
                local_field = np.exp(1j * (dx*0.2 + dy*0.1)/cwidth) * np.exp(-r**2/(4*cwidth**2))
                
            flow_strength = 0.2 * (scale_factor ** (-level/2))
            velocity_field += flow_strength * local_field
        
        velocity_field = np.exp(1j * np.angle(velocity_field))
        
        u = u * velocity_field
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
                
        return u
   
        
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
            elif phenomenon_type == 'free_singularity_adapted':
                sample = self.free_singularity_adapted(**params)
            elif phenomenon_type == 'logarithmic_singularity_adapted':
                sample = self.logarithmic_singularity_adapted(**params) 
            elif phenomenon_type == 'ring_soliton':
                sample = self.ring_soliton(**params)
            elif phenomenon_type == 'multi_ring':
                sample = self.multi_ring(**params) 
            elif phenomenon_type == 'turbulent_condensate':
                sample = self.turbulent_condensate(**params)
            elif phenomenon_type == 'topological_defect_network':
                sample = self.topological_defect_network(**params)
            elif phenomenon_type == 'self_similar_pattern':
                sample = self.self_similar_pattern(**params)
            elif phenomenon_type == 'akhmediev_breather':
                sample = self.akhmediev_breather(**params)
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            samples.append(sample)
        
        if n_samples == 1:
            return samples[0]
        else:
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
                return 0.0
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
            elif phenomenon_type == 'ring_soliton': # tested with trajectories, good
                sample = self.ring_soliton(**params)
            elif phenomenon_type == 'multi_ring': # tested with trajectories, good
                sample = self.multi_ring(**params) 
            elif phenomenon_type == 'turbulent_condensate': # good?
                sample = self.turbulent_condensate(**params)
            elif phenomenon_type == 'topological_defect_network': # good?
                sample = self.topological_defect_network(**params)
            elif phenomenon_type == 'self_similar_pattern': # shit sampler -- background though?
                sample = self.self_similar_pattern(**params)
            elif phenomenon_type == 'akhmediev_breather':
                sample = self.akhmediev(**params)
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

class NLSE3DSampler:
    def __init__(self, nx: int, ny: int, nz: int, L: float):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.L = L

        self._setup_grid()

    def _setup_grid(self):
        self.x = np.linspace(-self.L, self.L, self.nx)
        self.y = np.linspace(-self.L, self.L, self.ny)
        self.z = np.linspace(-self.L, self.L, self.nz)

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)
        self.r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.theta = np.arctan2(self.Y, self.X)

        self.dx = 2 * self.L / (self.nx - 1)
        self.dy = 2 * self.L / (self.ny - 1)
        self.dz = 2 * self.L / (self.nz - 1)

        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=2 * self.L / self.nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=2 * self.L / self.ny)
        self.kz = 2 * np.pi * np.fft.fftfreq(self.nz, d=2 * self.L / self.nz)

        self.KX, self.KY, self.KZ = np.meshgrid(self.kx, self.ky, self.kz)
        self.K_mag = np.sqrt(self.KX**2 + self.KY**2 + self.KZ**2)
    
    def _sech(self, x):
        return 1.0 / np.cosh(x)
    
    def _envelope(self, u, width):
        envelope = np.exp(-(self.X**2 + self.Y**2 + self.Z**2) / (width * self.L)**2)
        return u * envelope

    def fundamental_soliton(self, system_type, amplitude=1.0, width=1.0, position=(0, 0, 0),
                           phase=0.2, velocity=(0.0, 0.0, 0.0), sigma1=1.0, sigma2=-0.1,
                           kappa=1.0, apply_envelope=True, envelope_width=0.7, Lambda=0.1,
                           chirp_factor=0.0, aspect_ratio_x=1.0, aspect_ratio_y=1.0, 
                           orientation_xy=0.0, orientation_xz=0.0, orientation_yz=0.0, order=1):
        x0, y0, z0 = position
        vx, vy, vz = velocity
        
        X1 = (self.X - x0) * np.cos(orientation_xy) + (self.Y - y0) * np.sin(orientation_xy)
        Y1 = -(self.X - x0) * np.sin(orientation_xy) + (self.Y - y0) * np.cos(orientation_xy)
        Z1 = (self.Z - z0)
        
        X2 = X1 * np.cos(orientation_xz) + Z1 * np.sin(orientation_xz)
        Y2 = Y1
        Z2 = -X1 * np.sin(orientation_xz) + Z1 * np.cos(orientation_xz)
        
        X_rot = X2
        Y_rot = Y2 * np.cos(orientation_yz) + Z2 * np.sin(orientation_yz)
        Z_rot = -Y2 * np.sin(orientation_yz) + Z2 * np.cos(orientation_yz)
        
        r_local = np.sqrt((X_rot/aspect_ratio_x)**2 + (Y_rot/aspect_ratio_y)**2 + Z_rot**2)
        
        momentum_phase = vx * (self.X - x0) + vy * (self.Y - y0) + vz * (self.Z - z0)
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
                          chirp_range=(-0.1, 0.1), aspect_ratio_x_range=(1.0, 1.5), 
                          aspect_ratio_y_range=(1.0, 1.5), phase_value=0.0):
        u = np.zeros_like(self.X, dtype=complex)
        n_solitons = np.random.randint(3, 12) 
        
        if arrangement == 'linear':
            base_positions = [(i - (n_solitons-1)/2) * separation for i in range(n_solitons)]
            positions = [(pos, 0, 0) for pos in base_positions]
        elif arrangement == 'planar_grid':
            side = int(np.ceil(np.sqrt(n_solitons)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_solitons:
                        x = (i - (side-1)/2) * separation
                        y = (j - (side-1)/2) * separation
                        positions.append((x, y, 0))
        elif arrangement == 'circular':
            positions = []
            for i in range(n_solitons):
                angle = 2 * np.pi * i / n_solitons
                x = separation * np.cos(angle)
                y = separation * np.sin(angle)
                positions.append((x, y, 0))
        elif arrangement == 'spherical':
            positions = []
            for i in range(n_solitons):
                phi = np.arccos(1 - 2 * i / n_solitons)
                theta = np.pi * (1 + 5**0.5) * i
                x = separation * np.sin(phi) * np.cos(theta)
                y = separation * np.sin(phi) * np.sin(theta)
                z = separation * np.cos(phi)
                positions.append((x, y, z))
        elif arrangement == 'random':
            positions = []
            for _ in range(n_solitons):
                x = np.random.normal(0, position_variance * self.L/4)
                y = np.random.normal(0, position_variance * self.L/4)
                z = np.random.normal(0, position_variance * self.L/4)
                positions.append((x, y, z))
        elif arrangement == 'lattice':
            side = int(np.ceil(n_solitons**(1/3)))
            positions = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        if len(positions) < n_solitons:
                            x = (i - (side-1)/2) * separation
                            y = (j - (side-1)/2) * separation
                            z = (k - (side-1)/2) * separation
                            positions.append((x, y, z))
        elif arrangement == 'hierarchical':
            positions = []
            if cluster_levels <= 1:
                cluster_centers = [(0, 0, 0)]
            else:
                cluster_centers = []
                for i in range(cluster_levels):
                    phi = np.arccos(1 - 2 * i / cluster_levels)
                    theta = np.pi * (1 + 5**0.5) * i
                    x = separation * 2 * np.sin(phi) * np.cos(theta)
                    y = separation * 2 * np.sin(phi) * np.sin(theta)
                    z = separation * 2 * np.cos(phi)
                    cluster_centers.append((x, y, z))
            
            solitons_per_cluster = n_solitons // len(cluster_centers)
            remainder = n_solitons % len(cluster_centers)
            
            for i, (cx, cy, cz) in enumerate(cluster_centers):
                cluster_size = solitons_per_cluster + (1 if i < remainder else 0)
                for j in range(cluster_size):
                    if j == 0 and cluster_levels > 1:
                        positions.append((cx, cy, cz))
                    else:
                        phi = np.arccos(1 - 2 * j / cluster_size)
                        theta = np.pi * (1 + 5**0.5) * j
                        x = cx + separation * 0.5 * np.sin(phi) * np.cos(theta)
                        y = cy + separation * 0.5 * np.sin(phi) * np.sin(theta)
                        z = cz + separation * 0.5 * np.cos(phi)
                        positions.append((x, y, z))

        

            
        
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        center_z = np.mean([p[2] for p in positions])
        
        if phase_pattern == 'random':
            phases = np.random.uniform(0, 2*np.pi, n_solitons)
        elif phase_pattern == 'alternating':
            phases = [i * np.pi for i in range(n_solitons)]
        elif phase_pattern == 'synchronized':
            phases = np.ones(n_solitons) * phase_value
        elif phase_pattern == 'vortex':
            phases = []
            for x, y, z in positions:
                phi = np.arctan2(y - center_y, x - center_x)
                phases.append(phi)
        elif phase_pattern == '3d_vortex':
            phases = []
            for x, y, z in positions:
                r = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
                theta = np.arccos((z-center_z)/max(r, 1e-10))
                phi = np.arctan2(y-center_y, x-center_x)
                phases.append(phi + theta)
        elif phase_pattern == 'radial':
            phases = []
            for x, y, z in positions:
                r = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
                phases.append(r)
        elif phase_pattern == 'spiral':
            phases = []
            for x, y, z in positions:
                phi = np.arctan2(y - center_y, x - center_x)
                r = np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2)
                phases.append(phi + r)
        elif phase_pattern == 'z_dependent':
            phases = []
            for x, y, z in positions:
                phases.append(z - center_z)
        elif phase_pattern == 'partial_coherence':
            base_phase = np.random.uniform(0, 2*np.pi)
            phases = []
            for _ in range(n_solitons):
                if np.random.random() < coherence:
                    phases.append(base_phase)
                else:
                    phases.append(np.random.uniform(0, 2*np.pi))
        
        for i, ((x0, y0, z0), phase) in enumerate(zip(positions, phases)):
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            Lambda = np.random.uniform(*Lambda_range)
            order = np.random.randint(*order_range)
            chirp = np.random.uniform(*chirp_range)
            aspect_ratio_x = np.random.uniform(*aspect_ratio_x_range)
            aspect_ratio_y = np.random.uniform(*aspect_ratio_y_range)
            orientation_xy = np.random.uniform(0, 2*np.pi)
            orientation_xz = np.random.uniform(0, 2*np.pi)
            orientation_yz = np.random.uniform(0, 2*np.pi)
            
            if velocity_scale > 0:
                if arrangement == 'spherical':
                    r_vec = np.array([x0, y0, z0])
                    r_norm = np.linalg.norm(r_vec)
                    if r_norm > 1e-10:
                        unit_vec = -r_vec / r_norm
                        vx, vy, vz = velocity_scale * unit_vec
                    else:
                        vx = vy = vz = 0
                elif arrangement == 'circular':
                    angle = 2 * np.pi * i / n_solitons
                    vx = -velocity_scale * np.cos(angle)
                    vy = -velocity_scale * np.sin(angle)
                    vz = 0
                else:
                    vx = np.random.normal(0, velocity_scale)
                    vy = np.random.normal(0, velocity_scale)
                    vz = np.random.normal(0, velocity_scale)
            else:
                vx, vy, vz = 0, 0, 0
            
            component = self.fundamental_soliton(
                system_type, amplitude=amplitude, width=width, 
                position=(x0, y0, z0), phase=phase, velocity=(vx, vy, vz),
                sigma1=sigma1, sigma2=sigma2, kappa=kappa,
                apply_envelope=False, Lambda=Lambda, order=order,
                chirp_factor=chirp, aspect_ratio_x=aspect_ratio_x, 
                aspect_ratio_y=aspect_ratio_y, orientation_xy=orientation_xy,
                orientation_xz=orientation_xz, orientation_yz=orientation_yz
            )
            
            if interaction_strength < 1.0 and i > 0:
                u = u + component * interaction_strength
            else:
                u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)

        # # very artifical "boundary envelope" trying to reduce BC effects
        # buffer_x_mask = np.abs(self.X) > .8 * self.L
        # buffer_y_mask = np.abs(self.Y) > .8 * self.L 
        # buffer_z_mask = np.abs(self.Z) > .8 * self.L
        # clamping_mask = np.logical_and(
        #         np.logical_and(buffer_x_mask, buffer_y_mask),
        #         buffer_z_mask)
        # u[clamping_mask] = 0.1 * u[clamping_mask]
    
        return u

    def skyrmion_tube(self, system_type='cubic', amplitude_range=(0.8, 1.5), radius_range=(1.0, 3.0), 
                     width_range=(0.5, 1.5), position_variance=0.5, phase_range=(0, 2*np.pi),
                     winding_range=(1, 3), k_z_range=(0.1, 1.0), velocity_scale=0.3,
                     chirp_range=(-0.1, 0.1), tube_count_range=(1, 5),
                     apply_envelope=True, envelope_width=0.7, tube_arrangement='random',
                     interaction_strength=0.5, deformation_factor=0.2):
        u = np.zeros_like(self.X, dtype=np.complex128)
        n_tubes = np.random.randint(*tube_count_range)
        
        if tube_arrangement == 'random':
            positions = []
            for _ in range(n_tubes):
                x = np.random.normal(0, position_variance * self.L/4)
                y = np.random.normal(0, position_variance * self.L/4)
                z = 0
                positions.append((x, y, z))
        elif tube_arrangement == 'circular':
            positions = []
            radius = self.L / 4
            for i in range(n_tubes):
                angle = 2 * np.pi * i / n_tubes
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 0
                positions.append((x, y, z))
        elif tube_arrangement == 'linear':
            positions = []
            spacing = self.L / 3
            for i in range(n_tubes):
                x = (i - (n_tubes-1)/2) * spacing
                y = 0
                z = 0
                positions.append((x, y, z))
        elif tube_arrangement == 'lattice':
            side = int(np.ceil(np.sqrt(n_tubes)))
            positions = []
            spacing = self.L / 4
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_tubes:
                        x = (i - (side-1)/2) * spacing
                        y = (j - (side-1)/2) * spacing
                        z = 0
                        positions.append((x, y, z))
        
        for i, (x0, y0, z0) in enumerate(positions):
            amplitude = np.random.uniform(*amplitude_range)
            radius = np.random.uniform(*radius_range)
            width = np.random.uniform(*width_range)
            phase = np.random.uniform(*phase_range)
            winding = np.random.randint(*winding_range)
            k_z = np.random.uniform(*k_z_range)
            chirp = np.random.uniform(*chirp_range)
            
            if velocity_scale > 0:
                vx = np.random.normal(0, velocity_scale)
                vy = np.random.normal(0, velocity_scale)
                vz = np.random.normal(0, velocity_scale)
            else:
                vx = vy = vz = 0
            
            rho = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            phi = np.arctan2(self.Y - y0, self.X - x0)
            
            deformation = 1.0 + deformation_factor * np.cos(phi * np.random.randint(1, 4))
            
            profile = amplitude * np.exp(-((rho - radius*deformation)**2 + (self.Z - z0)**2) / width**2)
            
            momentum_phase = vx * (self.X - x0) + vy * (self.Y - y0) + vz * (self.Z - z0)
            chirp_term = chirp * ((self.X - x0)**2 + (self.Y - y0)**2 + (self.Z - z0)**2)
            
            phase_term = winding * phi + k_z * (self.Z - z0) + phase + momentum_phase + chirp_term
            
            component = profile * np.exp(1j * phase_term)
            
            if interaction_strength < 1.0 and i > 0:
                u = u + component * interaction_strength
            else:
                u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u

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
            
            if phenomenon_type == 'multi_soliton_state':
                sample = self.multi_soliton_state(system_type=system_type, **params)
            elif phenomenon_type == 'skyrmion_tube':
                sample = self.skyrmion_tube(system_type=system_type, **params) 
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            samples.append(sample)
        
        if n_samples == 1:
            return samples[0]
        else:
            return samples

    def generate_initial_condition(self, system_type: str = 'cubic',
                                   phenomenon_type: List[str] = None, **params) -> Tuple[np.ndarray, np.ndarray]:
        if phenomenon_type is None:
            raise Exception
        u0 = self.generate_ensemble(system_type=system_type,
                                      phenomenon_type=phenomenon_type,
                                      n_samples=1,
                                      **params)
        dV = self.dx * self.dy * self.dz

        mass = np.sum(np.abs(u0)**2) * dV
        target_mass = .2
        if mass > 1e-15:
            scale = np.sqrt(target_mass / mass)
            u0_normalized = u0 * scale
            
        # u0 = u0 / np.max(np.abs(u0))
        return u0
