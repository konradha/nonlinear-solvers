import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.fft as fft
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Callable
from improved_nlse_samplers import make_grid, sample_gaussian_random_field


class NLSESampler:
    def __init__(self, Nx: int, Ny: int, L: float):
        self.Nx = Nx
        self.Ny = Ny
        self.L = L
        self.X, self.Y = make_grid(Nx, Ny, L)
        
        self.kx = 2 * np.pi * fft.fftfreq(Nx, d=2 * L / Nx)
        self.ky = 2 * np.pi * fft.fftfreq(Ny, d=2 * L / Ny)
        self.KX, self.KY = torch.meshgrid(self.kx, self.ky, indexing='ij')
        self.K = torch.sqrt(self.KX**2 + self.KY**2)
        
        self.state_families = {
            'single_vortex': 0.10,
            'vortex_cluster': 0.20,
            'multi_soliton': 0.3,
            'breather': 0.15,
            'turbulent': 0.15,
            'ground_state': 0.10,
        }
        
        self.E_min = 0.1
        self.E_max = 100

    def create_envelope(self, shape: str = 'star', width_factor: float = 0.1, order: float = 1.0):
        r = torch.sqrt(self.X**2 + self.Y**2)
        effective_L = 0.9 * self.L
        
        if shape == 'tanh':
            boundary_dist = effective_L - r
            envelope = torch.tanh(boundary_dist / (width_factor * self.L))
            envelope = torch.clamp(envelope, min=0.0)
            
        elif shape == 'star':
            theta = torch.atan2(self.Y, self.X)
            n_lobes = 4 
            angular_modulation = 0.2 + 0.3 * torch.abs(torch.cos(n_lobes * theta))
            effective_radius = effective_L * angular_modulation

            boundary_dist = effective_radius - r
            envelope = torch.tanh(boundary_dist / (width_factor * self.L))
            envelope = torch.clamp(envelope, min=0.0)

        elif shape =='hyper':
            curvature = 2.
            sharpness = 3.
            X_norm = self.X / self.L
            Y_norm = self.Y / self.L
            p = 2.0 / (1.0 + curvature)
            r_p = (torch.abs(X_norm)**(p) + torch.abs(Y_norm)**(p))**(1/p)
            envelope = 1.0 / (1.0 + torch.exp(sharpness * (r_p - 0.85)))

            
        elif shape == 'gaussian':
            envelope = torch.exp(-r**2 / (2 * (effective_L * 0.6)**2))
            
        elif shape == 'super_gaussian':
            envelope = torch.exp(-(r / (effective_L * 0.6))**(2 * order))
            
        else:
            raise ValueError(f"Unknown envelope shape: {shape}")
            
        return envelope
    
    def apply_spectral_filter(self, psi: torch.Tensor, filter_type: str = 'power_law',
                              alpha: float = 5/3, k0: float = None, 
                              sigma_k: float = None, k_max: float = None):
        if k_max is None:
            k_max = 0.8 * torch.max(self.K)
            
        psi_hat = fft.fft2(psi)
        
        if filter_type == 'power_law':
            F_k = self.K**(-alpha) * torch.exp(-(self.K/k_max)**2)
            F_k[0, 0] = 0
            
        elif filter_type == 'band_limited':
            if k0 is None:
                k0 = 0.3 * torch.max(self.K)
            if sigma_k is None:
                sigma_k = 0.1 * k0
                
            F_k = torch.exp(-(self.K - k0)**2 / (2 * sigma_k**2))
            
        elif filter_type == 'multi_scale':
            if k0 is None:
                k_values = [0.1, 0.3, 0.6] * torch.max(self.K)
                amplitudes = [1.0, 0.6, 0.3]
                widths = [0.05, 0.1, 0.15] * torch.max(self.K)
            else:
                k_values = [k0 * 0.5, k0, k0 * 2.0]
                amplitudes = [0.5, 1.0, 0.5]
                widths = [sigma_k * 0.5, sigma_k, sigma_k * 1.5] if sigma_k else [0.05, 0.1, 0.15] * torch.max(self.K)
                
            F_k = torch.zeros_like(self.K)
            for k_val, amp, width in zip(k_values, amplitudes, widths):
                F_k += amp * torch.exp(-(self.K - k_val)**2 / (2 * width**2))
                
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        F_k = F_k / torch.max(F_k)
        psi_hat_filtered = psi_hat * F_k
        psi_filtered = fft.ifft2(psi_hat_filtered)
        
        return psi_filtered
        
    def sample_parameters(self, param_type: str, **kwargs):
        if param_type == 'vortex_charge':
            return random.choice([-2, -1, 1, 2])
            
        elif param_type == 'width':
            k = kwargs.get('k', 3.0)
            theta = kwargs.get('theta', 0.1 * self.L)
            max_width = kwargs.get('max_width', 0.4 * self.L)
            
            width = np.random.gamma(k, theta)
            return min(width, max_width)
            
        elif param_type == 'position':
            sigma = kwargs.get('sigma', self.L / 4)
            offset_x = torch.randn(1).item() * sigma
            offset_y = torch.randn(1).item() * sigma
            return torch.tensor([offset_x, offset_y])
            
        elif param_type == 'amplitude':
            a = kwargs.get('a', 2.0)
            b = kwargs.get('b', 2.0)
            scale = kwargs.get('scale', 1.0)
            
            amp = np.random.beta(a, b) * scale
            return amp
            
        elif param_type == 'phase':
            return 2 * np.pi * torch.rand(1).item()
            
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def create_ground_state(self, shape: str = 'gaussian', width: float = None, center: torch.Tensor = None):
        if width is None:
            width = self.sample_parameters('width', k=4.0, theta=0.08 * self.L)
            
        if center is None:
            center = self.sample_parameters('position', sigma=self.L / 5)
            
        x0, y0 = center
        r2 = (self.X - x0)**2 + (self.Y - y0)**2
        
        if shape == 'gaussian':
            psi = torch.exp(-r2 / (2 * width**2))
            
        elif shape == 'thomas_fermi':
            r_tf = width * 0.7
            psi = torch.maximum(1 - r2 / r_tf**2, torch.zeros_like(r2))
            psi = torch.sqrt(psi)
            
        elif shape == 'super_gaussian':
            order = 4.0
            psi = torch.exp(-(r2 / (2 * width**2))**order)
            
        else:
            raise ValueError(f"Unknown ground state shape: {shape}")
            
        return psi * torch.exp(1j * torch.zeros_like(self.X))
    
    def create_vortex(self, charge: int = None, width: float = None, 
                      center: torch.Tensor = None, core_size: float = None):
        if charge is None:
            charge = self.sample_parameters('vortex_charge')
            
        if width is None:
            width = self.sample_parameters('width', k=4.0, theta=0.07 * self.L)
            
        if center is None:
            center = self.sample_parameters('position', sigma=self.L / 5)
            
        if core_size is None:
            core_size = 0.05 * self.L
            
        x0, y0 = center
        r = torch.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = torch.atan2(self.Y - y0, self.X - x0)
        
        amplitude = torch.tanh(r / core_size)
        phase = charge * theta
        
        return amplitude * torch.exp(1j * phase)
    
    def create_vortex_cluster(self, num_vortices: int = None, arrangement: str = None, width: float = None):
        if num_vortices is None:
            num_vortices = random.randint(2, 8)
            
        if arrangement is None:
            arrangement = random.choice(['random', 'ring', 'lattice', 'clustered'])
            
        if width is None:
            width = self.sample_parameters('width', k=5.0, theta=0.1 * self.L)
            
        centers = []
        charges = []
        effective_L = 0.6 * self.L
        
        if arrangement == 'random':
            for _ in range(num_vortices):
                centers.append(effective_L * 0.5 * torch.randn(2))
                charges.append(random.choice([-1, 1]))
        
        elif arrangement == 'ring':
            radius = width * 0.3
            for i in range(num_vortices):
                angle = 2 * np.pi * i / num_vortices
                centers.append([radius * np.cos(angle), radius * np.sin(angle)])
                charges.append(1 if i % 2 == 0 else -1)
        
        elif arrangement == 'lattice':
            side_length = int(np.ceil(np.sqrt(num_vortices)))
            spacing = effective_L * 0.5 / side_length
            
            count = 0
            for i in range(side_length):
                for j in range(side_length):
                    if count < num_vortices:
                        offset = spacing * 0.5 if i % 2 == 1 else 0
                        centers.append([spacing * (j - side_length/2 + 0.5) + offset, 
                                       spacing * (i - side_length/2 + 0.5)])
                        charges.append(1 if (i + j) % 2 == 0 else -1)
                        count += 1
        
        elif arrangement == 'clustered':
            num_clusters = min(3, num_vortices)
            vortices_per_cluster = num_vortices // num_clusters
            remaining = num_vortices % num_clusters
            
            for c in range(num_clusters):
                cluster_center = effective_L * 0.3 * torch.randn(2)
                cluster_size = effective_L * 0.15
                
                vortices_in_this_cluster = vortices_per_cluster + (1 if c < remaining else 0)
                
                for _ in range(vortices_in_this_cluster):
                    offset = cluster_size * 0.7 * torch.randn(2)
                    centers.append(cluster_center + offset)
                    charges.append(random.choice([-1, 1]))
        
        psi = torch.ones_like(self.X, dtype=torch.complex128)
        
        for center, charge in zip(centers, charges):
            x0, y0 = center
            r_vortex = torch.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            theta = torch.atan2(self.Y - y0, self.X - x0)
            
            core_size = 0.05 * self.L
            amplitude_factor = r_vortex / torch.sqrt(r_vortex**2 + core_size**2)
            phase_factor = torch.exp(1j * charge * theta)
            psi = psi * amplitude_factor * phase_factor
        
        return psi
    
    def create_soliton(self, soliton_type: str = None, num_solitons: int = None, 
                       width: float = None, arrangement: str = None, velocity: float = None):
        if soliton_type is None:
            soliton_type = random.choice(['bright', 'dark'])
            
        if num_solitons is None:
            num_solitons = random.randint(1, 4)
            
        if width is None:
            width = self.sample_parameters('width', k=3.0, theta=0.1 * self.L)
            
        if arrangement is None:
            arrangement = random.choice(['parallel', 'crossing', 'radial'])
            
        if velocity is None and random.random() < 0.5:
            velocity = 2.0 * torch.rand(1).item()
        
        effective_width = min(width, 0.15 * self.L)
        if soliton_type == "dark":
            psi = torch.ones_like(self.X, dtype=torch.complex128)
            max_amplitude = 1.0
        else:
            psi = torch.zeros_like(self.X, dtype=torch.complex128)
            max_amplitude = 2.0
            
        distances = []
        angles = []
        velocities = []
        
        if arrangement == "parallel":
            spacing = 1.5 * effective_width
            max_offset = (num_solitons - 1) * spacing / 2
            
            for i in range(num_solitons):
                dist = spacing * i - max_offset
                distances.append(dist)
                angles.append(0) 
                if velocity is not None:
                    velocities.append(velocity * (-1 if i % 2 == 1 else 1))
                else:
                    velocities.append(0)
        
        elif arrangement == "crossing":
            for i in range(num_solitons):
                distances.append(0)
                angles.append(np.pi * i / num_solitons)
                if velocity is not None:
                    velocities.append(velocity)
                else:
                    velocities.append(0)
        
        elif arrangement == "radial":
            for i in range(num_solitons):
                distances.append(0)
                angles.append(2 * np.pi * i / num_solitons)
                if velocity is not None:
                    velocities.append(velocity)
                else:
                    velocities.append(0.5)
        
        for d, angle, vel in zip(distances, angles, velocities):
            X_rot = self.X * np.cos(angle) + self.Y * np.sin(angle)
            Y_rot = -self.X * np.sin(angle) + self.Y * np.cos(angle)
            
            if soliton_type == "bright":
                amplitude = max_amplitude / torch.cosh((X_rot - d) / effective_width)
                phase = vel * X_rot
                psi = psi + amplitude * torch.exp(1j * phase)
                
            else:
                tanh_factor = 1.5  
                profile = torch.tanh(tanh_factor * (X_rot - d) / effective_width)
                phase = vel * X_rot
                psi = psi * profile * torch.exp(1j * phase)
        
        return psi
    
    def create_breather(self, breather_type: str = None, width: float = None, 
                        oscillation: float = None, offset_position: bool = True):
        if breather_type is None:
            breather_type = random.choice(['kuznetsov', 'localized', 'vector', 'spatial'])
            
        if width is None:
            width = self.sample_parameters('width', k=3.0, theta=0.1 * self.L)
            
        if oscillation is None:
            oscillation = 1.0 + 2.0 * torch.rand(1).item()
            
        effective_width = min(width, 0.3*self.L) 
        if offset_position:
            max_offset = 0.3 * self.L
            x0 = max_offset * (2 * torch.rand(1).item() - 1)
            y0 = max_offset * (2 * torch.rand(1).item() - 1)
        else:
            x0, y0 = 0, 0
        
        X_shifted = self.X - x0
        Y_shifted = self.Y - y0
        
        theta_rotation = 2 * np.pi * torch.rand(1).item()
        X_rot = X_shifted * np.cos(theta_rotation) + Y_shifted * np.sin(theta_rotation)
        Y_rot = -X_shifted * np.sin(theta_rotation) + Y_shifted * np.cos(theta_rotation)
        
        r = torch.sqrt(X_rot**2 + Y_rot**2)
        theta = torch.atan2(Y_rot, X_rot)
        
        if breather_type == "kuznetsov":
            a = 0.5 / effective_width  
            b = oscillation 
            t = 0.0
            
            amplitude = 1.0 / torch.cosh(r / effective_width)
            c_param = np.sqrt(2) * a
            psi = amplitude * (torch.cos(torch.tensor(b * t)) + np.sqrt(2) * torch.sinh(a * r) / torch.cosh(a * r))
            
        elif breather_type == "localized":
            k = oscillation
            envelope = 1.0 / (1.0 + (r / effective_width)**2)
            oscillatory = torch.cos(k * r)
            psi = envelope * (1.0 + 3.0 * oscillatory * envelope)
        
        elif breather_type == "vector":
            k1 = oscillation
            envelope1 = 1.0 / (1.0 + (r / effective_width)**2)
            osc1 = torch.cos(k1 * r)
            comp1 = envelope1 * (1.0 + 2.0 * osc1 * envelope1)
            
            k2 = oscillation * 1.2
            phase_shift = np.pi/2
            envelope2 = 1.0 / (1.0 + (r / (0.8 * effective_width))**2)
            osc2 = torch.cos(k2 * r + phase_shift)
            comp2 = envelope2 * (1.0 + 2.0 * osc2 * envelope2) 
            
            angular_weight = 0.5 + 0.5 * torch.cos(2 * theta)
            psi = comp1 * angular_weight + comp2 * (1 - angular_weight)
        
        elif breather_type == "spatial":
            kx = oscillation
            ky = oscillation * 1.5 
            spatial_osc = torch.cos(kx * X_rot) * torch.cos(ky * Y_rot)
            psi = (1.0 + 2.0 * spatial_osc * (1.0 / (1.0 + r/effective_width)))
        
        phase_gradient = 0.5 * (X_rot + Y_rot) / self.L
        psi = psi * torch.exp(1j * phase_gradient)
        
        return psi
    
    def create_turbulent_state(self, energy_spectrum: str = None, num_vortices: int = None):
        if energy_spectrum is None:
            energy_spectrum = random.choice(['kolmogorov', 'gaussian', 'white'])
            
        if num_vortices is None:
            num_vortices = random.randint(10, 50)
            
        effective_L = 0.6 * self.L
        psi = torch.ones_like(self.X, dtype=torch.complex128)
        
        centers = []
        charges = []
        for _ in range(num_vortices):
            centers.append(effective_L * 0.5 * torch.randn(2))
            charges.append(random.choice([-1, 1]))
            
        for center, charge in zip(centers, charges):
            x0, y0 = center
            r = torch.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            theta = torch.atan2(self.Y - y0, self.X - x0)
            
            core_size = 0.05 * self.L + 0.05 * self.L * torch.rand(1).item()
            vortex_amplitude = torch.tanh(r / core_size)
            vortex_phase = torch.exp(1j * charge * theta)
            
            psi = psi * (vortex_amplitude * vortex_phase + (1 - vortex_amplitude))
        
        if energy_spectrum == "kolmogorov":
            power_law = -5/3
        elif energy_spectrum == "gaussian":
            power_law = -2
        else:
            power_law = -1
        
        filter_scale = 10.0 / self.L
        fluctuations = sample_gaussian_random_field(self.X.shape[0], self.X.shape[1], 
                                                  effective_L, 
                                                  power_law=power_law,
                                                  filter_scale=filter_scale)
        
        phase_fluctuation_strength = a = 0.2
        psi = psi * torch.exp(1j * phase_fluctuation_strength * fluctuations.real)
        
        return psi
    
    def create_rogue_wave(self, wave_type: str = None, peak_amplitude: float = None, 
                          background_amplitude: float = None):
        if wave_type is None:
            wave_type = random.choice(["peregrine", "akhmediev", "kuznetsov", "superposition"])
            
        if peak_amplitude is None:
            peak_amplitude = 2.0 + 2.0 * torch.rand(1).item()
            
        if background_amplitude is None:
            background_amplitude = 0.3 + 0.4 * torch.rand(1).item()
        
        x0 = 0.3 * self.L * (2 * torch.rand(1).item() - 1)
        y0 = 0.3 * self.L * (2 * torch.rand(1).item() - 1)
        X_shifted = self.X - x0
        Y_shifted = self.Y - y0
        
        theta = 2 * np.pi * torch.rand(1).item()
        stretch_factor = 1.0 + 2.0 * torch.rand(1).item()
        X_rot = X_shifted * np.cos(theta) + Y_shifted * np.sin(theta)
        Y_rot = -X_shifted * np.sin(theta) + Y_shifted * np.cos(theta)
    
        X_stretched = X_rot
        Y_stretched = Y_rot / stretch_factor
        
        r2 = X_stretched**2 + Y_stretched**2
        
        if wave_type == "peregrine":
            scale_factor = 6.0 + 6.0 * torch.rand(1).item()
            xi = r2 / (scale_factor * self.L**2)
            
            t = -0.1 + 0.2 * torch.rand(1).item()
            denominator = 1.0 + 4.0 * xi * (1.0 + 2.0j * t)
            numerator = 4.0 * (1.0 + 2.0j * t)
            
            profile = background_amplitude * (1.0 - numerator / denominator)
            
        elif wave_type == "akhmediev":
            a = 0.25 + 0.2 * torch.rand(1).item()  
            k = 2.0 + 2.0 * torch.rand(1).item() 
            
            cos_term = torch.cos(k * X_stretched)
            cosh_term = torch.cosh(2 * a * Y_stretched) + a * torch.cos(k * X_stretched)
            
            profile = background_amplitude * (1.0 + 2 * (1 - 2*a) * cos_term / cosh_term)
        
        elif wave_type == "kuznetsov":
            b = 0.4 + 0.3 * torch.rand(1).item()
            phi = 2.0 * np.pi * torch.rand(1).item()
     
            r = torch.sqrt(r2)
            
            cos_term = torch.cos(b * r + phi)
            sin_term = torch.sin(b * r + phi)
            cosh_term = torch.cosh(torch.tensor(np.sqrt(2) * b * 0.2)) + torch.sqrt(torch.tensor(2.0)) * cos_term
            
            profile = background_amplitude * (1.0 + 2 * np.sqrt(2) * b * cos_term / cosh_term)
        
        elif wave_type == "superposition":
            k1 = 1.0 + torch.rand(1).item()
            k2 = 1.0 + torch.rand(1).item()
            phase = torch.rand(1).item() * np.pi
            
            background_wave = background_amplitude * torch.cos(k1*self.X + k2*self.Y + phase)
            
            scale_factor = 8.0
            xi = r2 / (scale_factor * self.L**2)
            peak = (peak_amplitude - background_amplitude) * (4.0 / (1.0 + 4.0 * xi))
            
            profile = background_wave + peak
        
        phase_scale = 0.2 + 0.3 * torch.rand(1).item()
        background_phase = phase_scale * (self.X + self.Y) + 0.1 * (self.X**2 - self.Y**2) / self.L**2
        
        psi = profile * torch.exp(1j * background_phase)
        
        return psi
    
    def create_composite_state(self, num_components: int = None, component_types: List[str] = None):
        if num_components is None:
            num_components = random.randint(2, 4)
            
        if component_types is None:
            all_types = ['ground_state', 'vortex', 'soliton', 'breather', 'rogue_wave']
            component_types = random.choices(all_types, k=num_components)
        
        psi_base = torch.zeros_like(self.X, dtype=torch.complex128)
        
        for comp_type in component_types:
            if comp_type == 'ground_state':
                comp = self.create_ground_state()
            elif comp_type == 'vortex':
                comp = self.create_vortex()
            elif comp_type == 'soliton':
                comp = self.create_soliton()
            elif comp_type == 'breather':
                comp = self.create_breather()
            elif comp_type == 'rogue_wave':
                comp = self.create_rogue_wave()
            else:
                continue
                
            amp = self.sample_parameters('amplitude', a=1.5, b=1.5, scale=1.0)
            phase = self.sample_parameters('phase')
            
            psi_base = psi_base + amp * comp * torch.exp(1j * phase)
            
        return psi_base
    
    def sample_state(self, state_type: str = None):
        if state_type is None:
            state_types = list(self.state_families.keys())
            state_probs = list(self.state_families.values())
            state_type = random.choices(state_types, weights=state_probs, k=1)[0]
            
        if state_type == 'single_vortex':
            psi_base = self.create_vortex()
            description = "Single vortex state"
            
        elif state_type == 'vortex_cluster':
            psi_base = self.create_vortex_cluster()
            description = "Vortex cluster state"
            
        elif state_type == 'multi_soliton':
            psi_base = self.create_soliton()
            description = "Multi-soliton state"
            
        elif state_type == 'breather':
            psi_base = self.create_breather()
            description = "Breather state"
            
        elif state_type == 'turbulent':
            psi_base = self.create_turbulent_state()
            description = "Turbulent state"
            
        elif state_type == 'ground_state':
            psi_base = self.create_ground_state()
            description = "Ground state"
            
        elif state_type == 'composite':
            psi_base = self.create_composite_state()
            description = "Composite state"
            
        else:
            raise ValueError(f"Unknown state type: {state_type}")
            
        return psi_base, description
    
    def apply_envelope_and_normalize(self, psi: torch.Tensor, envelope_shape: str = 'star',
                                    width_factor: float = 0.1, order: float = 3.0):
        envelope = self.create_envelope(shape=envelope_shape, width_factor=width_factor, order=order)
        psi_with_envelope = psi * envelope
        
        norm = torch.sqrt(
                torch.sum(torch.abs(psi_with_envelope)**2) * (2*self.L/(self.Nx - 1) * (2*self.L/(self.Ny - 1))))
        psi_normalized = psi_with_envelope / norm
        
        return psi_normalized
    
    def generate_nlse_state(self, state_type: str = None, 
                         envelope_shape: str = 'star',
                         apply_spectral_filter: bool = False,
                         filter_type: str = 'power_law',
                         alpha: float = 5/3,
                         normalize: bool = True):
        psi_base, description = self.sample_state(state_type)
        
        if apply_spectral_filter:
            psi_base = self.apply_spectral_filter(psi_base, filter_type=filter_type, alpha=alpha)
            
        if normalize:
            psi_final = self.apply_envelope_and_normalize(psi_base, envelope_shape=envelope_shape)
        else:
            envelope = self.create_envelope(shape=envelope_shape)
            psi_final = psi_base * envelope
            
        return psi_final, description
    
    def calculate_energy(self, psi: torch.Tensor):
        dx = 2 * self.L / self.Nx
        dy = 2 * self.L / self.Ny
        
        grad_x = torch.zeros_like(psi, dtype=torch.complex128)
        grad_y = torch.zeros_like(psi, dtype=torch.complex128)
        
        grad_x[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dx)
        grad_y[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)
        
        kinetic_energy = 0.5 * (torch.sum(torch.abs(grad_x)**2) + torch.sum(torch.abs(grad_y)**2)) * dx * dy
        potential_energy = 0.5 * torch.sum(torch.abs(psi)**4) * dx * dy
        
        return kinetic_energy + potential_energy
    
    def is_physically_valid(self, psi: torch.Tensor):
        energy = self.calculate_energy(psi)
        
        if energy < self.E_min or energy > self.E_max:
            return False
            
        density = torch.abs(psi)**2
        total_density = torch.sum(density)
        boundary_density = (torch.sum(density[0, :]) + torch.sum(density[-1, :]) + 
                           torch.sum(density[1:-1, 0]) + torch.sum(density[1:-1, -1]))
        
        boundary_ratio = boundary_density / total_density
        if boundary_ratio > 0.05:
            return False
            
        return True
    
    def spectral_analysis(self, psi: torch.Tensor):
        psi_hat = fft.fft2(psi)
        psi_hat_shift = fft.fftshift(psi_hat)
        
        power_spectrum = torch.abs(psi_hat_shift)**2
        
        k_bins = np.linspace(0, torch.max(self.K).item(), 50)
        radial_profile = np.zeros(len(k_bins)-1)
        
        kx_centered = fft.fftshift(self.kx)
        ky_centered = fft.fftshift(self.ky)
        KX_centered, KY_centered = torch.meshgrid(kx_centered, ky_centered, indexing='ij')
        K_centered = torch.sqrt(KX_centered**2 + KY_centered**2)
        
        for i in range(len(k_bins)-1):
            k_min, k_max = k_bins[i], k_bins[i+1]
            mask = (K_centered >= k_min) & (K_centered < k_max)
            radial_profile[i] = torch.mean(power_spectrum[mask]).item() if torch.any(mask) else 0
            
        return power_spectrum, radial_profile, k_bins[:-1]
    
    def visualize_wavefunction(self, psi: torch.Tensor, title: str = None, figsize: tuple = (15, 10)):
        fig = plt.figure(figsize=figsize)
        
        ax1 = fig.add_subplot(2, 3, 1)
        amplitude = torch.abs(psi)
        im1 = ax1.imshow(amplitude, cmap='viridis', origin='lower', 
                        extent=[-self.L, self.L, -self.L, self.L])
        ax1.set_title('Amplitude')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        
        ax2 = fig.add_subplot(2, 3, 2)
        phase = torch.angle(psi)
        im2 = ax2.imshow(phase, cmap='twilight', origin='lower', 
                        extent=[-self.L, self.L, -self.L, self.L], vmin=-np.pi, vmax=np.pi)
        ax2.set_title('Phase')
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax2)
        
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        X_mesh, Y_mesh = np.meshgrid(
            np.linspace(-self.L, self.L, self.Nx),
            np.linspace(-self.L, self.L, self.Ny),
            indexing='ij'
        )
        stride = max(1, self.Nx // 50)
        surf = ax3.plot_surface(X_mesh[::stride, ::stride], 
                               Y_mesh[::stride, ::stride], 
                               amplitude.numpy()[::stride, ::stride],
                               cmap='viridis', edgecolor='none')
        ax3.set_title('3D Amplitude')
        
        ax4 = fig.add_subplot(2, 3, 4)
        real_part = torch.real(psi)
        im4 = ax4.imshow(real_part, cmap='RdBu', origin='lower', 
                        extent=[-self.L, self.L, -self.L, self.L])
        ax4.set_title('Real Part')
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax4)
        
        ax5 = fig.add_subplot(2, 3, 5)
        imag_part = torch.imag(psi)
        im5 = ax5.imshow(imag_part, cmap='RdBu', origin='lower', 
                        extent=[-self.L, self.L, -self.L, self.L])
        ax5.set_title('Imaginary Part')
        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im5, cax=cax5)
        
        power_spectrum, radial_profile, k_bins = self.spectral_analysis(psi)
        
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.loglog(k_bins, radial_profile)
        ax6.set_xlabel('Wavenumber k')
        ax6.set_ylabel('Power')
        ax6.set_title('Radial Power Spectrum')
        ax6.grid(True, which="both", ls="-", alpha=0.2)
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        
        return fig
    
    def visualize_spectral_properties(self, psi: torch.Tensor, figsize: tuple = (15, 7)):
        fig = plt.figure(figsize=figsize)
        
        power_spectrum, radial_profile, k_bins = self.spectral_analysis(psi)
        
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(torch.log10(power_spectrum + 1e-10), cmap='inferno', origin='lower')
        ax1.set_title('Log Power Spectrum')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.loglog(k_bins, radial_profile, 'b-', linewidth=2)
        
        k_range = np.logspace(-1, 1, 100)
        for slope, label in zip([-5/3, -2, -3], ['k^(-5/3) (Kolmogorov)', 'k^(-2) (Gaussian)', 'k^(-3) (steep)']):
            scale_factor = radial_profile[len(radial_profile)//5] / (k_bins[len(radial_profile)//5]**slope)
            ax2.loglog(k_range, scale_factor * k_range**slope, '--', label=label, alpha=0.6)
            
        ax2.set_xlabel('Wavenumber k')
        ax2.set_ylabel('Power')
        ax2.set_title('Radial Power Spectrum')
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        ax2.legend()
        
        
        return fig
    
    def visualize_phase_flow(self, psi: torch.Tensor, figsize: tuple = (10, 8)):
        fig = plt.figure(figsize=figsize)
        
        amplitude = torch.abs(psi)
        phase = torch.angle(psi)
        
        dx = 2 * self.L / self.Nx
        dy = 2 * self.L / self.Ny
        
        grad_x = torch.zeros_like(phase)
        grad_y = torch.zeros_like(phase)
        
        grad_x[:, 1:-1] = (phase[:, 2:] - phase[:, :-2]) / (2 * dx)
        grad_y[1:-1, :] = (phase[2:, :] - phase[:-2, :]) / (2 * dy)
        
        velocity_x = -grad_y
        velocity_y = grad_x
        
        x = np.linspace(-self.L, self.L, self.Nx)
        y = np.linspace(-self.L, self.L, self.Ny)
        X_mesh, Y_mesh = np.meshgrid(x, y, indexing='ij')
        
        ax = fig.add_subplot(1, 1, 1)
        
        stride = max(1, self.Nx // 25)
        
        im = ax.imshow(amplitude, extent=[-self.L, self.L, -self.L, self.L], 
                      origin='lower', cmap='viridis', alpha=0.8)
        
        quiver = ax.quiver(X_mesh[::stride, ::stride], Y_mesh[::stride, ::stride],
                          velocity_x.numpy()[::stride, ::stride], 
                          velocity_y.numpy()[::stride, ::stride],
                          scale=50, alpha=0.8)
        
        plt.colorbar(im, ax=ax, label='Amplitude')
        ax.set_title('Phase Flow')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        return fig
    
    def visualize_ensemble(self, num_samples: int = 9, state_type: str = None, 
                          rows: int = 3, cols: int = 3, figsize: tuple = (15, 15)):
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        
        if rows*cols == 1:
            axs = np.array([axs])
        axs = axs.flatten()
        
        for i in range(min(rows*cols, num_samples)):
            psi, description = self.generate_nlse_state(state_type)
            
            im = axs[i].imshow(torch.abs(psi), cmap='viridis', origin='lower',
                              extent=[-self.L, self.L, -self.L, self.L])
            
            axs[i].set_title(description)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
        if state_type:
            fig.suptitle(f"Ensemble of {state_type} states", fontsize=16)
        else:
            fig.suptitle("Ensemble of NLSE states", fontsize=16)
            
        
        return fig
    
    def run_analysis_suite(self, psi: torch.Tensor, title: str = None, save_path: str = None):
        figs = []
        
        figs.append(self.visualize_wavefunction(psi, title=title))
        figs.append(self.visualize_spectral_properties(psi))
        figs.append(self.visualize_phase_flow(psi))
        
        if save_path:
            for i, fig in enumerate(figs):
                fig.savefig(f"{save_path}_{i}.png", dpi=300)
                
        return figs


if __name__ == "__main__":
    Nx, Ny = 256, 256
    L = 10.0
    
    sampler = NLSESampler(Nx, Ny, L) 
    psi, description = sampler.generate_nlse_state(state_type="breather")
    
    fig = sampler.visualize_wavefunction(psi, title=description)
    plt.show()
    
    fig_spectral = sampler.visualize_spectral_properties(psi)
    plt.show()
    
    fig_flow = sampler.visualize_phase_flow(psi)
    plt.show()
    
    ensemble_fig = sampler.visualize_ensemble(state_type='vortex_cluster')
    plt.show()
   
    figs = sampler.run_analysis_suite(psi, title=description)
    plt.show()
