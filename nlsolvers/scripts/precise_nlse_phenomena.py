import numpy as np
from scipy import special
import torch
import matplotlib.pyplot as plt

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
                           phase=.2, velocity=(0.0, 0.0), sigma1=1.0, sigma2=-0.1,
                           kappa=1.0, apply_envelope=True, envelope_width=0.7, Lambda=0.1):
        x0, y0 = position
        vx, vy = velocity
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        momentum_phase = vx * (self.X - x0) + vy * (self.Y - y0)
        total_phase = momentum_phase + phase

        if system_type == 'cubic':
            profile = amplitude * self._sech(r_local/width)
        elif system_type == 'cubic_quintic':
            beta = -sigma2 * amplitude**2 / sigma1
            if beta > 0:
                profile = amplitude * self._sech(r_local/width) / np.sqrt(1 + beta * self._sech(r_local/width)**2)
            else:
                profile = amplitude * self._sech(r_local/width)
        elif system_type == 'saturable':
            sech_term = self._sech(r_local/width)
            denom = np.sqrt(1 + kappa * amplitude**2 * sech_term**2)
            profile = amplitude * sech_term / denom
        elif system_type == 'glasner_allen_flowers':
            sech_term = self._sech(np.sqrt(Lambda) * r_local)
            profile = amplitude * sech_term / np.sqrt(9 - 48 * Lambda * sech_term**2 + 31)
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
                          envelope_width=0.7, Lambda_range=(0.04, 0.14)):
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
        
        for i, ((x0, y0), phase) in enumerate(zip(positions, phases)):
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            Lambda = np.random.uniform(*Lambda_range)
            
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
                apply_envelope=False, Lambda=Lambda
            )
            u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    def spectral_method(self, spectrum_type='kolmogorov', n_modes=50, 
                      amplitude=1.0, k_min=0.5, k_max=5.0, spectrum_slope=-5/3,
                      randomize_phases=True, apply_envelope=False, envelope_width=0.7):
        spectrum = np.zeros((self.nx, self.ny), dtype=complex)
        
        if spectrum_type == 'kolmogorov':
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
                
                spot_width = np.random.uniform(0.1, 0.5)
                
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
        
        u = np.fft.ifft2(np.fft.ifftshift(spectrum))
        u = u / np.std(np.abs(u)) * amplitude
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
   
    def chaotic_field(self, amplitude=1.0, spectral_exponent=-1.5, coherent_structures=True,
                    n_structures=3, apply_envelope=True, envelope_width=0.7):
        u = self.spectral_method(
            spectrum_type='kolmogorov', amplitude=amplitude,
            spectrum_slope=spectral_exponent, apply_envelope=False
        )
        
        if coherent_structures:
            for _ in range(n_structures):
                structure_type = np.random.choice(['vortex', 'soliton'])
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
                
                weight = np.random.uniform(0.2, 0.5)
                u += weight * structure
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def vortex_solution(self, amplitude=1.0, position=(0, 0), charge=1, 
                       core_size=1.0, apply_envelope=True, envelope_width=0.7):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)
        
        profile = amplitude * np.tanh(r_local / core_size)
        phase = charge * theta_local
        
        u = profile * np.exp(1j * phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def vortex_lattice(self, amplitude=1.0, n_vortices=5, arrangement='random',
                      separation=2.0, charge_distribution='alternating', apply_envelope=True, 
                      envelope_width=0.8):
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
        else:  # random
            charges = np.random.choice([-1, 1], n_vortices)
        
        for (x0, y0), charge in zip(positions, charges):
            r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            theta_local = np.arctan2(self.Y - y0, self.X - x0)
            
            core_size = np.random.uniform(0.5, 1.5)
            u *= (r_local / core_size) * np.exp(1j * charge * theta_local) / np.sqrt(r_local**2 + core_size**2)
        
        u = amplitude * u / np.max(np.abs(u))
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def dark_soliton(self, amplitude=1.0, width=1.0, position=(0, 0), orientation=0,
                    velocity=(0, 0), apply_envelope=True, envelope_width=0.7):
        x0, y0 = position
        vx, vy = velocity
        
        rotated_X = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        
        profile = amplitude * np.tanh(rotated_X / width)
        momentum_phase = vx * (self.X - x0) + vy * (self.Y - y0)
        
        u = profile * np.exp(1j * momentum_phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def solitary_wave_with_ambient_field(self, system_type='cubic', solitary_amplitude=1.0, 
                                        solitary_width=1.0, solitary_position=(0, 0), 
                                        solitary_phase=0.0, solitary_velocity=(0.0, 0.0),
                                        ambient_amplitude=0.3, ambient_wavenumber=2.0,
                                        ambient_direction=0.0, ambient_phase=0.0,
                                        ambient_width=3.0, sigma1=1.0, sigma2=-0.1,
                                        kappa=1.0, Lambda=0.1, epsilon=0.025):
        soliton = self.fundamental_soliton(
            system_type, amplitude=solitary_amplitude/epsilon, width=solitary_width*epsilon,
            position=solitary_position, phase=solitary_phase, 
            velocity=solitary_velocity, sigma1=sigma1, sigma2=sigma2, 
            kappa=kappa, Lambda=Lambda
        )
        
        x0, y0 = solitary_position
        kx = ambient_wavenumber * np.cos(ambient_direction)
        ky = ambient_wavenumber * np.sin(ambient_direction)
        
        gaussian_envelope = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * ambient_width**2))
        ambient_wave = ambient_amplitude * np.exp(1j * (kx * self.X + ky * self.Y + ambient_phase)) * gaussian_envelope
        
        return soliton + ambient_wave
        
    def logarithmic_singularity(self, position=(0, 0), amplitude=1.0, m_lambda=0.5,
                               background_type='random', background_amplitude=0.3):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2 + 1e-10)  # avoid division by zero
        
        singular_part = amplitude * np.log(r_local)
        phase = np.random.uniform(0, 2*np.pi)
        
        if background_type == 'random':
            background = self.spectral_method(amplitude=background_amplitude, apply_envelope=False)
        elif background_type == 'gaussian':
            background = background_amplitude * np.exp(-r_local**2 / (2 * (self.L/4)**2))
        else:
            background = np.zeros_like(self.X)
        
        u = (singular_part * m_lambda + background) * np.exp(1j * phase)
        
        mask = (r_local < 0.5)
        smooth_factor = np.ones_like(self.X)
        smooth_factor[mask] = r_local[mask] / 0.5
        
        return u * smooth_factor
        
    def free_singularity_solution(self, position=(0, 0), amplitude=1.0, m_lambda=0.5,
                                 epsilon=0.01, chi=0.5, background_type='random', 
                                 background_amplitude=0.3):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2 + 1e-10)
        
        A = amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi))
        log_term = chi * np.log(r_local/epsilon) * m_lambda
        
        if background_type == 'random':
            background = self.spectral_method(amplitude=background_amplitude, apply_envelope=False)
        elif background_type == 'gaussian':
            background = background_amplitude * np.exp(-r_local**2 / (2 * (self.L/4)**2))
        else:
            background = np.zeros_like(self.X)
        
        u = A + A * log_term + background
        
        mask = (r_local < 0.5)
        smooth_factor = np.ones_like(self.X)
        smooth_factor[mask] = r_local[mask] / 0.5
        
        return u * smooth_factor
        
    def transparent_solitary_wave(self, amplitude=1.0, width=1.0, position=(0, 0),
                                phase=0.0, velocity=(0.0, 0.0), apply_envelope=True,
                                envelope_width=0.7):
        Lambda = 0.077  # The special value where m(λ) ≈ 0 according to the paper
        
        return self.fundamental_soliton(
            'glasner_allen_flowers', amplitude=amplitude, width=width,
            position=position, phase=phase, velocity=velocity,
            apply_envelope=apply_envelope, envelope_width=envelope_width,
            Lambda=Lambda
        )
        
    def colliding_solitary_waves(self, system_type='cubic', n_waves=2, angle_range=(0, 2*np.pi),
                               amplitude_range=(0.8, 1.2), width_range=(0.8, 1.2),
                               velocity_magnitude=2.0, separation=5.0, impact_parameter=0,
                               sigma1=1.0, sigma2=-0.1, kappa=1.0, Lambda_range=(0.04, 0.14)):
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
                sigma1=sigma1, sigma2=sigma2, kappa=kappa, Lambda=Lambda
            )
            
            u += soliton
            
        return u
        
    def oscillating_breather(self, amplitude=1.0, frequency=1.0, width=1.0, position=(0, 0),
                           phase=0.0, apply_envelope=True, envelope_width=0.7):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        
        envelope = np.exp(-r_local**2/(2*width**2))
        oscillator = np.cos(frequency * r_local + phase)
        
        u = amplitude * envelope * oscillator * np.exp(1j * phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
        
    def ring_soliton(self, amplitude=1.0, radius=3.0, width=0.5, position=(0, 0),
                   phase=0.0, apply_envelope=True, envelope_width=0.7):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        
        profile = amplitude * np.exp(-(r_local - radius)**2/(2*width**2))
        
        u = profile * np.exp(1j * phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width) 
        return u

    def multi_ring(self, amplitude_range=(0.8, 1.2), 
                          width_range=(0.8, 1.2), position_variance=1.0, velocity_scale=1.0, 
                          phase_pattern='vortex', arrangement='random', separation=5.0, 
                          sigma1=1.0, sigma2=-0.1, kappa=1.0, apply_envelope=False, 
                          envelope_width=0.7, Lambda_range=(0.04, 0.14)):
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
        
        for i, ((x0, y0), phase) in enumerate(zip(positions, phases)):
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            Lambda = np.random.uniform(*Lambda_range)
            
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
            radius = np.random.normal(0, self.L/10)
            phase = np.random.uniform(0, 2*np.pi)
            component = self.ring_soliton(amplitude, radius=radius, width=width, position=(x0, y0),
                   phase=phase, apply_envelope=False,)
            u += component
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            
        return u
    
    
    def free_singularity_adapted(self, position=None, amplitude=None, Lambda=None,
                              epsilon=None, background_type=None,
                              background_amplitude=None, phase=None):
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

        if position is None:
            x0 = np.random.uniform(-0.5*self.L, 0.5*self.L)
            y0 = np.random.uniform(-0.5*self.L, 0.5*self.L)
            position = (x0, y0)

        if amplitude is None:
            amplitude = np.random.uniform(0.5, 2.0)

        if Lambda is None:
            options = [0.04, 0.06, 0.077, 0.085, 0.092, 0.11, 0.13]
            weights = [0.15, 0.1, 0.3, 0.1, 0.2, 0.1, 0.05]
            Lambda = np.random.choice(options, p=weights)

        if epsilon is None:
            epsilon = np.random.uniform(0.005, 0.05)

        if background_type is None:
            background_type = np.random.choice(['random', 'gaussian', 'none'])

        if background_amplitude is None:
            background_amplitude = np.random.uniform(0.1, 0.5)

        if phase is None:
            phase = np.random.uniform(0, 2*np.pi)

        x0, y0 = position
        X_shifted = self.X - x0
        Y_shifted = self.Y - y0
        r_local = np.sqrt(X_shifted**2 + Y_shifted**2 + 1e-12)

        m_lambda = m_lambda_function(Lambda)
        a_lambda = a_lambda_function(Lambda)

        chi_epsilon = 1.0 / (max(-m_lambda * np.log(epsilon) + a_lambda, 0.01))

        A = amplitude * np.exp(1j * phase)

        if background_type == 'random':
            psi_R = self.spectral_method(amplitude=background_amplitude, apply_envelope=True)
        elif background_type == 'gaussian':
            psi_R = background_amplitude * np.exp(-r_local**2 / (2 * (self.L/4)**2)) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        else:
            psi_R = np.zeros_like(self.X, dtype=complex)

        G = create_greens_function(X_shifted, Y_shifted)

        singular_part = 2 * np.pi * A * m_lambda * chi_epsilon * G

        cutoff_radius = 3 * epsilon
        smooth_factor = np.ones_like(self.X)
        mask = (r_local < cutoff_radius)
        smooth_factor[mask] = (r_local[mask] / cutoff_radius)**2

        psi_R_at_singular_point = A
        psi_R = psi_R - psi_R[self.nx//2, self.ny//2] + psi_R_at_singular_point

        u = psi_R + singular_part * smooth_factor

        return u

    def logarithmic_singularity_adapted(self, position=None, amplitude=None, Lambda=None,
                              epsilon=None, background_type=None,
                              background_amplitude=None, phase=None):
        return self.free_singularity_adapted(position, amplitude, Lambda,
                                         epsilon, background_type,
                                         background_amplitude, phase)


        
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
            else:
                raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
            
            samples.append(sample)
        
        return samples

if __name__ == '__main__':
    n = 200
    L = 10.
    sampler = NLSEPhenomenonSampler(n, n, L)
    
    all_types = [
        'fundamental_soliton', 
        'multi_soliton', 
        'spectral',
        'chaotic',
        'vortex', 
        'vortex_lattice',
        'dark_soliton',
        'solitary_wave_with_ambient',
        'logarithmic_singularity',
        'free_singularity',
        'transparent_solitary_wave',
        'colliding_solitary_waves',
        'oscillating_breather',
        'ring_soliton'
    ]
    
    for phenomenon_type in ['logarithmic_singularity_adapted', 'free_singularity_adapted']:
        ensemble = sampler.generate_ensemble(phenomenon_type, n_samples=5)
        
        for i, sample in enumerate(ensemble):
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(np.abs(sample), cmap='viridis')
            plt.title(f"{phenomenon_type} - Amplitude {i+1}")
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(np.angle(sample), cmap='hsv')
            plt.title(f"{phenomenon_type} - Phase {i+1}")
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()
