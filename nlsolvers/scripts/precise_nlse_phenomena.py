import numpy as np
from scipy import special
import torch
import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy import special

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
                          phase=0.0, velocity=(0.0, 0.0), sigma1=1.0, sigma2=-0.1,
                          kappa=1.0, apply_envelope=False, envelope_width=0.7):
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
       else:
           raise ValueError(f"Unknown system type: {system_type}")

       u = profile * np.exp(1j * total_phase)

       if apply_envelope:
           u = self._envelope(u, envelope_width)

       return u
    
   def multi_soliton_state(self, system_type, amplitude_range=(0.8, 1.2), 
                         width_range=(0.8, 1.2), position_variance=1.0, velocity_scale=1.0, 
                         phase_pattern='random', arrangement='random', separation=3.0, 
                         sigma1=1.0, sigma2=-0.1, kappa=1.0, apply_envelope=False, 
                         envelope_width=0.7):
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
               apply_envelope=False
           )
           u += component
       
       if apply_envelope:
           u = self._envelope(u, envelope_width)
           
       return u
    
   def spectral_method(self, spectrum_type='kolmogorov', n_modes=50, 
                     amplitude=1.0, k_min=0.5, k_max=5.0, spectrum_slope=-5/3,
                     randomize_phases=True, apply_envelope=True, envelope_width=0.7):
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
           
           
           if phenomenon_type == 'multi_soliton':
               sample = self.multi_soliton_state(system_type=system_type, **params)
           elif phenomenon_type == 'spectral':
               sample = self.spectral_method(**params)
           elif phenomenon_type == 'chaotic':
               sample = self.chaotic_field(**params)
           else:
               raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
           
           samples.append(sample)
       
       return samples

if __name__ == '__main__':
    n = 200
    L = 10.
    sampler = NLSEPhenomenonSampler(n, n, L) 
    types = ['multi_soliton', 'spectral', 'chaotic']
    for pi in types:
        ensemble = sampler.generate_ensemble(pi, n_samples=10)
        for e in ensemble:
            plt.imshow(np.abs(e))
            plt.show()

            plt.imshow(np.angle(e))
            plt.show()
