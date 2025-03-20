import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

class RealWaveEquationSampler:
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
        
        self.tensor_X = torch.from_numpy(self.X)
        self.tensor_Y = torch.from_numpy(self.Y)
        self.tensor_r = torch.from_numpy(self.r)
        self.tensor_theta = torch.from_numpy(self.theta)
    
    def _envelope(self, field, width_factor=0.7):
        envelope_width = width_factor * self.L
        envelope = np.exp(-self.r**2/(2*envelope_width**2))
        return field * envelope
    
    def _envelope_tensor(self, field, width_factor=0.7):
        envelope_width = width_factor * self.L
        envelope = torch.exp(-self.tensor_r**2/(2*envelope_width**2))
        return field * envelope
    
    def _normalize_amplitude(self, field, target_amplitude):
        current_amplitude = np.max(np.abs(field))
        if current_amplitude > 0:
            return field * (target_amplitude / current_amplitude)
        return field
    
    def _tensor_to_numpy(self, tensor_field):
        if isinstance(tensor_field, torch.Tensor):
            return tensor_field.numpy()
        return tensor_field
    
    def kink_solution(self, system_type='sine_gordon', amplitude=1.0, width=1.0, 
                     position=(0, 0), orientation=0.0, velocity=(0.0, 0.0),
                     apply_envelope=True, envelope_width=0.7, order=1,
                     kink_type='standard', multi_kink=False, stacking=1,
                     velocity_type='fitting'):
        x0, y0 = position
        
        if velocity_type == 'zero':
            vx, vy = 0.0, 0.0
        elif velocity_type == 'fitting':
            vx, vy = velocity
        else:
            vx, vy = velocity
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        if multi_kink:
            spacing = width * 2 * stacking
            kink_positions = np.arange(-stacking, stacking+1) * spacing
            u = np.zeros_like(X_rot)
            v = np.zeros_like(X_rot)
            
            for i, pos in enumerate(kink_positions):
                if i % 2 == 0:
                    sign = 1
                else:
                    sign = -1
                    
                if system_type == 'sine_gordon':
                    u_component = sign * 4 * np.arctan(np.exp((X_rot - pos) / width))
                    if velocity_type == 'fitting':
                        v_component = sign * vx * 4 / (width * (np.cosh((X_rot - pos) / width)**2))
                    elif velocity_type == 'zero':
                        v_component = np.zeros_like(u_component)
                    else:
                        v_component = sign * vx * 4 / (width * (np.cosh((X_rot - pos) / width)**2))
                elif system_type == 'phi4':
                    u_component = sign * amplitude * np.tanh((X_rot - pos) / width)
                    if velocity_type == 'fitting':
                        v_component = sign * vx * amplitude / (width * (np.cosh((X_rot - pos) / width)**2))
                    elif velocity_type == 'zero':
                        v_component = np.zeros_like(u_component)
                    else:
                        v_component = sign * vx * amplitude / (width * (np.cosh((X_rot - pos) / width)**2))
                elif system_type == 'double_sine_gordon':
                    lambda_param = 0.3
                    k = 1 / np.sqrt(1 + lambda_param)
                    u_component = sign * 4 * np.arctan(np.sqrt((1+lambda_param)/lambda_param) * np.tanh(np.sqrt(lambda_param) * (X_rot - pos) / (2*width)))
                    if velocity_type == 'fitting':
                        v_component = sign * vx * 4 * np.sqrt((1+lambda_param)/lambda_param) * np.sqrt(lambda_param) / (2*width) * (1 - np.tanh(np.sqrt(lambda_param) * (X_rot - pos) / (2*width))**2)
                    elif velocity_type == 'zero':
                        v_component = np.zeros_like(u_component)
                    else:
                        v_component = sign * vx * 4 * np.sqrt((1+lambda_param)/lambda_param) * np.sqrt(lambda_param) / (2*width) * (1 - np.tanh(np.sqrt(lambda_param) * (X_rot - pos) / (2*width))**2)
                else:
                    u_component = sign * 4 * np.arctan(np.exp((X_rot - pos) / width))
                    if velocity_type == 'fitting':
                        v_component = sign * vx * 4 / (width * (np.cosh((X_rot - pos) / width)**2))
                    elif velocity_type == 'zero':
                        v_component = np.zeros_like(u_component)
                    else:
                        v_component = sign * vx * 4 / (width * (np.cosh((X_rot - pos) / width)**2))
                
                u += u_component
                v += v_component
        else:
            if system_type == 'sine_gordon':
                if kink_type == 'standard':
                    u = 4 * np.arctan(np.exp(X_rot / width))
                    if velocity_type == 'fitting':
                        v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
                elif kink_type == 'anti':
                    u = -4 * np.arctan(np.exp(X_rot / width))
                    if velocity_type == 'fitting':
                        v = -vx * 4 / (width * (np.cosh(X_rot / width)**2))
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = -vx * 4 / (width * (np.cosh(X_rot / width)**2))
                elif kink_type == 'double':
                    u = 4 * np.arctan(np.exp(X_rot / width)) + 4 * np.arctan(np.exp((X_rot - 2*width) / width))
                    if velocity_type == 'fitting':
                        v = vx * 4 / (width * (np.cosh(X_rot / width)**2)) + vx * 4 / (width * (np.cosh((X_rot - 2*width) / width)**2))
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = vx * 4 / (width * (np.cosh(X_rot / width)**2)) + vx * 4 / (width * (np.cosh((X_rot - 2*width) / width)**2))
            elif system_type == 'phi4':
                if kink_type == 'standard':
                    u = amplitude * np.tanh(X_rot / width)
                    if velocity_type == 'fitting':
                        v = vx * amplitude / (width * (np.cosh(X_rot / width)**2))
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = vx * amplitude / (width * (np.cosh(X_rot / width)**2))
                elif kink_type == 'anti':
                    u = -amplitude * np.tanh(X_rot / width)
                    if velocity_type == 'fitting':
                        v = -vx * amplitude / (width * (np.cosh(X_rot / width)**2))
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = -vx * amplitude / (width * (np.cosh(X_rot / width)**2))
                elif kink_type == 'double':
                    u = amplitude * np.tanh(X_rot / width) - amplitude * np.tanh((X_rot - 4*width) / width)
                    if velocity_type == 'fitting':
                        v = vx * amplitude / (width * (np.cosh(X_rot / width)**2)) - vx * amplitude / (width * (np.cosh((X_rot - 4*width) / width)**2))
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = vx * amplitude / (width * (np.cosh(X_rot / width)**2)) - vx * amplitude / (width * (np.cosh((X_rot - 4*width) / width)**2))
            elif system_type == 'double_sine_gordon':
                lambda_param = 0.3
                k = 1 / np.sqrt(1 + lambda_param)
                if kink_type == 'standard':
                    u = 4 * np.arctan(np.sqrt((1+lambda_param)/lambda_param) * np.tanh(np.sqrt(lambda_param) * X_rot / (2*width)))
                    if velocity_type == 'fitting':
                        v = vx * 4 * np.sqrt((1+lambda_param)/lambda_param) * np.sqrt(lambda_param) / (2*width) * (1 - np.tanh(np.sqrt(lambda_param) * X_rot / (2*width))**2)
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = vx * 4 * np.sqrt((1+lambda_param)/lambda_param) * np.sqrt(lambda_param) / (2*width) * (1 - np.tanh(np.sqrt(lambda_param) * X_rot / (2*width))**2)
                elif kink_type == 'anti':
                    u = -4 * np.arctan(np.sqrt((1+lambda_param)/lambda_param) * np.tanh(np.sqrt(lambda_param) * X_rot / (2*width)))
                    if velocity_type == 'fitting':
                        v = -vx * 4 * np.sqrt((1+lambda_param)/lambda_param) * np.sqrt(lambda_param) / (2*width) * (1 - np.tanh(np.sqrt(lambda_param) * X_rot / (2*width))**2)
                    elif velocity_type == 'zero':
                        v = np.zeros_like(u)
                    else:
                        v = -vx * 4 * np.sqrt((1+lambda_param)/lambda_param) * np.sqrt(lambda_param) / (2*width) * (1 - np.tanh(np.sqrt(lambda_param) * X_rot / (2*width))**2)
            else:
                u = 4 * np.arctan(np.exp(X_rot / width))
                if velocity_type == 'fitting':
                    v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
                elif velocity_type == 'zero':
                    v = np.zeros_like(u)
                else:
                    v = vx * 4 / (width * (np.cosh(X_rot / width)**2))
                    
        if velocity_type == 'grf':
            v = self.anisotropic_grf(length_scale=width*2.0, amplitude=np.max(np.abs(u))*0.2)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def breather_solution(self, system_type='sine_gordon', amplitude=0.5, frequency=0.9,
                         width=1.0, position=(0, 0), phase=0.0, orientation=0.0,
                         apply_envelope=True, envelope_width=0.7, breather_type='standard',
                         time_param=0.0, velocity_type='fitting'):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        if amplitude >= 1.0 and system_type == 'sine_gordon':
            amplitude = 0.999
            
        if system_type == 'sine_gordon':
            omega = np.sqrt(1 - amplitude**2)
            
            if breather_type == 'standard':
                xi = X_rot / width
                tau = time_param
                
                u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) / 
                                (omega * np.cosh(amplitude * xi)))
                
                if velocity_type == 'fitting':
                    v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                        omega * np.cosh(amplitude * xi) * 
                        (1 + (amplitude**2/omega**2) * np.sin(omega * tau + phase)**2)
                    )
                elif velocity_type == 'zero':
                    v = np.zeros_like(u)
                elif velocity_type == 'grf':
                    v = self.anisotropic_grf(length_scale=width*2.0, amplitude=np.max(np.abs(u))*0.2)
                else:
                    v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                        omega * np.cosh(amplitude * xi) * 
                        (1 + (amplitude**2/omega**2) * np.sin(omega * tau + phase)**2)
                    )
            elif breather_type == 'radial':
                r_local = np.sqrt(X_rot**2 + Y_rot**2)
                xi = r_local / width
                tau = time_param
                
                u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) / 
                                (omega * np.cosh(amplitude * xi)))
                
                if velocity_type == 'fitting':
                    v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                        omega * np.cosh(amplitude * xi) * 
                        (1 + (amplitude**2/omega**2) * np.sin(omega * tau + phase)**2)
                    )
                elif velocity_type == 'zero':
                    v = np.zeros_like(u)
                elif velocity_type == 'grf':
                    v = self.anisotropic_grf(length_scale=width*2.0, amplitude=np.max(np.abs(u))*0.2)
                else:
                    v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                        omega * np.cosh(amplitude * xi) * 
                        (1 + (amplitude**2/omega**2) * np.sin(omega * tau + phase)**2)
                    )
        elif system_type == 'phi4':
            xi = X_rot / width
            tau = time_param
            epsilon = amplitude
            
            if breather_type == 'standard':
                u = amplitude * np.sqrt(2) * np.tanh(xi) / np.cosh(epsilon * tau)
                if velocity_type == 'fitting':
                    v = amplitude * np.sqrt(2) * epsilon * np.tanh(xi) * np.sinh(epsilon * tau) / np.cosh(epsilon * tau)**2
                elif velocity_type == 'zero':
                    v = np.zeros_like(u)
                elif velocity_type == 'grf':
                    v = self.anisotropic_grf(length_scale=width*2.0, amplitude=np.max(np.abs(u))*0.2)
                else:
                    v = amplitude * np.sqrt(2) * epsilon * np.tanh(xi) * np.sinh(epsilon * tau) / np.cosh(epsilon * tau)**2
            elif breather_type == 'radial':
                r_local = np.sqrt(X_rot**2 + Y_rot**2)
                xi = r_local / width
                u = amplitude * np.sqrt(2) * np.tanh(xi) / np.cosh(epsilon * tau)
                if velocity_type == 'fitting':
                    v = amplitude * np.sqrt(2) * epsilon * np.tanh(xi) * np.sinh(epsilon * tau) / np.cosh(epsilon * tau)**2
                elif velocity_type == 'zero':
                    v = np.zeros_like(u)
                elif velocity_type == 'grf':
                    v = self.anisotropic_grf(length_scale=width*2.0, amplitude=np.max(np.abs(u))*0.2)
                else:
                    v = amplitude * np.sqrt(2) * epsilon * np.tanh(xi) * np.sinh(epsilon * tau) / np.cosh(epsilon * tau)**2
        else:
            omega = np.sqrt(1 - amplitude**2)
            xi = X_rot / width
            tau = time_param
            
            u = 4 * np.arctan(amplitude * np.sin(omega * tau + phase) / 
                            (omega * np.cosh(amplitude * xi)))
            
            if velocity_type == 'fitting':
                v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                    omega * np.cosh(amplitude * xi) * 
                    (1 + (amplitude**2/omega**2) * np.sin(omega * tau + phase)**2)
                )
            elif velocity_type == 'zero':
                v = np.zeros_like(u)
            elif velocity_type == 'grf':
                v = self.anisotropic_grf(length_scale=width*2.0, amplitude=np.max(np.abs(u))*0.2)
            else:
                v = 4 * amplitude * omega * np.cos(omega * tau + phase) / (
                    omega * np.cosh(amplitude * xi) * 
                    (1 + (amplitude**2/omega**2) * np.sin(omega * tau + phase)**2)
                )
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def oscillon_solution(self, system_type='phi4', amplitude=0.5, frequency=0.9,
                         width=1.0, position=(0, 0), phase=0.0, orientation=0.0,
                         apply_envelope=True, envelope_width=0.7, profile='gaussian',
                         time_param=0.0):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        r_local = np.sqrt(X_rot**2 + Y_rot**2)
        
        if profile == 'gaussian':
            spatial_profile = np.exp(-r_local**2 / (2 * width**2))
        elif profile == 'sech':
            spatial_profile = 1 / np.cosh(r_local / width)
        elif profile == 'polynomial':
            spatial_profile = 1 / (1 + (r_local / width)**4)
        else:
            spatial_profile = np.exp(-r_local**2 / (2 * width**2))
        
        if system_type == 'phi4':
            u = amplitude * spatial_profile * np.cos(frequency * time_param + phase)
            v = -amplitude * spatial_profile * frequency * np.sin(frequency * time_param + phase)
        elif system_type == 'sine_gordon':
            u = amplitude * spatial_profile * np.cos(frequency * time_param + phase)
            v = -amplitude * spatial_profile * frequency * np.sin(frequency * time_param + phase)
        else:
            u = amplitude * spatial_profile * np.cos(frequency * time_param + phase)
            v = -amplitude * spatial_profile * frequency * np.sin(frequency * time_param + phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def multi_oscillon_state(self, system_type='phi4', n_oscillons=5, 
                            amplitude_range=(0.3, 0.7), width_range=(0.5, 2.0),
                            frequency_range=(0.8, 0.99), position_variance=1.0, 
                            arrangement='random', separation=3.0, profile='gaussian',
                            apply_envelope=True, envelope_width=0.7, interaction_strength=0.7,
                            time_param=0.0):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        if arrangement == 'linear':
            base_positions = [(i - (n_oscillons-1)/2) * separation for i in range(n_oscillons)]
            positions = [(pos, 0) for pos in base_positions]
        elif arrangement == 'circular':
            positions = []
            for i in range(n_oscillons):
                angle = 2 * np.pi * i / n_oscillons
                x = separation * np.cos(angle)
                y = separation * np.sin(angle)
                positions.append((x, y))
        elif arrangement == 'random':
            positions = []
            for _ in range(n_oscillons):
                x = np.random.normal(0, position_variance * self.L/4)
                y = np.random.normal(0, position_variance * self.L/4)
                positions.append((x, y))
        elif arrangement == 'lattice':
            side = int(np.ceil(np.sqrt(n_oscillons)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_oscillons:
                        x = (i - (side-1)/2) * separation
                        y = (j - (side-1)/2) * separation
                        positions.append((x, y))
        
        for i, (x0, y0) in enumerate(positions):
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            frequency = np.random.uniform(*frequency_range)
            phase = np.random.uniform(0, 2*np.pi)
            
            u_comp, v_comp = self.oscillon_solution(
                system_type=system_type,
                amplitude=amplitude,
                frequency=frequency,
                width=width,
                position=(x0, y0),
                phase=phase,
                apply_envelope=False,
                profile=profile,
                time_param=time_param
            )
            
            if i == 0:
                u = u_comp
                v = v_comp
            else:
                u = u + interaction_strength * u_comp
                v = v + interaction_strength * v_comp
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def ring_soliton(self, system_type='sine_gordon', amplitude=1.0, radius=2.0, 
                     width=0.5, position=(0, 0), velocity=0.0, apply_envelope=True, 
                     envelope_width=0.7, ring_type='expanding', modulation_strength=0.0, 
                     modulation_mode=2, time_param=0.0):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)
        
        if ring_type == 'expanding':
            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / (width * (np.cosh((radius - r_local) / width)**2))
            elif system_type == 'phi4':
                u = amplitude * np.tanh((radius - r_local) / width)
                v = -velocity * amplitude / (width * (np.cosh((radius - r_local) / width)**2))
            else:
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / (width * (np.cosh((radius - r_local) / width)**2))
        elif ring_type == 'kink_antikink':
            inner_radius = radius - width
            outer_radius = radius + width
            
            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp((inner_radius - r_local) / (width/2))) - 4 * np.arctan(np.exp((outer_radius - r_local) / (width/2)))
                v = -velocity * 4 / (width/2 * (np.cosh((inner_radius - r_local) / (width/2))**2)) + velocity * 4 / (width/2 * (np.cosh((outer_radius - r_local) / (width/2))**2))
            elif system_type == 'phi4':
                u = amplitude * np.tanh((inner_radius - r_local) / (width/2)) - amplitude * np.tanh((outer_radius - r_local) / (width/2))
                v = -velocity * amplitude / (width/2 * (np.cosh((inner_radius - r_local) / (width/2))**2)) + velocity * amplitude / (width/2 * (np.cosh((outer_radius - r_local) / (width/2))**2))
            else:
                u = 4 * np.arctan(np.exp((inner_radius - r_local) / (width/2))) - 4 * np.arctan(np.exp((outer_radius - r_local) / (width/2)))
                v = -velocity * 4 / (width/2 * (np.cosh((inner_radius - r_local) / (width/2))**2)) + velocity * 4 / (width/2 * (np.cosh((outer_radius - r_local) / (width/2))**2))
        else:
            if system_type == 'sine_gordon':
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / (width * (np.cosh((radius - r_local) / width)**2))
            elif system_type == 'phi4':
                u = amplitude * np.tanh((radius - r_local) / width)
                v = -velocity * amplitude / (width * (np.cosh((radius - r_local) / width)**2))
            else:
                u = 4 * np.arctan(np.exp((radius - r_local) / width))
                v = -velocity * 4 / (width * (np.cosh((radius - r_local) / width)**2))
        
        if modulation_strength > 0:
            modulation = 1 + modulation_strength * np.cos(modulation_mode * theta_local)
            u *= modulation
            v *= modulation
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def multi_ring_state(self, system_type='sine_gordon', n_rings=3, 
                        radius_range=(1.0, 5.0), width_range=(0.3, 0.8),
                        position_variance=0.5, arrangement='concentric',
                        separation=2.0, apply_envelope=True, envelope_width=0.7, 
                        interaction_strength=0.7, modulation_strength=0.2,
                        modulation_mode_range=(1, 4)):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        if arrangement == 'concentric':
            positions = [(0, 0)] * n_rings
        elif arrangement == 'random':
            positions = []
            for _ in range(n_rings):
                x = np.random.normal(0, position_variance * self.L/4)
                y = np.random.normal(0, position_variance * self.L/4)
                positions.append((x, y))
        elif arrangement == 'circular':
            positions = []
            for i in range(n_rings):
                angle = 2 * np.pi * i / n_rings
                x = separation * np.cos(angle)
                y = separation * np.sin(angle)
                positions.append((x, y))
        
        for i, (x0, y0) in enumerate(positions):
            if arrangement == 'concentric':
                radius = radius_range[0] + (radius_range[1] - radius_range[0]) * i / (n_rings - 1) if n_rings > 1 else radius_range[0]
            else:
                radius = np.random.uniform(*radius_range)
            
            width = np.random.uniform(*width_range)
            velocity = np.random.uniform(-0.2, 0.2)
            ring_type = np.random.choice(['expanding', 'kink_antikink'])
            
            if modulation_strength > 0:
                mod_mode = np.random.randint(*modulation_mode_range)
            else:
                mod_mode = 0
            
            u_comp, v_comp = self.ring_soliton(
                system_type=system_type,
                amplitude=1.0,
                radius=radius,
                width=width,
                position=(x0, y0),
                velocity=velocity,
                apply_envelope=False,
                ring_type=ring_type,
                modulation_strength=modulation_strength,
                modulation_mode=mod_mode
            )
            
            if i == 0:
                u = u_comp
                v = v_comp
            else:
                u = u + interaction_strength * u_comp
                v = v + interaction_strength * v_comp
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def skyrmion_solution(self, system_type='sine_gordon', amplitude=1.0, 
                         radius=1.0, position=(0, 0), charge=1, apply_envelope=True,
                         envelope_width=0.7, profile='standard'):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)
        
        if profile == 'standard':
            phi = 2 * np.arctan(r_local / radius)
        elif profile == 'compact':
            phi = np.pi * (1 - np.exp(-(r_local / radius)**2))
        elif profile == 'exponential':
            phi = np.pi * (1 - np.exp(-r_local / radius))
        else:
            phi = 2 * np.arctan(r_local / radius)
        
        u = amplitude * np.sin(phi) * np.cos(charge * theta_local)
        v = amplitude * np.sin(phi) * np.sin(charge * theta_local)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def skyrmion_lattice(self, system_type='sine_gordon', n_skyrmions=5,
                        radius_range=(0.5, 1.5), amplitude=1.0,
                        arrangement='triangular', separation=3.0,
                        charge_distribution='alternating', apply_envelope=True,
                        envelope_width=0.7):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        positions = []
        if arrangement == 'triangular':
            rows = int(np.ceil(np.sqrt(n_skyrmions * 2 / np.sqrt(3))))
            for i in range(rows):
                offset = (i % 2) * 0.5 * separation
                for j in range(int(np.ceil(n_skyrmions / rows))):
                    if len(positions) < n_skyrmions:
                        x = (j - int(np.ceil(n_skyrmions / rows) - 1)/2) * separation + offset
                        y = (i - (rows-1)/2) * separation * np.sqrt(3)/2
                        positions.append((x, y))
        elif arrangement == 'square':
            side = int(np.ceil(np.sqrt(n_skyrmions)))
            for i in range(side):
                for j in range(side):
                    if len(positions) < n_skyrmions:
                        x = (i - (side-1)/2) * separation
                        y = (j - (side-1)/2) * separation
                        positions.append((x, y))
        elif arrangement == 'random':
            for _ in range(n_skyrmions):
                x = np.random.uniform(-self.L/3, self.L/3)
                y = np.random.uniform(-self.L/3, self.L/3)
                positions.append((x, y))
        
        charges = []
        if charge_distribution == 'alternating':
            charges = [(-1)**i for i in range(n_skyrmions)]
        elif charge_distribution == 'random':
            charges = [np.random.choice([-1, 1]) for _ in range(n_skyrmions)]
        elif charge_distribution == 'same':
            charges = [1] * n_skyrmions
        
        for i, ((x0, y0), charge) in enumerate(zip(positions, charges)):
            radius = np.random.uniform(*radius_range)
            profile = np.random.choice(['standard', 'compact', 'exponential'])
            
            u_comp, v_comp = self.skyrmion_solution(
                system_type=system_type,
                amplitude=amplitude,
                radius=radius,
                position=(x0, y0),
                charge=charge,
                apply_envelope=False,
                profile=profile
            )
            
            u += u_comp
            v += v_comp
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def anisotropic_grf(self, length_scale=1.0, anisotropy_ratio=2.0, 
                        theta=30.0, power=2.0, amplitude=1.0):
        theta_rad = np.deg2rad(theta)
        ell_x = length_scale * np.sqrt(anisotropy_ratio)
        ell_y = length_scale / np.sqrt(anisotropy_ratio)
        
        KX_rot = self.KX*np.cos(theta_rad) - self.KY*np.sin(theta_rad)
        KY_rot = self.KX*np.sin(theta_rad) + self.KY*np.cos(theta_rad)
        
        spectrum = np.exp(-((KX_rot/ell_x)**2 + (KY_rot/ell_y)**2)**(power/2))
        noise = np.random.randn(self.nx, self.ny) + 1j*np.random.randn(self.nx, self.ny)
        field = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum)).real
        
        field = field / np.std(field) * amplitude
        
        return field
    
    def wavelet_superposition(self, n_wavelets=20, scale_range=(0.1, 2.0), kappa=0.5, freq_range=(0.5, 3.0), amplitude=1.0):
        v0 = np.zeros((self.nx, self.ny))
        
        for _ in range(n_wavelets):
            scale = scale_range[0] + (scale_range[1]-scale_range[0])*np.random.rand()
            theta = 2*np.pi*np.random.rand()
            x0 = self.L*(np.random.rand()-0.5)
            y0 = self.L*(np.random.rand()-0.5)
            k0 = (freq_range[0] + (freq_range[1]-freq_range[0])*np.random.rand()) * (2*np.pi/(scale*self.L))
            
            envelope = np.exp(-((self.X-x0)**2 + (self.Y-y0)**2)/(2*(scale*self.L)**2))
            
            wavelet_type = np.random.choice(['cosine', 'gaussian_deriv', 'morlet'])
            
            if wavelet_type == 'cosine':
                carrier = np.cos(k0*((self.X-x0)*np.cos(theta) + (self.Y-y0)*np.sin(theta)))
            elif wavelet_type == 'gaussian_deriv':
                z = ((self.X-x0)*np.cos(theta) + (self.Y-y0)*np.sin(theta)) / (scale*self.L)
                carrier = -z * np.exp(-z**2/2)
            else:
                z = ((self.X-x0)*np.cos(theta) + (self.Y-y0)*np.sin(theta))
                carrier = np.cos(k0*z) * np.exp(-(z/(scale*self.L))**2/2)
            
            amp = (1 - kappa) + kappa*np.random.rand()
            v0 += amp * envelope * carrier

        return v0 / np.max(np.abs(v0)) * amplitude
    
    def spiral_wave_field(self, num_arms=2, decay_rate=0.5, amplitude=1.0, position=None, 
                          phase=0.0, k_factor=None, apply_envelope=True, envelope_width=0.7):
        if position is None:
            x0 = self.L * (2*np.random.rand() - 1)
            y0 = self.L * (2*np.random.rand() - 1)
        else:
            x0, y0 = position
            
        if k_factor is None:
            k = 1.0 + 2.0 * np.random.rand()
        else:
            k = k_factor
        
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = np.arctan2(self.Y - y0, self.X - x0)
        
        spiral_phase = theta + k * r / self.L + phase
        
        pattern = np.cos(num_arms * spiral_phase)
        
        decay = np.exp(-decay_rate * r / self.L)
        
        u = amplitude * pattern * decay
        v = amplitude * 0.1 * self.anisotropic_grf(length_scale=self.L/5)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def multi_spiral_state(self, n_spirals=3, amplitude_range=(0.5, 1.5), 
                          num_arms_range=(1, 4), decay_rate_range=(0.3, 0.7),
                          position_variance=1.0, apply_envelope=True, envelope_width=0.7,
                          interaction_strength=0.7):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        for i in range(n_spirals):
            amplitude = np.random.uniform(*amplitude_range)
            num_arms = np.random.randint(*num_arms_range)
            decay_rate = np.random.uniform(*decay_rate_range)
            
            x0 = np.random.normal(0, position_variance * self.L/4)
            y0 = np.random.normal(0, position_variance * self.L/4)
            phase = np.random.uniform(0, 2*np.pi)
            k_factor = 1.0 + 2.0 * np.random.rand()
            
            u_comp, v_comp = self.spiral_wave_field(
                num_arms=num_arms, 
                decay_rate=decay_rate,
                amplitude=amplitude,
                position=(x0, y0),
                phase=phase,
                k_factor=k_factor,
                apply_envelope=False
            )
            
            if i == 0:
                u = u_comp
                v = v_comp
            else:
                u = u + interaction_strength * u_comp
                v = v + interaction_strength * v_comp
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def rogue_wave(self, system_type='sine_gordon', amplitude=2.0, background_level=0.2,
                 width=1.0, position=None, apply_envelope=True, envelope_width=0.8):
        if position is None:
            x0 = self.L * (2*np.random.rand() - 1) * 0.5
            y0 = self.L * (2*np.random.rand() - 1) * 0.5
        else:
            x0, y0 = position
            
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        
        envelope = np.exp(-r_local**2/(2*width**2))
        
        if system_type == 'sine_gordon':
            peak = 4 * np.arctan(amplitude * envelope)
        elif system_type == 'phi4':
            peak = amplitude * np.tanh(2 * envelope)
        else:
            peak = amplitude * envelope
        
        background = background_level * self.wavelet_superposition(
            n_wavelets=10, scale_range=(self.L/10, self.L/3), amplitude=1.0)
        
        u = peak + background
        v = 0.1 * self.anisotropic_grf(length_scale=width)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def multi_rogue_state(self, system_type='sine_gordon', n_rogues=3, 
                         amplitude_range=(1.5, 3.0), width_range=(0.5, 2.0),
                         background_level=0.2, position_variance=0.5,
                         apply_envelope=True, envelope_width=0.8):
        positions = []
        for _ in range(n_rogues):
            x = np.random.normal(0, position_variance * self.L/4)
            y = np.random.normal(0, position_variance * self.L/4)
            positions.append((x, y))
        
        background = background_level * self.wavelet_superposition(
            n_wavelets=15, scale_range=(self.L/10, self.L/3))
        
        u = background.copy()
        v = 0.1 * self.anisotropic_grf(length_scale=self.L/5)
        
        for x0, y0 in positions:
            amplitude = np.random.uniform(*amplitude_range)
            width = np.random.uniform(*width_range)
            
            r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            envelope = np.exp(-r_local**2/(2*width**2))
            
            if system_type == 'sine_gordon':
                peak = 4 * np.arctan(amplitude * envelope)
            elif system_type == 'phi4':
                peak = amplitude * np.tanh(2 * envelope)
            else:
                peak = amplitude * envelope
                
            u += peak
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def fractal_kink(self, system_type='sine_gordon', levels=3, base_width=1.0,
                   scale_factor=2.0, amplitude=1.0, position=(0, 0), 
                   orientation=0.0, apply_envelope=True, envelope_width=0.7):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        u = np.zeros_like(X_rot)
        v = np.zeros_like(X_rot)
        
        for i in range(levels):
            width = base_width / (scale_factor**i)
            position_scale = base_width * (1 - 1/(2**i)) if i > 0 else 0
            
            if system_type == 'sine_gordon':
                u_level = 4 * np.arctan(np.exp((X_rot - position_scale) / width))
                v_level = np.zeros_like(u_level)
            elif system_type == 'phi4':
                u_level = amplitude * np.tanh((X_rot - position_scale) / width)
                v_level = np.zeros_like(u_level)
            else:
                u_level = 4 * np.arctan(np.exp((X_rot - position_scale) / width))
                v_level = np.zeros_like(u_level)
                
            level_weight = 1.0 / (2**i)
            u += level_weight * u_level
            v += level_weight * v_level
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def domain_wall_network(self, system_type='phi4', n_walls=6, width_range=(0.5, 2.0),
                          apply_envelope=True, envelope_width=0.7, orientation_variance=0.5,
                          interaction_strength=0.7):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        for _ in range(n_walls):
            x0 = self.L * (2*np.random.rand() - 1) * 0.7
            y0 = self.L * (2*np.random.rand() - 1) * 0.7
            orientation = np.pi * np.random.randn() * orientation_variance
            width = np.random.uniform(*width_range)
            
            X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
            
            if system_type == 'sine_gordon':
                u_wall = 4 * np.arctan(np.exp(X_rot / width))
                v_wall = np.zeros_like(u_wall)
            elif system_type == 'phi4':
                u_wall = np.tanh(X_rot / width)
                v_wall = np.zeros_like(u_wall)
            else:
                u_wall = 4 * np.arctan(np.exp(X_rot / width))
                v_wall = np.zeros_like(u_wall)
            
            if np.random.rand() > 0.5:
                u_wall = -u_wall
                
            u += interaction_strength * u_wall
            v += interaction_strength * v_wall
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def soliton_gas(self, system_type='sine_gordon', n_solitons=15, width_range=(0.5, 1.5),
                  velocity_scale=0.3, apply_envelope=True, envelope_width=0.7,
                  interaction_strength=0.7):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        for _ in range(n_solitons):
            x0 = self.L * (2*np.random.rand() - 1) * 0.7
            y0 = self.L * (2*np.random.rand() - 1) * 0.7
            orientation = 2 * np.pi * np.random.rand()
            width = np.random.uniform(*width_range)
            vx = np.random.normal(0, velocity_scale)
            vy = np.random.normal(0, velocity_scale)
            kink_type = np.random.choice(['standard', 'anti'])
            
            u_sol, v_sol = self.kink_solution(
                system_type=system_type,
                width=width,
                position=(x0, y0),
                orientation=orientation,
                velocity=(vx, vy),
                apply_envelope=False,
                kink_type=kink_type
            )
            
            u += interaction_strength * u_sol
            v += interaction_strength * v_sol
            
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def q_ball_solution(self, system_type='phi4', amplitude=1.0, radius=1.0,
                      position=(0, 0), phase=0.0, frequency=0.8, charge=1,
                      apply_envelope=True, envelope_width=0.7, time_param=0.0):
        x0, y0 = position
        r_local = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta_local = np.arctan2(self.Y - y0, self.X - x0)
        
        profile = amplitude * np.exp(-r_local**2 / (2 * radius**2))
        
        u = profile * np.cos(charge * theta_local + frequency * time_param + phase)
        v = -profile * frequency * np.sin(charge * theta_local + frequency * time_param + phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def multi_q_ball_state(self, system_type='phi4', n_qballs=3,
                         amplitude_range=(0.5, 1.5), radius_range=(0.5, 2.0),
                         frequency_range=(0.6, 0.9), position_variance=1.0,
                         apply_envelope=True, envelope_width=0.7,
                         interaction_strength=0.7, time_param=0.0):
        u = np.zeros_like(self.X)
        v = np.zeros_like(self.X)
        
        for i in range(n_qballs):
            amplitude = np.random.uniform(*amplitude_range)
            radius = np.random.uniform(*radius_range)
            frequency = np.random.uniform(*frequency_range)
            phase = np.random.uniform(0, 2*np.pi)
            charge = np.random.choice([-1, 1])
            
            x0 = np.random.normal(0, position_variance * self.L/4)
            y0 = np.random.normal(0, position_variance * self.L/4)
            
            u_q, v_q = self.q_ball_solution(
                system_type=system_type,
                amplitude=amplitude,
                radius=radius,
                position=(x0, y0),
                phase=phase,
                frequency=frequency,
                charge=charge,
                apply_envelope=False,
                time_param=time_param
            )
            
            if i == 0:
                u = u_q
                v = v_q
            else:
                u = u + interaction_strength * u_q
                v = v + interaction_strength * v_q
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def vibrational_kink_mode(self, system_type='sine_gordon', amplitude=1.0, 
                            width=1.0, position=(0, 0), orientation=0.0, 
                            mode_amplitude=0.3, mode_frequency=0.5, phase=0.0,
                            apply_envelope=True, envelope_width=0.7, time_param=0.0):
        x0, y0 = position
        
        X_rot = (self.X - x0) * np.cos(orientation) + (self.Y - y0) * np.sin(orientation)
        Y_rot = -(self.X - x0) * np.sin(orientation) + (self.Y - y0) * np.cos(orientation)
        
        if system_type == 'sine_gordon':
            base_profile = 4 * np.arctan(np.exp(X_rot / width))
            
            shape_mode = mode_amplitude * np.sech(X_rot / width) * np.tanh(X_rot / width)
            vibration = shape_mode * np.cos(mode_frequency * time_param + phase)
            
            u = base_profile + vibration
            v = -shape_mode * mode_frequency * np.sin(mode_frequency * time_param + phase)
        elif system_type == 'phi4':
            base_profile = amplitude * np.tanh(X_rot / width)
            
            shape_mode = mode_amplitude * np.sech(X_rot / width)**2
            vibration = shape_mode * np.cos(mode_frequency * time_param + phase)
            
            u = base_profile + vibration
            v = -shape_mode * mode_frequency * np.sin(mode_frequency * time_param + phase)
        else:
            base_profile = 4 * np.arctan(np.exp(X_rot / width))
            
            shape_mode = mode_amplitude * np.sech(X_rot / width) * np.tanh(X_rot / width)
            vibration = shape_mode * np.cos(mode_frequency * time_param + phase)
            
            u = base_profile + vibration
            v = -shape_mode * mode_frequency * np.sin(mode_frequency * time_param + phase)
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def radiation_soliton_interaction(self, system_type='sine_gordon', 
                                    soliton_width=1.0, soliton_position=(0, 0),
                                    soliton_orientation=0.0, radiation_amplitude=0.3,
                                    radiation_wavelength=0.5, radiation_direction=0.0,
                                    apply_envelope=True, envelope_width=0.7):
        x0, y0 = soliton_position
        
        X_rot_soliton = (self.X - x0) * np.cos(soliton_orientation) + (self.Y - y0) * np.sin(soliton_orientation)
        Y_rot_soliton = -(self.X - x0) * np.sin(soliton_orientation) + (self.Y - y0) * np.cos(soliton_orientation)
        
        if system_type == 'sine_gordon':
            soliton = 4 * np.arctan(np.exp(X_rot_soliton / soliton_width))
        elif system_type == 'phi4':
            soliton = np.tanh(X_rot_soliton / soliton_width)
        else:
            soliton = 4 * np.arctan(np.exp(X_rot_soliton / soliton_width))
        
        k_x = np.cos(radiation_direction) * 2 * np.pi / radiation_wavelength
        k_y = np.sin(radiation_direction) * 2 * np.pi / radiation_wavelength
        
        radiation = radiation_amplitude * np.sin(k_x * self.X + k_y * self.Y)
        radiation_v = radiation_amplitude * 0.5 * np.cos(k_x * self.X + k_y * self.Y)
        
        u = soliton + radiation
        v = radiation_v
        
        if apply_envelope:
            u = self._envelope(u, envelope_width)
            v = self._envelope(v, envelope_width)
        
        return u, v
    
    def combined_solution(self, system_type='sine_gordon', solution_types=None, 
                        weights=None, apply_envelope=True, envelope_width=0.8):
        if solution_types is None:
            solution_types = ['kink', 'breather', 'ring', 'spiral', 'rogue']
            
        if weights is None:
            weights = np.ones(len(solution_types)) / len(solution_types)
        
        u_combined = np.zeros_like(self.X)
        v_combined = np.zeros_like(self.X)
        
        for solution_type, weight in zip(solution_types, weights):
            if solution_type == 'kink':
                u, v = self.kink_solution(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    orientation=2 * np.pi * np.random.rand(),
                    apply_envelope=False
                )
            elif solution_type == 'breather':
                u, v = self.breather_solution(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    amplitude=0.3 + 0.6 * np.random.rand(),
                    apply_envelope=False
                )
            elif solution_type == 'oscillon':
                u, v = self.oscillon_solution(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    amplitude=0.3 + 0.6 * np.random.rand(),
                    apply_envelope=False
                )
            elif solution_type == 'ring':
                u, v = self.ring_soliton(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    radius=0.5 + 1.5 * np.random.rand(),
                    apply_envelope=False
                )
            elif solution_type == 'skyrmion':
                u, v = self.skyrmion_solution(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    radius=0.5 + 1.5 * np.random.rand(),
                    apply_envelope=False
                )
            elif solution_type == 'spiral':
                u, v = self.spiral_wave_field(
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    num_arms=np.random.randint(1, 5),
                    apply_envelope=False
                )
            elif solution_type == 'rogue':
                u, v = self.rogue_wave(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    apply_envelope=False
                )
            elif solution_type == 'fractal':
                u, v = self.fractal_kink(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    orientation=2 * np.pi * np.random.rand(),
                    apply_envelope=False
                )
            elif solution_type == 'qball':
                u, v = self.q_ball_solution(
                    system_type=system_type,
                    position=(self.L * 0.5 * (np.random.rand() - 0.5), self.L * 0.5 * (np.random.rand() - 0.5)),
                    radius=0.5 + 1.5 * np.random.rand(),
                    apply_envelope=False
                )
            else:
                continue
                
            u_combined += weight * u
            v_combined += weight * v
        
        if apply_envelope:
            u_combined = self._envelope(u_combined, envelope_width)
            v_combined = self._envelope(v_combined, envelope_width)
        
        return u_combined, v_combined
    
    def generate_sample(self, system_type='sine_gordon', phenomenon_type='kink',
                        apply_envelope=True, envelope_width=0.7, velocity_type="fitting", **params): 
        if phenomenon_type == 'kink':
            return self.kink_solution(system_type, apply_envelope=apply_envelope, 
                                     envelope_width=envelope_width, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'kink_winding':
            return self.kink_winding_solution(system_type, apply_envelope=apply_envelope, 
                                            envelope_width=envelope_width, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'kink_array':
            return self.kink_array_field(system_type, apply_envelope=apply_envelope, 
                                       envelope_width=envelope_width, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'breather':
            return self.breather_solution(system_type, apply_envelope=apply_envelope, 
                                         envelope_width=envelope_width, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'multi_breather':
            return self.multi_breather_field(system_type, apply_envelope=apply_envelope, 
                                           envelope_width=envelope_width, velocity_type=velocity_type, **params)
        elif phenomenon_type == 'oscillon':
            return self.oscillon_solution(system_type, apply_envelope=apply_envelope, 
                                        envelope_width=envelope_width, **params)
        elif phenomenon_type == 'multi_oscillon':
            return self.multi_oscillon_state(system_type, apply_envelope=apply_envelope, 
                                           envelope_width=envelope_width, **params)
        elif phenomenon_type == 'ring':
            return self.ring_soliton(system_type, apply_envelope=apply_envelope, 
                                    envelope_width=envelope_width, **params)
        elif phenomenon_type == 'multi_ring':
            return self.multi_ring_state(system_type, apply_envelope=apply_envelope, 
                                       envelope_width=envelope_width, **params)
        elif phenomenon_type == 'skyrmion':
            return self.skyrmion_solution(system_type, apply_envelope=apply_envelope, 
                                        envelope_width=envelope_width, **params)
        elif phenomenon_type == 'skyrmion_lattice':
            return self.skyrmion_lattice(system_type, apply_envelope=apply_envelope, 
                                        envelope_width=envelope_width, **params)
        elif phenomenon_type == 'spiral':
            return self.spiral_wave_field(apply_envelope=apply_envelope, 
                                        envelope_width=envelope_width, **params)
        elif phenomenon_type == 'multi_spiral':
            return self.multi_spiral_state(apply_envelope=apply_envelope, 
                                         envelope_width=envelope_width, **params)
        elif phenomenon_type == 'rogue':
            return self.rogue_wave(system_type, apply_envelope=apply_envelope, 
                                  envelope_width=envelope_width, **params)
        elif phenomenon_type == 'multi_rogue':
            return self.multi_rogue_state(system_type, apply_envelope=apply_envelope, 
                                        envelope_width=envelope_width, **params)
        elif phenomenon_type == 'fractal_kink':
            return self.fractal_kink(system_type, apply_envelope=apply_envelope, 
                                   envelope_width=envelope_width, **params)
        elif phenomenon_type == 'domain_wall_network':
            return self.domain_wall_network(system_type, apply_envelope=apply_envelope, 
                                          envelope_width=envelope_width, **params)
        elif phenomenon_type == 'soliton_gas':
            return self.soliton_gas(system_type, apply_envelope=apply_envelope, 
                                  envelope_width=envelope_width, **params)
        elif phenomenon_type == 'qball':
            return self.q_ball_solution(system_type, apply_envelope=apply_envelope, 
                                      envelope_width=envelope_width, **params)
        elif phenomenon_type == 'multi_qball':
            return self.multi_q_ball_state(system_type, apply_envelope=apply_envelope, 
                                         envelope_width=envelope_width, **params)
        elif phenomenon_type == 'vibrational_kink':
            return self.vibrational_kink_mode(system_type, apply_envelope=apply_envelope, 
                                            envelope_width=envelope_width, **params)
        elif phenomenon_type == 'radiation_soliton':
            return self.radiation_soliton_interaction(system_type, apply_envelope=apply_envelope, 
                                                   envelope_width=envelope_width, **params)
        elif phenomenon_type == 'combined':
            return self.combined_solution(system_type, apply_envelope=apply_envelope, 
                                        envelope_width=envelope_width, **params)
        else:
            raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
    
    def generate_ensemble(self, system_type='sine_gordon', phenomenon_type='kink', 
                        n_samples=10, parameter_ranges=None, **fixed_params):
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
                    raise ValueError(f"Invalid parameter range: {range_values}")
            
            sample = self.generate_sample(system_type, phenomenon_type, **params)
            samples.append(sample)
        
        if n_samples == 1:
            return samples[0]
        else:
            return samples
    
    def generate_diverse_ensemble(self, system_type='sine_gordon', phenomenon_type='kink',
                               n_samples=10, parameter_ranges=None, similarity_threshold=0.2,
                               max_attempts=100, diversity_metric='l2', **fixed_params):
        samples = []
        attempts = 0
        
        if parameter_ranges is None:
            parameter_ranges = {}
            
        def diversity_distance(s1, s2):
            s1_u, s1_v = s1
            s2_u, s2_v = s2
            
            if diversity_metric == 'l2':
                s1_norm = np.sqrt(np.sum(s1_u**2 + s1_v**2))
                s2_norm = np.sqrt(np.sum(s2_u**2 + s2_v**2))
                
                if s1_norm == 0 or s2_norm == 0:
                    return 1.0
                    
                s1_u_norm = s1_u / s1_norm
                s1_v_norm = s1_v / s1_norm
                s2_u_norm = s2_u / s2_norm
                s2_v_norm = s2_v / s2_norm
                
                u_dist = np.sqrt(np.sum((s1_u_norm - s2_u_norm)**2))
                v_dist = np.sqrt(np.sum((s1_v_norm - s2_v_norm)**2))
                
                return (u_dist + v_dist) / 2
                
            elif diversity_metric == 'spectral':
                s1_u_fft = np.fft.fft2(s1_u)
                s1_v_fft = np.fft.fft2(s1_v)
                s2_u_fft = np.fft.fft2(s2_u)
                s2_v_fft = np.fft.fft2(s2_v)
                
                s1_u_fft_abs = np.abs(s1_u_fft)
                s1_v_fft_abs = np.abs(s1_v_fft)
                s2_u_fft_abs = np.abs(s2_u_fft)
                s2_v_fft_abs = np.abs(s2_v_fft)
                
                s1_u_fft_norm = np.sqrt(np.sum(s1_u_fft_abs**2))
                s1_v_fft_norm = np.sqrt(np.sum(s1_v_fft_abs**2))
                s2_u_fft_norm = np.sqrt(np.sum(s2_u_fft_abs**2))
                s2_v_fft_norm = np.sqrt(np.sum(s2_v_fft_abs**2))
                
                if s1_u_fft_norm == 0 or s2_u_fft_norm == 0 or s1_v_fft_norm == 0 or s2_v_fft_norm == 0:
                    return 1.0
                    
                u_overlap = np.sum(s1_u_fft_abs * s2_u_fft_abs) / (s1_u_fft_norm * s2_u_fft_norm)
                v_overlap = np.sum(s1_v_fft_abs * s2_v_fft_abs) / (s1_v_fft_norm * s2_v_fft_norm)
                
                return 1.0 - (u_overlap + v_overlap) / 2
            else:
                return 0.0
        
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
                    raise ValueError(f"Invalid parameter range: {range_values}")
            
            sample = self.generate_sample(system_type, phenomenon_type, **params)
            
            if len(samples) == 0:
                samples.append(sample)
            else:
                is_diverse = True
                for existing_sample in samples:
                    dist = diversity_distance(sample, existing_sample)
                    if dist < similarity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    samples.append(sample)
            
            attempts += 1
        
        return samples
    
    def generate_initial_condition(self, system_type='sine_gordon', 
                                 phenomenon_types=None, weights=None,
                                 randomize_parameters=True, apply_envelope=True,
                                 envelope_width=0.7, normalize=True, time_param=0.0,
                                 max_amplitude=1.0, velocity_type='fitting'):
        if phenomenon_types is None:
            phenomenon_types = ['kink', 'breather', 'ring', 'oscillon']
            
        if weights is None:
            weights = np.ones(len(phenomenon_types)) / len(phenomenon_types)
            
        if randomize_parameters:
            selected_idx = np.random.choice(len(phenomenon_types), p=weights)
            phenomenon_type = phenomenon_types[selected_idx]
            
            if phenomenon_type == 'kink':
                params = {
                    'width': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'orientation': np.random.uniform(0, 2*np.pi),
                    'velocity': (np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)),
                    'kink_type': np.random.choice(['standard', 'anti', 'double']),
                    'velocity_type': velocity_type
                }
            elif phenomenon_type == 'kink_winding':
                params = {
                    'winding_x': np.random.randint(-2, 3),
                    'winding_y': np.random.randint(-2, 3),
                    'width_range': (0.5, 2.0),
                    'randomize_positions': True,
                    'velocity_scale': np.random.uniform(0.1, 0.4),
                    'velocity_type': velocity_type
                }
                if params['winding_x'] == 0 and params['winding_y'] == 0:
                    params['winding_x'] = 1

            elif phenomenon_type == 'kink_array':
                params = {
                    'num_kinks_x': np.random.randint(1, 5),
                    'num_kinks_y': np.random.randint(0, 5),
                    'width_range': (0.5, 2.0),
                    'jitter': np.random.uniform(0.1, 0.5),
                    'velocity_scale': np.random.uniform(0.1, 0.4),
                    'velocity_type': velocity_type
                }
            elif phenomenon_type == 'breather':
                params = {
                    'amplitude': np.random.uniform(0.2, 0.9),
                    'frequency': np.random.uniform(0.6, 0.95),
                    'width': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'orientation': np.random.uniform(0, 2*np.pi),
                    'breather_type': np.random.choice(['standard', 'radial']),
                    'time_param': time_param,
                    'velocity_type': velocity_type
                }
            elif phenomenon_type == 'multi_breather':
                params = {
                    'num_breathers': np.random.randint(2, 5),
                    'position_type': np.random.choice(['random', 'circle', 'line']),
                    'amplitude_range': (0.2, 0.8),
                    'frequency_range': (0.6, 0.95),
                    'width_range': (0.5, 2.0),
                    'time_param': time_param,
                    'velocity_type': velocity_type
                }
            elif phenomenon_type == 'oscillon':
                params = {
                    'amplitude': np.random.uniform(0.3, 0.8),
                    'frequency': np.random.uniform(0.7, 0.95),
                    'width': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'profile': np.random.choice(['gaussian', 'sech', 'polynomial']),
                    'time_param': time_param
                }
            elif phenomenon_type == 'multi_oscillon':
                params = {
                    'n_oscillons': np.random.randint(3, 8),
                    'amplitude_range': (0.3, 0.7),
                    'width_range': (0.5, 2.0),
                    'frequency_range': (0.7, 0.95),
                    'position_variance': np.random.uniform(0.5, 1.0),
                    'arrangement': np.random.choice(['random', 'circular', 'lattice']),
                    'interaction_strength': np.random.uniform(0.5, 0.9),
                    'time_param': time_param
                }
            elif phenomenon_type == 'ring':
                params = {
                    'amplitude': 1.0,
                    'radius': np.random.uniform(1.0, min(self.L/4, 5.0)),
                    'width': np.random.uniform(0.3, 1.0),
                    'position': (np.random.uniform(-0.1, 0.1) * self.L, np.random.uniform(-0.1, 0.1) * self.L),
                    'velocity': np.random.uniform(-0.2, 0.2),
                    'ring_type': np.random.choice(['expanding', 'kink_antikink']),
                    'modulation_strength': np.random.uniform(0, 0.4),
                    'modulation_mode': np.random.randint(1, 6),
                    'time_param': time_param
                }
            elif phenomenon_type == 'multi_ring':
                params = {
                    'n_rings': np.random.randint(2, 6),
                    'radius_range': (1.0, min(self.L/4, 5.0)),
                    'width_range': (0.3, 1.0),
                    'arrangement': np.random.choice(['concentric', 'random', 'circular']),
                    'interaction_strength': np.random.uniform(0.5, 0.9),
                    'modulation_strength': np.random.uniform(0, 0.4),
                    'modulation_mode_range': (1, 6)
                }
            elif phenomenon_type == 'skyrmion':
                params = {
                    'amplitude': np.random.uniform(0.5, 1.5),
                    'radius': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'charge': np.random.choice([-1, 1]),
                    'profile': np.random.choice(['standard', 'compact', 'exponential'])
                }
            elif phenomenon_type == 'skyrmion_lattice':
                params = {
                    'n_skyrmions': np.random.randint(3, 10),
                    'radius_range': (0.5, 1.5),
                    'amplitude': np.random.uniform(0.5, 1.5),
                    'arrangement': np.random.choice(['triangular', 'square', 'random']),
                    'separation': np.random.uniform(2.0, 4.0),
                    'charge_distribution': np.random.choice(['alternating', 'random', 'same'])
                }
            elif phenomenon_type == 'spiral':
                params = {
                    'num_arms': np.random.randint(1, 6),
                    'decay_rate': np.random.uniform(0.3, 0.8),
                    'amplitude': np.random.uniform(0.5, 1.5),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'k_factor': np.random.uniform(1.0, 3.0)
                }
            elif phenomenon_type == 'multi_spiral':
                params = {
                    'n_spirals': np.random.randint(2, 5),
                    'amplitude_range': (0.5, 1.5),
                    'num_arms_range': (1, 5),
                    'decay_rate_range': (0.3, 0.8),
                    'position_variance': np.random.uniform(0.5, 1.0),
                    'interaction_strength': np.random.uniform(0.5, 0.9)
                }
            elif phenomenon_type == 'rogue':
                params = {
                    'amplitude': np.random.uniform(1.5, 3.0),
                    'background_level': np.random.uniform(0.1, 0.3),
                    'width': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L)
                }
            elif phenomenon_type == 'multi_rogue':
                params = {
                    'n_rogues': np.random.randint(2, 5),
                    'amplitude_range': (1.5, 3.0),
                    'width_range': (0.5, 2.0),
                    'background_level': np.random.uniform(0.1, 0.3),
                    'position_variance': np.random.uniform(0.3, 0.7)
                }
            elif phenomenon_type == 'fractal_kink':
                params = {
                    'levels': np.random.randint(2, 5),
                    'base_width': np.random.uniform(0.5, 2.0),
                    'scale_factor': np.random.uniform(1.5, 3.0),
                    'amplitude': 1.0,
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'orientation': np.random.uniform(0, 2*np.pi)
                }
            elif phenomenon_type == 'domain_wall_network':
                params = {
                    'n_walls': np.random.randint(4, 10),
                    'width_range': (0.5, 2.0),
                    'orientation_variance': np.random.uniform(0.3, 1.0),
                    'interaction_strength': np.random.uniform(0.5, 0.9)
                }
            elif phenomenon_type == 'soliton_gas':
                params = {
                    'n_solitons': np.random.randint(10, 20),
                    'width_range': (0.5, 1.5),
                    'velocity_scale': np.random.uniform(0.1, 0.5),
                    'interaction_strength': np.random.uniform(0.5, 0.9),
                    'velocity_type': velocity_type
                }
            elif phenomenon_type == 'qball':
                params = {
                    'amplitude': np.random.uniform(0.5, 1.5),
                    'radius': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'frequency': np.random.uniform(0.6, 0.9),
                    'charge': np.random.choice([-1, 1]),
                    'time_param': time_param
                }
            elif phenomenon_type == 'multi_qball':
                params = {
                    'n_qballs': np.random.randint(2, 5),
                    'amplitude_range': (0.5, 1.5),
                    'radius_range': (0.5, 2.0),
                    'frequency_range': (0.6, 0.9),
                    'position_variance': np.random.uniform(0.5, 1.0),
                    'interaction_strength': np.random.uniform(0.5, 0.9),
                    'time_param': time_param
                }
            elif phenomenon_type == 'vibrational_kink':
                params = {
                    'amplitude': 1.0,
                    'width': np.random.uniform(0.5, 2.0),
                    'position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'orientation': np.random.uniform(0, 2*np.pi),
                    'mode_amplitude': np.random.uniform(0.1, 0.5),
                    'mode_frequency': np.random.uniform(0.3, 0.7),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'time_param': time_param
                }
            elif phenomenon_type == 'radiation_soliton':
                params = {
                    'soliton_width': np.random.uniform(0.5, 2.0),
                    'soliton_position': (np.random.uniform(-0.3, 0.3) * self.L, np.random.uniform(-0.3, 0.3) * self.L),
                    'soliton_orientation': np.random.uniform(0, 2*np.pi),
                    'radiation_amplitude': np.random.uniform(0.1, 0.5),
                    'radiation_wavelength': np.random.uniform(0.3, 1.0),
                    'radiation_direction': np.random.uniform(0, 2*np.pi)
                }
            elif phenomenon_type == 'combined':
                n_types = np.random.randint(2, 4)
                all_types = ['kink', 'kink_winding', 'breather', 'ring', 'oscillon', 'skyrmion', 'spiral', 'rogue']
                selected_types = np.random.choice(all_types, size=n_types, replace=False)
                random_weights = np.random.rand(n_types)
                random_weights /= random_weights.sum()
                
                params = {
                    'solution_types': selected_types,
                    'weights': random_weights,
                    'velocity_type': velocity_type
                }
            else:
                params = {}


            if 'velocity_type' not in params.keys():
                u, v = self.generate_sample(
                    system_type, 
                    phenomenon_type, 
                    apply_envelope=apply_envelope,
                    envelope_width=envelope_width,
                    velocity_type=velocity_type,
                    **params
                )
            else:
                u, v = self.generate_sample(
                    system_type, 
                    phenomenon_type, 
                    apply_envelope=apply_envelope,
                    envelope_width=envelope_width,
                    **params
                )

        else:
            u, v = self.combined_solution(
                system_type, 
                solution_types=phenomenon_types,
                weights=weights,
                apply_envelope=apply_envelope,
                envelope_width=envelope_width
            )
        
        if normalize:
            max_u = np.max(np.abs(u))
            
            u = u / max_u
                
            max_v = np.max(np.abs(v))
            v = v / max_v

        
        return u, v

if __name__ == '__main__':
    nx = ny = 128
    L = 3.
    sampler = RealWaveEquationSampler(nx, ny, L)
    for _ in range(5):
        u, v = sampler.generate_initial_condition(system_type='sine_gordon', 
                                 phenomenon_types=None, weights=None,
                                 randomize_parameters=True, apply_envelope=True,
                                 envelope_width=0.7, normalize=True, time_param=0.0,
                                 max_amplitude=1.0,)
        plt.imshow(v)
        plt.show()
    
