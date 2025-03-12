import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import torch.fft as fft

def sample_anisotropic_grf(Nx, Ny, L, length_scale=1.0, anisotropy_ratio=2.0, theta=30.0, power=2.0):
    theta_rad = np.deg2rad(theta)
    ell_x = length_scale * np.sqrt(anisotropy_ratio)
    ell_y = length_scale / np.sqrt(anisotropy_ratio)
    kx = 2*np.pi*fft.fftfreq(Nx, d=2 * L/Nx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=2 * L/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    KX_rot = KX*np.cos(theta_rad) - KY*np.sin(theta_rad)
    KY_rot = KX*np.sin(theta_rad) + KY*np.cos(theta_rad) 
    
    spectrum = torch.exp(-((KX_rot/ell_x)**2 + (KY_rot/ell_y)**2)**(power/2))
    noise = torch.randn(Nx, Ny) + 1j*torch.randn(Nx, Ny)
    field = fft.ifft2(fft.fft2(noise) * torch.sqrt(spectrum)).real
    return field / torch.std(field)

def sample_wavelet_superposition(Nx, Ny, L, n_wavelets=20, scale_range=(0.1, 2.0), kappa=0.5, freq_range=(0.5, 3.0)): 
    v0 = torch.zeros(Nx, Ny)
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    for _ in range(n_wavelets):
        scale = scale_range[0] + (scale_range[1]-scale_range[0])*torch.rand(1)
        theta = 2*np.pi*torch.rand(1)
        x0 = L*(torch.rand(1)-0.5)
        y0 = L*(torch.rand(1)-0.5)
        k0 = (freq_range[0] + (freq_range[1]-freq_range[0])*torch.rand(1)) * (2*np.pi/(scale*L))
        
        envelope = torch.exp(-((X-x0)**2 + (Y-y0)**2)/(2*(scale*L)**2))
        
        wavelet_type = random.choice(['cosine', 'gaussian_deriv', 'morlet'])
        
        if wavelet_type == 'cosine':
            carrier = torch.cos(k0*((X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta)))
        elif wavelet_type == 'gaussian_deriv':
            z = ((X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta)) / (scale*L)
            carrier = -z * torch.exp(-z**2/2)
        else:
            z = ((X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta))
            carrier = torch.cos(k0*z) * torch.exp(-(z/(scale*L))**2/2)
        
        amp = (1 - kappa) + kappa*torch.rand(1)
        v0 += amp * envelope * carrier

    return v0 / torch.max(torch.abs(v0))

def sample_kink_field(X, Y, L, winding_x, winding_y, width_range=(0.5, 3.0), randomize_positions=True):
    u0 = torch.zeros_like(X)
    
    if winding_x != 0:
        width_x = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(1)
        positions_x = []
        
        if randomize_positions:
            for i in range(abs(winding_x)):
                positions_x.append(L * (2 * torch.rand(1) - 1))
        else:
            for i in range(abs(winding_x)):
                positions_x.append(L * (-0.8 + 1.6 * i / (abs(winding_x))))
        
        sign_x = 1 if winding_x > 0 else -1
        for x0 in positions_x:
            kink_width = width_x * (0.8 + 0.4 * torch.rand(1))
            u0 += sign_x * 4 * torch.atan(torch.exp((X - x0) / kink_width))
    
    if winding_y != 0:
        width_y = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(1)
        positions_y = []
        
        if randomize_positions:
            for i in range(abs(winding_y)):
                positions_y.append(L * (2 * torch.rand(1) - 1))
        else:
            for i in range(abs(winding_y)):
                positions_y.append(L * (-0.8 + 1.6 * i / (abs(winding_y))))
        
        sign_y = 1 if winding_y > 0 else -1
        for y0 in positions_y:
            kink_width = width_y * (0.8 + 0.4 * torch.rand(1))
            u0 += sign_y * 4 * torch.atan(torch.exp((Y - y0) / kink_width))
    
    return u0

def sample_kink_array_field(X, Y, L, num_kinks_x, num_kinks_y, width_range=(0.5, 2.0), jitter=0.3):
    u0 = torch.zeros_like(X)
    
    if num_kinks_x > 0:
        width_x = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(1)
        spacing_x = 2.0 * L / (num_kinks_x + 1)
        
        for i in range(num_kinks_x):
            x0 = -L + (i + 1) * spacing_x
            if jitter > 0:
                x0 = x0 + jitter * spacing_x * (2 * torch.rand(1) - 1)
            
            sign_x = 1 if torch.rand(1) > 0.5 else -1
            kink_width = width_x * (0.8 + 0.4 * torch.rand(1))
            u0 += sign_x * 4 * torch.atan(torch.exp((X - x0) / kink_width))
    
    if num_kinks_y > 0:
        width_y = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(1)
        spacing_y = 2.0 * L / (num_kinks_y + 1)
        
        for i in range(num_kinks_y):
            y0 = -L + (i + 1) * spacing_y
            if jitter > 0:
                y0 = y0 + jitter * spacing_y * (2 * torch.rand(1) - 1)
            
            sign_y = 1 if torch.rand(1) > 0.5 else -1
            kink_width = width_y * (0.8 + 0.4 * torch.rand(1))
            u0 += sign_y * 4 * torch.atan(torch.exp((Y - y0) / kink_width))
    
    return u0

def sample_breather_field(X, Y, L, num_breathers=1, position_type='random'):
    u0 = torch.zeros_like(X)
    
    positions = []
    if position_type == 'random':
        for _ in range(num_breathers):
            x0 = L * (2 * torch.rand(1) - 1)
            y0 = L * (2 * torch.rand(1) - 1)
            positions.append((x0, y0))
    elif position_type == 'circle':
        radius = 0.6 * L * torch.rand(1)
        for i in range(num_breathers):
            angle = 2 * np.pi * i / num_breathers
            x0 = radius * torch.tensor(np.cos(angle))
            y0 = radius * torch.tensor(np.sin(angle))
            positions.append((x0, y0))
    elif position_type == 'line':
        for i in range(num_breathers):
            pos = -L + 2 * L * i / (num_breathers - 1 if num_breathers > 1 else 1)
            if random.choice([True, False]):
                positions.append((torch.tensor([pos]), torch.tensor([0.0])))
            else:
                positions.append((torch.tensor([0.0]), torch.tensor([pos])))
    
    for x0, y0 in positions:
        width = 0.5 + 2.5 * torch.rand(1)
        amplitude = 0.1 + 0.8 * torch.rand(1)
        phase = 2 * np.pi * torch.rand(1)
        
        omega = torch.sqrt(1.0 - amplitude**2)
        t = 0.0
        
        direction = random.choice(['x', 'y', 'radial'])
        if direction == 'x':
            xi = (X - x0) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                              (omega * torch.cosh(amplitude * xi)))
        elif direction == 'y':
            yi = (Y - y0) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                              (omega * torch.cosh(amplitude * yi)))
        else:
            ri = torch.sqrt((X - x0)**2 + (Y - y0)**2) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                              (omega * torch.cosh(amplitude * ri)))
    
    return u0

def sample_multi_frequency_breather(X, Y, L, num_modes=2):
    u0 = torch.zeros_like(X)
    
    x0 = L * (2 * torch.rand(1) - 1)
    y0 = L * (2 * torch.rand(1) - 1)
    
    base_width = 0.5 + 2.5 * torch.rand(1)
    
    for i in range(num_modes):
        width = base_width * (0.5 + 1.0 * i/num_modes)
        amplitude = (0.1 + 0.8 * torch.rand(1)) / (i + 1)**0.5
        phase = 2 * np.pi * torch.rand(1)
        
        omega = torch.sqrt(1.0 - amplitude**2)
        t = 0.0
        
        direction = random.choice(['x', 'y', 'radial', 'angular'])
        
        if direction == 'x':
            xi = (X - x0) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                              (omega * torch.cosh(amplitude * xi)))
        elif direction == 'y':
            yi = (Y - y0) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                              (omega * torch.cosh(amplitude * yi)))
        elif direction == 'radial':
            ri = torch.sqrt((X - x0)**2 + (Y - y0)**2) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                              (omega * torch.cosh(amplitude * ri)))
        else:
            theta = torch.atan2(Y - y0, X - x0)
            r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
            decay = torch.exp(-r**2/(2*L**2))
            u0 += decay * amplitude * torch.sin((i+1) * theta + phase)
    
    return u0

def sample_spiral_wave_field(X, Y, L, num_arms=2, decay_rate=0.5):
    u0 = torch.zeros_like(X)
    
    x0 = L * (2 * torch.rand(1) - 1)
    y0 = L * (2 * torch.rand(1) - 1)
    
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)
    
    k = 1.0 + 2.0 * torch.rand(1)
    spiral_phase = theta + k * r / L
    
    pattern = torch.cos(num_arms * spiral_phase)
    
    decay = torch.exp(-decay_rate * r / L)
    
    u0 = 4 * torch.atan(pattern * decay)
    
    return u0

def sample_colliding_rings(X, Y, L, num_rings=2, ring_type='random'):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)

    if ring_type == 'random':
        for _ in range(num_rings):
            x0 = L * (2*torch.rand(1) - 1)
            y0 = L * (2*torch.rand(1) - 1)
            r0 = 0.1*L + 0.6*L*torch.rand(1)
            width = 0.5 + 2.5*torch.rand(1)
            direction = 1 if torch.rand(1) > 0.5 else -1

            r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
            if np.random.choice(2):
                u0 += 4 * torch.atan(torch.exp((r - r0)/width))
                v0 += direction * torch.exp(-(r - r0)**2/(2*width**2))
            else:
                u0 -= 4 * torch.atan(torch.exp((r - r0)/width))
                v0 -= direction * torch.exp(-(r - r0)**2/(2*width**2))
    
    elif ring_type == 'concentric':
        x0 = L * (2*torch.rand(1) - 1)
        y0 = L * (2*torch.rand(1) - 1)
        
        for i in range(num_rings):
            r0 = (0.2 + 0.6 * i/num_rings) * L
            width = 0.5 + 1.5*torch.rand(1)
            direction = 1 if i % 2 == 0 else -1
            
            r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
            u0 += direction * 4 * torch.atan(torch.exp((r - r0)/width))
            v0 += direction * torch.exp(-(r - r0)**2/(2*width**2))
    
    elif ring_type == 'nested':
        for i in range(num_rings):
            max_offset = 0.3 * L * i / num_rings
            x0 = max_offset * (2*torch.rand(1) - 1)
            y0 = max_offset * (2*torch.rand(1) - 1)
            r0 = (0.2 + 0.5 * (num_rings-i)/num_rings) * L
            width = 0.5 + 1.5*torch.rand(1)
            direction = 1 if i % 2 == 0 else -1
            
            r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
            u0 += direction * 4 * torch.atan(torch.exp((r - r0)/width))
            v0 += direction * torch.exp(-(r - r0)**2/(2*width**2))

    return u0, v0

def sample_elliptical_soliton(X, Y, L, complexity='simple'):
    if complexity == 'simple':
        x0, y0 = (L/2) * (2*torch.rand(2) - 1)
        a = (0.1*L + 0.2*L*torch.rand(1))
        b = a * (0.2 + 0.8*torch.rand(1))


        theta = torch.pi*torch.rand(1)
        phase = 2*torch.pi*torch.rand(1)

        X_rot = (X - x0)*torch.cos(theta) + (Y - y0)*torch.sin(theta)
        Y_rot = -(X - x0)*torch.sin(theta) + (Y - y0)*torch.cos(theta)

        r_ellipse = torch.sqrt((X_rot/a)**2 + (Y_rot/b)**2)

        omega = (1.0 - 0.3**2)**.5
        u0 = 4 * torch.atan(0.3 * torch.sin(phase) /
                          (omega * torch.cosh(0.3 * r_ellipse)))

        v0 = 4 * 0.3 * omega * torch.cos(phase + omega*0.0) / (
            omega * torch.cosh(0.3 * r_ellipse) *
            (1 + (0.3**2/omega**2) * torch.sin(phase + omega*0.0)**2)
        )
    elif complexity == 'complex':
        u0 = torch.zeros_like(X)
        v0 = torch.zeros_like(X)
        
        num_features = random.randint(2, 4)
        
        for _ in range(num_features):
            x0, y0 = (L/2) * (2*torch.rand(2) - 1)
            a = (0.1*L + 0.2*L*torch.rand(1))
            b = a * (0.2 + 0.8*torch.rand(1))
            
            theta = torch.pi*torch.rand(1)
            phase = 2*torch.pi*torch.rand(1)            
            X_rot = (X - x0)*torch.cos(theta) + (Y - y0)*torch.sin(theta)
            Y_rot = -(X - x0)*torch.sin(theta) + (Y - y0)*torch.cos(theta)
            
            r_ellipse = torch.sqrt((X_rot/a)**2 + (Y_rot/b)**2)
            
            amplitude = 0.2 + 0.3 * torch.rand(1)
            omega = (1.0 - amplitude**2)**.5
            
            u0 += 4 * torch.atan(amplitude * torch.sin(phase) /
                              (omega * torch.cosh(amplitude * r_ellipse)))
            
            v0 += 4 * amplitude * omega * torch.cos(phase) / (
                omega * torch.cosh(amplitude * r_ellipse) *
                (1 + (amplitude**2/omega**2) * torch.sin(phase)**2)
            )

    return u0, v0

def sample_soliton_antisoliton_pair(X, Y, L, pattern_type='auto'):
    if pattern_type == 'auto':
        pattern_type = random.choice(['radial', 'linear', 'angular', 'nested'])
        
    width = 0.8 + 2.2*torch.rand(1)
    x0, y0 = L*(2*torch.rand(2) - 1)

    if pattern_type == 'radial':
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        u0 = 4*torch.atan(torch.exp(r/width)) - 4*torch.atan(torch.exp((r - 0.5*width)/width))
    elif pattern_type == 'linear':
        theta = torch.pi*torch.rand(1)
        x_rot = (X - x0)*torch.cos(theta) + (Y - y0)*torch.sin(theta)
        u0 = 4*torch.atan(torch.exp(x_rot/width)) - 4*torch.atan(torch.exp(-x_rot/width))
    elif pattern_type == 'angular':
        phi = torch.atan2(Y - y0, X - x0)
        u0 = 4*torch.atan(torch.exp(torch.sin(phi)/width)) - 4*torch.atan(torch.exp(-torch.sin(phi)/width))
    else:
        r1 = 0.3*L + 0.1*L*torch.rand(1)
        r2 = 0.6*L + 0.1*L*torch.rand(1)
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        u0 = 4*torch.atan(torch.exp((r-r1)/width)) - 4*torch.atan(torch.exp((r-r2)/width))

    v0 = sample_anisotropic_grf(X.shape[0], X.shape[1], L,
                               length_scale=width, anisotropy_ratio=2.0) * 0.2

    return u0, v0

def sample_skyrmion_like_field_nonsmooth(X, Y, L, num_skyrmions=1):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    
    for _ in range(num_skyrmions):
        x0, y0 = L * (2*torch.rand(2) - 1)
        radius = 0.1*L + 0.3*L*torch.rand(1)
        
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        phi = torch.atan2(Y - y0, X - x0)
        
        profile = 2 * torch.atan(r/radius)
        
        u0 += 2 * torch.cos(phi) * (1 - torch.tanh(r/radius))
        v0 += 2 * torch.sin(phi) * (1 - torch.tanh(r/radius))
    
    skyrmion_field = 4 * torch.atan2(u0, v0)
    
    return skyrmion_field, 0.1 * sample_anisotropic_grf(X.shape[0], X.shape[1], L)

def sample_skyrmion_like_field(X, Y, L, num_skyrmions=1):
    phi = torch.zeros_like(X)
    
    for _ in range(num_skyrmions):
        x0, y0 = L * (2*torch.rand(2) - 1)
        lambda_size = 0.2*L + 0.4*L*torch.rand(1)
        q = random.choice([-1, 1])
        alpha = 2 * np.pi * torch.rand(1)
        z = (X - x0) + 1j * (Y - y0) 
        # rational map
        if q > 0:
            w = z / (lambda_size + torch.abs(z))
        else:
            w = z.conj() / (lambda_size + torch.abs(z))
    
        angle = torch.angle(w * torch.exp(1j * alpha))

        r = torch.abs(z)
        profile = 2 * torch.arctan2(lambda_size, r)
        
        skyrmion_contribution = 2 * profile * angle / torch.pi
        
        cutoff = torch.exp(-(r/(0.8*L))**4)
        phi += cutoff * skyrmion_contribution
    
    return phi, 0.05 * sample_anisotropic_grf(X.shape[0], X.shape[1], L)

def sample_combined_field(X, Y, L, solution_types=None, weights=None):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    
    if solution_types is None:
        solution_types = ['kink', 'breather', 'rings', 'soliton_pair', 'spiral']
        
    if weights is None:
        weights = torch.ones(len(solution_types))
    weights = weights / weights.sum()
    
    for solution_type, weight in zip(solution_types, weights):
        if solution_type == 'kink':
            winding_x = random.randint(-2, 2)
            winding_y = random.randint(-2, 2)
            while winding_x == 0 and winding_y == 0:
                winding_x = random.randint(-2, 2)
            kink_u0 = sample_kink_field(X, Y, L, winding_x, winding_y)
            u0 += weight * kink_u0
            
        elif solution_type == 'kink_array':
            num_kinks_x = random.randint(0, 3)
            num_kinks_y = random.randint(0, 3)
            while num_kinks_x == 0 and num_kinks_y == 0:
                num_kinks_x = random.randint(0, 3)
            kink_u0 = sample_kink_array_field(X, Y, L, num_kinks_x, num_kinks_y)
            u0 += weight * kink_u0
            
        elif solution_type == 'breather':
            num_breathers = random.randint(1, 3)
            position_type = random.choice(['random', 'circle', 'line'])
            breather_u0 = sample_breather_field(X, Y, L, num_breathers, position_type)
            u0 += weight * breather_u0
            
        elif solution_type == 'multi_breather':
            multi_u0 = sample_multi_frequency_breather(X, Y, L, random.randint(2, 4))
            u0 += weight * multi_u0
            
        elif solution_type == 'rings':
            num_rings = random.randint(1, 4)
            ring_type = random.choice(['random', 'concentric', 'nested'])
            ring_u0, ring_v0 = sample_colliding_rings(X, Y, L, num_rings, ring_type)
            u0 += weight * ring_u0
            v0 += weight * ring_v0
            
        elif solution_type == 'elliptical':
            complexity = random.choice(['simple', 'complex'])
            ellip_u0, ellip_v0 = sample_elliptical_soliton(X, Y, L, complexity)
            u0 += weight * ellip_u0
            v0 += weight * ellip_v0
            
        elif solution_type == 'soliton_pair':
            pattern_type = random.choice(['radial', 'linear', 'angular', 'nested'])
            pair_u0, pair_v0 = sample_soliton_antisoliton_pair(X, Y, L, pattern_type)
            u0 += weight * pair_u0
            v0 += weight * pair_v0
            
        elif solution_type == 'spiral':
            num_arms = random.randint(1, 4)
            spiral_u0 = sample_spiral_wave_field(X, Y, L, num_arms)
            u0 += weight * spiral_u0
            
        elif solution_type == 'skyrmion':
            num_skyrmions = random.randint(1, 3)
            sky_u0, sky_v0 = sample_skyrmion_like_field(X, Y, L, num_skyrmions)
            u0 += weight * sky_u0
            v0 += weight * sky_v0
    
    return u0, v0

def sample_sine_gordon_solution(Nx, Ny, L, solution_type='auto'):
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    if solution_type == 'auto':
        solution_types = ['kink', 'breather', 'combined', 'rings', 'kink_array', 
                         'multi_breather', 'spiral', 'elliptical', 'skyrmion', 'random_mix']
        solution_type = random.choice(solution_types)
    
    v0_amplitude = 0.05 + 0.1 * torch.rand(1)
    v0 = sample_wavelet_superposition(Nx, Ny, L, n_wavelets=15, 
                                     scale_range=(0.1, 0.8), kappa=0.3) * v0_amplitude
    
    if solution_type == 'kink':
        winding_x = random.randint(-2, 2)
        winding_y = random.randint(-2, 2)
        
        while winding_x == 0 and winding_y == 0:
            winding_x = random.randint(-2, 2)
            winding_y = random.randint(-2, 2)
            
        u0 = sample_kink_field(X, Y, L, winding_x, winding_y)
        solution_info = f"Kink (nx={winding_x}, ny={winding_y})"
        
    elif solution_type == 'kink_array':
        num_kinks_x = random.randint(1, 4)
        num_kinks_y = random.randint(1, 4)
        u0 = sample_kink_array_field(X, Y, L, num_kinks_x, num_kinks_y)
        solution_info = f"Kink Array ({num_kinks_x}x{num_kinks_y})"
        
    elif solution_type == 'breather':
        num_breathers = random.randint(1, 4)
        position_type = random.choice(['random', 'circle', 'line'])
        u0 = sample_breather_field(X, Y, L, num_breathers, position_type)
        solution_info = f"Breather (n={num_breathers}, {position_type})"
        
    elif solution_type == 'multi_breather':
        num_modes = random.randint(2, 4)
        u0 = sample_multi_frequency_breather(X, Y, L, num_modes)
        solution_info = f"Multi-frequency Breather (modes={num_modes})"
        
    elif solution_type == 'spiral':
        num_arms = random.randint(1, 4)
        u0 = sample_spiral_wave_field(X, Y, L, num_arms)
        solution_info = f"Spiral Wave (arms={num_arms})"
        
    elif solution_type == 'combined':
        combo_types = random.sample(['kink', 'breather', 'rings', 'spiral', 'soliton_pair'], 
                                   k=random.randint(2, 3))
        u0, v0_extra = sample_combined_field(X, Y, L, combo_types)
        v0 += v0_extra
        solution_info = f"Combined ({'+'.join(combo_types)})"

    elif solution_type == 'rings':
        num_rings = random.randint(1, 5)
        ring_type = random.choice(['random', 'concentric', 'nested'])
        u0, v0 = sample_colliding_rings(X, Y, L, num_rings, ring_type)
        solution_info = f"Rings ({num_rings}, {ring_type})"
      
    elif solution_type == 'elliptical':
        complexity = random.choice(['simple', 'complex'])
        u0, v0 = sample_elliptical_soliton(X, Y, L, complexity)
        solution_info = f"Elliptical Soliton ({complexity})"
    
    elif solution_type == 'skyrmion':
        num_skyrmions = random.randint(1, 3)
        u0, v0_extra = sample_skyrmion_like_field(X, Y, L, num_skyrmions)
        v0 += v0_extra
        solution_info = f"Skyrmion-like ({num_skyrmions})"
        
    elif solution_type == 'random_mix':
        all_types = ['kink', 'breather', 'rings', 'elliptical', 'soliton_pair', 'spiral', 'skyrmion']
        num_types = random.randint(2, 4)
        selected_types = random.sample(all_types, k=num_types)
        weights = torch.rand(num_types)
        u0, v0_extra = sample_combined_field(X, Y, L, selected_types, weights)
        v0 += v0_extra
        solution_info = f"Random Mix ({'+'.join(selected_types)})"
    
    anisotropy_ratio = 1.0 + 3.0 * torch.rand(1)
    theta = 360.0 * torch.rand(1)
    length_scale = L / (1.0 + 5.0 * torch.rand(1))
    power = 1.0 + 1.0 * torch.rand(1)
    
    anisotropy = sample_anisotropic_grf(Nx, Ny, L, length_scale=length_scale, 
                                      anisotropy_ratio=anisotropy_ratio, theta=theta,
                                      power=power)
    
    m = 1.0 + torch.nn.functional.relu(anisotropy)
    
    mask = torch.abs(m) > 5.0
    m[mask] = 5.0

    m /= torch.max(m)
    
    return u0, v0, m, solution_info

def main():
    Nx = 128
    Ny = 128
    L = 10.0
    
    
    x = torch.linspace(-L, L, Nx).numpy()
    y = torch.linspace(-L, L, Ny).numpy()
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    #solution_types = ['rings', 'kink', 'breather', 'combined', 'kink_array', 
    #                'multi_breather', 'spiral', 'elliptical']
    """
    fig = plt.figure(figsize=(20, 15))
    solution_types = ['skyrmion'] * 4
    
    for i, solution_type in enumerate(solution_types):
        u0, v0, m, solution_info = sample_sine_gordon_solution(Nx, Ny, L, solution_type)
        
        
        
        ax1 = fig.add_subplot(3, 4, i+1)
        im1 = ax1.pcolormesh(X, Y, m.numpy(), cmap='viridis', shading='auto')
        ax1.set_title(f'{solution_info} (m)')
        fig.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(3, 4, i+5)
        im2 = ax2.pcolormesh(X, Y, u0.numpy(), cmap='coolwarm', shading='auto', vmin=-6, vmax=6)
        ax2.set_title(f'{solution_info} (u)')
        fig.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(3, 4, i+9)
        im3 = ax3.pcolormesh(X, Y, v0.numpy(), cmap='coolwarm', shading='auto', vmin=-1, vmax=1)
        ax3.set_title(f'{solution_info} (v)')
        fig.colorbar(im3, ax=ax3)
    plt.tight_layout() 
    plt.show()
    """
    
    fig = plt.figure(figsize=(20, 15)) 
    for i in range(8):
        u0, v0, m, solution_info = sample_sine_gordon_solution(Nx, Ny, L)
        
        ax1 = fig.add_subplot(2, 4, i+1, projection='3d')
        surf = ax1.plot_surface(X[::4,::4], Y[::4,::4], u0.numpy()[::4,::4], 
                              cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax1.set_title(f'{solution_info}')
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    main()
