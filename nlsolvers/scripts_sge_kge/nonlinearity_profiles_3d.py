import numpy as np
from scipy.ndimage import gaussian_filter

def make_grid(n, L):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    z = np.linspace(-L, L, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z

def piecewise_constant(n, L, num_layers=3, base_value=1.0, contrast_factor=2.0):
    X, Y, Z = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    layer_width = 2*L / num_layers
    
    for i in range(num_layers):
        if i % 2 == 1:
            x_min = -L + i * layer_width
            x_max = -L + (i + 1) * layer_width
            mask = (X >= x_min) & (X < x_max)
            profile[mask] = base_value * contrast_factor
    
    return profile

def smooth_gradient(n, L, direction='x', base_value=1.0, gradient_strength=0.5):
    X, Y, Z = make_grid(n, L)
    
    if direction == 'x':
        norm_coord = (X + L) / (2*L)
        profile = base_value * (1 + gradient_strength * norm_coord)
    elif direction == 'y':
        norm_coord = (Y + L) / (2*L)
        profile = base_value * (1 + gradient_strength * norm_coord)
    elif direction == 'z':
        norm_coord = (Z + L) / (2*L)
        profile = base_value * (1 + gradient_strength * norm_coord)
    elif direction == 'radial':
        R = np.sqrt(X**2 + Y**2 + Z**2) / (np.sqrt(3)*L)
        profile = base_value * (1 + gradient_strength * R)
    
    return profile

def localized_perturbations(n, L, num_perturbations=5, base_value=1.0, 
                           perturbation_strength=0.5, width=0.1, 
                           strategic_placement=False):
    X, Y, Z = make_grid(n, L)
    profile = np.ones((n, n, n)) * base_value
    width_scaled = width * L
    
    if strategic_placement:
        positions = []
        for i in range(num_perturbations):
            if i == 0:
                positions.append((0, 0, 0))
            elif i == 1:
                positions.append((0.5*L, 0, 0))
            elif i == 2:
                positions.append((-0.5*L, 0, 0))
            elif i == 3:
                positions.append((0, 0.5*L, 0))
            elif i == 4:
                positions.append((0, 0, 0.5*L))
        
        for x0, y0, z0 in positions:
            dist_sq = (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2
            gaussian = np.exp(-dist_sq / (2 * width_scaled**2))
            profile += base_value * perturbation_strength * gaussian
    else:
        for _ in range(num_perturbations):
            x0 = np.random.uniform(-0.8*L, 0.8*L)
            y0 = np.random.uniform(-0.8*L, 0.8*L)
            z0 = np.random.uniform(-0.8*L, 0.8*L)
            
            dist_sq = (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2
            gaussian = np.exp(-dist_sq / (2 * width_scaled**2))
            profile += base_value * perturbation_strength * gaussian
    
    return profile

def periodic_structure(n, L, base_value=1.0, amplitude=0.5, frequency=3):
    X, Y, Z = make_grid(n, L)
    
    k = np.pi * frequency / L
    
    profile = base_value * (1 + amplitude * (
        np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
    ))
    
    return profile

def random_media(n, L, base_value=1.0, std=0.2, correlation_length=0.1):
    white_noise = np.random.randn(n, n, n) 
    sigma = correlation_length * n / (2*L) 
    smoothed = gaussian_filter(white_noise, sigma=sigma) 
    smoothed = (smoothed - np.mean(smoothed)) / np.std(smoothed) 
    return base_value * (1 + std * smoothed) 

def sign_changing_mass(n, L, base_value=1.0, regions='checkerboard', scale=2, 
                      sharpness=5.0):
    X, Y, Z = make_grid(n, L)
    
    if regions == 'checkerboard':
        cell_size = L / scale
        pattern = np.sin(np.pi * X / cell_size) * np.sin(np.pi * Y / cell_size) * np.sin(np.pi * Z / cell_size) 
        if sharpness > 0:
            profile = base_value * np.tanh(sharpness * pattern)
        else:
            profile = base_value * np.sign(pattern) 
    elif regions == 'half_space':
        profile = base_value * np.tanh(sharpness * X / L)
    
    return profile

def topological_mass(n, L, type='vortex', base_value=1.0, strength=1.0):
    X, Y, Z = make_grid(n, L)
    
    if type == 'vortex':
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        profile = base_value * (1 + strength * np.tanh(r/L) * np.cos(theta)) 
    elif type == 'monopole':
        r = np.sqrt(X**2 + Y**2 + Z**2)
        profile = base_value * (1 + strength * (1 - L/np.maximum(r, L/n)))
    
    return profile

def focusing_lens_c(n, L, base_value=1.0, strength=1.5, focus_point=(0,0,0), width=0.3):
    X, Y, Z = make_grid(n, L)
    x0, y0, z0 = focus_point
    width_scaled = width * L 
    dist_sq = (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2 
    c_profile = base_value * (1 + strength * (1 - np.exp(-dist_sq / (2 * width_scaled**2)))) 
    return c_profile

def soliton_enhancing_m(n, L, base_value=1.0, transition_width=0.1):
    X, Y, Z = make_grid(n, L) 
    R = np.sqrt(X**2 + Y**2 + Z**2) / L
    transition_scaled = transition_width * 10 
    m_profile = base_value * np.tanh((0.7 - R) / transition_scaled) 
    return m_profile

def resonant_cavity_pair(n, L, base_c=1.0, base_m=1.0, cavity_size=0.4, 
                        c_contrast=3.0, m_contrast=2.0):
    X, Y, Z = make_grid(n, L) 
    R = np.sqrt(X**2 + Y**2 + Z**2) / L 
    c_profile = base_c * (1 + (c_contrast - 1) * (R > cavity_size)) 
    m_profile = base_m * (m_contrast * (R <= cavity_size) - (R > cavity_size)) 
    return c_profile, m_profile

def multiscale_grf(n, L, base_value=1.0, scales=[0.05, 0.2, 0.5], weights=[0.6, 0.3, 0.1]):
    profile = np.ones((n, n, n)) * base_value
    
    for scale, weight in zip(scales, weights):
        white_noise = np.random.randn(n, n, n)
        sigma = scale * n / (2*L)
        smoothed = gaussian_filter(white_noise, sigma=sigma)
        smoothed = (smoothed - np.mean(smoothed)) / np.std(smoothed)
        profile += base_value * weight * smoothed
    
    return profile

def fractal_interfaces(n, L, base_value=1.0, num_octaves=4, lacunarity=2.0, persistence=0.5):
    X, Y, Z = make_grid(n, L)
    profile = np.zeros((n, n, n))
    
    freq = 1.0
    amp = 1.0
    
    for _ in range(num_octaves):
        phase = np.random.uniform(0, 2*np.pi)
        k = np.pi * freq / L
        
        noise = np.sin(k * X + phase) * np.sin(k * Y + phase) * np.sin(k * Z + phase)
        profile += amp * noise
        
        freq *= lacunarity
        amp *= persistence
    
    profile = profile / np.max(np.abs(profile))
    profile = base_value * np.tanh(3.0 * profile)
    
    return profile

def nonlinear_waveguide(n, L, base_c=1.0, base_m=1.0, width=0.2, length=0.8):
    X, Y, Z = make_grid(n, L)
    width_scaled = width * L
    length_scaled = length * L
    
    waveguide_mask = np.logical_and.reduce((
        np.abs(Y) < width_scaled,
        np.abs(Z) < width_scaled,
        np.abs(X) < length_scaled
    ))
    
    c_profile = np.ones((n, n, n)) * base_c
    c_profile[waveguide_mask] = base_c * 0.5
    
    m_profile = np.ones((n, n, n)) * base_m
    m_profile[waveguide_mask] = -base_m
    
    return c_profile, m_profile

def nonlinear_grf_pair(n, L, base_c=1.0, base_m=1.0, c_scale=0.2, m_scale=0.15):
    X, Y, Z = make_grid(n, L)
    
    white_noise_c = np.random.randn(n, n, n)
    white_noise_m = np.random.randn(n, n, n)
    
    sigma_c = c_scale * n / (2*L)
    sigma_m = m_scale * n / (2*L)
    
    grf_c = gaussian_filter(white_noise_c, sigma=sigma_c)
    grf_m = gaussian_filter(white_noise_m, sigma=sigma_m)
    
    grf_c = (grf_c - np.mean(grf_c)) / np.std(grf_c)
    grf_m = (grf_m - np.mean(grf_m)) / np.std(grf_m)
    
    c_profile = base_c * (1 + 0.4 * grf_c)
    
    m_threshold = 0.7
    m_profile = base_m * np.where(grf_m > m_threshold, 1.0, -0.5)
    
    return c_profile, m_profile

def anisotropic_wavespeed(n, L, base_value=1.0, strength=0.8):
    X, Y, Z = make_grid(n, L)
    
    theta = np.arctan2(Y, X)
    phi = np.arctan2(np.sqrt(X**2 + Y**2), Z)
    
    anisotropy = np.abs(np.sin(2*theta) * np.sin(phi))
    
    c_profile = base_value * (1 + strength * anisotropy)
    
    return c_profile


def highlight_profiles(n, L):
    profiles = {} 
    profiles['optimal'] = (
        localized_perturbations(n, L, 
                              num_perturbations=4, 
                              perturbation_strength=0.9,
                              width=0.15,
                              strategic_placement=True),
        sign_changing_mass(n, L, 
                         base_value=1.0,
                         regions='checkerboard', 
                         scale=2.5,
                         sharpness=8.0)
    )
    
    profiles['resonant_cavity'] = resonant_cavity_pair(n, L, 
                                                     base_c=1.0, 
                                                     base_m=1.0,
                                                     cavity_size=0.4,
                                                     c_contrast=2.5, 
                                                     m_contrast=1.5)
    
    profiles['focusing_soliton'] = (
        focusing_lens_c(n, L, base_value=1.0, strength=1.5),
        soliton_enhancing_m(n, L, base_value=1.0, transition_width=0.08)
    )
    
    profiles['sharp_interfaces'] = (
        piecewise_constant(n, L, num_layers=2, contrast_factor=4.0),
        sign_changing_mass(n, L, regions='half_space', sharpness=10.0)
    )
    
    profiles['multi_scale'] = (
        periodic_structure(n, L, amplitude=0.6, frequency=5),
        topological_mass(n, L, type='vortex', strength=1.2)
    )
    
    profiles['fractal_nonlinear'] = (
        multiscale_grf(n, L, base_value=1.0, scales=[0.05, 0.15, 0.3], weights=[0.5, 0.3, 0.2]),
        fractal_interfaces(n, L, base_value=1.0, num_octaves=5, persistence=0.6)
    )
    
    profiles['waveguide'] = nonlinear_waveguide(n, L, base_c=1.0, base_m=1.0, width=0.15, length=0.9)
    
    profiles['grf_threshold'] = nonlinear_grf_pair(n, L, base_c=1.0, base_m=1.0, c_scale=0.2, m_scale=0.15)
    
    profiles['anisotropic'] = (
        anisotropic_wavespeed(n, L, base_value=1.0, strength=0.8),
        sign_changing_mass(n, L, regions='checkerboard', scale=3, sharpness=6.0)
    )

    profiles['maybe_blowup'] = (
        focusing_lens_c(n, L, base_value=1.0, strength=-0.95,
                               focus_point=(0,0,0), width=0.2),
        sign_changing_mass(n, L, base_value=-2.0,
                                 regions='checkerboard', scale=2,
                                 sharpness=10.0)
        )
    
    return profiles
