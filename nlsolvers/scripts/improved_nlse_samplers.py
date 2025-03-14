import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.fft as fft
from matplotlib import cm


def make_grid(Nx, Ny, L):
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y


def sample_gaussian_random_field(Nx, Ny, L, power_law=-3.0, filter_scale=5.0):
    kx = 2*np.pi*fft.fftfreq(Nx, d=2*L/Nx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=2*L/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K = torch.sqrt(KX**2 + KY**2)
    power = K**power_law
    power[0, 0] = 0 
    power = power * torch.exp(-(K/filter_scale)**2)
    noise_real = torch.randn(Nx, Ny)
    noise_imag = torch.randn(Nx, Ny)
    noise = noise_real + 1j*noise_imag
    field = fft.ifft2(fft.fft2(noise) * torch.sqrt(power))
    field = field / torch.std(torch.abs(field))
    
    return field

def ground_state(X, Y, shape="gaussian", width=1.0, center=None, L=None, phase_pattern=None):
    # improved to have better control of locality
    if L is None:
        L = torch.max(X).item()

    if center is None:
        x0, y0 = 0.0, 0.0
    else:
        x0, y0 = center

    r2 = (X - x0)**2 + (Y - y0)**2

    if shape == "gaussian":
        amplitude = torch.exp(-r2/(2*width**2))

    elif shape == "thomas_fermi":
        tf_radius = width * 0.5
        amplitude = torch.clamp(1 - r2/tf_radius**2, min=0.0)
        amplitude = torch.sqrt(amplitude)

    elif shape == "plateau":
        plateau_radius = width * 0.5
        edge_width = width * 0.2
        amplitude = 0.5 * (1 - torch.tanh((torch.sqrt(r2) - plateau_radius)/edge_width))

    elif shape == "super_gaussian":
        order = 4.0
        amplitude = torch.exp(-(r2/(2*width**2))**order)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    boundary_distance = torch.minimum(
        torch.minimum(L - torch.abs(X), L - torch.abs(Y)),
        torch.minimum(L + X, L + Y)
    )
    boundary_weight = 1.0 - torch.exp(-boundary_distance/(0.1*L))

    psi = amplitude * boundary_weight * torch.exp(1j * torch.zeros_like(X))
    psi = psi

    return psi


def vortex_state(X, Y, charge=1, width=1.0, core_size=0.2, center=None, L=None):
    if L is None:
        L = torch.max(X).item()

    if center is None:
        x0, y0 = 0.0, 0.0
    else:
        x0, y0 = center

    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)
    amplitude = torch.tanh(r / core_size)
    gaussian_envelope = torch.exp(-r**2/(2*width**2))
    boundary_dist = 0.75 * L - torch.sqrt(torch.clamp((X)**2 + (Y)**2, min=0))
    boundary_factor = torch.sigmoid(boundary_dist / (0.1 * L))
    amplitude = amplitude * gaussian_envelope * boundary_factor
    phase = charge * theta
    psi = amplitude * torch.exp(1j * phase)
    return psi



def multi_vortex_state(X, Y, num_vortices=3, arrangement="random", width=2.0, L=None):
    if L is None:
        L = torch.max(X).item()
    
    effective_L = 0.6 * L
    r2 = X**2 + Y**2
    background = torch.exp(-r2/(2*(0.7*L)**2))
    
    psi = background * torch.exp(1j * torch.zeros_like(X))
   
    centers = []
    charges = []
    
    if arrangement == "random":
        for _ in range(num_vortices):
            centers.append(effective_L * 0.5 * torch.randn(2))
            charges.append(random.choice([-1, 1]))
    
    elif arrangement == "ring":
        radius = width * 0.3
        for i in range(num_vortices):
            angle = 2 * np.pi * i / num_vortices
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            charges.append(1 if i % 2 == 0 else -1)
    
    elif arrangement == "lattice":
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
    
    elif arrangement == "clustered":
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
    
    for center, charge in zip(centers, charges):
        x0, y0 = center
        r_vortex = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        theta = torch.atan2(Y - y0, X - x0)
        
        core_size = 0.05 * L
        
        amplitude_factor = r_vortex / torch.sqrt(r_vortex**2 + core_size**2)
        
        phase_factor = torch.exp(1j * charge * theta)
        psi = psi * amplitude_factor * phase_factor
    
    order = 2.0
    envelope = torch.exp(-(r2/(2*(0.9*L)**2))**order)
   
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    
    psi = psi * envelope * boundary_factor
    psi = psi
    
    return psi

def soliton_state(X, Y, soliton_type="bright", num_solitons=1, width=1.0, 
                           arrangement="parallel", velocity=None, L=None):
    if L is None:
        L = torch.max(X).item()
   
    effective_width = min(width, 0.15 * L) 
    if soliton_type == "dark":
        r2 = X**2 + Y**2
        psi = torch.exp(-r2/(2*(0.8*L)**2))
        max_amplitude = 1.0
    else:  # bright
        psi = torch.zeros_like(X, dtype=torch.complex128)
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
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        Y_rot = -X * np.sin(angle) + Y * np.cos(angle)
        
        if soliton_type == "bright":
            amplitude = max_amplitude / torch.cosh((X_rot - d) / effective_width)
            phase = vel * X_rot
            psi = psi + amplitude * torch.exp(1j * phase)
            
        else:  # dark
            tanh_factor = 1.5  
            profile = torch.tanh(tanh_factor * (X_rot - d) / effective_width)
            phase = vel * X_rot
            psi = psi * profile * torch.exp(1j * phase)
    
    # global envelope
    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(0.75*L)**2))
    
    # some non-physical things to not immediately introduce noise to the evolution
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    
    psi = psi * envelope * boundary_factor
     
    return psi


def lattice_state(X, Y, lattice_type="square", num_sites=16, 
                        amplitude_var=0.2, phase_var=0.5, L=None):
    if L is None:
        L = torch.max(X).item()
    
    effective_L = 0.7 * L 
    sites_per_side = min(6, int(np.sqrt(num_sites)))   
    centers = []
    if lattice_type == "square":
        spacing = 2 * effective_L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                centers.append([spacing * (j - sites_per_side/2 + 0.5), 
                               spacing * (i - sites_per_side/2 + 0.5)])
    
    elif lattice_type == "triangular":
        spacing = 2 * effective_L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                centers.append([spacing * (j - sites_per_side/2 + 0.5) + offset, 
                               spacing * (i - sites_per_side/2 + 0.5)])
    
    elif lattice_type == "honeycomb":
        spacing = 2 * effective_L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                base_x = spacing * (j - sites_per_side/2 + 0.5) + offset
                base_y = spacing * (i - sites_per_side/2 + 0.5)
                if i*sites_per_side + j < num_sites/2:
                    centers.append([base_x - spacing/4, base_y])
                    centers.append([base_x + spacing/4, base_y])
    
    elif lattice_type == "kagome":
        spacing = 2 * effective_L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                if i*sites_per_side + j >= num_sites/3:
                    continue
                offset = spacing * 0.5 if i % 2 == 1 else 0
                base_x = spacing * (j - sites_per_side/2 + 0.5) + offset
                base_y = spacing * (i - sites_per_side/2 + 0.5)
                centers.append([base_x, base_y])
                centers.append([base_x + spacing/2, base_y])
                centers.append([base_x + spacing/4, base_y + spacing*np.sqrt(3)/4])
   
    centers = centers[:num_sites]
    
    psi = torch.zeros_like(X, dtype=torch.complex128)
    site_width = spacing * 0.3  
    for center in centers:
        x0, y0 = center
        r2 = (X - x0)**2 + (Y - y0)**2
        amplitude = 1.0 + amplitude_var * (2 * torch.rand(1).item() - 1)
        phase = phase_var * 2 * np.pi * torch.rand(1).item()
        psi += amplitude * torch.exp(-r2/(2*site_width**2)) * torch.exp(torch.tensor(1j * phase))
   
    r2 = X**2 + Y**2
    order = 3.0  # setting a fairly sharp cutoff
    envelope = torch.exp(-(r2/(2*(0.7*L)**2))**order)
    
    psi = psi * envelope
    
    return psi

def dynamic_lattice_state(X, Y, lattice_type="square", num_sites=16,
                       dynamics_type="phase_gradient", L=None):
    if L is None:
        L = torch.max(X).item()

    effective_L = 0.7 * L
    sites_per_side = min(3, int(np.sqrt(num_sites)))
    spacing = 2 * effective_L / sites_per_side

    centers = []
    if lattice_type == "square":
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                centers.append([spacing * (j - sites_per_side/2 + 0.5),
                               spacing * (i - sites_per_side/2 + 0.5)])

    elif lattice_type == "triangular":
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                centers.append([spacing * (j - sites_per_side/2 + 0.5) + offset,
                               spacing * (i - sites_per_side/2 + 0.5)])

    elif lattice_type == "honeycomb":
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                base_x = spacing * (j - sites_per_side/2 + 0.5) + offset
                base_y = spacing * (i - sites_per_side/2 + 0.5)
                centers.append([base_x - spacing/4, base_y])
                centers.append([base_x + spacing/4, base_y])

    centers = centers[:num_sites]
    psi = torch.zeros_like(X, dtype=torch.complex128)
    site_width = spacing * 0.25
    if dynamics_type == "phase_gradient":
        flow_direction = torch.randn(2)
        flow_direction = flow_direction / torch.norm(flow_direction)
        flow_strength = 2.0
        global_phase = flow_strength * (X * flow_direction[0] + Y * flow_direction[1])

        for center in centers:
            x0, y0 = center
            r2 = (X - x0)**2 + (Y - y0)**2
            amplitude = torch.exp(-r2/(2*site_width**2))
            psi += amplitude * torch.exp(1j * global_phase)

    elif dynamics_type == "phase_vortex":
        center_x, center_y = 0, 0
        r_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        theta_center = torch.atan2(Y - center_y, X - center_x)

        charge = random.choice([-1, 1])
        for center in centers:
            x0, y0 = center
            r2 = (X - x0)**2 + (Y - y0)**2
            amplitude = torch.exp(-r2/(2*site_width**2))
            phase = charge * theta_center
            psi += amplitude * torch.exp(1j * phase)

    elif dynamics_type == "interference":
        for i, center in enumerate(centers):
            x0, y0 = center
            r2 = (X - x0)**2 + (Y - y0)**2
            amplitude = torch.exp(-r2/(2*site_width**2))

            phase = np.pi * (i % 2)
            psi += amplitude * torch.exp(torch.tensor(1j * phase))


    elif dynamics_type == "josephson":
        site_pairs = []
        remaining_centers = centers.copy()

        while len(remaining_centers) >= 2:
            c1 = remaining_centers.pop(0)
            distances = [torch.sqrt(torch.tensor((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)) for c2 in remaining_centers]
            nearest_idx = distances.index(min(distances))
            c2 = remaining_centers.pop(nearest_idx)

            site_pairs.append((c1, c2))
        if remaining_centers:
            site_pairs.append((remaining_centers[0], None))
        for pair in site_pairs:
            c1, c2 = pair
            x0, y0 = c1
            r2 = (X - x0)**2 + (Y - y0)**2
            amplitude = torch.exp(-r2/(2*site_width**2))
            phase1 = 0
            psi += amplitude * torch.exp(torch.tensor(torch.tensor(1j * phase1)))
            if c2 is not None:
                x0, y0 = c2
                r2 = (X - x0)**2 + (Y - y0)**2
                amplitude = torch.exp(-r2/(2*site_width**2))
                phase2 = np.pi  # π phase difference -- not sure how well-defined this is
                psi += amplitude * torch.exp(torch.tensor(1j * phase2))
    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(0.8*L)**2))
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)

    psi = psi * envelope * boundary_factor
    
    return psi


def quantum_turbulence(X, Y, energy_spectrum="kolmogorov", num_vortices=20, L=None):
    if L is None:
        L = torch.max(X).item()
    
    effective_L = 0.6 * L
    psi = torch.ones_like(X, dtype=torch.complex128)
    centers = []
    charges = []
    for _ in range(num_vortices):
        centers.append(effective_L * 0.5 * torch.randn(2))
        charges.append(random.choice([-1, 1]))
    for center, charge in zip(centers, charges):
        x0, y0 = center
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        theta = torch.atan2(Y - y0, X - x0)
        
        core_size = 0.05 * L + 0.05 * L * torch.rand(1).item()
        vortex_amplitude = torch.tanh(r / core_size)
        vortex_phase = torch.exp(1j * charge * theta)
        
        psi = psi * (vortex_amplitude * vortex_phase + (1 - vortex_amplitude))
   
    if energy_spectrum == "kolmogorov":
        power_law = -5/3
    elif energy_spectrum == "gaussian":
        power_law = -2
    else:
        power_law = -1
    
    filter_scale = 10.0 / L
    fluctuations = sample_gaussian_random_field(X.shape[0], X.shape[1], 
                                              effective_L, 
                                              power_law=power_law,
                                              filter_scale=filter_scale)
    
    phase_fluctuation_strength = 0.2
    psi = psi * torch.exp(1j * phase_fluctuation_strength * fluctuations.real)

    r2 = X**2 + Y**2
    order = 3.0
    envelope = torch.exp(-(r2/(2*(0.65*L)**2))**order)
    
    psi = psi * envelope
    psi = psi / torch.max(torch.abs(psi))
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    psi = psi * boundary_factor
    
    return psi


def rogue_wave(X, Y, peak_amplitude=3.0, background_amplitude=0.5, wave_type="peregrine", L=None):
    if L is None:
        L = torch.max(X).item()
    

    x0 = 0.3 * L * (2 * torch.rand(1).item() - 1)
    y0 = 0.3 * L * (2 * torch.rand(1).item() - 1)
    X_shifted = X - x0
    Y_shifted = Y - y0
    
    theta = 2 * np.pi * torch.rand(1).item()
    stretch_factor = 1.0 + 2.0 * torch.rand(1).item()
    X_rot = X_shifted * np.cos(theta) + Y_shifted * np.sin(theta)
    Y_rot = -X_shifted * np.sin(theta) + Y_shifted * np.cos(theta)

    X_stretched = X_rot
    Y_stretched = Y_rot / stretch_factor
    

    r2 = X_stretched**2 + Y_stretched**2
   
    if wave_type == "peregrine":
        # Peregrine soliton
        scale_factor = 6.0 + 6.0 * torch.rand(1).item()  # scale (6-12)
        xi = r2 / (scale_factor * L**2)
        
        t = -0.1 + 0.2 * torch.rand(1).item()
        denominator = 1.0 + 4.0 * xi * (1.0 + 2.0j * t)
        numerator = 4.0 * (1.0 + 2.0j * t)
        
        profile = background_amplitude * (1.0 - numerator / denominator)
        
    elif wave_type == "akhmediev":
        # Akhmediev breather which should be periodic in space
        a = 0.25 + 0.2 * torch.rand(1).item()  
        k = 2.0 + 2.0 * torch.rand(1).item() 
        
        cos_term = torch.cos(k * X_stretched)
        cosh_term = torch.cosh(2 * a * Y_stretched) + a * torch.cos(k * X_stretched)
        
        profile = background_amplitude * (1.0 + 2 * (1 - 2*a) * cos_term / cosh_term)
    
    elif wave_type == "kuznetsov":
        # Kuznetsov-Ma breather which should be periodic in time
        b = 0.4 + 0.3 * torch.rand(1).item()  # modulation kinda difficult
        phi = 2.0 * np.pi * torch.rand(1).item()
 
        r = torch.sqrt(r2)
        
        # KM profile
        cos_term = torch.cos(b * r + phi)
        sin_term = torch.sin(b * r + phi)
        cosh_term = torch.cosh(torch.tensor(np.sqrt(2) * b * 0.2)) + torch.sqrt(torch.tensor(2.0)) * cos_term
        
        profile = background_amplitude * (1.0 + 2 * np.sqrt(2) * b * cos_term / cosh_term)
    
    elif wave_type == "superposition":

        k1 = 1.0 + torch.rand(1).item()
        k2 = 1.0 + torch.rand(1).item()
        phase = torch.rand(1).item() * np.pi
        
        background_wave = background_amplitude * torch.cos(k1*X + k2*Y + phase)
        
        # handrolled, modified Peregrine
        scale_factor = 8.0
        xi = r2 / (scale_factor * L**2)
        peak = (peak_amplitude - background_amplitude) * (4.0 / (1.0 + 4.0 * xi))
        
        profile = background_wave + peak
    else:
        raise NotImplemented
    
   
    phase_scale = 0.2 + 0.3 * torch.rand(1).item() # small
    background_phase = phase_scale * (X + Y) + 0.1 * (X**2 - Y**2) / L**2
    
    psi = profile * torch.exp(1j * background_phase)
    
    r2_origin = X**2 + Y**2
    envelope = torch.exp(-(r2_origin/(2*(0.8*L)**2))**2)
    
    boundary_dist = 0.9 * L - torch.sqrt(r2_origin)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    
    psi = psi * envelope * boundary_factor
    max_amp = torch.max(torch.abs(psi))
    norm_factor = peak_amplitude / max_amp
    psi = psi * norm_factor
    
    return psi


def breather_state(X, Y, breather_type="kuznetsov", width=1.0, oscillation=2.0, L=None, offset_position=True):
    if L is None:
        L = torch.max(X).item()
    
    effective_width = min(width, 0.3*L) 
    if offset_position:
        max_offset = 0.3 * L
        x0 = max_offset * (2 * torch.rand(1).item() - 1)
        y0 = max_offset * (2 * torch.rand(1).item() - 1)
    else:
        x0, y0 = 0, 0
    
    X_shifted = X - x0
    Y_shifted = Y - y0
    
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
        omega_param = 2 * a**2
        mod_term = torch.cos(torch.tensor(b * t)) * torch.cosh(c_param * r) + np.sqrt(2) * torch.sinh(c_param * r)
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
        envelope = torch.exp(-r**2/(2*effective_width**2)) 
        psi = envelope * (1.0 + 2.0 * spatial_osc * (1.0 / (1.0 + r/effective_width)))
    else:
        raise NotImplemented
   
    base_field = torch.ones_like(X, dtype=torch.complex128)
    phase_gradient = 0.5 * (X_rot + Y_rot) / L
    psi = psi * torch.exp(1j * phase_gradient)
   
    r2 = X**2 + Y**2
    order = 2.5
    envelope = torch.exp(-(r2/(2*(0.8*L)**2))**order)
    
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    
    psi = psi * envelope * boundary_factor
   
    
    return psi


def condensate_with_sound(X, Y, condensate_shape="gaussian", sound_type="random", 
                              sound_amplitude=0.1, sound_wavelength=0.5, L=None):
    if L is None:
        L = torch.max(X).item()
    
    effective_width = 0.5 * L
    
    if condensate_shape == "gaussian":
        amplitude = torch.exp(-(X**2 + Y**2)/(2*effective_width**2))
    elif condensate_shape == "thomas_fermi":
        r2 = X**2 + Y**2
        tf_radius = 0.6 * L
        amplitude = torch.clamp(1 - r2/tf_radius**2, min=0.0)
        amplitude = torch.sqrt(amplitude)
    elif condensate_shape == "super_gaussian":
        r2 = X**2 + Y**2
        order = 4.0
        amplitude = torch.exp(-(r2/(2*effective_width**2))**order)
    else:
        raise NotImplemented
    
    sound = torch.zeros_like(X)
    
    if sound_type == "random":
        k = 2 * np.pi / (sound_wavelength * L)
        filter_scale = k * 2.0 # what we've seen often as approximation for leading wavenum
        sound = sample_gaussian_random_field(X.shape[0], X.shape[1], L, 
                                           power_law=-2.0, filter_scale=filter_scale)
        sound = sound.real
        
    elif sound_type == "standing":
        k = 2 * np.pi / (sound_wavelength * L)
        kx = 2.0 + 1.0 * torch.rand(1).item()
        ky = 2.0 + 1.0 * torch.rand(1).item() 
        r2 = X**2 + Y**2
        wave_envelope = torch.exp(-r2/(2*(0.7*L)**2))
        
        sound = torch.cos(kx * k * X) * torch.cos(ky * k * Y) * wave_envelope
        
    elif sound_type == "radial":
        k = 2 * np.pi / (sound_wavelength * L)
        r = torch.sqrt(X**2 + Y**2) 
        sound = torch.cos(k * r) * torch.exp(-r/(0.6*L))
    
    elif sound_type == "vortex_sound":
        r = torch.sqrt(X**2 + Y**2)
        theta = torch.atan2(Y, X)
        k_r = 2 * np.pi / (sound_wavelength * L)
        m = 3  # just mimic angular wavenumbers? 
        sound = torch.cos(k_r * r + m * theta) * torch.exp(-r/(0.6*L))
    else:
        raise NotImplemented
    amplitude_mask = (amplitude > 0.05)
    phase = torch.zeros_like(X)
    phase[amplitude_mask] = sound_amplitude * sound[amplitude_mask]
    
    psi = amplitude * torch.exp(1j * phase)
    r2 = X**2 + Y**2
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    psi = psi * boundary_factor
    
    return psi


def spiral_vortex_state(X, Y, charge=1, width=1.0, spiral_factor=2.0, L=None):
    if L is None:
        L = torch.max(X).item()
    
    effective_width = min(width, 0.4*L)
    spiral_scale = spiral_factor / L 
    x0, y0 =  np.random.randn(2) * L / 5
    
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)
    amplitude = torch.tanh(r / (0.1 * L)) * torch.exp(-r**2/(2*effective_width**2)) 
    phase = charge * (theta + spiral_scale * r)
    
    
    psi = amplitude * torch.exp(torch.tensor(1j * phase))
    r2 = X**2 + Y**2
    order = 3.0
    envelope = torch.exp(-(r2/(2*(0.6*L)**2))**order)
    psi = psi * envelope
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    psi = psi * boundary_factor
    
    return psi


def multi_spiral_vortex(X, Y, num_vortices=3, arrangement="random", width=2.0, spiral_factor=2.0):
    L = int(torch.max(X))
    psi = torch.zeros_like(X, dtype=torch.complex128)
    centers = []
    charges = []

    if arrangement == "random":
        for _ in range(num_vortices):
            centers.append(L * 0.6 * torch.randn(2))
            charges.append(random.choice([-2, 2]))

    elif arrangement == "ring":
        radius = width * 0.4
        for i in range(num_vortices):
            angle = 2 * np.pi * i / num_vortices
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            charges.append(1 if i % 2 == 0 else -1)

    elif arrangement == "lattice":
        side_length = int(np.ceil(np.sqrt(num_vortices)))
        spacing = width * 0.6 / side_length

        count = 0
        for i in range(side_length):
            for j in range(side_length):
                if count < num_vortices:
                    offset = spacing * 0.5 if i % 2 == 1 else 0
                    centers.append([spacing * (j - side_length/2) + offset,
                                   spacing * (i - side_length/2)])
                    charges.append(1 if (i + j) % 2 == 0 else -1)
                    count += 1
    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")

    for center, charge in zip(centers, charges):
        spiral_variation = spiral_factor * (0.8 + 0.4 * torch.rand(1).item())
        psi += spiral_vortex_state(X, Y, charge=charge, spiral_factor=spiral_variation)

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(width*1.2)**2))
    psi = psi * envelope
    return psi


def skyrmion_state(X, Y, width=1.0, skyrmion_radius=0.5, edge_width=0.1, L=None):
    if L is None:
        L = int(torch.max(X))
    
    effective_width = min(width, 0.4*L)
    effective_radius = min(skyrmion_radius, 0.3*L)
    effective_edge = min(edge_width, 0.1*L)
    

    x0, y0 = np.random.randn(2) * L * .8
    
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)
    amplitude = torch.exp(-r**2/(2*effective_width**2))
    profile_transition = torch.tanh((r - effective_radius)/effective_edge)
    z_component = -profile_transition
    
    xy_component = torch.sqrt(torch.clamp(1 - z_component**2, min=0.0, max=1.0))
    
    real_part = xy_component * torch.cos(theta)
    imag_part = xy_component * torch.sin(theta)
    psi = amplitude * (real_part + 1j * imag_part)
    
    r2 = X**2 + Y**2
    order = 3.0
    envelope = torch.exp(-(r2/(2*(0.6*L)**2))**order)
    psi = psi * envelope
   
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    psi = psi * boundary_factor

    
    return psi


def multi_skyrmion_state(X, Y, num_skyrmions=3, arrangement="random", width=2.0):
    L = torch.max(X).item()
    psi = torch.zeros_like(X, dtype=torch.complex128)

    centers = []

    if arrangement == "random":
        for _ in range(num_skyrmions):
            centers.append(L * torch.randn(2))

    elif arrangement == "ring":
        radius = width * 0.4
        for i in range(num_skyrmions):
            angle = 2 * np.pi * i / num_skyrmions
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])

    elif arrangement == "lattice":
        side_length = int(np.ceil(np.sqrt(num_skyrmions)))
        spacing = width * 0.6 / side_length

        count = 0
        for i in range(side_length):
            for j in range(side_length):
                if count < num_skyrmions:
                    offset = spacing * 0.5 if i % 2 == 1 else 0
                    centers.append([spacing * (j - side_length/2) + offset,
                                   spacing * (i - side_length/2)])
                    count += 1

    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")

    for center in centers:
        skyrmion_radius = 0.3 + 0.4 * torch.rand(1).item()
        edge_width = 0.05 + 0.1 * torch.rand(1).item()
        psi += skyrmion_state(X, Y, width=width, skyrmion_radius=skyrmion_radius, edge_width=edge_width)

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(width*1.2)**2))
    psi = psi * envelope
    return psi


def vortex_dipole(X, Y, separation=1.0, width=2.0, orientation=0, L=None):
    if L is None:
        L = int(torch.max(X))
    
    effective_width = min(width, 0.4*L)
    effective_separation = min(separation, 0.3*L)
   
    x0, y0 = np.random.randn(2) * L * .1
    
    dx = 0.5 * effective_separation * np.cos(orientation)
    dy = 0.5 * effective_separation * np.sin(orientation)
   
    center1 = [x0 + dx, y0 + dy]
    center2 = [x0 - dx, y0 - dy]
  
    x1, y1 = center1
    r1 = torch.sqrt((X - x1)**2 + (Y - y1)**2)
    theta1 = torch.atan2(Y - y1, X - x1)
    
    core_size = 0.05 * L
    vortex1_amplitude = torch.tanh(r1 / core_size)
    vortex1_phase = torch.exp(1j * theta1)
    
    x2, y2 = center2
    r2 = torch.sqrt((X - x2)**2 + (Y - y2)**2)
    theta2 = torch.atan2(Y - y2, X - x2)
    
    vortex2_amplitude = torch.tanh(r2 / core_size)
    vortex2_phase = torch.exp(-1j * theta2)
    
    psi = torch.ones_like(X, dtype=torch.complex128)
    psi = psi * (vortex1_amplitude * vortex1_phase + (1 - vortex1_amplitude))
    psi = psi * (vortex2_amplitude * vortex2_phase + (1 - vortex2_amplitude))

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*effective_width**2))
    
    propagation_phase = 0.5 * (-X * np.sin(orientation) + Y * np.cos(orientation))
    propagation_factor = torch.exp(1j * propagation_phase)
    
    psi = psi * envelope * propagation_factor
   
    boundary_dist = 0.9 * L - torch.sqrt(r2)
    boundary_factor = torch.sigmoid(5 * boundary_dist / L)
    psi = psi * boundary_factor
     
    return psi



def sample_nlse_initial_condition(Nx, Ny, L, condition_type='auto'):
    X, Y = make_grid(Nx, Ny, L)

    if condition_type == 'auto':
        condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton',
                         'lattice', 'turbulence', 'rogue', 'breather',
                         'sound', 'combined',  'spiral_vortex',
                         'multi_spiral', 'skyrmion', 'multi_skyrmion',
                         'vortex_dipole', 'dynamic_lattice']
        condition_type = random.choice(condition_types)

    if condition_type == 'multi_spiral':
        num_vortices = random.randint(2, 8)
        arrangement = random.choice(['random', 'ring', 'lattice'])
        spiral_factor = 0.5 + 3.0*torch.rand(1).item()
        psi = multi_spiral_vortex(X, Y,
                num_vortices=num_vortices, arrangement=arrangement,
                spiral_factor=spiral_factor)
        description = f"Multi-spiral vortices ({num_vortices}, {arrangement})"

    elif condition_type == 'skyrmion':
        width = 0.2*L + 0.1*L*torch.rand(1).item()
        skyrmion_radius = 0.1*L + 0.3*L*torch.rand(1).item()
        edge_width = 0.05*L + 0.1*L*torch.rand(1).item()
        psi = skyrmion_state(X, Y, width=width, skyrmion_radius=skyrmion_radius, edge_width=edge_width)
        description = f"Skyrmion (radius={skyrmion_radius:.1f})"

    elif condition_type == 'multi_skyrmion':
        num_skyrmions = random.randint(3, 10)
        arrangement = random.choice(['random', 'ring', 'lattice'])
        width = 0.2*L + 0.5*L*torch.rand(1).item()
        psi = multi_skyrmion_state(X, Y, num_skyrmions=num_skyrmions,
                                 arrangement=arrangement, width=width)
        description = f"Multi-skyrmion ({num_skyrmions}, {arrangement})"

    elif condition_type == 'vortex_dipole':
        separation = 0.2*L + 0.4*L*torch.rand(1).item()
        width = 0.3*L + 0.5*L*torch.rand(1).item()
        orientation = 2*np.pi*torch.rand(1).item()
        psi = vortex_dipole(X, Y, separation=separation, width=width, orientation=orientation)
        description = f"Vortex dipole (separation={separation:.1f}), [{orientation:.2f}]"

    elif condition_type == 'vortex_lattice':
        lattice_type = random.choice(['square', 'triangular', 'honeycomb'])
        vortex_density = 0.3 + 0.8*torch.rand(1).item()
        alternating = random.choice([True, False])
        width = 0.7*L + 0.2*L*torch.rand(1).item()
        psi = vortex_lattice(X, Y, lattice_type=lattice_type, vortex_density=vortex_density,
                           alternating=alternating, width=width)
        polarity = "alternating" if alternating else "same"
        description = f"Vortex lattice ({lattice_type}, {polarity} polarity)"

    elif condition_type == 'ground':
        shape = random.choice(['gaussian', 'thomas_fermi', 'plateau'])
        width = 0.2*L + 0.6*L*torch.rand(1).item()
        
        if torch.rand(1).item() < 0.3:
            def phase_pattern(X, Y, x0, y0):
                vx, vy = torch.randn(2)
                return vx * (X - x0) + vy * (Y - y0)
        else:
            phase_pattern = None
            
        psi = ground_state(X, Y, shape=shape, width=width, phase_pattern=phase_pattern)
        description = f"Ground state ({shape})"
        
    elif condition_type == 'vortex':
        charge = random.choice([-2, -1, 1, 2])
        width = 0.2*L + 0.5*L*torch.rand(1).item()
        psi = vortex_state(X, Y, charge=charge, width=width)
        description = f"Vortex (charge={charge})"
        
    elif condition_type == 'multi_vortex':
        num_vortices = random.randint(2, 8)
        arrangement = random.choice(['random', 'ring', 'lattice'])
        psi = multi_vortex_state(X, Y, num_vortices=num_vortices, arrangement=arrangement)
        description = f"Multi-vortex ({num_vortices}, {arrangement})"
        
    elif condition_type == 'soliton':
        soliton_type = random.choice(['bright', 'dark'])
        num_solitons = random.randint(1, 4)
        arrangement = random.choice(['parallel', 'crossing', 'radial'])
        if torch.rand(1).item() < 0.5:
            velocity = 2.0 * torch.rand(1).item()
        else:
            velocity = None
            
        psi = soliton_state(X, Y, soliton_type=soliton_type, num_solitons=num_solitons,
                          arrangement=arrangement, velocity=velocity)
        
        vel_str = f", v={velocity:.1f}" if velocity is not None else ""
        description = f"{soliton_type.capitalize()} soliton ({num_solitons}, {arrangement}{vel_str})"
        
    elif condition_type == 'lattice':
        lattice_type = random.choice(['square', 'triangular', 'honeycomb'])
        num_sites = random.randint(9, 36)
        psi = lattice_state(X, Y, lattice_type=lattice_type, num_sites=num_sites)
        description = f"Lattice ({lattice_type}, {num_sites} sites)"

    elif condition_type == 'dynamic_lattice':
        lattice_type = random.choice(['square', 'triangular', 'honeycomb'])
        dynamics = random.choice(["phase_gradient","phase_vortex", "interference", "josephson"])
        num_sites = random.randint(5, 25)
        psi = dynamic_lattice_state(X, Y, lattice_type=lattice_type, num_sites=num_sites,
                       dynamics_type=dynamics)
        description = f"Lattice {lattice_type}, {num_sites} sites, {dynamics} dynamism" 
        
    elif condition_type == 'turbulence':
        energy_spectrum = random.choice(['kolmogorov', 'white'])
        num_vortices = random.randint(10, 50)
        psi = quantum_turbulence(X, Y, energy_spectrum=energy_spectrum,
                               num_vortices=num_vortices)
        description = f"Quantum turbulence ({energy_spectrum}, {num_vortices} vortices)"
        
    elif condition_type == 'rogue':
        peak_amplitude = 2.0 + 2.0 * torch.rand(1).item()
        ty = random.choice(["peregrine", "akhmediev", "kuznetsov", "superposition"])
        psi = rogue_wave(X, Y, peak_amplitude=peak_amplitude, wave_type=ty) 
        description = f"Rogue wave, type={ty}"
        
    elif condition_type == 'breather':
        breather_type = random.choice(['kuznetsov',
            'localized', 'vector', 'spatial'])
        width = 0.2*L + 0.4*L*torch.rand(1).item()
        psi = breather_state(X, Y, breather_type=breather_type, width=width)
        description = f"{breather_type} breather"
        
    elif condition_type == 'sound':
        condensate_shape = random.choice(['gaussian', 'thomas_fermi', 'super_gaussian'])
        sound_type = random.choice(['random', 'standing', 'radial', 'vortex_sound'])

        sound_amplitude = 0.05 + 0.15 * torch.rand(1).item()
        sound_wavelength = 0.2 + 0.8 * torch.rand(1).item()
        
        psi = condensate_with_sound(X, Y, condensate_shape=condensate_shape,
                                  sound_type=sound_type, sound_amplitude=sound_amplitude,
                                  sound_wavelength=sound_wavelength)
 
        description = f"Condensate ({condensate_shape}) with {sound_type} sound"
       

    elif condition_type == 'spiral_vortex':
        charge = np.random.randint(-2, 2)
        width = np.random.random(1) * L / 3
        spiral_factor = charge * 2.
        psi = spiral_vortex_state(X, Y, charge=charge, width=width, spiral_factor=spiral_factor)
        description = f"Spiral vortex, {charge=}"
        
    else:
        raise NotImplemented

    return psi, description


def visualize_nlse_initial_condition(psi, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(torch.abs(psi), cmap='viridis')
    axes[0].set_title('Amplitude')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(torch.angle(psi), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Phase')
    plt.colorbar(im1, ax=axes[1])
    X, Y = np.meshgrid(
        np.linspace(-1, 1, psi.shape[0]),
        np.linspace(-1, 1, psi.shape[1]),
        indexing='ij'
    )
    
    ax2 = axes[2]
    ax2.remove()
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax2.plot_surface(
        X, Y, torch.abs(psi).numpy(),
        cmap='viridis',
        linewidth=0,
        antialiased=True
    )
    ax2.set_title('3D Amplitude')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    Nx, Ny = 300, 300
    L = 10.0
   
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    
    #condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton', 
    #                  'lattice', 'turbulence', 'rogue', 'breather', ]
    condition_types = ['vortex_dipole'] * 9
    
    for i, ctype in enumerate(condition_types):
        psi, description = sample_nlse_initial_condition(Nx, Ny, L, ctype)
        im = axs[i].imshow(torch.abs(psi), cmap='viridis')
        axs[i].set_title(description)
        axs[i].axis('off')
        cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)     
    fig.suptitle("Amplitude")
    plt.show()

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    for i, ctype in enumerate(condition_types):
        psi, description = sample_nlse_initial_condition(Nx, Ny, L, ctype)
        im = axs[i].imshow(torch.real(psi), cmap='viridis')
        axs[i].set_title(description)
        axs[i].axis('off')
        cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)     
    fig.suptitle("Real")
    plt.show()

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    for i, ctype in enumerate(condition_types):
        psi, description = sample_nlse_initial_condition(Nx, Ny, L, ctype)
        im = axs[i].imshow(torch.imag(psi), cmap='viridis')
        axs[i].set_title(description)
        axs[i].axis('off')
        cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)     
    fig.suptitle("Imag")
    plt.show()
