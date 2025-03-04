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


def enforce_no_flux_boundary(psi, X, Y, L, width=0.1):
    envelope = (1 - torch.exp(-(L - torch.abs(X))/(width*L))) * \
               (1 - torch.exp(-(L - torch.abs(Y))/(width*L)))
    return psi * envelope


def ground_state(X, Y, shape="gaussian", width=1.0, center=None, phase_pattern=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0 = L * 0.5 * torch.randn(2)
    else:
        x0, y0 = center
    
    r2 = (X - x0)**2 + (Y - y0)**2
    if shape == "gaussian":
        amplitude = torch.exp(-r2/(2*width**2))
    
    elif shape == "thomas_fermi":
        tf_radius = width * torch.max(X).item() * 0.5
        amplitude = torch.clamp(1 - r2/tf_radius**2, min=0.0)
        amplitude = torch.sqrt(amplitude)
    
    elif shape == "plateau":
        plateau_radius = width * 0.5
        edge_width = width * 0.3
        amplitude = 0.5 * (1 - torch.tanh((torch.sqrt(r2) - plateau_radius)/edge_width))
    
    else:
        raise ValueError(f"Unknown shape: {shape}")
    if phase_pattern is None:
        phase = torch.zeros_like(X)
    else:
        phase = phase_pattern(X, Y, x0, y0)
    psi = amplitude * torch.exp(1j * phase)
    
    return psi


def vortex_phase(X, Y, x0, y0, charge=1):
    return charge * torch.atan2(Y - y0, X - x0)


def vortex_state(X, Y, charge=1, width=1.0, core_size=0.2, center=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0 = L * 0.5 * torch.randn(2)
    else:
        x0, y0 = center
    
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    amplitude = r / torch.sqrt(r**2 + core_size**2)
    amplitude = amplitude * torch.exp(-r**2/(2*width**2))
    phase = vortex_phase(X, Y, x0, y0, charge)
    psi = amplitude * torch.exp(1j * phase)
    
    return psi


def multi_vortex_state(X, Y, num_vortices=3, arrangement="random", width=2.0):
    L = torch.max(X).item()
    psi = torch.zeros_like(X, dtype=torch.complex128)
    
    centers = []
    charges = []
    
    if arrangement == "random":
        for _ in range(num_vortices):
            centers.append(L * 0.6 * torch.randn(2))
            charges.append(random.choice([-1, 1]))
    
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
        core_size = 0.1 + 0.2 * torch.rand(1).item()
        psi += vortex_state(X, Y, charge, width, core_size, center)
    
    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(width*1.2)**2))
    psi = psi * envelope / torch.max(torch.abs(psi))
    
    return psi


def soliton_state(X, Y, soliton_type="bright", num_solitons=1, width=1.0, 
                 arrangement="parallel", velocity=None):
    L = torch.max(X).item()
    
    if soliton_type == "dark":
        psi = torch.ones_like(X, dtype=torch.complex128)
    else:
        psi = torch.zeros_like(X, dtype=torch.complex128)
    distances = []
    angles = []
    
    if arrangement == "parallel":
        spacing = 2.0 * width
        for i in range(num_solitons):
            distances.append(spacing * (i - (num_solitons-1)/2))
            angles.append(0)
            
    elif arrangement == "crossing":
        for i in range(num_solitons):
            distances.append(width * torch.randn(1).item())
            angles.append(np.pi * i / num_solitons)
    
    elif arrangement == "radial":
        for i in range(num_solitons):
            distances.append(0)
            angles.append(np.pi * i / num_solitons)
    
    for d, angle in zip(distances, angles):
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        Y_rot = -X * np.sin(angle) + Y * np.cos(angle)
        
        if soliton_type == "bright":
            # Bright; sech profile
            profile = 1.0 / torch.cosh((X_rot - d) / width)
            
        else:  # dark
            profile = torch.tanh((X_rot - d) / width)
        if velocity is not None:
            phase = velocity * X_rot
            phase_factor = torch.exp(1j * phase)
        else:
            phase_factor = 1.0
        
        if soliton_type == "bright":
            psi = psi + profile * phase_factor
        else:
            psi = psi * profile * phase_factor
    
    r2 = X**2 + Y**2
    if soliton_type == "bright":
        psi = psi / torch.max(torch.abs(psi))
    else:
        envelope = 0.5 * (1 + torch.tanh((2*L - torch.sqrt(r2))/(0.2*L)))
        psi = psi * envelope
    
    return psi


def lattice_state(X, Y, lattice_type="square", num_sites=16, amplitude_var=0.2, phase_var=0.5):
    L = torch.max(X).item()
    sites_per_side = int(np.sqrt(num_sites))
    centers = []
    if lattice_type == "square":
        spacing = 2 * L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                centers.append([spacing * (j - sites_per_side/2 + 0.5), 
                               spacing * (i - sites_per_side/2 + 0.5)])
    
    elif lattice_type == "triangular":
        spacing = 2 * L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                centers.append([spacing * (j - sites_per_side/2 + 0.5) + offset, 
                               spacing * (i - sites_per_side/2 + 0.5)])
    
    elif lattice_type == "honeycomb":
        spacing = 2 * L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                base_x = spacing * (j - sites_per_side/2 + 0.5) + offset
                base_y = spacing * (i - sites_per_side/2 + 0.5)
                centers.append([base_x - spacing/4, base_y])
                centers.append([base_x + spacing/4, base_y])
    
    psi = torch.zeros_like(X, dtype=torch.complex128)
    width = L / sites_per_side * 0.8
    
    for center in centers:
        x0, y0 = center
        r2 = (X - x0)**2 + (Y - y0)**2
        amplitude = 1.0 + amplitude_var * (2 * torch.rand(1).item() - 1)
        phase = phase_var * 2 * np.pi * torch.rand(1).item()
        psi += amplitude * torch.exp(-r2/(2*width**2)) * torch.exp(torch.tensor(1j * phase))
    psi = psi / torch.max(torch.abs(psi))
    
    return psi


def quantum_turbulence(X, Y, energy_spectrum="kolmogorov", num_vortices=20):
    L = torch.max(X).item()
    psi = multi_vortex_state(X, Y, num_vortices=num_vortices, arrangement="random", width=L*0.8)
    
    power_law = -5/3 if energy_spectrum == "kolmogorov" else 0
    fluctuations = sample_gaussian_random_field(X.shape[0], X.shape[1], L, power_law=power_law)
    phase_fluctuation_strength = 0.3
    psi = psi * torch.exp(1j * phase_fluctuation_strength * fluctuations.real)
    psi = enforce_no_flux_boundary(psi, X, Y, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi


def rogue_wave(X, Y, background_amplitude=0.5, peak_amplitude=3.0):
    L = torch.max(X).item()
    x0, y0 = 0.2 * L * torch.randn(2)
    r2 = (X - x0)**2 + (Y - y0)**2
    
    xi = r2 / (4 * L**2) 
    # background * (1 - 4(1+2it)/(1+4x^2+4t^2))
    peregrine = background_amplitude * (1.0 - (4.0 / (1.0 + 4.0 * xi)) * torch.exp(torch.tensor(1j * np.pi/2)))
    background_phase = 0.2 * torch.randn_like(X)
    psi = peregrine * torch.exp(1j * background_phase)
    psi = enforce_no_flux_boundary(psi, X, Y, L)
    psi = psi * (peak_amplitude / torch.max(torch.abs(psi)))
    
    return psi


def breather_state(X, Y, breather_type="kuznetsov", width=1.0, oscillation=2.0):
    L = torch.max(X).item()
    x0, y0 = 0.2 * L * torch.randn(2)
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    
    if breather_type == "kuznetsov":
        # Kuznetsov-Ma breather: sech(x) * (cos(b) + sqrt(2)*sinh(a*x)/cosh(a*x))
        a = 0.5 / width
        b = oscillation
        
        amplitude = 1.0 / torch.cosh(r / width)
        modulation = torch.cos(torch.tensor(b)) + np.sqrt(2) * torch.sinh(torch.tensor(a) * r) / torch.cosh(torch.tensor(a) * r)
        psi = amplitude * modulation
        
    else:  # akhmediev breather
        # (1 + a*cos(b*x)/cosh(c*x))
        # similar to SGE
        a = 2.0
        b = oscillation
        c = 0.5 / width
        theta = torch.atan2(Y - y0, X - x0)

        modulation = 1.0 + a * torch.cos(b * theta) / torch.cosh(c * r)
        envelope = torch.exp(-r**2 / (4 * L**2))
        psi = modulation * envelope
    
    phase = 0.5 * (X + Y)
    psi = psi * torch.exp(1j * phase)
    psi = enforce_no_flux_boundary(psi, X, Y, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi


def condensate_with_sound(X, Y, condensate_shape="gaussian", sound_type="random", 
                          sound_amplitude=0.1, sound_wavelength=0.5):
    # acoustic
    L = torch.max(X).item()
    psi = ground_state(X, Y, shape=condensate_shape, width=0.8*L)
    amplitude = torch.abs(psi)
    if sound_type == "random":
        k = 2 * np.pi / sound_wavelength
        sound = sample_gaussian_random_field(X.shape[0], X.shape[1], L, 
                                           power_law=-2.0, filter_scale=k)
        
    elif sound_type == "standing":
        k = 2 * np.pi / sound_wavelength
        kx = 1.0 + 0.5 * torch.rand(1).item()
        ky = 1.0 + 0.5 * torch.rand(1).item()
        sound = torch.cos(kx * k * X) * torch.cos(ky * k * Y)
        
    elif sound_type == "radial":
        k = 2 * np.pi / sound_wavelength
        x0, y0 = 0.2 * L * torch.randn(2)
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        sound = torch.cos(k * r)
    
    psi = amplitude * torch.exp(1j * (torch.angle(psi) + sound_amplitude * sound.real))
    psi = enforce_no_flux_boundary(psi, X, Y, L)
    
    return psi


def combined_state(X, Y, components=None, weights=None):
    L = torch.max(X).item()
    
    if components is None:
        components = ["vortex", "ground", "soliton", "sound"]
    
    if weights is None:
        weights = torch.ones(len(components))
    
    weights = weights / weights.sum()  # Normalize weights
    
    psi = torch.zeros_like(X, dtype=torch.complex128)
    
    for component, weight in zip(components, weights):
        if component == "vortex":
            component_psi = multi_vortex_state(X, Y, num_vortices=random.randint(1, 5))
        
        elif component == "ground":
            shape = random.choice(["gaussian", "thomas_fermi", "plateau"])
            component_psi = ground_state(X, Y, shape=shape)
        
        elif component == "soliton":
            soliton_type = random.choice(["bright", "dark"])
            arrangement = random.choice(["parallel", "crossing", "radial"])
            component_psi = soliton_state(X, Y, soliton_type=soliton_type,
                                        num_solitons=random.randint(1, 3),
                                        arrangement=arrangement)
        
        elif component == "sound":
            sound_type = random.choice(["random", "standing", "radial"])
            component_psi = condensate_with_sound(X, Y, sound_type=sound_type)
        
        elif component == "turbulence":
            component_psi = quantum_turbulence(X, Y, num_vortices=random.randint(10, 30))
        
        elif component == "lattice":
            lattice_type = random.choice(["square", "triangular", "honeycomb"])
            component_psi = lattice_state(X, Y, lattice_type=lattice_type)
        
        elif component == "breather":
            breather_type = random.choice(["kuznetsov", "akhmediev"])
            component_psi = breather_state(X, Y, breather_type=breather_type)
        
        elif component == "rogue":
            component_psi = rogue_wave(X, Y)
        
        else:
            raise ValueError(f"Unknown component: {component}")
        
        psi += weight * component_psi
    psi = enforce_no_flux_boundary(psi, X, Y, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi

def true_vortex_state(X, Y, charge=1, width=1.0, vortex_radius=0.3, center=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0 = L * 0.5 * torch.randn(2)
    else:
        x0, y0 = center

    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)

    amplitude = torch.tanh(r / vortex_radius) * torch.exp(-r**2/(2*width**2))
    phase = charge * theta
    psi = amplitude * torch.exp(1j * phase)

    return psi


def spiral_vortex_state(X, Y, charge=1, width=1.0, spiral_factor=2.0, center=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0 = L * 0.5 * torch.randn(2)
    else:
        x0, y0 = center

    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)

    amplitude = torch.exp(-r**2/(2*width**2))
    phase = charge * (theta + spiral_factor * r)
    psi = amplitude * torch.exp(1j * phase)

    return psi


def multi_spiral_vortex(X, Y, num_vortices=3, arrangement="random", width=2.0, spiral_factor=2.0):
    L = torch.max(X).item()
    psi = torch.zeros_like(X, dtype=torch.complex128)

    centers = []
    charges = []

    if arrangement == "random":
        for _ in range(num_vortices):
            centers.append(L * 0.6 * torch.randn(2))
            charges.append(random.choice([-1, 1]))

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
        psi += spiral_vortex_state(X, Y, charge, width, spiral_variation, center)

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(width*1.2)**2))
    psi = psi * envelope / torch.max(torch.abs(psi))

    return psi


def skyrmion_state(X, Y, width=1.0, skyrmion_radius=0.5, edge_width=0.1, center=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0 = L * 0.5 * torch.randn(2)
    else:
        x0, y0 = center

    r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = torch.atan2(Y - y0, X - x0)

    amplitude = torch.exp(-r**2/(2*width**2))

    z_component = 2 * (0.5 - torch.exp(-(r - skyrmion_radius)**2 / (2*edge_width**2))) - 1
    z_component = torch.clamp(z_component, min=-1.0, max=1.0)

    xy_component = torch.sqrt(1 - z_component**2)

    real_part = xy_component * torch.cos(theta)
    imag_part = xy_component * torch.sin(theta)

    psi = amplitude * (real_part + 1j * imag_part)

    return psi


def multi_skyrmion_state(X, Y, num_skyrmions=3, arrangement="random", width=2.0):
    L = torch.max(X).item()
    psi = torch.zeros_like(X, dtype=torch.complex128)

    centers = []

    if arrangement == "random":
        for _ in range(num_skyrmions):
            centers.append(L * 0.6 * torch.randn(2))

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
        psi += skyrmion_state(X, Y, width, skyrmion_radius, edge_width, center)

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*(width*1.2)**2))
    psi = psi * envelope / torch.max(torch.abs(psi))

    return psi


def vortex_dipole(X, Y, separation=1.0, width=2.0, orientation=0, center=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0 = L * 0.3 * torch.randn(2)
    else:
        x0, y0 = center

    dx = 0.5 * separation * np.cos(orientation)
    dy = 0.5 * separation * np.sin(orientation)

    center1 = [x0 + dx, y0 + dy]
    center2 = [x0 - dx, y0 - dy]

    psi1 = true_vortex_state(X, Y, charge=1, width=width, vortex_radius=0.2, center=center1)
    psi2 = true_vortex_state(X, Y, charge=-1, width=width, vortex_radius=0.2, center=center2)

    psi = psi1 + psi2
    r2 = (X - x0)**2 + (Y - y0)**2
    envelope = torch.exp(-r2/(2*(width*1.2)**2))
    psi = psi * envelope / torch.max(torch.abs(psi))

    return psi


def vortex_lattice(X, Y, lattice_type="square", vortex_density=0.7, alternating=True, width=None):
    L = torch.max(X).item()
    if width is None:
        width = L * 0.8

    grid_size = int(np.sqrt(L * vortex_density))
    spacing = 2 * L / grid_size

    psi = torch.ones_like(X, dtype=torch.complex128)

    centers = []
    charges = []

    if lattice_type == "square":
        for i in range(grid_size):
            for j in range(grid_size):
                x = spacing * (j - grid_size/2 + 0.5)
                y = spacing * (i - grid_size/2 + 0.5)
                centers.append([x, y])
                if alternating:
                    charges.append(1 if (i + j) % 2 == 0 else -1)
                else:
                    charges.append(1)

    elif lattice_type == "triangular":
        for i in range(grid_size):
            for j in range(grid_size):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                x = spacing * (j - grid_size/2 + 0.5) + offset
                y = spacing * (i - grid_size/2 + 0.5)
                centers.append([x, y])
                if alternating:
                    charges.append(1 if (i + j) % 2 == 0 else -1)
                else:
                    charges.append(1)

    elif lattice_type == "honeycomb":
        for i in range(grid_size):
            for j in range(grid_size):
                offset = spacing * 0.5 if i % 2 == 1 else 0
                x_base = spacing * (j - grid_size/2 + 0.5) + offset
                y_base = spacing * (i - grid_size/2 + 0.5)

                centers.append([x_base - spacing/4, y_base])
                centers.append([x_base + spacing/4, y_base])

                if alternating:
                    charges.append(1)
                    charges.append(-1)
                else:
                    charges.append(1)
                    charges.append(1)

    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    for center, charge in zip(centers, charges):
        x0, y0 = center
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        theta = torch.atan2(Y - y0, X - x0)

        core_size = 0.1 * spacing
        amplitude_factor = torch.tanh(r / core_size)
        phase_factor = torch.exp(1j * charge * theta)

        psi = psi * (amplitude_factor * phase_factor + (1 - amplitude_factor))

    r2 = X**2 + Y**2
    envelope = torch.exp(-r2/(2*width**2))
    psi = psi * envelope
    psi = psi / torch.max(torch.abs(psi))

    return psi

"""
def sample_nlse_initial_condition(Nx, Ny, L, condition_type='auto'):
    X, Y = make_grid(Nx, Ny, L)
    
    if condition_type == 'auto':
        condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton',
                         'lattice', 'turbulence', 'rogue', 'breather',
                         'sound', 'combined']
        condition_type = random.choice(condition_types)
    
    if condition_type == 'ground':
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
        
    elif condition_type == 'turbulence':
        energy_spectrum = random.choice(['kolmogorov', 'white'])
        num_vortices = random.randint(10, 50)
        psi = quantum_turbulence(X, Y, energy_spectrum=energy_spectrum,
                               num_vortices=num_vortices)
        description = f"Quantum turbulence ({energy_spectrum}, {num_vortices} vortices)"
        
    elif condition_type == 'rogue':
        peak_amplitude = 2.0 + 2.0 * torch.rand(1).item()
        psi = rogue_wave(X, Y, peak_amplitude=peak_amplitude)
        description = f"Rogue wave (amplitude={peak_amplitude:.1f})"
        
    elif condition_type == 'breather':
        breather_type = random.choice(['kuznetsov', 'akhmediev'])
        width = 0.2*L + 0.4*L*torch.rand(1).item()
        oscillation = 1.0 + 3.0 * torch.rand(1).item()
        psi = breather_state(X, Y, breather_type=breather_type,
                           width=width, oscillation=oscillation)
        description = f"{breather_type.capitalize()} breather"
        
    elif condition_type == 'sound':
        condensate_shape = random.choice(['gaussian', 'thomas_fermi'])
        sound_type = random.choice(['random', 'standing', 'radial'])
        sound_amplitude = 0.05 + 0.15 * torch.rand(1).item()
        sound_wavelength = 0.2 + 0.8 * torch.rand(1).item()
        
        psi = condensate_with_sound(X, Y, condensate_shape=condensate_shape,
                                  sound_type=sound_type, sound_amplitude=sound_amplitude,
                                  sound_wavelength=sound_wavelength)
        
        description = f"Condensate with {sound_type} sound"
        
    elif condition_type == 'combined':
        all_types = ['vortex', 'ground', 'soliton', 'sound', 'turbulence', 
                    'lattice', 'breather']
        num_types = random.randint(2, 4)
        selected_types = random.sample(all_types, k=num_types)
        weights = torch.rand(num_types)
        
        psi = combined_state(X, Y, components=selected_types, weights=weights)
        description = f"Combined state ({'+'.join(selected_types)})"
        
    else:
        raise ValueError(f"Unknown condition type: {condition_type}")
    
    #psi = enforce_no_flux_boundary(psi, X, Y, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi, description
"""
def sample_nlse_initial_condition(Nx, Ny, L, condition_type='auto'):
    X, Y = make_grid(Nx, Ny, L)

    if condition_type == 'auto':
        condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton',
                         'lattice', 'turbulence', 'rogue', 'breather',
                         'sound', 'combined', 'true_vortex', 'spiral_vortex',
                         'multi_spiral', 'skyrmion', 'multi_skyrmion',
                         'vortex_dipole', 'vortex_lattice']
        condition_type = random.choice(condition_types)

    if condition_type == 'true_vortex':
        charge = random.choice([-2, -1, 1, 2])
        width = 0.2*L + 0.5*L*torch.rand(1).item()
        vortex_radius = 0.1*L + 0.2*L*torch.rand(1).item()
        psi = true_vortex_state(X, Y, charge=charge, width=width, vortex_radius=vortex_radius)
        description = f"True vortex (charge={charge})"

    elif condition_type == 'spiral_vortex':
        charge = random.choice([-2, -1, 1, 2])
        width = 0.2*L + 0.5*L*torch.rand(1).item()
        spiral_factor = 0.5 + 3.0*torch.rand(1).item()
        psi = spiral_vortex_state(X, Y, charge=charge, width=width, spiral_factor=spiral_factor)
        description = f"Spiral vortex (charge={charge}, spiral={spiral_factor:.1f})"

    elif condition_type == 'multi_spiral':
        num_vortices = random.randint(2, 8)
        arrangement = random.choice(['random', 'ring', 'lattice'])
        spiral_factor = 0.5 + 3.0*torch.rand(1).item()
        psi = multi_spiral_vortex(X, Y, num_vortices=num_vortices,
                                arrangement=arrangement, spiral_factor=spiral_factor)
        description = f"Multi-spiral vortices ({num_vortices}, {arrangement})"

    elif condition_type == 'skyrmion':
        width = 0.2*L + 0.5*L*torch.rand(1).item()
        skyrmion_radius = 0.1*L + 0.3*L*torch.rand(1).item()
        edge_width = 0.05*L + 0.1*L*torch.rand(1).item()
        psi = skyrmion_state(X, Y, width=width, skyrmion_radius=skyrmion_radius, edge_width=edge_width)
        description = f"Skyrmion (radius={skyrmion_radius:.1f})"

    elif condition_type == 'multi_skyrmion':
        num_skyrmions = random.randint(2, 8)
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
        description = f"Vortex dipole (separation={separation:.1f})"

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
        
    elif condition_type == 'turbulence':
        energy_spectrum = random.choice(['kolmogorov', 'white'])
        num_vortices = random.randint(10, 50)
        psi = quantum_turbulence(X, Y, energy_spectrum=energy_spectrum,
                               num_vortices=num_vortices)
        description = f"Quantum turbulence ({energy_spectrum}, {num_vortices} vortices)"
        
    elif condition_type == 'rogue':
        peak_amplitude = 2.0 + 2.0 * torch.rand(1).item()
        psi = rogue_wave(X, Y, peak_amplitude=peak_amplitude)
        description = f"Rogue wave (amplitude={peak_amplitude:.1f})"
        
    elif condition_type == 'breather':
        breather_type = random.choice(['kuznetsov', 'akhmediev'])
        width = 0.2*L + 0.4*L*torch.rand(1).item()
        oscillation = 1.0 + 3.0 * torch.rand(1).item()
        psi = breather_state(X, Y, breather_type=breather_type,
                           width=width, oscillation=oscillation)
        description = f"{breather_type.capitalize()} breather"
        
    elif condition_type == 'sound':
        condensate_shape = random.choice(['gaussian', 'thomas_fermi'])
        sound_type = random.choice(['random', 'standing', 'radial'])
        sound_amplitude = 0.05 + 0.15 * torch.rand(1).item()
        sound_wavelength = 0.2 + 0.8 * torch.rand(1).item()
        
        psi = condensate_with_sound(X, Y, condensate_shape=condensate_shape,
                                  sound_type=sound_type, sound_amplitude=sound_amplitude,
                                  sound_wavelength=sound_wavelength)
        
        description = f"Condensate with {sound_type} sound"
        
    elif condition_type == 'combined':
        all_types = ['vortex', 'ground', 'soliton', 'sound', 'turbulence', 
                    'lattice', 'breather']
        num_types = random.randint(2, 4)
        selected_types = random.sample(all_types, k=num_types)
        weights = torch.rand(num_types)
        
        psi = combined_state(X, Y, components=selected_types, weights=weights)
        description = f"Combined state ({'+'.join(selected_types)})"
        

    else:
        raise NotImplemented

    psi = enforce_no_flux_boundary(psi, X, Y, L)
    psi = psi / torch.max(torch.abs(psi))

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
    Nx, Ny = 128, 128
    L = 10.0
   
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    
    condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton', 
                      'lattice', 'turbulence', 'rogue', 'breather', 'combined']
    
    for i, ctype in enumerate(condition_types):
        psi, description = sample_nlse_initial_condition(Nx, Ny, L, ctype)
        axs[i].imshow(torch.abs(psi), cmap='viridis')
        axs[i].set_title(description)
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()
