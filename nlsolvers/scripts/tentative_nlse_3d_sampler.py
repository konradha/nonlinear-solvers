import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.fft as fft
from matplotlib import cm
from skimage import measure

def make_grid_3d(Nx, Ny, Nz, L):
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    z = torch.linspace(-L, L, Nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z

def sample_gaussian_random_field_3d(Nx, Ny, Nz, L, power_law=-3.0, filter_scale=5.0):
    kx = 2*np.pi*fft.fftfreq(Nx, d=2*L/Nx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=2*L/Ny)
    kz = 2*np.pi*fft.fftfreq(Nz, d=2*L/Nz)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K = torch.sqrt(KX**2 + KY**2 + KZ**2)
    power = K**power_law
    power[0, 0, 0] = 0 
    power = power * torch.exp(-(K/filter_scale)**2)
    noise_real = torch.randn(Nx, Ny, Nz)
    noise_imag = torch.randn(Nx, Ny, Nz)
    noise = noise_real + 1j*noise_imag
    field = fft.ifftn(fft.fftn(noise) * torch.sqrt(power))
    field = field / torch.std(torch.abs(field))
    
    return field

def enforce_no_flux_boundary_3d(psi, X, Y, Z, L, width=0.1):
    envelope = (1 - torch.exp(-(L - torch.abs(X))/(width*L))) * \
               (1 - torch.exp(-(L - torch.abs(Y))/(width*L))) * \
               (1 - torch.exp(-(L - torch.abs(Z))/(width*L)))
    return psi * envelope

def ground_state_3d(X, Y, Z, shape="gaussian", width=1.0, center=None, phase_pattern=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0, z0 = L * 0.5 * torch.randn(3)
    else:
        x0, y0, z0 = center
    
    r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
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
        phase = phase_pattern(X, Y, Z, x0, y0, z0)
    
    psi = amplitude * torch.exp(1j * phase)
    
    return psi

def vortex_phase_3d(X, Y, Z, x0, y0, z0, vortex_type="line", charge=1):
    if vortex_type == "line":
        return charge * torch.atan2(Y - y0, X - x0)
    elif vortex_type == "ring":
        r_xy = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        z_dist = Z - z0
        r_to_ring = torch.sqrt((r_xy - 0.5*torch.max(X).item())**2 + z_dist**2)
        theta = torch.atan2(z_dist, r_xy - 0.5*torch.max(X).item())
        return charge * theta
    else:
        return torch.zeros_like(X)

def vortex_state_3d(X, Y, Z, vortex_type="line", charge=1, width=1.0, core_size=0.2, center=None):
    if center is None:
        L = torch.max(X).item()
        x0, y0, z0 = L * 0.5 * torch.randn(3)
    else:
        x0, y0, z0 = center
    
    if vortex_type == "line":
        r_xy = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        amplitude = r_xy / torch.sqrt(r_xy**2 + core_size**2)
        amplitude = amplitude * torch.exp(-(r_xy**2 + (Z - z0)**2)/(2*width**2))
        phase = vortex_phase_3d(X, Y, Z, x0, y0, z0, vortex_type, charge)
    
    elif vortex_type == "ring":
        r_xy = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        z_dist = Z - z0
        r_to_ring = torch.sqrt((r_xy - 0.5*torch.max(X).item())**2 + z_dist**2)
        amplitude = r_to_ring / torch.sqrt(r_to_ring**2 + core_size**2)
        amplitude = amplitude * torch.exp(-r_to_ring**2/(2*width**2))
        phase = vortex_phase_3d(X, Y, Z, x0, y0, z0, vortex_type, charge)
    
    elif vortex_type == "knot":
        r_major = 0.5 * torch.max(X).item()
        r_minor = 0.2 * torch.max(X).item()
        theta = torch.atan2(Y - y0, X - x0)
        
        knot_x = x0 + r_major * torch.cos(theta)
        knot_y = y0 + r_major * torch.sin(theta)
        knot_z = z0 + r_minor * torch.sin(2 * theta)
        
        r_to_knot = torch.sqrt((X - knot_x)**2 + (Y - knot_y)**2 + (Z - knot_z)**2)
        amplitude = r_to_knot / torch.sqrt(r_to_knot**2 + core_size**2)
        amplitude = amplitude * torch.exp(-r_to_knot**2/(2*width**2))
        
        phase = torch.atan2(Z - knot_z, torch.sqrt((X - knot_x)**2 + (Y - knot_y)**2))
        phase = charge * phase
    
    psi = amplitude * torch.exp(1j * phase)
    
    return psi

def multi_vortex_state_3d(X, Y, Z, num_vortices=3, arrangement="random", width=2.0):
    L = torch.max(X).item()
    psi = torch.zeros_like(X, dtype=torch.complex128)
    
    centers = []
    charges = []
    vortex_types = []
    
    if arrangement == "random":
        for _ in range(num_vortices):
            centers.append(L * 0.6 * torch.randn(3))
            charges.append(random.choice([-1, 1]))
            vortex_types.append(random.choice(["line", "ring"]))
    
    elif arrangement == "grid":
        count = 0
        side = int(np.ceil(np.cbrt(num_vortices)))
        spacing = width * 0.8 / side
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if count < num_vortices:
                        centers.append([spacing * (i - side/2), 
                                       spacing * (j - side/2),
                                       spacing * (k - side/2)])
                        charges.append(1 if (i + j + k) % 2 == 0 else -1)
                        vortex_types.append("line")
                        count += 1
    
    elif arrangement == "tangle":
        for i in range(num_vortices):
            if i < num_vortices // 2:
                centers.append([0, 0, L * 0.8 * (i - num_vortices/4) / (num_vortices/4)])
                charges.append(1)
                vortex_types.append("line")
            else:
                angle = 2 * np.pi * (i - num_vortices//2) / (num_vortices - num_vortices//2)
                radius = 0.6 * L
                centers.append([radius * np.cos(angle), radius * np.sin(angle), 0])
                charges.append(-1)
                vortex_types.append("line")
    
    for center, charge, v_type in zip(centers, charges, vortex_types):
        core_size = 0.1 + 0.2 * torch.rand(1).item()
        psi += vortex_state_3d(X, Y, Z, vortex_type=v_type, charge=charge, 
                             width=width, core_size=core_size, center=center)
    
    r2 = X**2 + Y**2 + Z**2
    envelope = torch.exp(-r2/(2*(width*1.5)**2))
    psi = psi * envelope / torch.max(torch.abs(psi))
    
    return psi

def soliton_state_3d(X, Y, Z, soliton_type="bright", num_solitons=1, width=1.0, 
                    arrangement="parallel", velocity=None):
    L = torch.max(X).item()
    
    if soliton_type == "dark":
        psi = torch.ones_like(X, dtype=torch.complex128)
    else:
        psi = torch.zeros_like(X, dtype=torch.complex128)
    
    distances = []
    direction_x = []
    direction_y = []
    direction_z = []
    centers = []
    
    if arrangement == "parallel":
        spacing = 2.0 * width
        for i in range(num_solitons):
            distances.append(spacing * (i - (num_solitons-1)/2))
            direction_x.append(1)
            direction_y.append(0)
            direction_z.append(0)
            centers.append([0, 0, 0])
    
    elif arrangement == "crossing":
        for i in range(num_solitons):
            distances.append(width * torch.randn(1).item())
            theta = 2 * np.pi * i / num_solitons
            phi = np.pi * torch.rand(1).item()
            direction_x.append(np.sin(phi) * np.cos(theta))
            direction_y.append(np.sin(phi) * np.sin(theta))
            direction_z.append(np.cos(phi))
            centers.append([L * 0.3 * torch.randn(1).item() for _ in range(3)])
    
    elif arrangement == "radial":
        central_point = [L * 0.3 * torch.randn(1).item() for _ in range(3)]
        for i in range(num_solitons):
            theta = 2 * np.pi * i / num_solitons
            phi = np.pi * torch.rand(1).item()
            dir_x = np.sin(phi) * np.cos(theta)
            dir_y = np.sin(phi) * np.sin(theta)
            dir_z = np.cos(phi)
            distances.append(0)
            direction_x.append(dir_x)
            direction_y.append(dir_y)
            direction_z.append(dir_z)
            centers.append(central_point)
    
    for d, dx, dy, dz, center in zip(distances, direction_x, direction_y, direction_z, centers):
        cx, cy, cz = center
        X_rot = (X - cx) * dx + (Y - cy) * dy + (Z - cz) * dz
        
        if soliton_type == "bright":
            profile = 1.0 / torch.cosh((X_rot - d) / width)
        else:
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
    
    r2 = X**2 + Y**2 + Z**2
    if soliton_type == "bright":
        envelope = torch.exp(-r2/(2*(L*0.8)**2))
        psi = psi * envelope / torch.max(torch.abs(psi))
    else:
        envelope = 0.5 * (1 + torch.tanh((2*L - torch.sqrt(r2))/(0.2*L)))
        psi = psi * envelope
    
    return psi

def lattice_state_3d(X, Y, Z, lattice_type="cubic", num_sites=64, amplitude_var=0.2, phase_var=0.5):
    L = torch.max(X).item()
    sites_per_side = int(np.round(num_sites**(1/3)))
    centers = []
    
    if lattice_type == "cubic":
        spacing = 2 * L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                for k in range(sites_per_side):
                    centers.append([spacing * (i - sites_per_side/2 + 0.5), 
                                   spacing * (j - sites_per_side/2 + 0.5),
                                   spacing * (k - sites_per_side/2 + 0.5)])
    
    elif lattice_type == "bcc":
        spacing = 2 * L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                for k in range(sites_per_side):
                    if (i + j + k) % 2 == 0:
                        centers.append([spacing * (i - sites_per_side/2 + 0.5), 
                                       spacing * (j - sites_per_side/2 + 0.5),
                                       spacing * (k - sites_per_side/2 + 0.5)])
                    else:
                        centers.append([spacing * (i - sites_per_side/2 + 0.5) + spacing/2, 
                                       spacing * (j - sites_per_side/2 + 0.5) + spacing/2,
                                       spacing * (k - sites_per_side/2 + 0.5) + spacing/2])
    
    elif lattice_type == "fcc":
        spacing = 2 * L / sites_per_side
        for i in range(sites_per_side):
            for j in range(sites_per_side):
                for k in range(sites_per_side):
                    centers.append([spacing * (i - sites_per_side/2 + 0.5), 
                                   spacing * (j - sites_per_side/2 + 0.5),
                                   spacing * (k - sites_per_side/2 + 0.5)])
                    centers.append([spacing * (i - sites_per_side/2 + 0.5) + spacing/2, 
                                   spacing * (j - sites_per_side/2 + 0.5) + spacing/2,
                                   spacing * (k - sites_per_side/2 + 0.5)])
                    centers.append([spacing * (i - sites_per_side/2 + 0.5) + spacing/2, 
                                   spacing * (j - sites_per_side/2 + 0.5),
                                   spacing * (k - sites_per_side/2 + 0.5) + spacing/2])
                    centers.append([spacing * (i - sites_per_side/2 + 0.5), 
                                   spacing * (j - sites_per_side/2 + 0.5) + spacing/2,
                                   spacing * (k - sites_per_side/2 + 0.5) + spacing/2])
    
    psi = torch.zeros_like(X, dtype=torch.complex128)
    width = L / sites_per_side * 0.4
    
    for center in centers:
        x0, y0, z0 = center
        r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
        amplitude = 1.0 + amplitude_var * (2 * torch.rand(1).item() - 1)
        phase = phase_var * 2 * np.pi * torch.rand(1)
        psi += amplitude * torch.exp(-r2/(2*width**2)) * torch.exp(1j * phase)
    
    psi = psi / torch.max(torch.abs(psi))
    
    return psi

def quantum_turbulence_3d(X, Y, Z, energy_spectrum="kolmogorov", num_vortices=20):
    L = torch.max(X).item()
    psi = multi_vortex_state_3d(X, Y, Z, num_vortices=num_vortices, arrangement="random", width=L*0.8)
    
    power_law = -5/3 if energy_spectrum == "kolmogorov" else -2
    fluctuations = sample_gaussian_random_field_3d(X.shape[0], X.shape[1], X.shape[2], 
                                                 L, power_law=power_law)
    phase_fluctuation_strength = 0.3
    psi = psi * torch.exp(1j * phase_fluctuation_strength * fluctuations.real)
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi

def rogue_wave_3d(X, Y, Z, background_amplitude=0.5, peak_amplitude=3.0):
    L = torch.max(X).item()
    x0, y0, z0 = 0.2 * L * torch.randn(3)
    r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    
    xi = r2 / (4 * L**2)
    peregrine = background_amplitude * (1.0 - (4.0 / (1.0 + 4.0 * xi)) * torch.exp(torch.tensor(1j * np.pi/2)))
    background_phase = 0.2 * torch.randn_like(X)
    psi = peregrine * torch.exp(1j * background_phase)
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    psi = psi * (peak_amplitude / torch.max(torch.abs(psi)))
    
    return psi

def breather_state_3d(X, Y, Z, breather_type="kuznetsov", width=1.0, oscillation=2.0):
    L = torch.max(X).item()
    x0, y0, z0 = 0.2 * L * torch.randn(3)
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)
    
    if breather_type == "kuznetsov":
        a = torch.tensor(0.5 / width) 
        b = torch.tensor(oscillation)
        
        amplitude = 1.0 / torch.cosh(r / width)
        modulation = torch.cos(b) + np.sqrt(2) * torch.sinh(a * r) / torch.cosh(a * r)
        psi = amplitude * modulation
        
    else:
        a = torch.tensor(2.0)
        b = torch.tensor(oscillation) 
        c = torch.tensor(0.5 / width)
        theta = torch.atan2(Y - y0, X - x0)

        modulation = 1.0 + a * torch.cos(b * theta) / torch.cosh(c * r)
        envelope = torch.exp(-r**2 / (4 * L**2))
        psi = modulation * envelope
    
    phase = 0.5 * (X + Y + Z)
    psi = psi * torch.exp(1j * phase)
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi

def condensate_with_sound_3d(X, Y, Z, condensate_shape="gaussian", sound_type="random", 
                           sound_amplitude=0.1, sound_wavelength=0.5):
    L = torch.max(X).item()
    psi = ground_state_3d(X, Y, Z, shape=condensate_shape, width=0.8*L)
    amplitude = torch.abs(psi)
    
    if sound_type == "random":
        k = 2 * np.pi / sound_wavelength
        sound = sample_gaussian_random_field_3d(X.shape[0], X.shape[1], X.shape[2], 
                                              L, power_law=-2.0, filter_scale=k)
        
    elif sound_type == "standing":
        k = 2 * np.pi / sound_wavelength
        kx = 1.0 + 0.5 * torch.rand(1).item()
        ky = 1.0 + 0.5 * torch.rand(1).item()
        kz = 1.0 + 0.5 * torch.rand(1).item()
        sound = torch.cos(kx * k * X) * torch.cos(ky * k * Y) * torch.cos(kz * k * Z)
        
    elif sound_type == "radial":
        k = 2 * np.pi / sound_wavelength
        x0, y0, z0 = 0.2 * L * torch.randn(3)
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)
        sound = torch.cos(k * r)
    
    psi = amplitude * torch.exp(torch.tensor(1j * (torch.angle(psi) + sound_amplitude * sound.real)))
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    
    return psi

def droplet_state_3d(X, Y, Z, num_droplets=1, width=1.0):
    L = torch.max(X).item()
    psi = torch.zeros_like(X, dtype=torch.complex128)
    
    for _ in range(num_droplets):
        x0, y0, z0 = L * 0.6 * torch.randn(3)
        r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
        
        profile = torch.exp(-r2/(2*width**2)) * torch.sqrt(torch.clamp(1 - r2/width**2, min=0.0))
        
        velocity = 0.5 * torch.randn(3)
        phase = velocity[0] * (X - x0) + velocity[1] * (Y - y0) + velocity[2] * (Z - z0)
        
        psi += profile * torch.exp(1j * phase)
    
    psi = psi / torch.max(torch.abs(psi))
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    
    return psi

def combined_state_3d(X, Y, Z, components=None, weights=None):
    L = torch.max(X).item()
    
    if components is None:
        components = ["vortex", "ground", "soliton", "sound", "droplet"]
    
    if weights is None:
        weights = torch.ones(len(components))
    
    weights = weights / weights.sum()
    
    psi = torch.zeros_like(X, dtype=torch.complex128)
    
    for component, weight in zip(components, weights):
        if component == "vortex":
            vortex_type = random.choice(["line", "ring", "knot"])
            component_psi = multi_vortex_state_3d(X, Y, Z, num_vortices=random.randint(1, 5))
        
        elif component == "ground":
            shape = random.choice(["gaussian", "thomas_fermi", "plateau"])
            component_psi = ground_state_3d(X, Y, Z, shape=shape)
        
        elif component == "soliton":
            soliton_type = random.choice(["bright", "dark"])
            arrangement = random.choice(["parallel", "crossing", "radial"])
            component_psi = soliton_state_3d(X, Y, Z, soliton_type=soliton_type,
                                           num_solitons=random.randint(1, 3),
                                           arrangement=arrangement)
        
        elif component == "sound":
            sound_type = random.choice(["random", "standing", "radial"])
            component_psi = condensate_with_sound_3d(X, Y, Z, sound_type=sound_type)
        
        elif component == "turbulence":
            component_psi = quantum_turbulence_3d(X, Y, Z, num_vortices=random.randint(10, 30))
        
        elif component == "lattice":
            lattice_type = random.choice(["cubic", "bcc", "fcc"])
            component_psi = lattice_state_3d(X, Y, Z, lattice_type=lattice_type)
        
        elif component == "breather":
            breather_type = random.choice(["kuznetsov", "akhmediev"])
            component_psi = breather_state_3d(X, Y, Z, breather_type=breather_type)
        
        elif component == "rogue":
            component_psi = rogue_wave_3d(X, Y, Z)
            
        elif component == "droplet":
            component_psi = droplet_state_3d(X, Y, Z, num_droplets=random.randint(1, 3))
        
        else:
            raise ValueError(f"Unknown component: {component}")
        
        psi += weight * component_psi
    
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi

def sample_nlse_initial_condition_3d(Nx, Ny, Nz, L, condition_type='auto'):
    X, Y, Z = make_grid_3d(Nx, Ny, Nz, L)
    
    if condition_type == 'auto':
        condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton',
                         'lattice', 'turbulence', 'rogue', 'breather',
                         'sound', 'droplet', 'combined']
        condition_type = random.choice(condition_types)
    
    if condition_type == 'ground':
        shape = random.choice(['gaussian', 'thomas_fermi', 'plateau'])
        width = 0.2*L + 0.6*L*torch.rand(1).item()
        
        if torch.rand(1).item() < 0.3:
            def phase_pattern(X, Y, Z, x0, y0, z0):
                vx, vy, vz = torch.randn(3)
                return vx * (X - x0) + vy * (Y - y0) + vz * (Z - z0)
        else:
            phase_pattern = None
            
        psi = ground_state_3d(X, Y, Z, shape=shape, width=width, phase_pattern=phase_pattern)
        description = f"Ground state ({shape})"
        
    elif condition_type == 'vortex':
        vortex_type = random.choice(['line', 'ring', 'knot'])
        charge = random.choice([-2, -1, 1, 2])
        width = 0.2*L + 0.5*L*torch.rand(1).item()
        psi = vortex_state_3d(X, Y, Z, vortex_type=vortex_type, charge=charge, width=width)
        description = f"Vortex ({vortex_type}, charge={charge})"
        
    elif condition_type == 'multi_vortex':
        num_vortices = random.randint(2, 8)
        arrangement = random.choice(['random', 'grid', 'tangle'])
        psi = multi_vortex_state_3d(X, Y, Z, num_vortices=num_vortices, arrangement=arrangement)
        description = f"Multi-vortex ({num_vortices}, {arrangement})"
        
    elif condition_type == 'soliton':
        soliton_type = random.choice(['bright', 'dark'])
        num_solitons = random.randint(1, 4)
        arrangement = random.choice(['parallel', 'crossing', 'radial'])
        if torch.rand(1).item() < 0.5:
            velocity = 2.0 * torch.rand(1).item()
        else:
            velocity = None
            
        psi = soliton_state_3d(X, Y, Z, soliton_type=soliton_type, num_solitons=num_solitons,
                             arrangement=arrangement, velocity=velocity)
        
        vel_str = f", v={velocity:.1f}" if velocity is not None else ""
        description = f"{soliton_type.capitalize()} soliton ({num_solitons}, {arrangement}{vel_str})"
        
    elif condition_type == 'lattice':
        lattice_type = random.choice(['cubic', 'bcc', 'fcc'])
        num_sites = random.randint(27, 125)
        psi = lattice_state_3d(X, Y, Z, lattice_type=lattice_type, num_sites=num_sites)
        description = f"Lattice ({lattice_type}, {num_sites} sites)"
        
    elif condition_type == 'turbulence':
        energy_spectrum = random.choice(['kolmogorov', 'white'])
        num_vortices = random.randint(10, 50)
        psi = quantum_turbulence_3d(X, Y, Z, energy_spectrum=energy_spectrum,
                                  num_vortices=num_vortices)
        description = f"Quantum turbulence ({energy_spectrum}, {num_vortices} vortices)"
        
    elif condition_type == 'rogue':
        peak_amplitude = 2.0 + 2.0 * torch.rand(1).item()
        psi = rogue_wave_3d(X, Y, Z, peak_amplitude=peak_amplitude)
        description = f"Rogue wave (amplitude={peak_amplitude:.1f})"
        
    elif condition_type == 'breather':
        breather_type = random.choice(['kuznetsov', 'akhmediev'])
        width = 0.2*L + 0.4*L*torch.rand(1).item()
        oscillation = 1.0 + 3.0 * torch.rand(1).item()
        psi = breather_state_3d(X, Y, Z, breather_type=breather_type,
                              width=width, oscillation=oscillation)
        description = f"{breather_type.capitalize()} breather"
        
    elif condition_type == 'sound':
        condensate_shape = random.choice(['gaussian', 'thomas_fermi'])
        sound_type = random.choice(['random', 'standing', 'radial'])
        sound_amplitude = 0.05 + 0.15 * torch.rand(1).item()
        sound_wavelength = 0.2 + 0.8 * torch.rand(1).item()
        
        psi = condensate_with_sound_3d(X, Y, Z, condensate_shape=condensate_shape,
                                     sound_type=sound_type, sound_amplitude=sound_amplitude,
                                     sound_wavelength=sound_wavelength)
        
        description = f"Condensate with {sound_type} sound"
        
    elif condition_type == 'droplet':
        num_droplets = random.randint(1, 4)
        width = 0.2*L + 0.4*L*torch.rand(1).item()
        psi = droplet_state_3d(X, Y, Z, num_droplets=num_droplets, width=width)
        description = f"Droplet state ({num_droplets})"
        
    elif condition_type == 'combined':
        all_types = ['vortex', 'ground', 'soliton', 'sound', 'turbulence', 
                    'lattice', 'breather', 'droplet']
        num_types = random.randint(2, 4)
        selected_types = random.sample(all_types, k=num_types)
        weights = torch.rand(num_types)
        
        psi = combined_state_3d(X, Y, Z, components=selected_types, weights=weights)
        description = f"Combined state ({'+'.join(selected_types)})"
        
    else:
        raise ValueError(f"Unknown condition type: {condition_type}")
    
    m = torch.ones_like(X)
    anisotropy = sample_gaussian_random_field_3d(X.shape[0], X.shape[1], X.shape[2], L, 
                                               power_law=-3.0, filter_scale=5.0)
    m = 1.0 + 0.5 * torch.nn.functional.relu(anisotropy.real)
    
    psi = enforce_no_flux_boundary_3d(psi, X, Y, Z, L)
    psi = psi / torch.max(torch.abs(psi))
    
    return psi, m, description

def visualize_isosurfaces(psi, m, description, L):
    fig = plt.figure(figsize=(15, 5))
    
    psi_abs = torch.abs(psi).numpy()
    psi_angle = torch.angle(psi).numpy()
    m_np = m.numpy()
    
    verts_to_coords = lambda verts: verts / np.array(psi.shape) * 2*L - L
    
    amp_min, amp_max = psi_abs.min(), psi_abs.max()
    phase_min, phase_max = psi_angle.min(), psi_angle.max()
    m_min, m_max = m_np.min(), m_np.max()
    
    level_amp = amp_min + 0.6 * (amp_max - amp_min)
    level_m = m_min + 0.7 * (m_max - m_min)
    
    ax1 = fig.add_subplot(131, projection='3d')
    if amp_max > amp_min:
        try:
            verts, faces, _, _ = measure.marching_cubes(psi_abs, level=level_amp)
            verts = verts_to_coords(verts)
            ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                           cmap='viridis', lw=0, alpha=0.8)
        except (ValueError, RuntimeError):
            pass
    ax1.set_title(f'{description} (|ψ|)')
    
    ax2 = fig.add_subplot(132, projection='3d')
    if phase_max > phase_min:
        phase_mask = psi_abs > 0.4 * amp_max
        masked_phase = psi_angle.copy()
        masked_phase[~phase_mask] = 0
        
        if np.sum(phase_mask) > 0:
            unique_values = np.unique(masked_phase[phase_mask])
            if len(unique_values) > 0:
                level_phase = unique_values[len(unique_values) // 2]
                try:
                    verts, faces, _, _ = measure.marching_cubes(masked_phase, level=level_phase)
                    verts = verts_to_coords(verts)
                    ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                                   cmap='twilight', lw=0, alpha=0.8)
                except (ValueError, RuntimeError):
                    pass
    ax2.set_title(f'{description} (phase)')
    
    ax3 = fig.add_subplot(133, projection='3d')
    if m_max > m_min:
        try:
            verts, faces, _, _ = measure.marching_cubes(m_np, level=level_m)
            verts = verts_to_coords(verts)
            ax3.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                           cmap='plasma', lw=0, alpha=0.8)
        except (ValueError, RuntimeError):
            pass
    ax3.set_title(f'{description} (m)')
    
    plt.tight_layout()
    return fig

def main():
    Nx, Ny, Nz = 64, 64, 64
    L = 10.0
    
    condition_types = ['ground', 'vortex', 'multi_vortex', 'soliton',
                      'lattice', 'turbulence', 'rogue', 'breather', 'sound', 'droplet']
    
    #for condition_type in condition_types:
    #    psi, m, description = sample_nlse_initial_condition_3d(Nx, Ny, Nz, L, condition_type)
    #    fig = visualize_isosurfaces(psi, m, description, L)
    #    plt.show()
    #    plt.close(fig)
    
    for i in range(2):
        psi, m, description = sample_nlse_initial_condition_3d(Nx, Ny, Nz, L)
        fig = visualize_isosurfaces(psi, m, description, L)
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    main()
