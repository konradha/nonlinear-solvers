import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch.fft as fft
from matplotlib import cm
from skimage import measure

def sample_anisotropic_grf_3d(Nx, Ny, Nz, L, length_scale=1.0, anisotropy_ratio=2.0, theta=30.0, phi=45.0, power=2.0):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    ell_x = length_scale * np.sqrt(anisotropy_ratio)
    ell_y = length_scale / np.sqrt(anisotropy_ratio)
    ell_z = length_scale * (0.5 + 0.5 * np.sqrt(anisotropy_ratio))
    
    kx = 2*np.pi*fft.fftfreq(Nx, d=2 * L/Nx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=2 * L/Ny)
    kz = 2*np.pi*fft.fftfreq(Nz, d=2 * L/Nz)
    
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    
    KX_rot = KX*np.cos(theta_rad)*np.cos(phi_rad) - KY*np.sin(theta_rad) + KZ*np.cos(theta_rad)*np.sin(phi_rad)
    KY_rot = KX*np.sin(theta_rad)*np.cos(phi_rad) + KY*np.cos(theta_rad) + KZ*np.sin(theta_rad)*np.sin(phi_rad)
    KZ_rot = -KX*np.sin(phi_rad) + KZ*np.cos(phi_rad)
    
    spectrum = torch.exp(-((KX_rot/ell_x)**2 + (KY_rot/ell_y)**2 + (KZ_rot/ell_z)**2)**(power/2))
    noise = torch.randn(Nx, Ny, Nz) + 1j*torch.randn(Nx, Ny, Nz)
    field = fft.ifftn(fft.fftn(noise) * torch.sqrt(spectrum)).real
    
    return field / torch.std(field)

def sample_wavelet_superposition_3d(Nx, Ny, Nz, L, n_wavelets=20, scale_range=(0.1, 2.0), kappa=0.5, freq_range=(0.5, 3.0)):
    v0 = torch.zeros(Nx, Ny, Nz)
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    z = torch.linspace(-L, L, Nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    for _ in range(n_wavelets):
        scale = scale_range[0] + (scale_range[1]-scale_range[0])*torch.rand(1)
        theta = 2*np.pi*torch.rand(1)
        phi = np.pi*torch.rand(1)
        x0 = L*(torch.rand(1)-0.5)
        y0 = L*(torch.rand(1)-0.5)
        z0 = L*(torch.rand(1)-0.5)
        k0 = (freq_range[0] + (freq_range[1]-freq_range[0])*torch.rand(1)) * (2*np.pi/(scale*L))
        
        envelope = torch.exp(-((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)/(2*(scale*L)**2))
        
        wavelet_type = random.choice(['cosine', 'gaussian_deriv', 'morlet'])
        
        nx = torch.cos(phi)*torch.cos(theta)
        ny = torch.cos(phi)*torch.sin(theta)
        nz = torch.sin(phi)
        direction = (X-x0)*nx + (Y-y0)*ny + (Z-z0)*nz
        
        if wavelet_type == 'cosine':
            carrier = torch.cos(k0*direction)
        elif wavelet_type == 'gaussian_deriv':
            z_dir = direction / (scale*L)
            carrier = -z_dir * torch.exp(-z_dir**2/2)
        else:
            carrier = torch.cos(k0*direction) * torch.exp(-(direction/(scale*L))**2/2)
        
        amp = (1 - kappa) + kappa*torch.rand(1)
        v0 += amp * envelope * carrier

    return v0 / torch.max(torch.abs(v0))

def sample_kink_field_3d(X, Y, Z, L, winding_x, winding_y, winding_z, width_range=(0.5, 3.0), randomize_positions=True):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    
    kink_velocities = []
    
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
            kink_term = sign_x * 4 * torch.atan(torch.exp((X - x0) / kink_width))
            u0 += kink_term
            
            velocity = (-0.2 - 0.5 * torch.rand(1)) * sign_x
            kink_velocities.append((x0, 'x', velocity, kink_width))
    
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
            kink_term = sign_y * 4 * torch.atan(torch.exp((Y - y0) / kink_width))
            u0 += kink_term
            
            velocity = (-0.2 - 0.5 * torch.rand(1)) * sign_y
            kink_velocities.append((y0, 'y', velocity, kink_width))
    
    if winding_z != 0:
        width_z = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(1)
        positions_z = []
        
        if randomize_positions:
            for i in range(abs(winding_z)):
                positions_z.append(L * (2 * torch.rand(1) - 1))
        else:
            for i in range(abs(winding_z)):
                positions_z.append(L * (-0.8 + 1.6 * i / (abs(winding_z))))
        
        sign_z = 1 if winding_z > 0 else -1
        for z0 in positions_z:
            kink_width = width_z * (0.8 + 0.4 * torch.rand(1))
            kink_term = sign_z * 4 * torch.atan(torch.exp((Z - z0) / kink_width))
            u0 += kink_term
            
            velocity = (-0.2 - 0.5 * torch.rand(1)) * sign_z
            kink_velocities.append((z0, 'z', velocity, kink_width))
    
    for pos, direction, vel, width in kink_velocities:
        if direction == 'x':
            derivative = 1.0 / (width * torch.cosh((X - pos) / width)**2)
            v0 -= vel * derivative
        elif direction == 'y':
            derivative = 1.0 / (width * torch.cosh((Y - pos) / width)**2)
            v0 -= vel * derivative
        else:  # z
            derivative = 1.0 / (width * torch.cosh((Z - pos) / width)**2)
            v0 -= vel * derivative
    
    return u0, v0

def sample_breather_field_3d(X, Y, Z, L, num_breathers=1, position_type='random'):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    
    positions = []
    if position_type == 'random':
        for _ in range(num_breathers):
            x0 = L * (2 * torch.rand(1) - 1)
            y0 = L * (2 * torch.rand(1) - 1)
            z0 = L * (2 * torch.rand(1) - 1)
            positions.append((x0, y0, z0))
    elif position_type == 'sphere':
        radius = 0.6 * L * torch.rand(1)
        for i in range(num_breathers):
            theta = 2 * np.pi * i / num_breathers
            phi = np.pi * torch.rand(1)
            x0 = radius * torch.tensor(np.sin(phi) * np.cos(theta))
            y0 = radius * torch.tensor(np.sin(phi) * np.sin(theta))
            z0 = radius * torch.tensor(np.cos(phi))
            positions.append((x0, y0, z0))
    elif position_type == 'line':
        for i in range(num_breathers):
            pos = -L + 2 * L * i / (num_breathers - 1 if num_breathers > 1 else 1)
            axis = random.choice(['x', 'y', 'z'])
            if axis == 'x':
                positions.append((torch.tensor([pos]), torch.tensor([0.0]), torch.tensor([0.0])))
            elif axis == 'y':
                positions.append((torch.tensor([0.0]), torch.tensor([pos]), torch.tensor([0.0])))
            else:
                positions.append((torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([pos])))
    
    for x0, y0, z0 in positions:
        width = 0.5 + 2.5 * torch.rand(1)
        amplitude = 0.1 + 0.8 * torch.rand(1)
        phase = 2 * np.pi * torch.rand(1)
        
        omega = torch.sqrt(1.0 - amplitude**2)
        t = 0.0
        
        direction = random.choice(['x', 'y', 'z', 'radial'])
        if direction == 'x':
            xi = (X - x0) / width
            u_term = 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                                  (omega * torch.cosh(amplitude * xi)))
            
            numerator = amplitude * omega * torch.cos(omega * t + phase)
            denominator = omega * torch.cosh(amplitude * xi)
            denominator_squared = denominator**2 + (amplitude * torch.sin(omega * t + phase))**2
            v_term = 4 * numerator / denominator_squared
            
            u0 += u_term
            v0 += v_term
            
        elif direction == 'y':
            yi = (Y - y0) / width
            u_term = 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                                  (omega * torch.cosh(amplitude * yi)))
            
            numerator = amplitude * omega * torch.cos(omega * t + phase)
            denominator = omega * torch.cosh(amplitude * yi)
            denominator_squared = denominator**2 + (amplitude * torch.sin(omega * t + phase))**2
            v_term = 4 * numerator / denominator_squared
            
            u0 += u_term
            v0 += v_term
            
        elif direction == 'z':
            zi = (Z - z0) / width
            u_term = 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                                  (omega * torch.cosh(amplitude * zi)))
            
            numerator = amplitude * omega * torch.cos(omega * t + phase)
            denominator = omega * torch.cosh(amplitude * zi)
            denominator_squared = denominator**2 + (amplitude * torch.sin(omega * t + phase))**2
            v_term = 4 * numerator / denominator_squared
            
            u0 += u_term
            v0 += v_term
            
        else:
            ri = torch.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / width
            u_term = 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                                  (omega * torch.cosh(amplitude * ri)))
            
            numerator = amplitude * omega * torch.cos(omega * t + phase)
            denominator = omega * torch.cosh(amplitude * ri)
            denominator_squared = denominator**2 + (amplitude * torch.sin(omega * t + phase))**2
            v_term = 4 * numerator / denominator_squared
            
            u0 += u_term
            v0 += v_term
    
    return u0, v0

def sample_spherical_soliton_3d(X, Y, Z, L, num_solitons=1):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
   
    target_x = 0.2*L * torch.rand(1)
    target_y = 0.2*L * torch.rand(1)
    target_z = 0.2*L * torch.rand(1)
    
    for i in range(num_solitons):
        distance = 0.5*L + 0.3*L*torch.rand(1)
        theta = 2*np.pi * i / num_solitons
        phi = np.pi * torch.rand(1)

        theta = torch.tensor(theta) if not isinstance(theta, torch.Tensor) else theta
        phi   = torch.tensor(phi) if not isinstance(phi, torch.Tensor) else phi
        
        x0 = target_x + distance * torch.sin(phi) * torch.cos(theta)
        y0 = target_y + distance * torch.sin(phi) * torch.sin(theta)
        z0 = target_z + distance * torch.cos(phi)
        
        r0 = 0.1*L + 0.3*L*torch.rand(1)
        width = 0.5 + 1.5*torch.rand(1)
        
        direction_x = (target_x - x0) 
        direction_y = (target_y - y0)
        direction_z = (target_z - z0)
        dir_magnitude = torch.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        direction_x /= dir_magnitude
        direction_y /= dir_magnitude
        direction_z /= dir_magnitude
        speed = 0.2 + 0.3*torch.rand(1)
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)
        
        sign = 1 if i % 2 == 0 else -1
        u0 += sign * 4 * torch.atan(torch.exp((r - r0)/width))
        
        dr_dx = (X - x0) / (r + 1e-10)
        dr_dy = (Y - y0) / (r + 1e-10)
        dr_dz = (Z - z0) / (r + 1e-10)
    
        projection = dr_dx*direction_x + dr_dy*direction_y + dr_dz*direction_z
        derivative = projection * sign * (1.0 / width) * torch.exp((r - r0)/width) / (1 + torch.exp((r - r0)/width))**2
        v0 += speed * derivative

    return u0, v0

def sample_spiral_field_3d(X, Y, Z, L, num_arms=2):
    u0 = torch.zeros_like(X)
    
    x0 = L * (2 * torch.rand(1) - 1)
    y0 = L * (2 * torch.rand(1) - 1)
    z0 = L * (2 * torch.rand(1) - 1)
    
    r = torch.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)
    theta = torch.atan2(Y - y0, X - x0)
    phi = torch.acos((Z - z0) / (r + 1e-10))
    
    k = 1.0 + 2.0 * torch.rand(1)
    spiral_phase = theta + phi + k * r / L
    
    pattern = torch.cos(num_arms * spiral_phase)
    decay = torch.exp(-0.5 * r / L)
    
    u0 = 4 * torch.atan(pattern * decay)
    
    return u0

def sample_skyrmion_field_3d(X, Y, Z, L, num_skyrmions=1):
    phi = torch.zeros_like(X)
    
    for _ in range(num_skyrmions):
        x0, y0, z0 = L * (2*torch.rand(3) - 1)
        lambda_size = 0.2*L + 0.4*L*torch.rand(1)
        q = random.choice([-1, 1])
        alpha = 2 * np.pi * torch.rand(1)
        
        r_xy = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        z_rel = Z - z0
        r_3d = torch.sqrt(r_xy**2 + z_rel**2)
        theta_xy = torch.atan2(Y - y0, X - x0)
        
        profile = 2 * torch.arctan2(lambda_size, r_3d)
        angle = theta_xy + q * alpha
        
        skyrmion_contribution = 2 * profile * angle / torch.pi
        
        cutoff = torch.exp(-(r_3d/(0.8*L))**4)
        phi += cutoff * skyrmion_contribution
    
    return phi, 0.05 * sample_anisotropic_grf_3d(X.shape[0], X.shape[1], X.shape[2], L)

def sample_combined_field_3d(X, Y, Z, L, solution_types=None, weights=None):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    
    if solution_types is None:
        solution_types = ['kink', 'breather', 'spherical', 'spiral', 'skyrmion']
        
    if weights is None:
        weights = torch.ones(len(solution_types))
    weights = weights / weights.sum()
    
    for solution_type, weight in zip(solution_types, weights):
        if solution_type == 'kink':
            winding_x = random.randint(-2, 2)
            winding_y = random.randint(-2, 2)
            winding_z = random.randint(-2, 2)
            while winding_x == 0 and winding_y == 0 and winding_z == 0:
                winding_x = random.randint(-2, 2)
            kink_u0, kink_v0 = sample_kink_field_3d(X, Y, Z, L, winding_x, winding_y, winding_z)
            u0 += weight * kink_u0
            v0 += weight * kink_v0
            
        elif solution_type == 'breather':
            num_breathers = random.randint(1, 3)
            position_type = random.choice(['random', 'sphere', 'line'])
            breather_u0, breather_v0 = sample_breather_field_3d(X, Y, Z, L, num_breathers, position_type)
            u0 += weight * breather_u0
            v0 += weight * breather_v0
            
        elif solution_type == 'spherical':
            num_solitons = random.randint(1, 4)
            sph_u0, sph_v0 = sample_spherical_soliton_3d(X, Y, Z, L, num_solitons)
            u0 += weight * sph_u0
            v0 += weight * sph_v0
            
        elif solution_type == 'spiral':
            num_arms = random.randint(1, 4)
            spiral_u0 = sample_spiral_field_3d(X, Y, Z, L, num_arms)
            
            dx = torch.zeros_like(spiral_u0)
            dy = torch.zeros_like(spiral_u0)
            dz = torch.zeros_like(spiral_u0)
            
            x = torch.linspace(-L, L, X.shape[0])
            y = torch.linspace(-L, L, X.shape[1])
            z = torch.linspace(-L, L, X.shape[2])
            
            dx[1:-1, :, :] = (spiral_u0[2:, :, :] - spiral_u0[:-2, :, :]) / (2 * (x[1] - x[0]))
            dy[:, 1:-1, :] = (spiral_u0[:, 2:, :] - spiral_u0[:, :-2, :]) / (2 * (y[1] - y[0]))
            dz[:, :, 1:-1] = (spiral_u0[:, :, 2:] - spiral_u0[:, :, :-2]) / (2 * (z[1] - z[0]))
            
            spiral_v0 = 0.3 * (-dy + dx)
            
            u0 += weight * spiral_u0
            v0 += weight * spiral_v0
            
        elif solution_type == 'skyrmion':
            num_skyrmions = random.randint(1, 3)
            sky_u0, sky_v0 = sample_skyrmion_field_3d(X, Y, Z, L, num_skyrmions)
            u0 += weight * sky_u0
            v0 += weight * sky_v0
    
    return u0, v0

def sample_sine_gordon_solution_3d(Nx, Ny, Nz, L, solution_type='auto'):
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    z = torch.linspace(-L, L, Nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    if solution_type == 'auto':
        solution_types = ['kink', 'breather', 'combined', 'spherical', 'spiral', 'skyrmion', 'random_mix']
        solution_type = random.choice(solution_types)
    
    if solution_type == 'kink':
        winding_x = random.randint(-2, 2)
        winding_y = random.randint(-2, 2)
        winding_z = random.randint(-2, 2)
        
        while winding_x == 0 and winding_y == 0 and winding_z == 0:
            winding_x = random.randint(-2, 2)
            winding_y = random.randint(-2, 2)
            winding_z = random.randint(-2, 2)
            
        u0, v0 = sample_kink_field_3d(X, Y, Z, L, winding_x, winding_y, winding_z)
        solution_info = f"Kink (nx={winding_x}, ny={winding_y}, nz={winding_z})"
        
    elif solution_type == 'breather':
        num_breathers = random.randint(1, 4)
        position_type = random.choice(['random', 'sphere', 'line'])
        u0, v0 = sample_breather_field_3d(X, Y, Z, L, num_breathers, position_type)
        solution_info = f"Breather (n={num_breathers}, {position_type})"
        
    elif solution_type == 'spiral':
        num_arms = random.randint(1, 4)
        u0 = sample_spiral_field_3d(X, Y, Z, L, num_arms)
        
        dx = torch.zeros_like(u0)
        dy = torch.zeros_like(u0)
        dz = torch.zeros_like(u0)
        
        dx[1:-1, :, :] = (u0[2:, :, :] - u0[:-2, :, :]) / (2 * (x[1] - x[0]))
        dy[:, 1:-1, :] = (u0[:, 2:, :] - u0[:, :-2, :]) / (2 * (y[1] - y[0]))
        dz[:, :, 1:-1] = (u0[:, :, 2:] - u0[:, :, :-2]) / (2 * (z[1] - z[0]))
   
        # somewhat of a cross product merited?
        v0 = 0.3 * (-dy + dx)
        
        solution_info = f"Spiral Wave (arms={num_arms})"
        
    elif solution_type == 'combined':
        combo_types = random.sample(['kink', 'breather', 'spherical'], 
                                   k=random.randint(2, 3))
        u0, v0 = sample_combined_field_3d(X, Y, Z, L, combo_types)
        solution_info = f"Combined ({'+'.join(combo_types)})"

    elif solution_type == 'spherical':
        num_solitons = random.randint(1, 4)
        u0, v0 = sample_spherical_soliton_3d(X, Y, Z, L, num_solitons)
        solution_info = f"Spherical Soliton ({num_solitons})"
    
    elif solution_type == 'skyrmion':
        num_skyrmions = random.randint(1, 3)
        u0, v0 = sample_skyrmion_field_3d(X, Y, Z, L, num_skyrmions)
        solution_info = f"Skyrmion-like ({num_skyrmions})"
        
    elif solution_type == 'random_mix':
        all_types = ['kink', 'breather', 'spherical', 'spiral']
        num_types = random.randint(2, 3)
        selected_types = random.sample(all_types, k=num_types)
        weights = torch.rand(num_types)
        u0, v0 = sample_combined_field_3d(X, Y, Z, L, selected_types, weights)
        solution_info = f"Random Mix ({'+'.join(selected_types)})"
    
    v0 += 0.01 * sample_wavelet_superposition_3d(Nx, Ny, Nz, L, n_wavelets=5, 
                                             scale_range=(0.1, 0.5), kappa=0.3)
    
    anisotropy_ratio = 1.0 + 3.0 * torch.rand(1)
    theta = 360.0 * torch.rand(1)
    phi = 180.0 * torch.rand(1)
    length_scale = L / (1.0 + 5.0 * torch.rand(1))
    power = 1.0 + 1.0 * torch.rand(1)
    
    anisotropy = sample_anisotropic_grf_3d(Nx, Ny, Nz, L, length_scale=length_scale, 
                                         anisotropy_ratio=anisotropy_ratio, theta=theta, phi=phi,
                                         power=power)
    
    m = 1.0 + torch.nn.functional.relu(anisotropy)
    
    mask = torch.abs(m) > 5.0
    m[mask] = 5.0

    m /= torch.max(m)
    
    return u0, v0, m, solution_info

def visualize_isosurfaces(u, v, m, solution_info, L):
    fig = plt.figure(figsize=(15, 5))
    
    u_np = u.numpy()
    v_np = v.numpy()
    m_np = m.numpy()
    
    u_min, u_max = u_np.min(), u_np.max()
    v_min, v_max = v_np.min(), v_np.max()
    m_min, m_max = m_np.min(), m_np.max()
    
    level_u = u_min + 0.5 * (u_max - u_min)
    level_v = v_min + 0.6 * (v_max - v_min)
    level_m = m_min + 0.7 * (m_max - m_min)
    
    verts_to_coords = lambda verts: verts / np.array(u.shape) * 2*L - L
    
    ax1 = fig.add_subplot(131, projection='3d')
    if u_max > u_min:
        verts, faces, _, _ = measure.marching_cubes(u_np, level=level_u)
        verts = verts_to_coords(verts)
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                       cmap='coolwarm', lw=0, alpha=0.8)
    ax1.set_title(f'{solution_info} (u)')
    
    ax2 = fig.add_subplot(132, projection='3d')
    if v_max > v_min:
        verts, faces, _, _ = measure.marching_cubes(v_np, level=level_v)
        verts = verts_to_coords(verts)
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                       cmap='coolwarm', lw=0, alpha=0.8)
    ax2.set_title(f'{solution_info} (v)')
    
    ax3 = fig.add_subplot(133, projection='3d')
    if m_max > m_min:
        verts, faces, _, _ = measure.marching_cubes(m_np, level=level_m)
        verts = verts_to_coords(verts)
        ax3.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                       cmap='viridis', lw=0, alpha=0.8)
    ax3.set_title(f'{solution_info} (m)')
    
    plt.tight_layout()
    return fig

def main():
    Nx = 64
    Ny = 64
    Nz = 64
    L = 10.0
    
    solution_types = ['kink', 'breather', 'combined', 'spherical', 'spiral', 'skyrmion']
    
    for i, solution_type in enumerate(solution_types):
        u0, v0, m, solution_info = sample_sine_gordon_solution_3d(Nx, Ny, Nz, L, solution_type)
        fig = visualize_isosurfaces(u0, v0, m, solution_info, L)
        plt.show()
    
    for i in range(2):
        u0, v0, m, solution_info = sample_sine_gordon_solution_3d(Nx, Ny, Nz, L)
        fig = visualize_isosurfaces(u0, v0, m, solution_info, L)
        plt.show()

if __name__ == "__main__":
    main()
