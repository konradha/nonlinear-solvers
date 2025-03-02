import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import torch.fft as fft

def sample_anisotropic_grf(Nx, Ny, L, length_scale=1.0, anisotropy_ratio=2.0, theta=30.0):
    theta_rad = np.deg2rad(theta)
    ell_x = length_scale * np.sqrt(anisotropy_ratio)
    ell_y = length_scale / np.sqrt(anisotropy_ratio)
    kx = 2*np.pi*fft.fftfreq(Nx, d=2 * L/Nx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=2 * L/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    KX_rot = KX*np.cos(theta_rad) - KY*np.sin(theta_rad)
    KY_rot = KX*np.sin(theta_rad) + KY*np.cos(theta_rad) 
    
    spectrum = torch.exp(-( (KX_rot/ell_x)**2 + (KY_rot/ell_y)**2 )) 
    noise = torch.randn(Nx, Ny) + 1j*torch.randn(Nx, Ny)
    field = fft.ifft2(fft.fft2(noise) * torch.sqrt(spectrum)).real
    return field / torch.std(field)

def sample_wavelet_superposition(Nx, Ny, L, n_wavelets=20, scale_range=(0.1, 2.0), kappa=0.5): 
    v0 = torch.zeros(Nx, Ny)
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    for _ in range(n_wavelets):
        scale = scale_range[0] + (scale_range[1]-scale_range[0])*torch.rand(1)
        theta = 2*np.pi*torch.rand(1)
        x0 = L*(torch.rand(1)-0.5)
        y0 = L*(torch.rand(1)-0.5)
        k0 = 2*np.pi/(scale*L)
        
        envelope = torch.exp(-((X-x0)**2 + (Y-y0)**2)/(2*(scale*L)**2))
        carrier = torch.cos(k0*((X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta)))
        
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

def sample_breather_field(X, Y, L, num_breathers=1):
    u0 = torch.zeros_like(X)
    
    for _ in range(num_breathers):
        x0 = L * (2 * torch.rand(1) - 1)
        y0 = L * (2 * torch.rand(1) - 1)
        
        width = 0.5 + 2.5 * torch.rand(1)
        amplitude = 0.1 + 0.8 * torch.rand(1)
        phase = 2 * np.pi * torch.rand(1)
        
        omega = torch.sqrt(1.0 - amplitude**2)
        t = 0.0
        
        direction = 'x' if torch.rand(1) > 0.5 else 'y'
        if direction == 'x':
            xi = (X - x0) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                               (omega * torch.cosh(amplitude * xi)))
        else:
            yi = (Y - y0) / width
            u0 += 4 * torch.atan(amplitude * torch.sin(omega * t + phase) / 
                               (omega * torch.cosh(amplitude * yi)))
    
    return u0

def sample_colliding_rings(X, Y, L, num_rings=2):
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)

    for _ in range(num_rings):
        x0 = L * (2*torch.rand(1) - 1)
        y0 = L * (2*torch.rand(1) - 1)
        r0 = 0.1*L + 0.6*L*torch.rand(1)
        width = 0.5 + 2.5*torch.rand(1)
        direction = 1 if torch.rand(1) > 0.5 else -1

        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        u0 += 4 * torch.atan(torch.exp((r - r0)/width))
        v0 += direction * torch.exp(-(r - r0)**2/(2*width**2))

    return u0, v0

def sample_elliptical_soliton(X, Y, L):
    x0, y0 = (L/2) * (2*torch.rand(2) - 1)
    a = (0.3*L + 0.4*L*torch.rand(1)) 
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

    return u0, v0

def sample_soliton_antisoliton_pair(X, Y, L):
    pattern_type = random.choice(['radial', 'linear', 'angular'])
    width = 0.8 + 2.2*torch.rand(1)
    x0, y0 = L*(2*torch.rand(2) - 1)

    if pattern_type == 'radial':
        r = torch.sqrt((X - x0)**2 + (Y - y0)**2)
        u0 = 4*torch.atan(torch.exp(r/width)) - 4*torch.atan(torch.exp((r - 0.5*width)/width))
    elif pattern_type == 'linear':
        theta = torch.pi*torch.rand(1)
        x_rot = (X - x0)*torch.cos(theta) + (Y - y0)*torch.sin(theta)
        u0 = 4*torch.atan(torch.exp(x_rot/width)) - 4*torch.atan(torch.exp(-x_rot/width))
    else:
        phi = torch.atan2(Y - y0, X - x0)
        u0 = 4*torch.atan(torch.exp(torch.sin(phi)/width)) - 4*torch.atan(torch.exp(-torch.sin(phi)/width))

    v0 = sample_anisotropic_grf(X.shape[0], X.shape[1], L,
                               length_scale=width, anisotropy_ratio=2.0) * 0.2

    return u0, v0


def sample_combined_field(X, Y, L, winding_x, winding_y, num_breathers=0):
    u0 = sample_kink_field(X, Y, L, winding_x, winding_y)
    
    if num_breathers > 0:
        u0 += sample_breather_field(X, Y, L, num_breathers)
    
    return u0

def sample_sine_gordon_solution(Nx, Ny, L, solution_type='auto'):
    x = torch.linspace(-L, L, Nx)
    y = torch.linspace(-L, L, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    if solution_type == 'auto':
        solution_type = random.choice(['kink', 'breather', 'combined', 'rings'])
    
    v0_amplitude = 0.05 + torch.rand(1)
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
        
    elif solution_type == 'breather':
        num_breathers = random.randint(1, 3)
        u0 = sample_breather_field(X, Y, L, num_breathers)
        solution_info = f"Breather (n={num_breathers})"
        
    elif solution_type == 'combined':
        winding_x = random.randint(-1, 1)
        winding_y = random.randint(-1, 1)
        num_breathers = random.randint(1, 2)
        
        u0 = sample_combined_field(X, Y, L, winding_x, winding_y, num_breathers)
        solution_info = f"Combined (nx={winding_x}, ny={winding_y}, breathers={num_breathers})"

    elif solution_type == 'rings':
        u0, v0 = sample_colliding_rings(X, Y, L, num_rings=random.randint(1, 10)) 
        solution_info = f"Elliptical collision"
      
    anisotropy_ratio = 1.0 + 3.0 * torch.rand(1)
    theta = 360.0 * torch.rand(1)
    length_scale = L / (1.0 + 5.0 * torch.rand(1))
    
    anisotropy = sample_anisotropic_grf(Nx, Ny, L, length_scale=length_scale, 
                                      anisotropy_ratio=anisotropy_ratio, theta=theta)
    
    m = 1.0 + torch.nn.functional.relu(anisotropy)
    
    mask = torch.abs(m) > 5.0
    m[mask] = 5.0
    
    return u0, v0, m, solution_info

def main():
    Nx = 128
    Ny = 128
    L = 10.0
    
    fig = plt.figure(figsize=(20, 15))
    
    solution_types = 4 * ['rings']# ['rings', 'kink', 'breather', 'combined']
    
    for i, solution_type in enumerate(solution_types):
        u0, v0, m, solution_info = sample_sine_gordon_solution(Nx, Ny, L, solution_type)
        
        x = torch.linspace(-L, L, Nx).numpy()
        y = torch.linspace(-L, L, Ny).numpy()
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        ax1 = fig.add_subplot(3, 4, i+1)
        im1 = ax1.pcolormesh(X, Y, u0.numpy(), cmap='coolwarm', shading='auto', vmin=-6, vmax=6)
        ax1.set_title(f'{solution_info}')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(3, 4, i+5)
        im2 = ax2.pcolormesh(X, Y, v0.numpy(), cmap='coolwarm', shading='auto')
        plt.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(3, 4, i+9, projection='3d')
        surf = ax3.plot_surface(X[::4,::4], Y[::4,::4], u0.numpy()[::4,::4], 
                              cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    plt.show()
    
    """ 
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    for i in range(8):
        u0, v0, m, solution_info = sample_sine_gordon_solution(Nx, Ny, L)
        im = axs[i].pcolormesh(X, Y, u0.numpy(), cmap='coolwarm', shading='auto', vmin=-6, vmax=6)
        axs[i].set_title(f'{solution_info}')
        plt.colorbar(im, ax=axs[i])
    
    plt.tight_layout()
    plt.show()
    """
    

if __name__ == "__main__":
    main()
