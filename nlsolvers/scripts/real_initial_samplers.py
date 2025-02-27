import torch
import matplotlib.pyplot as plt
import torch.fft as fft
import numpy as np


def show(X, Y, u0, v0, m):
    fig, axs = plt.subplots(figsize=(20, 20), ncols=3, subplot_kw={"projection":'3d'}) 

    mp = [(u0, "u0"), (v0, "v0"), (m, "m")]

    for i in range(len(mp)):
        axs[i].plot_surface(X, Y, mp[i][0], cmap='coolwarm')
        axs[i].set_title(mp[i][1])

    plt.show()

def sample_grf(Nx, Ny, Lx, Ly, length_scale=2.5, variance=1.):
    kx = 2.0 * torch.pi * torch.fft.fftfreq(Nx, Lx/Nx)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(Ny, Ly/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = (KX ** 2 + KY ** 2)
    spectrum = variance * torch.exp(-0.5 ** 2 * length_scale**2 * K2)
    spectrum = spectrum / torch.sum(spectrum)
    white_noise = torch.randn(Nx, Ny, dtype=torch.complex64) + \
                 1j * torch.randn(Nx, Ny, dtype=torch.complex64)

    k = white_noise[0,0].clone()
    white_noise[0,0] = k.real
    field_k = torch.sqrt(spectrum) * white_noise
    field = torch.fft.ifft2(field_k).real

    dx = Lx / Nx
    dy = Ly / Ny

    norm = torch.sqrt(torch.sum(torch.abs(field)**2) * dx * dy)

    return field / norm

def sample_mixture_kpi_grf(Nx, Ny, Lx, Ly, n_wavelets=5, epsilon=1e-10):
    kx = 2.0 * torch.pi * torch.fft.fftfreq(Nx, Lx/Nx)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(Ny, Ly/Ny)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    KX_reg = torch.where(abs(KX) < epsilon, epsilon, KX)
    K2 = abs(KX)**3 + (KY**2)/abs(KX_reg)

    wavelet_centers_x = torch.rand(n_wavelets) * Nx
    wavelet_centers_y = torch.rand(n_wavelets) * Ny
    wavelet_scales = torch.rand(n_wavelets) * 0.5 + 0.5

    field_k = torch.zeros((Nx, Ny), dtype=torch.complex64)
    for i in range(n_wavelets):
        phase = torch.exp(-1j * (KX * wavelet_centers_x[i] + KY * wavelet_centers_y[i]))
        morlet = torch.exp(-(K2 * wavelet_scales[i]**2))
        field_k += torch.randn(2)[0] * morlet * phase

    spectrum = 1.0 / (1.0 + K2)**2
    field_k = field_k * torch.sqrt(spectrum)
    k = field_k[0,0].clone()
    field_k[0,0] = k.real
    field = torch.fft.ifft2(field_k).real
    norm = torch.sqrt(torch.sum(torch.abs(field)**2) * (Lx/Nx) * (Ly/Ny))
    return field / norm

def sampling_idea(Nx, Ny, Lx, Ly):
    # get initial sample
    W = (sample_mixture_kpi_grf(Nx, Ny, Lx, Ly, n_wavelets=5, epsilon=1e-10))
    # natural solution to sine-Gordon
    u0 = torch.atan(torch.exp(W))
    # small initial velocities
    v0 = torch.randn_like(u0) * 0.1
    # small positive wave speeds
    c = torch.exp(0.1*W)
    m = 1 + torch.sigmoid(W)
    return u0, v0, c, m



def sample_grf(Nx, Ny, L, length_scale=1.0, variance=1.0, nu=1.5):
    # Matern type GRF
    dx = 2 * L / Nx
    dy = 2 * L / Ny
    
    kx = 2*np.pi*fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=dy)
    KX, KY = torch.meshgrid(kx.clone().detach(), ky.clone().detach(), indexing='ij')
    
    # (1 + |k|^2)^(-ν-1)
    K2 = KX**2 + KY**2
    spectrum = variance * (1 + (length_scale**2)*K2)**(-nu-1)
    
    noise_real = torch.randn(Nx, Ny)
    noise_imag = torch.randn(Nx, Ny)
    noise = noise_real + 1j*0.5*(noise_imag + noise_imag.conj().T) 
    field = fft.ifft2(fft.fft2(noise) * torch.sqrt(spectrum)).real
    return field / torch.std(field)

def sample_anisotropic_grf(Nx, Ny, L, length_scale=1.0, 
                          anisotropy_ratio=2.0, theta=30.0):
    theta_rad = np.deg2rad(theta)
    ell_x = length_scale * np.sqrt(anisotropy_ratio)
    ell_y = length_scale / np.sqrt(anisotropy_ratio)
    kx = 2*np.pi*fft.fftfreq(Nx, d=2 * L/Nx)
    ky = 2*np.pi*fft.fftfreq(Ny, d=2 * L/Ny)
    KX, KY = torch.meshgrid(kx.clone().detach(), ky.clone().detach(), indexing='ij')
    KX_rot = KX*np.cos(theta_rad) - KY*np.sin(theta_rad)
    KY_rot = KX*np.sin(theta_rad) + KY*np.cos(theta_rad) 
    # anisotropy
    spectrum = torch.exp(-( (KX_rot/ell_x)**2 + (KY_rot/ell_y)**2 )) 
    noise = torch.randn(Nx, Ny) + 1j*torch.randn(Nx, Ny)
    field = fft.ifft2(fft.fft2(noise) * torch.sqrt(spectrum)).real
    return field / (L**2 * torch.std(field))

def sample_wavelet_superposition(Nx, Ny, L, n_wavelets=20, 
                                scale_range=(0.1, 2.0), kappa=0.5): 
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
        # Morlet
        envelope = torch.exp( -((X-x0)**2 + (Y-y0)**2)/(2*(scale*L)**2) )
        carrier = torch.cos(k0*( (X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta) )) 
        # Anisotropy
        amp = (1 - kappa) + kappa*torch.rand(1)
        v0 += amp * envelope * carrier

    return v0 / torch.max(torch.abs(v0))

def sample_matern_field(Nx, Ny, L, nu=1.5, length_scale=1.0):
    field = sample_grf(Nx, Ny, L, length_scale=length_scale, nu=nu)
    return torch.nn.functional.relu(field)

def sample_initial_conditions(Nx, Ny, L):
    # L/4 correlation length
    W = sample_grf(Nx, Ny, L, length_scale=L/4, variance=0.5) 
    # typical soliton-like solution
    u0 = 4 * torch.atan(torch.exp(3*W))  # some amplification 
    # mixing scales
    v0 = sample_wavelet_superposition(Nx, Ny, L, n_wavelets=30, 
                                     scale_range=(0.1, 0.5), kappa=0.3)
    
    # Anisotropic wave speed
    anisotropy = sample_anisotropic_grf(Nx, Ny, L, length_scale=L/3, 
                                            anisotropy_ratio=3.0, theta=45.0)
    c = torch.exp(0.1*anisotropy)
    m = sample_matern_field(Nx, Ny, L, nu=1.0, length_scale=L/2)

    # do not make m-field explode anywhere
    mask = torch.abs(m) > 100.
    m[mask] = 0.
    
    return u0, v0, m, W, anisotropy  


if __name__ == '__main__':
    nx = ny = 100
    Lx = Ly = 10

    u0, v0, m, _, _ = sample_initial_conditions(nx, ny, Lx) 

    xn, yn = torch.linspace(-Lx, Lx, nx), torch.linspace(-Ly, Ly, ny)
    X, Y = torch.meshgrid(xn, yn, indexing='ij')
    show(X, Y, u0, v0, m)
