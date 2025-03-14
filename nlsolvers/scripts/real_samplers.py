import numpy as np

def generate_grf(nx, ny, Lx, Ly, scale=2.0, mean=1.0, std=0.5, seed=None):
    # mostly used for (de)focussing term m = m(x,y)
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(nx, d=2*Lx/(nx-1)) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=2*Ly/(ny-1)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    envelope = np.exp(-(KX**2 + KY**2) / (2 * scale**2))
    white_noise = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
    coeffs = white_noise * envelope
    field = np.real(np.fft.ifft2(coeffs))
    field = (field - np.mean(field)) / np.std(field) * std + mean
    field = np.maximum(field, 1.)
    return field.astype(np.complex128)

import numpy as np

def generate_wavelet(nx, ny, Lx, Ly, num_wavelets=30, scale=0.15, frequency=3.0,
                           amplitude=1.0, mean=1.0, std=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    field = np.zeros((nx, ny))

    for _ in range(num_wavelets):
        x0 = np.random.uniform(-Lx, Lx)
        y0 = np.random.uniform(-Ly, Ly)
        s = np.random.uniform(0.5, 1.5) * scale * min(Lx, Ly)

        k = np.random.uniform(0.7, 1.3) * frequency * np.pi / min(Lx, Ly)
        theta = np.random.uniform(0, 2*np.pi)

        kx = k * np.cos(theta)
        ky = k * np.sin(theta)

        amp = np.random.uniform(0.7, 1.3) * amplitude

        r2 = ((X - x0)**2 + (Y - y0)**2) / s**2
        wavelet = amp * np.exp(-0.5 * r2) * np.cos(kx * (X - x0) + ky * (Y - y0))
        field += wavelet

    field = (field - np.mean(field)) / np.std(field) * std + mean
    field = np.maximum(field, 0.1)
    return field


# TODO replace the below samplers for KGE and SGE with physically meaningful Wavelet-enhanced GRF samplers
def sample_random_kink(X, Y, width=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    X_rot = c*X + s*Y
    offset = np.random.uniform(-min(np.max(X), np.max(Y))/2, min(np.max(X), np.max(Y))/2)
    profile = 4 * np.arctan(np.exp((X_rot - offset) / width))
    return profile

def sample_breather(X, Y, frequency=0.9, amplitude=4.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    omega = frequency
    x0 = np.random.uniform(-np.max(X)/2, np.max(X)/2)
    y0 = np.random.uniform(-np.max(Y)/2, np.max(Y)/2)
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    gamma = 1/np.sqrt(1 - omega**2)
    profile = 4 * np.arctan(amplitude * np.sqrt(1-omega**2) / (omega * np.cosh(R)))
    return profile

def sample_random_bumps(X, Y, num_bumps=5, max_amplitude=np.pi, min_width=0.5, max_width=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    field = np.zeros_like(X)
    for _ in range(num_bumps):
        x0 = np.random.uniform(X.min(), X.max())
        y0 = np.random.uniform(Y.min(), Y.max())
        amplitude = np.random.uniform(-max_amplitude, max_amplitude)
        width = np.random.uniform(min_width, max_width)
        R2 = (X - x0)**2 + (Y - y0)**2
        bump = amplitude * np.exp(-R2 / (2 * width**2))
        field += bump

    return field
