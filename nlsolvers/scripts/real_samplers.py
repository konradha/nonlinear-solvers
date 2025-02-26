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
