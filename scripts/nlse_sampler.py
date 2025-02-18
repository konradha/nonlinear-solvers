import numpy as np

def vortex_pair(X, Y, vortex_distance=1.0, core_size=0.2, amplitude=1.0):
    # hm, stationary
    x0 = vortex_distance / 2.0
    theta1 = np.arctan2(Y, X - x0)
    theta2 = np.arctan2(Y, X + x0)
    phase = theta1 - theta2
    amplitude_field = amplitude * np.exp(-(((X - x0)**2 + Y**2) / (2 * core_size**2))) * \
                              np.exp(-(((X + x0)**2 + Y**2) / (2 * core_size**2)))
    u = amplitude_field * np.exp(1j * phase)
    return u

def sampler_random_vortices(X, Y, num_vortices=10, core_size=0.3, seed=None):
    # nice, dynamic (somewhat) -- not very smooth however
    if seed is not None:
        np.random.seed(seed) 
    u = np.ones_like(X, dtype=np.complex128)
    for _ in range(num_vortices):
        x0 = np.random.uniform(X.min(), X.max())
        y0 = np.random.uniform(Y.min(), Y.max())
        charge = np.random.choice([-1, 1])
        phase = charge * np.arctan2(Y - y0, X - x0)
        envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * core_size**2))
        u *= (1 + envelope * (np.exp(1j * phase) - 1))
    return u

def sampler_random_solitons(X, Y, num_solitons=5, soliton_width=0.5, amp_range=(0.5, 2.0), seed=None):
    # nice, somewhat stable soliton structures, fairly smooth
    if seed is not None:
        np.random.seed(seed)
    u = np.zeros_like(X, dtype=complex)
    for _ in range(num_solitons):
        A = np.random.uniform(*amp_range) * (np.random.randn() + 1j*np.random.randn())
        x0 = np.random.uniform(X.min(), X.max())
        y0 = np.random.uniform(Y.min(), Y.max())
        phase0 = np.random.uniform(0, 2*np.pi)
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        soliton = A * (1/np.cosh(r/soliton_width)) * np.exp(1j * phase0)
        u += soliton
    return u

def sampler_random_fourier_localized(Nx=256, Ny=256, Lx=10, Ly=10, freq_scale=2.0, seed=None):
    # localized structured in random smooth field, seems to evolve somewhat nicely
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(Nx, d=2 *  Lx/(Nx - 1)) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=2 *  Ly/(Ny - 1)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    envelope = np.exp(- (KX**2 + KY**2) / (2 * freq_scale**2))
    coeff = (np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)) * envelope
    u = np.fft.ifft2(coeff)
    # localizing
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    window = np.exp(- (X**2 + Y**2) / (2 * (Lx/2)**2))
    return u * window

def sampler_random_eigenfunctions(X, Y, num_modes=20, seed=None):
    # Random matrix theory (of Laplacian?)
    # Hm, static
    if seed is not None:
        np.random.seed(seed)
    u = np.zeros_like(X, dtype=complex)
    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min() 
    for n in range(1, num_modes+1):
        for m in range(1, num_modes+1):
            coeff = np.random.randn() + 1j * np.random.randn()
            mode = np.sin(np.pi * n * (X - X.min()) / Lx) * np.sin(np.pi * m * (Y - Y.min()) / Ly)
            u += coeff * mode
    return u

