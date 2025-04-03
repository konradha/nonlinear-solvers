import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def make_grid(n, L):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def create_constant_m(X, Y, value=1.0):
    return np.ones_like(X) * value


def create_periodic_boxes(n, L, factor=1.5, num_boxes_per_dim=3, 
                         box_length=0.1, wall_dist=0.1, 
                         set_zero=False, change_sign=True):
    X, Y = make_grid(n, L)
    box_length *= L
    wall_dist *= L
    available_space = 2*L - 2*wall_dist
    total_boxes_length = num_boxes_per_dim * box_length
    spacing = (available_space - total_boxes_length)/(num_boxes_per_dim - 1) if num_boxes_per_dim > 1 else 0
    
    start_x = start_y = -L + wall_dist
    m = np.ones_like(X)
    
    masks = []
    for i in range(num_boxes_per_dim):
        for j in range(num_boxes_per_dim):
            x0 = start_x + i*(box_length + spacing)
            x1 = x0 + box_length
            y0 = start_y + j*(box_length + spacing)
            y1 = y0 + box_length
            mask = np.logical_and.reduce((
                X >= x0, X <= x1,
                Y >= y0, Y <= y1
            ))
            masks.append(mask)
            m[mask] *= -factor if change_sign else factor
            
    if set_zero:
        for mask in masks:
            m[~mask] = 0.
    
    return m

def create_periodic_gaussians(n, L, factor=1.5, num_gaussians_per_dim=3,
                             sigma=0.05, wall_dist=0.1,
                             change_sign=True, background=1.0):
    X, Y = make_grid(n, L)
    sigma *= L
    wall_dist *= L
    available_space = 2*L - 2*wall_dist
    spacing = available_space / (num_gaussians_per_dim - 1) if num_gaussians_per_dim > 1 else 0
    m = np.ones_like(X) * background
    for i in range(num_gaussians_per_dim):
        for j in range(num_gaussians_per_dim):
            x_center = -L + wall_dist + i * spacing
            y_center = -L + wall_dist + j * spacing
            gaussian = np.exp(-((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma**2))
            if change_sign:
                m += -factor * gaussian
            else:
                m += factor * gaussian

    return m


def create_grf(nx, ny, Lx, Ly, mean=1.0, std=0.5, scale=2.0):
    white_noise = np.random.randn(nx, ny)
    smoothed = gaussian_filter(white_noise, sigma=scale * min(nx, ny) / (2 * max(Lx, Ly)))
    
    smoothed = (smoothed - np.mean(smoothed)) / np.std(smoothed)
    m = mean + std * smoothed
    
    return m


def create_morlet_wavelet(X, Y, scale, angle=0):
    k0 = 5.0
    rotated_X = X * np.cos(angle) + Y * np.sin(angle)
    rotated_Y = -X * np.sin(angle) + Y * np.cos(angle)
    
    gauss_env = np.exp(-(rotated_X**2 + rotated_Y**2) / (2 * scale**2))
    complex_wave = np.exp(1j * k0 * rotated_X / scale)
    
    wavelet = gauss_env * complex_wave
    
    return np.real(wavelet)


def create_wavelet_modulated_grf(nx, ny, Lx, Ly, wavelet_scale=1.0, 
                               grf_scale=2.0, mean=1.0, std=0.5):
    X, Y = make_grid(nx, Ly)
    
    grf = create_grf(nx, ny, Lx, Ly, mean=0, std=1, scale=grf_scale)
    
    angles = np.linspace(0, np.pi, 4)
    wavelet_sum = np.zeros_like(X)
    
    for angle in angles:
        wavelet = create_morlet_wavelet(X, Y, scale=wavelet_scale, angle=angle)
        wavelet_sum += wavelet
    
    wavelet_sum = (wavelet_sum - np.min(wavelet_sum)) / (np.max(wavelet_sum) - np.min(wavelet_sum))
    
    m = mean + std * grf * wavelet_sum
    
    return m


def scale_m_to_range(m, min_val=1.0, max_val=2.0):
    current_min = np.min(m)
    current_max = np.max(m)
    
    if current_min == current_max:
        return np.ones_like(m) * min_val
    
    m_scaled = (m - current_min) / (current_max - current_min) * (max_val - min_val) + min_val
    
    return m_scaled
