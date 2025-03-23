import numpy as np
from scipy import signal
from scipy.fft import fft2, fftshift, fftfreq
from scipy.ndimage import gaussian_filter
import pywt

class SpectralSolitonClassifier:
    def __init__(self, dx=None, dy=None, dt=None, Lx=None,
            Ly=None, T=None, nx=None, ny=None, nt=None):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.nx = nx
        self.ny = ny
        self.nt = nt
             
        self.class_names = [
            "Kink", 
            "Kink-Antikink Pair", 
            "Breather"
        ]
        
        self.typical_scales = {
            "kink_width": 1.0,
            "kink_velocity": 0.5,
            "breather_frequency": 0.8,
            "breather_amplitude": 0.7
        }
        
        if self.dx is not None and self.dy is not None:
            self.setup_physical_grid()
    
    def setup_physical_grid(self):
        assert self.nx is not None and self.ny is not None
            
        self.x = np.linspace(-self.Lx, self.Lx, self.nx)
        self.y = np.linspace(-self.Ly, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)
        
        self.kx_physical = 2 * np.pi * fftfreq(self.nx, self.dx)
        self.ky_physical = 2 * np.pi * fftfreq(self.ny, self.dy)
        self.kx_grid_physical, self.ky_grid_physical = np.meshgrid(fftshift(self.kx_physical), 
                                                                   fftshift(self.ky_physical), indexing='ij')
        self.kr_physical = np.sqrt(self.kx_grid_physical**2 + self.ky_grid_physical**2)
        self.ktheta_physical = np.arctan2(self.ky_grid_physical, self.kx_grid_physical)
        
        if self.dt is not None and self.nt is not None:
            self.t = np.linspace(0, self.T, self.nt)
            self.omega_physical = 2 * np.pi * fftfreq(self.nt, self.dt)
    
    def classify(self, u, return_details=False):
        assert hasattr(self, 'nx') is True
        
        features = self._extract_key_features(u)
        classification, confidence_scores = self._classify_from_features(features)
        
        if return_details:
            return classification, confidence_scores, features
        else:
            return classification
    
    def _extract_key_features(self, u):
        features = {}
        
        u_windowed = self._apply_spectral_window(u)
        
        spatial_fft_seq = self._compute_spatial_fft_sequence(u_windowed)
        temporal_fft = self._compute_temporal_fft(u)
        
        features['energy_distribution'] = self._compute_energy_distribution(u)
        features['spatial_spectral'] = self._analyze_spatial_spectrum(spatial_fft_seq)
        features['temporal_spectral'] = self._analyze_temporal_spectrum(temporal_fft)
        features['dynamical'] = self._compute_dynamical_properties(u)
        features['conservation_laws'] = self._evaluate_conservation_laws(u)
        
        return features
    
    def _apply_spectral_window(self, u):
        window_x = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.nx) / (self.nx - 1)))
        window_y = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.ny) / (self.ny - 1)))
        window_2d = np.outer(window_x, window_y) 
        u_windowed = np.zeros_like(u)
        for t in range(self.nt):
            u_windowed[t] = u[t] * window_2d
            
        return u_windowed
    
    def _compute_spatial_fft_sequence(self, u):
        spatial_fft_seq = np.zeros((self.nt, self.nx, self.ny), dtype=complex)
        for t in range(self.nt):
            spatial_fft_seq[t] = fftshift(fft2(u[t]))
            
        return spatial_fft_seq
    
    def _compute_temporal_fft(self, u):
        return fftshift(np.abs(np.fft.fft(u, axis=0)))
    
    def _analyze_spatial_spectrum(self, spatial_fft_seq):
        mid_t = self.nt // 2
        spatial_power = np.abs(spatial_fft_seq[mid_t])**2
        
        radial_profile = self._compute_radial_profile(spatial_power)
        angular_profile = self._compute_angular_profile(spatial_power)
        
        spectral_entropy = self._compute_spectral_entropy(spatial_power)
        
        spectral_moments = self._compute_spectral_moments(spatial_power)
        
        peak_analysis = self._analyze_spectral_peaks(radial_profile, angular_profile)
        
        orientation_strength = self._detect_orientation_strength(spatial_power)
        
        return {
            'power': spatial_power,
            'spectral_entropy': spectral_entropy,
            'orientation_strength': orientation_strength,
            'peak_analysis': peak_analysis,
            'spectral_moments': spectral_moments
        }
    
    def _compute_radial_profile(self, power_spectrum):
        k_max = np.max(self.kr_physical)
        n_bins = min(20, int(np.sqrt(power_spectrum.shape[0])))
        
        radial_profile = np.zeros(n_bins)
        bin_centers = np.zeros(n_bins)
        
        for i in range(n_bins):
            k_min = i * k_max / n_bins
            k_max_bin = (i + 1) * k_max / n_bins
            mask = (self.kr_physical >= k_min) & (self.kr_physical < k_max_bin)
            if np.any(mask):
                radial_profile[i] = np.mean(power_spectrum[mask])
                bin_centers[i] = (k_min + k_max_bin) / 2
                
        return {
            'profile': radial_profile,
            'bin_centers': bin_centers
        }
    
    def _compute_angular_profile(self, power_spectrum):
        n_bins = 36
        angular_profile = np.zeros(n_bins)
        
        for i in range(n_bins):
            theta_min = i * 2 * np.pi / n_bins - np.pi
            theta_max = (i + 1) * 2 * np.pi / n_bins - np.pi
            mask = (self.ktheta_physical >= theta_min) & (self.ktheta_physical < theta_max)
            
            if np.any(mask):
                angular_profile[i] = np.mean(power_spectrum[mask])
                
        smoothed_profile = gaussian_filter(angular_profile, sigma=1.0, mode='wrap')
                
        return {
            'profile': angular_profile,
            'smoothed': smoothed_profile,
            'variance': np.var(angular_profile) / (np.mean(angular_profile) + 1e-10)
        }
    
    def _compute_spectral_entropy(self, power_spectrum):
        ps_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        entropy = -np.sum(ps_normalized * np.log2(ps_normalized + 1e-10))
        max_entropy = np.log2(power_spectrum.size)
        return entropy / max_entropy
    
    def _compute_spectral_moments(self, power_spectrum):
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return {
                'radial_mean': 0, 
                'radial_variance': 0,
                'circularity': 0
            }
        
        M1_r = np.sum(self.kr_physical * power_spectrum) / total_power
        M2_r = np.sum(self.kr_physical**2 * power_spectrum) / total_power
        
        M1_x = np.sum(self.kx_grid_physical * power_spectrum) / total_power
        M1_y = np.sum(self.ky_grid_physical * power_spectrum) / total_power
        
        circularity = 1.0 - np.sqrt(M1_x**2 + M1_y**2) / (M1_r + 1e-10)
        
        return {
            'radial_mean': M1_r, 
            'radial_variance': M2_r - M1_r**2,
            'circularity': circularity
        }
    
    def _analyze_spectral_peaks(self, radial_profile, angular_profile):
        radial_data = radial_profile['profile']
        angular_data = angular_profile['smoothed']
        
        filtered_radial = gaussian_filter(radial_data, sigma=1.0)
        radial_peaks, radial_props = signal.find_peaks(filtered_radial, height=0.05*np.max(filtered_radial), distance=2)
        
        angular_peaks, angular_props = signal.find_peaks(angular_data, height=0.05*np.max(angular_data), distance=3)
        
        peak_count_radial = len(radial_peaks)
        peak_count_angular = len(angular_peaks)
        
        peak_heights_radial = radial_props['peak_heights'] if peak_count_radial > 0 else np.array([])
        peak_heights_angular = angular_props['peak_heights'] if peak_count_angular > 0 else np.array([])
        
        return {
            'radial_peak_count': peak_count_radial,
            'angular_peak_count': peak_count_angular,
            'radial_peak_heights': peak_heights_radial,
            'angular_peak_heights': peak_heights_angular
        }
    
    def _detect_orientation_strength(self, power_spectrum):
        angular_data = self._compute_angular_profile(power_spectrum)['smoothed']
        
        angular_data_normalized = angular_data / (np.max(angular_data) + 1e-10)
        
        peaks, _ = signal.find_peaks(angular_data_normalized, height=0.3, distance=5)
        
        if len(peaks) == 0:
            return 0.0
        
        peak_heights = angular_data_normalized[peaks]
        peak_positions = peaks * (2*np.pi/36)
        
        if len(peaks) == 2:
            peak_separation = np.abs(peak_positions[0] - peak_positions[1])
            if np.abs(peak_separation - np.pi) < np.pi/4:
                return np.mean(peak_heights) * 2.0
                
        return np.mean(peak_heights)
    
    def _analyze_temporal_spectrum(self, temporal_fft):
        mean_temporal_power = np.mean(temporal_fft, axis=(1, 2))
        
        temporal_entropy = self._compute_spectral_entropy(mean_temporal_power)
        
        peaks, props = signal.find_peaks(mean_temporal_power, height=0.1*np.max(mean_temporal_power), distance=3)
        peak_count = len(peaks)
        peak_heights = props['peak_heights'] if peak_count > 0 else np.array([])
        
        dominant_freq = 0
        if peak_count > 0:
            dominant_idx = peaks[np.argmax(peak_heights)]
            dominant_freq = self.omega_physical[dominant_idx] if hasattr(self, 'omega_physical') else dominant_idx / self.nt
        
        peak_ratio = 0
        if peak_count > 0:
            background = np.mean(mean_temporal_power)
            if background > 0:
                peak_ratio = np.max(peak_heights) / background
        
        wavelet_features = self._compute_wavelet_features(temporal_fft)
        
        return {
            'power': mean_temporal_power,
            'entropy': temporal_entropy,
            'peak_count': peak_count,
            'peak_heights': peak_heights,
            'dominant_frequency': dominant_freq,
            'peak_ratio': peak_ratio,
            'wavelet_features': wavelet_features
        }
    
    def _compute_wavelet_features(self, temporal_fft):
        center_x, center_y = self.nx // 2, self.ny // 2
        time_series = temporal_fft[:, center_x, center_y]
        
        coeffs = pywt.wavedec(time_series, 'db4', level=5)
        energy_per_level = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energy_per_level) + 1e-10
        energy_ratio = [e / total_energy for e in energy_per_level]
        
        return {
            'energy_ratio': energy_ratio,
            'high_freq_ratio': sum(energy_ratio[1:3]) / total_energy if total_energy > 0 else 0,
            'mid_freq_ratio': sum(energy_ratio[3:5]) / total_energy if total_energy > 0 else 0,
            'low_freq_ratio': energy_ratio[0] / total_energy if total_energy > 0 else 0
        }
    
    def _compute_dynamical_properties(self, u):
        dx_u = np.zeros_like(u)
        dy_u = np.zeros_like(u)
        dt_u = np.zeros_like(u)
        
        dx_u[:, 1:self.nx-1, :] = (u[:, 2:self.nx, :] - u[:, 0:self.nx-2, :]) / (2 * self.dx)
        dy_u[:, :, 1:self.ny-1] = (u[:, :, 2:self.ny] - u[:, :, 0:self.ny-2]) / (2 * self.dy)
        
        dt_u[1:self.nt-1, :, :] = (u[2:self.nt, :, :] - u[0:self.nt-2, :, :]) / (2 * self.dt)
        
        energy_density = 0.5 * (dt_u**2 + dx_u**2 + dy_u**2) + (1 - np.cos(u))
        
        energy_center = self._track_energy_center(energy_density)
        
        energy_concentration = self._compute_energy_concentration(energy_density)
        
        total_energy = np.sum(energy_density, axis=(1, 2)) * self.dx * self.dy
        energy_conservation = np.std(total_energy) / np.mean(total_energy) if np.mean(total_energy) > 0 else 0
        
        momentum_x = np.sum(dt_u * dx_u, axis=(1, 2)) * self.dx * self.dy
        momentum_y = np.sum(dt_u * dy_u, axis=(1, 2)) * self.dx * self.dy
        
        return {
            'energy_center': energy_center,
            'energy_concentration': energy_concentration,
            'energy_conservation': energy_conservation,
            'total_energy': total_energy,
            'momentum_x': momentum_x,
            'momentum_y': momentum_y
        }
    
    def _track_energy_center(self, energy_density):
        x_grid, y_grid = np.meshgrid(np.linspace(-self.Lx, self.Lx, self.nx), 
                                      np.linspace(-self.Ly, self.Ly, self.ny), indexing='ij')
        
        energy_center_x = np.zeros(self.nt)
        energy_center_y = np.zeros(self.nt)
        
        for t in range(self.nt):
            total_energy = np.sum(energy_density[t]) + 1e-10
            energy_center_x[t] = np.sum(energy_density[t] * x_grid) / total_energy
            energy_center_y[t] = np.sum(energy_density[t] * y_grid) / total_energy
            
        if self.nt > 1:
            dx = energy_center_x[-1] - energy_center_x[0]
            dy = energy_center_y[-1] - energy_center_y[0]
            displacement = np.sqrt(dx**2 + dy**2)
            
            path_length = 0
            for t in range(1, self.nt):
                path_length += np.sqrt((energy_center_x[t] - energy_center_x[t-1])**2 + 
                                       (energy_center_y[t] - energy_center_y[t-1])**2)
                
            straightness = displacement / (path_length + 1e-10)
            
            velocity = path_length / (self.nt * self.dt)
            
            return {
                'x': energy_center_x,
                'y': energy_center_y,
                'displacement': displacement,
                'path_length': path_length,
                'straightness': straightness,
                'velocity': velocity
            }
        else:
            return {
                'x': energy_center_x,
                'y': energy_center_y,
                'displacement': 0,
                'path_length': 0,
                'straightness': 1,
                'velocity': 0
            }
    
    def _compute_energy_concentration(self, energy_density):
        max_energy = np.max(energy_density, axis=(1, 2))
        mean_energy = np.mean(energy_density, axis=(1, 2))
        
        concentration_ratio = max_energy / (mean_energy + 1e-10)
        
        r_grid = np.sqrt((self.X - np.mean(self.X))**2 + (self.Y - np.mean(self.Y))**2)
        
        energy_radius = np.zeros(self.nt)
        for t in range(self.nt):
            energy_radius[t] = np.sum(r_grid * energy_density[t]) / (np.sum(energy_density[t]) + 1e-10)
            
        if self.nt > 1:
            radius_change_rate = (energy_radius[-1] - energy_radius[0]) / (self.nt * self.dt)
        else:
            radius_change_rate = 0
            
        return {
            'ratio': np.mean(concentration_ratio),
            'radius': energy_radius,
            'radius_change_rate': radius_change_rate
        }
    
    def _evaluate_conservation_laws(self, u):
        dx_u = np.zeros_like(u)
        dy_u = np.zeros_like(u)
        dt_u = np.zeros_like(u)
        
        dx_u[:, 1:self.nx-1, :] = (u[:, 2:self.nx, :] - u[:, 0:self.nx-2, :]) / (2 * self.dx)
        dy_u[:, :, 1:self.ny-1] = (u[:, :, 2:self.ny] - u[:, :, 0:self.ny-2]) / (2 * self.dy)
        
        dt_u[1:self.nt-1, :, :] = (u[2:self.nt, :, :] - u[0:self.nt-2, :, :]) / (2 * self.dt)
        
        energy_density = 0.5 * (dt_u**2 + dx_u**2 + dy_u**2) + (1 - np.cos(u))
        momentum_density_x = dt_u * dx_u
        momentum_density_y = dt_u * dy_u
        
        valid_x_slice = slice(1, self.nx-1)
        valid_y_slice = slice(1, self.ny-1)
        valid_t_slice = slice(1, self.nt-1)
        
        total_energy = np.sum(energy_density[valid_t_slice, valid_x_slice, valid_y_slice], axis=(1, 2)) * self.dx * self.dy
        total_momentum_x = np.sum(momentum_density_x[valid_t_slice, valid_x_slice, valid_y_slice], axis=(1, 2)) * self.dx * self.dy
        total_momentum_y = np.sum(momentum_density_y[valid_t_slice, valid_x_slice, valid_y_slice], axis=(1, 2)) * self.dx * self.dy
        
        energy_deviation = np.std(total_energy) / np.mean(total_energy) if np.mean(total_energy) > 0 else 0
        momentum_deviation = np.std(np.sqrt(total_momentum_x**2 + total_momentum_y**2))
        
        return {
            'energy_deviation': energy_deviation,
            'momentum_deviation': momentum_deviation,
            'energy_profile': total_energy,
            'momentum_profile': np.sqrt(total_momentum_x**2 + total_momentum_y**2)
        }
    
    def _compute_energy_distribution(self, u):
        dx_u = np.zeros_like(u)
        dy_u = np.zeros_like(u)
        dt_u = np.zeros_like(u)
        
        dx_u[:, 1:self.nx-1, :] = (u[:, 2:self.nx, :] - u[:, 0:self.nx-2, :]) / (2 * self.dx)
        dy_u[:, :, 1:self.ny-1] = (u[:, :, 2:self.ny] - u[:, :, 0:self.ny-2]) / (2 * self.dy)
        
        dt_u[1:self.nt-1, :, :] = (u[2:self.nt, :, :] - u[0:self.nt-2, :, :]) / (2 * self.dt)
        
        kinetic_energy = 0.5 * dt_u**2
        gradient_energy = 0.5 * (dx_u**2 + dy_u**2)
        potential_energy = 1 - np.cos(u)
        
        total_kinetic = np.sum(kinetic_energy, axis=(1, 2)) * self.dx * self.dy
        total_gradient = np.sum(gradient_energy, axis=(1, 2)) * self.dx * self.dy
        total_potential = np.sum(potential_energy, axis=(1, 2)) * self.dx * self.dy
        
        kinetic_ratio = np.mean(total_kinetic) / (np.mean(total_kinetic + total_gradient + total_potential) + 1e-10)
        gradient_ratio = np.mean(total_gradient) / (np.mean(total_kinetic + total_gradient + total_potential) + 1e-10)
        potential_ratio = np.mean(total_potential) / (np.mean(total_kinetic + total_gradient + total_potential) + 1e-10)
        
        return {
            'kinetic_ratio': kinetic_ratio,
            'gradient_ratio': gradient_ratio,
            'potential_ratio': potential_ratio,
            'kinetic_profile': total_kinetic,
            'gradient_profile': total_gradient,
            'potential_profile': total_potential
        }
    
    def _classify_from_features(self, features):
        scores = np.zeros(len(self.class_names))
        
        spatial_spectral = features['spatial_spectral']
        temporal_spectral = features['temporal_spectral']
        dynamical = features['dynamical']
        conservation = features['conservation_laws']
        energy_distribution = features['energy_distribution']
        
        spectral_entropy = spatial_spectral['spectral_entropy']
        orientation_strength = spatial_spectral['orientation_strength']
        
        peak_analysis = spatial_spectral['peak_analysis']
        radial_peak_count = peak_analysis['radial_peak_count']
        angular_peak_count = peak_analysis['angular_peak_count']
        
        energy_center = dynamical['energy_center']
        energy_concentration = dynamical['energy_concentration']
        
        velocity = energy_center['velocity']
        straightness = energy_center['straightness']
        
        concentration_ratio = energy_concentration['ratio']
        radius_change_rate = energy_concentration['radius_change_rate']
        
        energy_deviation = conservation['energy_deviation']
        momentum_deviation = conservation['momentum_deviation']
        
        temporal_peak_ratio = temporal_spectral['peak_ratio']
        temporal_entropy = temporal_spectral['entropy']
        peak_count = temporal_spectral['peak_count']
        
        kinetic_ratio = energy_distribution['kinetic_ratio']
        gradient_ratio = energy_distribution['gradient_ratio']
        
        kink_score = self._calculate_kink_score(
            spectral_entropy, velocity, straightness, orientation_strength,
            radial_peak_count, gradient_ratio, kinetic_ratio
        )
        
        kink_antikink_score = self._calculate_kink_antikink_score(
            spectral_entropy, radial_peak_count, velocity, straightness, 
            angular_peak_count, gradient_ratio
        )
        
        breather_score = self._calculate_breather_score(
            temporal_peak_ratio, concentration_ratio, peak_count,
            temporal_entropy, kinetic_ratio, energy_deviation
        )
        
        scores[0] = kink_score
        scores[1] = kink_antikink_score
        scores[2] = breather_score
        
        classification = np.argmax(scores)
        
        return self.class_names[classification], scores
    
    def _calculate_kink_score(self, spectral_entropy, velocity, straightness, orientation_strength,
                              radial_peak_count, gradient_ratio, kinetic_ratio):
        score = 0
        
        if spectral_entropy < 0.6 and velocity > 0.1 * self.typical_scales['kink_velocity']:
            score = (1.0 - spectral_entropy) * (velocity / self.typical_scales['kink_velocity'])
            
            if straightness > 0.8:
                score *= straightness
                
            if orientation_strength > 0.5:
                score *= (1.0 + orientation_strength)
                
            if 1 <= radial_peak_count <= 2:
                score *= 1.5
                
            if gradient_ratio > 0.4:
                score *= (1.0 + gradient_ratio)
                
            if kinetic_ratio > 0.2:
                score *= (1.0 + kinetic_ratio)
                
        return max(0, min(1, score))
    
    def _calculate_kink_antikink_score(self, spectral_entropy, radial_peak_count, velocity, straightness, 
                                      angular_peak_count, gradient_ratio):
        score = 0
        
        if spectral_entropy < 0.7 and radial_peak_count >= 2 and velocity > 0.05 * self.typical_scales['kink_velocity']:
            score = (1.0 - spectral_entropy) * min(1.0, radial_peak_count / 2.0)
            
            if straightness > 0.7:
                score *= straightness
                
            if angular_peak_count >= 2:
                score *= min(1.5, angular_peak_count / 2.0)
                
            if gradient_ratio > 0.3:
                score *= (1.0 + gradient_ratio)
                
        return max(0, min(1, score))
    
    def _calculate_breather_score(self, temporal_peak_ratio, concentration_ratio, peak_count,
                                 temporal_entropy, kinetic_ratio, energy_deviation):
        score = 0
        
        if temporal_peak_ratio > 2.0 and concentration_ratio > 2.0:
            score = temporal_peak_ratio / 5.0 * concentration_ratio / 3.0
            
            if peak_count >= 1:
                score *= min(2.0, peak_count)
                
            if temporal_entropy < 0.5:
                score *= (1.0 - temporal_entropy)
                
            if kinetic_ratio > 0.3:
                score *= (1.0 + kinetic_ratio)
                
            if energy_deviation < 0.1:
                score *= (1.0 + (1.0 - 10.0 * energy_deviation))
                
        return max(0, min(1, score))

def compute_curl_field(u, classifier):
    dx_u = np.zeros((classifier.nx, classifier.ny))
    dy_u = np.zeros((classifier.nx, classifier.ny))
    
    dx_u[1:classifier.nx-1, :] = (u[2:classifier.nx, :] - u[0:classifier.nx-2, :]) / (2 * classifier.dx)
    dy_u[:, 1:classifier.ny-1] = (u[:, 2:classifier.ny] - u[:, 0:classifier.ny-2]) / (2 * classifier.dy)
    
    curl = np.zeros((classifier.nx, classifier.ny))
    
    dx_dy = (dy_u[2:classifier.nx, 1:classifier.ny-1] - dy_u[0:classifier.nx-2, 1:classifier.ny-1]) / (2 * classifier.dx)
    dy_dx = (dx_u[1:classifier.nx-1, 2:classifier.ny] - dx_u[1:classifier.nx-1, 0:classifier.ny-2]) / (2 * classifier.dy)
    
    curl[1:classifier.nx-1, 1:classifier.ny-1] = dx_dy - dy_dx
    
    return curl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.animation as animation

class SolitonVisualizer:
    def __init__(self, classifier, save_dir="soliton_analysis"):
        self.classifier = classifier
        self.save_dir = save_dir
        self._create_dir(save_dir)
        self.cmap = plt.cm.viridis
        self.cmap_diverging = plt.cm.RdBu_r
        self.figure_dpi = 300
        self.class_colors = sns.color_palette("Set1", len(classifier.class_names))

    def _create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def visualize_and_save(self, u, name=None):
        if name is None:
            name = f"soliton_analysis_{len(os.listdir(self.save_dir))}"

        classification, confidence_scores, features = self.classifier.classify(u, return_details=True)
        
        self._plot_spatial_field(u, name, classification)
        self._plot_energy_distribution(features, name)
        self._plot_spectral_analysis(features, name)
        self._plot_temporal_dynamics(features, name)
        self._plot_classification_results(classification, confidence_scores, name)
        self._create_animation(u, name)

        return classification, confidence_scores, features

    def _plot_spatial_field(self, u, name, classification):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Spatial Field Analysis - {classification}", fontsize=16)
        
        mid_t = u.shape[0] // 2
        
        ax = axes[0, 0]
        im = ax.imshow(u[0], cmap=self.cmap_diverging, origin='lower', 
                      extent=[-self.classifier.Lx, self.classifier.Lx, -self.classifier.Ly, self.classifier.Ly])
        ax.set_title(f"Initial Field (t=0)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = axes[0, 1]
        im = ax.imshow(u[mid_t], cmap=self.cmap_diverging, origin='lower',
                      extent=[-self.classifier.Lx, self.classifier.Lx, -self.classifier.Ly, self.classifier.Ly])
        ax.set_title(f"Mid-simulation Field (t={mid_t*self.classifier.dt:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = axes[1, 0]
        im = ax.imshow(u[-1], cmap=self.cmap_diverging, origin='lower',
                      extent=[-self.classifier.Lx, self.classifier.Lx, -self.classifier.Ly, self.classifier.Ly])
        ax.set_title(f"Final Field (t={self.classifier.T:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = axes[1, 1]
        curl = compute_curl_field(u[-1], self.classifier)
        im = ax.imshow(curl, cmap=self.cmap_diverging, origin='lower',
                      extent=[-self.classifier.Lx, self.classifier.Lx, -self.classifier.Ly, self.classifier.Ly])
        ax.set_title("Curl Field (Final State)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/{name}_spatial_field.png", dpi=self.figure_dpi)
        plt.close()

    def _plot_energy_distribution(self, features, name):
        energy_data = features["energy_distribution"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Energy Distribution Analysis", fontsize=16)
        
        ax = axes[0, 0]
        energy_ratios = [energy_data["kinetic_ratio"], energy_data["gradient_ratio"], energy_data["potential_ratio"]]
        labels = ["Kinetic", "Gradient", "Potential"]
        ax.pie(energy_ratios, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", 3))
        ax.set_title("Energy Components Ratio")
        
        ax = axes[0, 1]
        components = ["Kinetic", "Gradient", "Potential"]
        y_pos = np.arange(len(components))
        values = [energy_data["kinetic_ratio"], energy_data["gradient_ratio"], energy_data["potential_ratio"]]
        ax.barh(y_pos, values, align='center', color=sns.color_palette("viridis", 3))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.set_xlabel('Ratio')
        ax.set_title('Energy Component Distribution')
        
        ax = axes[1, 0]
        time_steps = np.arange(len(energy_data["kinetic_profile"]))
        ax.plot(time_steps, energy_data["kinetic_profile"], 'r-', label='Kinetic')
        ax.plot(time_steps, energy_data["gradient_profile"], 'g-', label='Gradient')
        ax.plot(time_steps, energy_data["potential_profile"], 'b-', label='Potential')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Profiles Over Time')
        ax.legend()
        
        ax = axes[1, 1]
        total_energy = energy_data["kinetic_profile"] + energy_data["gradient_profile"] + energy_data["potential_profile"]
        ax.plot(time_steps, total_energy, 'k-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Total Energy')
        ax.set_title(f'Total Energy (Deviation: {features["conservation_laws"]["energy_deviation"]:.4f})')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/{name}_energy_distribution.png", dpi=self.figure_dpi)
        plt.close()

    def _plot_spectral_analysis(self, features, name):
        spatial_spectral = features["spatial_spectral"]
        
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 3, figure=fig)
        fig.suptitle("Spatial Spectral Analysis", fontsize=16)
        
        power = spatial_spectral["power"]
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(np.log10(np.abs(power) + 1e-10), cmap=self.cmap, origin='lower')
        ax.set_title(f"Power Spectrum (Entropy: {spatial_spectral['spectral_entropy']:.4f})")
        ax.set_xlabel("kx index")
        ax.set_ylabel("ky index")
        plt.colorbar(im, ax=ax)
        
        ax = fig.add_subplot(gs[0, 1])
        peak_analysis = spatial_spectral["peak_analysis"]
        metrics = [
            peak_analysis["radial_peak_count"],
            peak_analysis["angular_peak_count"],
            spatial_spectral["orientation_strength"],
            spatial_spectral["spectral_entropy"]
        ]
        labels = ['Radial\nPeaks', 'Angular\nPeaks', 'Orientation\nStrength', 'Spectral\nEntropy']
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, metrics, align='center', color=sns.color_palette("viridis", len(labels)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title('Spectral Metrics')
        ax.set_xlabel('Value')
        
        ax = fig.add_subplot(gs[0, 2])
        energy_center = features["dynamical"]["energy_center"]
        x_traj = energy_center["x"]
        y_traj = energy_center["y"]
        ax.plot(x_traj, y_traj, 'b-', linewidth=2)
        ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Energy Center Trajectory (v={energy_center["velocity"]:.4f})')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/{name}_spectral_analysis.png", dpi=self.figure_dpi)
        plt.close()

    def _plot_temporal_dynamics(self, features, name):
        temporal = features["temporal_spectral"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Temporal Dynamics Analysis", fontsize=16)
        
        ax = axes[0]
        power = temporal["power"]
        freq = np.arange(len(power))
        ax.bar(freq, power, width=0.7, color='g')
        ax.set_xlabel('Frequency Index')
        ax.set_ylabel('Power')
        ax.set_title(f'Temporal Power Spectrum (Peaks: {temporal["peak_count"]})')
        
        ax = axes[1]
        wavelet = temporal["wavelet_features"]
        metrics = [
            wavelet["low_freq_ratio"],
            wavelet["mid_freq_ratio"],
            wavelet["high_freq_ratio"],
            temporal["entropy"]
        ]
        labels = ['Low Freq\nRatio', 'Mid Freq\nRatio', 'High Freq\nRatio', 'Temporal\nEntropy']
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, metrics, align='center', color=sns.color_palette("viridis", len(labels)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title('Temporal Frequency Analysis')
        ax.set_xlabel('Value')
        
        ax = axes[2]
        dynamical = features["dynamical"]
        energy_radius = dynamical["energy_concentration"]["radius"]
        time_steps = np.arange(len(energy_radius))
        ax.plot(time_steps, energy_radius, 'g-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Radius')
        ax.set_title(f'Energy Radius (Change Rate: {dynamical["energy_concentration"]["radius_change_rate"]:.4f})')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/{name}_temporal_dynamics.png", dpi=self.figure_dpi)
        plt.close()

    def _plot_classification_results(self, classification, confidence_scores, name):
        class_index = self.classifier.class_names.index(classification)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Classification Results: {classification}", fontsize=16)
        
        ax = axes[0]
        y_pos = np.arange(len(self.classifier.class_names))
        bars = ax.barh(y_pos, confidence_scores, align='center', color=[self.class_colors[i] for i in range(len(self.classifier.class_names))])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.classifier.class_names)
        ax.set_xlabel('Confidence Score')
        ax.set_title('Classification Confidence')
        
        for i, bar in enumerate(bars):
            if i == class_index:
                bar.set_alpha(1.0)
            else:
                bar.set_alpha(0.6)
        
        ax = axes[1]
        characteristic_features = self._get_characteristic_features(classification, confidence_scores, self.classifier.class_names)
        feature_names = list(characteristic_features.keys())
        feature_values = list(characteristic_features.values())
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, feature_values, align='center', color=self.class_colors[class_index])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Value')
        ax.set_title(f'Key Distinguishing Features for {classification}')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/{name}_classification.png", dpi=self.figure_dpi)
        plt.close()

    def _get_characteristic_features(self, classification, confidence_scores, class_names):
        if classification == "Kink":
            return {
                "Orientation Strength": confidence_scores[0] * 0.8,
                "Trajectory Straightness": confidence_scores[0] * 0.9,
                "Velocity": confidence_scores[0] * 0.7,
                "Gradient Energy": confidence_scores[0] * 0.6,
                "Temporal Stability": confidence_scores[0] * 0.5
            }
        elif classification == "Kink-Antikink Pair":
            return {
                "Angular Peaks": confidence_scores[1] * 0.9,
                "Radial Peaks": confidence_scores[1] * 0.8,
                "Interaction Strength": confidence_scores[1] * 0.7,
                "Boundary Structure": confidence_scores[1] * 0.6,
                "Energy Distribution": confidence_scores[1] * 0.5
            }
        elif classification == "Breather":
            return {
                "Temporal Oscillation": confidence_scores[2] * 0.9,
                "Energy Concentration": confidence_scores[2] * 0.8,
                "Kinetic Energy Ratio": confidence_scores[2] * 0.7,
                "Temporal Entropy": confidence_scores[2] * 0.6,
                "Frequency Peak Ratio": confidence_scores[2] * 0.5
            }
        else:
            return {"Unknown": 1.0}

    def create_dashboard(self, u, name=None):
        if name is None:
            name = f"soliton_dashboard_{len(os.listdir(self.save_dir))}"
        
        classification, confidence_scores, features = self.classifier.classify(u, return_details=True)
        
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        fig.suptitle(f"Soliton Analysis: {classification}", fontsize=20)
        
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(u[-1], cmap=self.cmap_diverging, origin='lower',
                      extent=[-self.classifier.Lx, self.classifier.Lx, -self.classifier.Ly, self.classifier.Ly])
        ax.set_title("Final Field Configuration")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = fig.add_subplot(gs[0, 1])
        y_pos = np.arange(len(self.classifier.class_names))
        bars = ax.barh(y_pos, confidence_scores, align='center', color=[self.class_colors[i] for i in range(len(self.classifier.class_names))])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.classifier.class_names)
        ax.set_xlabel('Confidence')
        ax.set_title('Classification Confidence')
        
        ax = fig.add_subplot(gs[0, 2])
        energy_data = features["energy_distribution"]
        energy_ratios = [energy_data["kinetic_ratio"], energy_data["gradient_ratio"], energy_data["potential_ratio"]]
        labels = ["Kinetic", "Gradient", "Potential"]
        ax.pie(energy_ratios, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", 3))
        ax.set_title("Energy Components")
        
        ax = fig.add_subplot(gs[1, 0])
        power = features["spatial_spectral"]["power"]
        im = ax.imshow(np.log10(np.abs(power) + 1e-10), cmap=self.cmap, origin='lower')
        ax.set_title("Power Spectrum (log scale)")
        ax.set_xlabel("kx index")
        ax.set_ylabel("ky index")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = fig.add_subplot(gs[1, 1])
        x_traj = features["dynamical"]["energy_center"]["x"]
        y_traj = features["dynamical"]["energy_center"]["y"]
        ax.plot(x_traj, y_traj, 'b-', linewidth=2)
        ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
        ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Energy Center Trajectory (v={features["dynamical"]["energy_center"]["velocity"]:.4f})')
        ax.grid(True)
        ax.legend()
        
        ax = fig.add_subplot(gs[1, 2])
        temporal = features["temporal_spectral"]
        power = temporal["power"]
        freq = np.arange(len(power))
        ax.bar(freq, power, width=0.7, color='g')
        ax.set_xlabel('Frequency Index')
        ax.set_ylabel('Power')
        ax.set_title(f'Temporal Spectrum (Peaks: {temporal["peak_count"]})')
        
        ax = fig.add_subplot(gs[2, 0:2])
        self._plot_key_differentiating_features(ax, features, classification)
        
        ax = fig.add_subplot(gs[2, 2])
        curl = compute_curl_field(u[-1], self.classifier)
        im = ax.imshow(curl, cmap=self.cmap_diverging, origin='lower',
                      extent=[-self.classifier.Lx, self.classifier.Lx, -self.classifier.Ly, self.classifier.Ly])
        ax.set_title("Curl Field")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.save_dir}/{name}.png", dpi=self.figure_dpi)
        plt.close()
        
        return classification, confidence_scores, features
        
    def _plot_key_differentiating_features(self, ax, features, classification):
        key_features = []
        feature_values = []
        
        if classification == "Kink":
            key_features = [
                "Velocity",
                "Path Straightness", 
                "Orientation Strength",
                "Radial Peak Count", 
                "Gradient Energy Ratio"
            ]
            feature_values = [
                features["dynamical"]["energy_center"]["velocity"] / self.classifier.typical_scales["kink_velocity"],
                features["dynamical"]["energy_center"]["straightness"],
                features["spatial_spectral"]["orientation_strength"],
                min(1.0, features["spatial_spectral"]["peak_analysis"]["radial_peak_count"] / 2.0),
                features["energy_distribution"]["gradient_ratio"]
            ]
        elif classification == "Kink-Antikink Pair":
            key_features = [
                "Radial Peak Count",
                "Angular Peak Count",
                "Orientation Strength",
                "Spectral Entropy",
                "Energy Conservation"
            ]
            feature_values = [
                min(1.0, features["spatial_spectral"]["peak_analysis"]["radial_peak_count"] / 3.0),
                min(1.0, features["spatial_spectral"]["peak_analysis"]["angular_peak_count"] / 3.0),
                features["spatial_spectral"]["orientation_strength"],
                1.0 - features["spatial_spectral"]["spectral_entropy"],
                1.0 - features["conservation_laws"]["energy_deviation"]
            ]
        elif classification == "Breather":
            key_features = [
                "Temporal Peak Ratio",
                "Energy Concentration",
                "Kinetic Energy Ratio",
                "Temporal Entropy",
                "Radius Change Rate"
            ]
            feature_values = [
                min(1.0, features["temporal_spectral"]["peak_ratio"] / 5.0),
                min(1.0, features["dynamical"]["energy_concentration"]["ratio"] / 5.0),
                features["energy_distribution"]["kinetic_ratio"] * 2,
                1.0 - features["temporal_spectral"]["entropy"],
                abs(features["dynamical"]["energy_concentration"]["radius_change_rate"]) * 10
            ]
        
        y_pos = np.arange(len(key_features))
        ax.barh(y_pos, feature_values, align='center', color=self.class_colors[self.classifier.class_names.index(classification)])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(key_features)
        ax.set_title(f'Key Features for {classification}')
        ax.set_xlabel('Normalized Value')
        ax.set_xlim(0, 1.2)
        
        for i, v in enumerate(feature_values):
            ax.text(max(0.05, min(v - 0.15, 1.0)), i, f"{v:.2f}", va='center')

def create_comparative_analysis(solutions, classifier, save_dir="soliton_analysis"):
    visualizer = SolitonVisualizer(classifier, save_dir)
    
    results = {}
    for name, u in solutions.items():
        print(f"Processing solution: {name}")
        classification, scores, features = visualizer.create_dashboard(u, name)
        results[name] = {
            "classification": classification,
            "confidence_scores": scores,
            "features": features
        }
    
    solution_names = list(results.keys())
    classifications = [r["classification"] for r in results.values()]
    
    unique_classes = sorted(list(set(classifications)))
    class_counts = {c: classifications.count(c) for c in unique_classes}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Comparative Analysis of Soliton Solutions", fontsize=16)
    
    ax = axes[0, 0]
    ax.bar(range(len(unique_classes)), [class_counts[c] for c in unique_classes], 
           color=[visualizer.class_colors[classifier.class_names.index(c)] for c in unique_classes])
    ax.set_xticks(range(len(unique_classes)))
    ax.set_xticklabels(unique_classes)
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Soliton Types")
    
    ax = axes[0, 1]
    for i, name in enumerate(solution_names):
        result = results[name]
        class_idx = classifier.class_names.index(result["classification"])
        ax.scatter(i, result["confidence_scores"][class_idx], 
                  color=visualizer.class_colors[class_idx], s=100, alpha=0.7)
        
    ax.set_xticks(range(len(solution_names)))
    ax.set_xticklabels([name[:10] for name in solution_names], rotation=45, ha="right")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Classification Confidence by Solution")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    temporal_metrics = {c: [] for c in unique_classes}
    for result in results.values():
        c = result["classification"]
        features = result["features"]
        temporal_metrics[c].append([
            features["temporal_spectral"]["peak_ratio"],
            features["temporal_spectral"]["peak_count"],
            features["temporal_spectral"]["entropy"]
        ])
    
    width = 0.25
    positions = np.arange(3)
    labels = ["Peak Ratio", "Peak Count", "Temporal Entropy"]
    
    for i, c in enumerate(unique_classes):
        if temporal_metrics[c]:
            means = np.mean(temporal_metrics[c], axis=0)
            ax.bar(positions + i*width, means, width, label=c, 
                  color=visualizer.class_colors[classifier.class_names.index(c)])
    
    ax.set_xticks(positions + width)
    ax.set_xticklabels(labels)
    ax.set_title("Temporal Characteristics by Type")
    ax.legend()
    
    ax = axes[1, 1]
    spatial_metrics = {c: [] for c in unique_classes}
    for result in results.values():
        c = result["classification"]
        features = result["features"]
        spatial_metrics[c].append([
            features["spatial_spectral"]["orientation_strength"],
            features["dynamical"]["energy_center"]["velocity"],
            features["dynamical"]["energy_concentration"]["ratio"],
            features["energy_distribution"]["kinetic_ratio"]
        ])
    
    positions = np.arange(4)
    labels = ["Orientation", "Velocity", "Concentration", "Kinetic Ratio"]
    
    for i, c in enumerate(unique_classes):
        if spatial_metrics[c]:
            means = np.mean(spatial_metrics[c], axis=0)
            ax.bar(positions + i*width, means, width, label=c, 
                  color=visualizer.class_colors[classifier.class_names.index(c)])
    
    ax.set_xticks(positions + width)
    ax.set_xticklabels(labels)
    ax.set_title("Spatial Characteristics by Type")
    ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_dir}/comparative_analysis.png", dpi=300)
    plt.close()
    
    return result

def batch_process_solutions(solution_dict, classifier, save_dir=None):
    assert save_dir is not None
    visualizer = SolitonVisualizer(classifier, save_dir)
    results = {}
    for name, u in solution_dict.items():
        print(f"Processing solution: {name}") 
        classification, scores, features = visualizer.create_dashboard(u, name)
        results[name] = {
            "classification": classification,
            "confidence_scores": scores,
            "features": features
        }

    return results
