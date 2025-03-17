import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from precise_nlse_phenomena import NLSEPhenomenonSampler


def generate_random_initial_conditions(num_samples, nx, ny, Lx, Ly):
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    initial_conditions = []
    sampler = NLSEPhenomenonSampler(nx, ny, Lx)
    phenomena = ["multi_soliton", "chaotic"]
    samples = sampler.generate_ensemble("chaotic", n_samples=num_samples) 
    return samples, X, Y

def generate_spatial_variations(num_samples, X, Y, Lx, Ly):
    spatial_variations = []
    
    for i in range(num_samples):
        variation_type = np.random.randint(0, 3)
        
        if variation_type == 0:
            sigma = 1.0 + 4.0 * np.random.random()
            m = np.exp(-(X**2 + Y**2)/(2*sigma**2))
        
        elif variation_type == 1:
            k1 = 0.5 + 1.5 * np.random.random()
            k2 = 0.5 + 1.5 * np.random.random()
            amp = 0.2 + 0.8 * np.random.random()
            m = 1.0 + amp * np.cos(k1*X) * np.cos(k2*Y)
        
        else:
            threshold = 1.0 + 3.0 * np.random.random()
            amp = 0.5 + 1.5 * np.random.random()
            r = np.sqrt(X**2 + Y**2)
            m = 1.0 + amp * (r < threshold)
        
        spatial_variations.append(m)
    
    return spatial_variations

def compute_laplacian(u, dx, dy):
    nx, ny = u.shape
    
    kx = 2 * np.pi * fftpack.fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftpack.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K_SQ = KX**2 + KY**2
    
    u_ft = fftpack.fft2(u)
    laplacian_ft = -K_SQ * u_ft
    laplacian = fftpack.ifft2(laplacian_ft)
    
    return laplacian

def cubic_nonlinearity(u):
    return np.abs(u)**2 * u

def cubic_quintic_nonlinearity(u, s1, s2):
    return (s1 * np.abs(u)**2 + s2 * np.abs(u)**4) * u

def saturating_nonlinearity(u, kappa):
    return (np.abs(u)**2 / (1 + kappa * np.abs(u)**2)) * u

def approach1_physical_scaling(u0, m, laplacian_u0, nonlinearity_func):
    max_laplacian = np.max(np.abs(laplacian_u0))
    max_m = np.max(m)
    
    nonlinear_term = nonlinearity_func(u0)
    max_nonlinear = np.max(np.abs(nonlinear_term))
    
    if max_m <= 1e-10 or max_nonlinear <= 1e-10:
        return 1.0
    
    scale = max_laplacian / (max_m * max_nonlinear)
    if nonlinearity_func.__name__ == 'cubic_nonlinearity':
        scale = np.clip(scale, 0.1, 100.0)
    elif 'cubic_quintic' in nonlinearity_func.__name__:
        scale = np.clip(scale, 0.01, 10.0)
    elif 'saturating' in nonlinearity_func.__name__:
        scale = np.clip(scale, 0.1, 1000.0)
    else:
        scale = np.clip(scale, 0.01, 100.0)
    
    return scale

def approach2_fourier_scaling(u0, m, nonlinearity_func, dx, dy):
    nx, ny = u0.shape
    
    kx = 2 * np.pi * fftpack.fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftpack.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K_SQ = KX**2 + KY**2
    
    u0_ft = fftpack.fft2(u0)
    
    nonlinear_term = nonlinearity_func(u0)
    m_nonlinear = m * nonlinear_term
    try:
        m_nonlinear_ft = fftpack.fft2(m_nonlinear)
        
        energy_laplacian = np.sum(K_SQ * np.abs(u0_ft)**2)
        energy_nonlinear = np.sum(np.abs(m_nonlinear_ft)**2)
        
        if energy_nonlinear < 1e-10 or not np.isfinite(energy_nonlinear):
            return 1.0
        
        scale = energy_laplacian / energy_nonlinear
    except:
        return 1.0

    if nonlinearity_func.__name__ == 'cubic_nonlinearity':
        scale = np.clip(scale, 0.1, 100.0)
    elif 'cubic_quintic' in nonlinearity_func.__name__:
        scale = np.clip(scale, 0.01, 50.0)
    elif 'saturating' in nonlinearity_func.__name__:
        scale = np.clip(scale, 0.1, 1000.0)
    else:
        scale = np.clip(scale, 0.01, 100.0)
    if not np.isfinite(scale):
        return 1.0
        
    return scale

def approach3_dominant_mode_scaling(u0, m, nonlinearity_func, dx, dy):
    nx, ny = u0.shape 
    kx = 2 * np.pi * fftpack.fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftpack.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K_SQ = KX**2 + KY**2
    
    try:
        u0_ft = fftpack.fft2(u0)
        abs_u0_ft = np.abs(u0_ft)
        max_idx = np.unravel_index(np.argmax(abs_u0_ft), u0_ft.shape)
        k_dominant_sq = K_SQ[max_idx]
        if k_dominant_sq < 1e-6:
            alternative_idx = np.where(K_SQ > 1e-6)
            if len(alternative_idx[0]) > 0:
                alt_max_idx = np.unravel_index(
                    np.argmax(abs_u0_ft[alternative_idx]), 
                    (len(alternative_idx[0]),)
                )
                max_idx = (alternative_idx[0][alt_max_idx], alternative_idx[1][alt_max_idx])
                k_dominant_sq = K_SQ[max_idx]
            else:
                return 1.0
        
        nonlinear_term = nonlinearity_func(u0)
        m_nonlinear = m * nonlinear_term
        m_nonlinear_ft = fftpack.fft2(m_nonlinear)
        
        energy_laplacian = k_dominant_sq * np.abs(u0_ft[max_idx])
        energy_nonlinear = np.abs(m_nonlinear_ft[max_idx])
        
        if energy_nonlinear < 1e-10 or not np.isfinite(energy_nonlinear):
            return 1.0
        
        scale = energy_laplacian / energy_nonlinear
    except:
        return 1.0
    if nonlinearity_func.__name__ == 'cubic_nonlinearity':
        scale = np.clip(scale, 0.5, 50.0)
    elif 'cubic_quintic' in nonlinearity_func.__name__:
        scale = np.clip(scale, 0.1, 10.0)
    elif 'saturating' in nonlinearity_func.__name__:
        scale = np.clip(scale, 1.0, 500.0)
    else:
        scale = np.clip(scale, 0.1, 50.0)
    if not np.isfinite(scale):
        return 1.0
        
    return scale

def calculate_balance_metrics(u0, m, laplacian_u0, scaling_approaches, nonlinearity_funcs, dx, dy):
    metrics = {}
    
    for nl_name, nl_func in nonlinearity_funcs.items():
        metrics[nl_name] = {}
        
        for approach_name, scaling_func in scaling_approaches.items():
            if approach_name == 'Physical':
                scale = scaling_func(u0, m, laplacian_u0, nl_func)
            else:
                scale = scaling_func(u0, m, nl_func, dx, dy)
            
            nl_term = nl_func(u0)
            scaled_nl_term = scale * m * nl_term
            
            metrics[nl_name][approach_name] = {
                'scale': scale,
                'max_ratio': np.max(np.abs(scaled_nl_term)) / np.max(np.abs(laplacian_u0)),
                'rms_ratio': np.sqrt(np.mean(np.abs(scaled_nl_term)**2)) / np.sqrt(np.mean(np.abs(laplacian_u0)**2)),
                'l1_error': np.sum(np.abs(scaled_nl_term - laplacian_u0)) / np.sum(np.abs(laplacian_u0)),
                'l2_error': np.sqrt(np.sum(np.abs(scaled_nl_term - laplacian_u0)**2)) / np.sqrt(np.sum(np.abs(laplacian_u0)**2)),
                'spatial_corr': np.corrcoef(np.abs(scaled_nl_term).flatten(), np.abs(laplacian_u0).flatten())[0, 1]
            }
    
    return metrics

def analyze_nlse_balance(num_samples=100, nx=128, ny=128, Lx=10.0, Ly=10.0):
    dx = 2 * Lx / (nx - 1)
    dy = 2 * Ly / (ny - 1)
    
    initial_conditions, X, Y = generate_random_initial_conditions(num_samples, nx, ny, Lx, Ly)
    spatial_variations = generate_spatial_variations(num_samples, X, Y, Lx, Ly)
    
    scaling_approaches = {
        'Physical': approach1_physical_scaling,
        'Fourier': approach2_fourier_scaling,
        'Dominant': approach3_dominant_mode_scaling
    }
    
    nonlinearity_funcs = {
        'Cubic': lambda u: cubic_nonlinearity(u),
        'CQ_s1-1_s2-0.1': lambda u: cubic_quintic_nonlinearity(u, s1=1.0, s2=-0.1),
        'CQ_s1-3_s2-1': lambda u: cubic_quintic_nonlinearity(u, s1=3.0, s2=-1.0),
        'Sat_k-0.5': lambda u: saturating_nonlinearity(u, kappa=0.5),
        'Sat_k-5': lambda u: saturating_nonlinearity(u, kappa=5.0)
    }
    
    all_metrics = {}
    
    for nl_name in nonlinearity_funcs.keys():
        all_metrics[nl_name] = {}
        for approach_name in scaling_approaches.keys():
            all_metrics[nl_name][approach_name] = {
                'scales': [],
                'max_ratios': [],
                'rms_ratios': [],
                'l1_errors': [],
                'l2_errors': [],
                'spatial_corrs': []
            }
    
    for i in range(num_samples):
        u0 = initial_conditions[i]
        m = spatial_variations[i]
        
        laplacian_u0 = compute_laplacian(u0, dx, dy)
        
        metrics = calculate_balance_metrics(u0, m, laplacian_u0, scaling_approaches, nonlinearity_funcs, dx, dy)
        
        for nl_name, nl_metrics in metrics.items():
            for approach_name, approach_metrics in nl_metrics.items():
                all_metrics[nl_name][approach_name]['scales'].append(approach_metrics['scale'])
                all_metrics[nl_name][approach_name]['max_ratios'].append(approach_metrics['max_ratio'])
                all_metrics[nl_name][approach_name]['rms_ratios'].append(approach_metrics['rms_ratio'])
                all_metrics[nl_name][approach_name]['l1_errors'].append(approach_metrics['l1_error'])
                all_metrics[nl_name][approach_name]['l2_errors'].append(approach_metrics['l2_error'])
                all_metrics[nl_name][approach_name]['spatial_corrs'].append(approach_metrics['spatial_corr'])
    
    return all_metrics

def plot_metrics_comparison(all_metrics):
    metrics_to_plot = ['max_ratios', 'rms_ratios', 'l1_errors', 'l2_errors', 'spatial_corrs']
    metric_titles = ['Max Amplitude Ratio', 'RMS Ratio', 'L1 Error', 'L2 Error', 'Spatial Correlation']
    
    nl_names = list(all_metrics.keys())
    approach_names = list(all_metrics[nl_names[0]].keys())
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)))
    plt.subplots_adjust(hspace=0.4)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(approach_names)))
    
    for i, (metric_name, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[i]
        
        for j, approach_name in enumerate(approach_names):
            data_to_plot = []
            
            for nl_name in nl_names:
                data = all_metrics[nl_name][approach_name][metric_name]
                
                if metric_name in ['max_ratios', 'rms_ratios', 'l1_errors', 'l2_errors']:
                    data = np.clip(data, 0, 5)
                
                if metric_name in ['spatial_corrs']:
                    data = np.clip(data, -1, 1)
                
                data_to_plot.append(data)
            
            box = ax.boxplot(data_to_plot, positions=np.arange(len(nl_names)) * (len(approach_names) + 1) + j,
                          widths=0.8, patch_artist=True)
            
            for patch in box['boxes']:
                patch.set_facecolor(colors[j])
            
            for whisker in box['whiskers']:
                whisker.set_color(colors[j])
            
            for cap in box['caps']:
                cap.set_color(colors[j])
            
            for median in box['medians']:
                median.set_color('black')
        
        ax.set_title(metric_title)
        ax.set_xticks(np.arange(len(nl_names)) * (len(approach_names) + 1) + (len(approach_names) - 1) / 2)
        ax.set_xticklabels(nl_names)
        ax.set_xlabel('Nonlinearity Type')
        
        if metric_name == 'spatial_corrs':
            ax.set_ylim(-0.1, 1.1)
        
        if i == 0:
            ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=colors[j]) for j in range(len(approach_names))],
                     approach_names, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_scale_distributions(all_metrics):
    nl_names = list(all_metrics.keys())
    approach_names = list(all_metrics[nl_names[0]].keys())
    
    fig, axes = plt.subplots(len(nl_names), 1, figsize=(12, 4 * len(nl_names)))
    plt.subplots_adjust(hspace=0.4)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(approach_names)))
    
    for i, nl_name in enumerate(nl_names):
        ax = axes[i] if len(nl_names) > 1 else axes
        
        for j, approach_name in enumerate(approach_names):
            scales = np.array(all_metrics[nl_name][approach_name]['scales'])
            
            scales = np.log10(scales)
            scales = np.clip(scales, -5, 5)
            
            ax.hist(scales, bins=30, alpha=0.7, color=colors[j], label=approach_name)
        
        ax.set_title(f'Scale Parameter Distribution for {nl_name} (log10 scale)')
        ax.set_xlabel('log10(Scale Parameter)')
        ax.set_ylabel('Frequency')
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    return fig

def generate_comparison_visualizations(num_samples=100, nx=128, ny=128, Lx=10.0, Ly=10.0):
    all_metrics = analyze_nlse_balance(num_samples, nx, ny, Lx, Ly) 
    metrics_fig = plot_metrics_comparison(all_metrics)
    scales_fig = plot_scale_distributions(all_metrics)
 
    return metrics_fig, scales_fig

if __name__ == "__main__":
    metrics_fig, scales_fig = generate_comparison_visualizations(num_samples=100) 
    metrics_fig.savefig('nlse_balance_metrics.png', dpi=300)
    scales_fig.savefig('nlse_scale_distributions.png', dpi=300)
    
    plt.show()
