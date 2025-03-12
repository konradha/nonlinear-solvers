import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import signal, fft
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
import networkx as nx
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams
from scipy.linalg import svd, pinv
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm



def modal_decomposition_entropy(trajectory, dx, dy, Lx, Ly, m=None):
    nt, nx, ny = trajectory.shape
    
    modal_entropy = np.zeros(nt)
    freq_entropy = np.zeros(nt)
    dominant_modes = np.zeros((nt, 3, 2), dtype=int)
    
    kx = 2 * np.pi * fft.fftfreq(nx, dx)
    ky = 2 * np.pi * fft.fftfreq(ny, dy)
    
    for t in range(nt):
        fft_coefs = fft.fft2(trajectory[t])
        
        power = np.abs(fft_coefs)**2
        total_power = np.sum(power)
        if total_power > 0:
            prob_dist = power / total_power
        else:
            prob_dist = np.ones_like(power) / (nx * ny)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            modal_entropy[t] = entropy(prob_dist.flatten())
        
        power_copy = power.copy()
        power_copy[0, 0] = 0
        
        for i in range(3):
            idx = np.unravel_index(np.argmax(power_copy), power_copy.shape)
            dominant_modes[t, i] = idx
            power_copy[idx] = 0
            
        freq_magnitudes = np.sqrt(kx[dominant_modes[t, :, 0]]**2 + ky[dominant_modes[t, :, 1]]**2)
        if np.sum(freq_magnitudes) > 0:
            freq_probs = freq_magnitudes / np.sum(freq_magnitudes)
            freq_entropy[t] = entropy(freq_probs)
    
    return modal_entropy, freq_entropy, dominant_modes

def spatiotemporal_mutual_information(trajectory, dx, dy, Lx, Ly, m=None, n_regions=4, time_lag=1):
    nt, Nx, Ny = trajectory.shape
    
    region_size_x = Nx // n_regions
    region_size_y = Ny // n_regions
    
    mi_matrix = np.zeros((n_regions*n_regions, n_regions*n_regions))
    region_data = {}
    
    for i in range(n_regions):
        for j in range(n_regions):
            region_idx = i * n_regions + j
            region_data[region_idx] = []
            
            x_start, x_end = i * region_size_x, (i + 1) * region_size_x
            y_start, y_end = j * region_size_y, (j + 1) * region_size_y
            
            for t in range(nt - time_lag):
                region_t = trajectory[t, x_start:x_end, y_start:y_end]
                region_data[region_idx].append(np.abs(region_t).flatten())
    
    for i in range(n_regions*n_regions):
        for j in range(n_regions*n_regions):
            if i == j:
                continue
                
            data_i = np.array(region_data[i][:-time_lag])
            data_j = np.array(region_data[j][time_lag:])
            
            bins = min(20, int(np.sqrt(len(data_i))))
            
            h_i = np.zeros(len(data_i))
            h_j = np.zeros(len(data_j))
            h_ij = np.zeros(len(data_i))
            
            for t in range(len(data_i)):
                hist_i, _ = np.histogram(data_i[t], bins=bins, density=True)
                hist_j, _ = np.histogram(data_j[t], bins=bins, density=True)
                
                h_i[t] = entropy(hist_i + 1e-10)
                h_j[t] = entropy(hist_j + 1e-10)
                
                hist_2d, _, _ = np.histogram2d(data_i[t], data_j[t], bins=bins)
                hist_2d_prob = hist_2d / np.sum(hist_2d)
                h_ij[t] = entropy((hist_2d_prob + 1e-10).flatten())
            
            mi = np.mean(h_i + h_j - h_ij)
            mi_matrix[i, j] = mi
    
    directed_mi = mi_matrix.copy()
    
    mi_network = nx.DiGraph()
    for i in range(n_regions*n_regions):
        for j in range(n_regions*n_regions):
            if directed_mi[i, j] > 0.1 * np.max(directed_mi):
                mi_network.add_edge(i, j, weight=directed_mi[i, j])
    
    return mi_matrix, directed_mi, mi_network

def persistent_homology_analysis(trajectory, dx, dy, Lx, Ly, m=None, time_indices=None):
    nt, nx, ny = trajectory.shape 
    if time_indices is None:
        time_indices = np.linspace(0, nt-1, min(10, nt), dtype=int)
    
    persistence_results = []
    betti_curves = []
    
    for t_idx in time_indices:
        data = np.abs(trajectory[t_idx])
        
        stride = max(1, min(nx, ny) // 20)
        grid_x = np.arange(0, nx, stride)
        grid_y = np.arange(0, ny, stride)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        points_x = (X.flatten() * dx - Lx)
        points_y = (Y.flatten() * dy - Ly)
        values = data[X.flatten(), Y.flatten()]
        
        mask = values > 0.1 * np.max(values)
        points_x = points_x[mask]
        points_y = points_y[mask]
        values = values[mask]
        
        max_points = min(300, len(values))
        if len(values) > max_points:
            indices = np.argsort(values)[-max_points:]
            points_x = points_x[indices]
            points_y = points_y[indices]
            values = values[indices]
        
        point_cloud = np.column_stack([points_x, points_y, values])
        
        results = ripser(point_cloud, maxdim=1)
        
        diagram = results['dgms']
        persistence_results.append(diagram)
        
        lifetimes = [dgm[:, 1] - dgm[:, 0] for dgm in diagram]
        
        betti_numbers = []
        max_death = max([np.max(dgm[:, 1]) if dgm.size > 0 else 0 for dgm in diagram])
        epsilon_range = np.linspace(0, max_death, 50)
        
        for dim in range(len(diagram)):
            betti_curve = np.zeros_like(epsilon_range)
            if diagram[dim].size > 0:
                births = diagram[dim][:, 0]
                deaths = diagram[dim][:, 1]
                for i, epsilon in enumerate(epsilon_range):
                    betti_curve[i] = np.sum((births <= epsilon) & (deaths > epsilon))
            betti_numbers.append(betti_curve)
        
        betti_curves.append(betti_numbers)
    
    return persistence_results, betti_curves, time_indices

def dynamic_mode_decomposition(trajectory, dx, dy, Lx, Ly, m=None, r=10):
    nt, nx, ny = trajectory.shape
    
    X = trajectory.reshape(nt, -1).T
    
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, Sigma, Vh = svd(X1, full_matrices=False)
    
    U_r = U[:, :r]
    Sigma_r = np.diag(Sigma[:r])
    Vh_r = Vh[:r, :]
    
    A_tilde = U_r.T @ X2 @ Vh_r.T @ np.linalg.inv(Sigma_r)
    
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    
    modes = X2 @ Vh_r.T @ np.linalg.inv(Sigma_r) @ eigenvectors
    
    for i in range(modes.shape[1]):
        modes[:, i] = modes[:, i] / np.linalg.norm(modes[:, i])
    
    x0 = trajectory[0].flatten()
    b = np.linalg.lstsq(modes, x0, rcond=None)[0]
    
    dmd_entropy = np.zeros(nt)
    for t in range(nt):
        dynamics = np.zeros_like(b, dtype=complex)
        for i in range(len(eigenvalues)):
            dynamics[i] = b[i] * eigenvalues[i]**t
        
        prob_dist = np.abs(dynamics)**2
        prob_dist = prob_dist / np.sum(prob_dist)
        
        dmd_entropy[t] = entropy(prob_dist)
    
    spatial_modes = np.zeros((r, nx, ny), dtype=complex)
    for i in range(r):
        spatial_modes[i] = modes[:, i].reshape(nx, ny)
    
    return eigenvalues, spatial_modes, dmd_entropy

def wavelet_transfer_entropy(trajectory, dx, dy, Lx, Ly, m=None, n_regions=2, wavelet_level=3):
    nt, Nx, Ny = trajectory.shape
    
    region_size_x = Nx // n_regions
    region_size_y = Ny // n_regions
    
    region_avg = np.zeros((n_regions*n_regions, nt))
    
    for i in range(n_regions):
        for j in range(n_regions):
            region_idx = i * n_regions + j
            x_start, x_end = i * region_size_x, (i + 1) * region_size_x
            y_start, y_end = j * region_size_y, (j + 1) * region_size_y
            
            for t in range(nt):
                region_avg[region_idx, t] = np.mean(np.abs(trajectory[t, x_start:x_end, y_start:y_end]))
    
    wavelet_coeffs = []
    for i in range(n_regions*n_regions):
        coeffs = []
        for level in range(1, wavelet_level + 1):
            filt_size = 2**level
            smoothed = gaussian_filter(region_avg[i], sigma=filt_size/2)
            detail = region_avg[i] - smoothed
            coeffs.append(detail)
        wavelet_coeffs.append(coeffs)
    
    te_matrices = []
    
    for level in range(wavelet_level):
        te_matrix = np.zeros((n_regions*n_regions, n_regions*n_regions))
        
        for i in range(n_regions*n_regions):
            for j in range(n_regions*n_regions):
                if i == j:
                    continue
                
                x = wavelet_coeffs[i][level]
                y = wavelet_coeffs[j][level]
                
                x_past = x[:-1]
                y_past = y[:-1]
                y_present = y[1:]
                
                bins = min(20, int(np.sqrt(len(x_past))))
                
                h_ypast_ypres = 0
                h_ypast_xpast_ypres = 0
                h_ypast = 0
                h_ypast_xpast = 0
                
                try:
                    x_past_binned = np.digitize(x_past, np.linspace(min(x_past), max(x_past), bins))
                    y_past_binned = np.digitize(y_past, np.linspace(min(y_past), max(y_past), bins))
                    y_pres_binned = np.digitize(y_present, np.linspace(min(y_present), max(y_present), bins))
                    
                    h_ypast_ypres_hist = np.zeros((bins, bins))
                    h_ypast_xpast_ypres_hist = np.zeros((bins, bins, bins))
                    h_ypast_hist = np.zeros(bins)
                    h_ypast_xpast_hist = np.zeros((bins, bins))
                    
                    for t in range(len(x_past_binned)):
                        xp, yp, ypr = x_past_binned[t]-1, y_past_binned[t]-1, y_pres_binned[t]-1
                        h_ypast_ypres_hist[yp, ypr] += 1
                        h_ypast_xpast_ypres_hist[yp, xp, ypr] += 1
                        h_ypast_hist[yp] += 1
                        h_ypast_xpast_hist[yp, xp] += 1
                    
                    h_ypast_ypres_hist = h_ypast_ypres_hist / np.sum(h_ypast_ypres_hist)
                    h_ypast_xpast_ypres_hist = h_ypast_xpast_ypres_hist / np.sum(h_ypast_xpast_ypres_hist)
                    h_ypast_hist = h_ypast_hist / np.sum(h_ypast_hist)
                    h_ypast_xpast_hist = h_ypast_xpast_hist / np.sum(h_ypast_xpast_hist)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        p = h_ypast_ypres_hist.flatten()
                        p = p[p > 0]
                        h_ypast_ypres = -np.sum(p * np.log2(p))
                        
                        p = h_ypast_xpast_ypres_hist.flatten()
                        p = p[p > 0]
                        h_ypast_xpast_ypres = -np.sum(p * np.log2(p))
                        
                        p = h_ypast_hist
                        p = p[p > 0]
                        h_ypast = -np.sum(p * np.log2(p))
                        
                        p = h_ypast_xpast_hist.flatten()
                        p = p[p > 0]
                        h_ypast_xpast = -np.sum(p * np.log2(p))
                    
                    te = h_ypast_ypres - h_ypast_xpast_ypres + h_ypast_xpast - h_ypast
                    
                    te_matrix[i, j] = max(0, te)
                
                except:
                    te_matrix[i, j] = 0
        
        te_matrices.append(te_matrix)
    
    te_networks = []
    for level in range(wavelet_level):
        G = nx.DiGraph()
        for i in range(n_regions*n_regions):
            for j in range(n_regions*n_regions):
                if i != j and te_matrices[level][i, j] > 0.1 * np.max(te_matrices[level]):
                    G.add_edge(i, j, weight=te_matrices[level][i, j])
        te_networks.append(G)
    
    return te_matrices, te_networks, wavelet_coeffs


def plot_analysis_results(analysis_type, results, output_prefix):
    if analysis_type == 'modal':
        modal_entropy = results['modal_entropy']
        freq_entropy = results['freq_entropy']
        
        plt.figure(figsize=(10, 6))
        plt.plot(modal_entropy, label='Modal Entropy')
        plt.plot(freq_entropy, label='Frequency Entropy')
        plt.title('Modal Decomposition Entropy')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        plt.legend() 
        plt.show()
        #plt.savefig(f"{output_prefix}_modal_entropy.png")
    
    elif analysis_type == 'mutual':
        mi_matrix = results['mi_matrix']
        mi_network = results['mi_network']
        
        plt.figure(figsize=(10, 8))
        plt.imshow(mi_matrix, cmap='viridis')
        plt.colorbar(label='Mutual Information')
        plt.title('Spatiotemporal Mutual Information')
        plt.xlabel('Region Index')
        plt.ylabel('Region Index')
        #plt.savefig(f"{output_prefix}_mutual_information.png")
        plt.show()
        
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(mi_network)
        edges = mi_network.edges(data=True)
        weights = [d['weight'] * 5.0 / max(e[2]['weight'] for e in edges) if edges else 1.0 for _, _, d in edges]
        nx.draw(mi_network, pos, with_labels=True, node_color='skyblue', 
                node_size=500, font_size=10, font_weight='bold',
                width=weights, edge_color='gray', arrows=True)
        plt.title('Mutual Information Network')
        plt.show()
        #plt.savefig(f"{output_prefix}_mi_network.png")
    
    elif analysis_type == 'topology':
        persistence_results = results['persistence_results']
        betti_curves = results['betti_curves']
        time_indices = results['time_indices']
        
        for i, t_idx in enumerate(time_indices):
            plt.figure(figsize=(10, 6))
            plot_diagrams(persistence_results[i], show=False)
            plt.title(f'Persistence Diagram at t={t_idx}')
            plt.savefig(f"{output_prefix}_persistence_t{t_idx}.png")
            
            plt.figure(figsize=(10, 6))
            for dim in range(len(betti_curves[i])):
                plt.plot(betti_curves[i][dim], label=f'Dim {dim}')
            plt.title(f'Betti Curves at t={t_idx}')
            plt.xlabel('Filtration Value')
            plt.ylabel('Betti Number')
            plt.legend()
            #plt.savefig(f"{output_prefix}_betti_t{t_idx}.png")
            plt.show()
     
    
    elif analysis_type == 'dmd':
        eigenvalues = results['eigenvalues']
        dmd_entropy = results['dmd_entropy']
        
        plt.figure(figsize=(8, 6))
        plt.scatter(eigenvalues.real, eigenvalues.imag)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        plt.axis('equal')
        plt.title('DMD Eigenvalues')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        #plt.savefig(f"{output_prefix}_dmd_eigenvalues.png")
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(dmd_entropy)
        plt.title('DMD Entropy')
        plt.xlabel('Time step')
        plt.ylabel('Entropy')
        #plt.savefig(f"{output_prefix}_dmd_entropy.png")
        plt.show()
    
    elif analysis_type == 'transfer':
        te_matrices = results['te_matrices']
        
        for level in range(len(te_matrices)):
            plt.figure(figsize=(8, 6))
            plt.imshow(te_matrices[level], cmap='viridis')
            plt.colorbar(label='Transfer Entropy')
            plt.title(f'Wavelet Transfer Entropy (Level {level+1})')
            plt.xlabel('Target Region')
            plt.ylabel('Source Region')
            #plt.savefig(f"{output_prefix}_transfer_entropy_level{level+1}.png")
            plt.show()
    else:
        # getting lazy now
        raise NotImplemented

def calculate_all_analyses(trajectory, dx, dy, Lx, Ly, m=None, output_prefix="analysis"):
    results = {}
    nt, nx, ny = trajectory.shape
    
    if m is None and nx > 0 and ny > 0:
        m = np.ones((nx, ny))
    
    modal_entropy, freq_entropy, dominant_modes = modal_decomposition_entropy(
        trajectory, dx, dy, Lx, Ly, m)
    results['modal'] = {
        'modal_entropy': modal_entropy,
        'freq_entropy': freq_entropy,
        'dominant_modes': dominant_modes
    }
    
    mi_matrix, directed_mi, mi_network = spatiotemporal_mutual_information(
        trajectory, dx, dy, Lx, Ly, m)
    results['mutual'] = {
        'mi_matrix': mi_matrix,
        'directed_mi': directed_mi,
        'mi_network': mi_network
    }
    
    time_indices = np.linspace(0, nt-1, min(10, nt), dtype=int)
    persistence_results, betti_curves, time_indices = persistent_homology_analysis(
        trajectory, dx, dy, Lx, Ly, m, time_indices)
    results['topology'] = {
        'persistence_results': persistence_results,
        'betti_curves': betti_curves,
        'time_indices': time_indices
    }
    
    
    dmd_eigenvalues, spatial_modes, dmd_entropy = dynamic_mode_decomposition(
        trajectory, dx, dy, Lx, Ly, m)
    results['dmd'] = {
        'eigenvalues': dmd_eigenvalues,
        'spatial_modes': spatial_modes,
        'dmd_entropy': dmd_entropy
    }
    
    te_matrices, te_networks, wavelet_coeffs = wavelet_transfer_entropy(
        trajectory, dx, dy, Lx, Ly, m)
    results['transfer'] = {
        'te_matrices': te_matrices,
        'te_networks': te_networks,
        'wavelet_coeffs': wavelet_coeffs
    }
    
    
    for analysis_type, analysis_results in results.items():
        plot_analysis_results(analysis_type, analysis_results, output_prefix)
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Information-theoretic analysis of SGE/NLSE trajectories')
    parser.add_argument('--trajectory', type=str, required=True, help='Path to trajectory data file (.npy)')
    parser.add_argument('--dx', type=float, default=0.1, help='Grid spacing in x')
    parser.add_argument('--dy', type=float, default=0.1, help='Grid spacing in y')
    parser.add_argument('--Lx', type=float, default=10.0, help='Half-size of domain in x')
    parser.add_argument('--Ly', type=float, default=10.0, help='Half-size of domain in y')
    parser.add_argument('--m_file', type=str, default=None, help='Path to nonlinearity field m(x,y) (.npy)')
    parser.add_argument('--output', type=str, default='analysis_results', help='Output directory for results')
    parser.add_argument('--analysis', type=str, default='all', 
                        choices=['all', 'modal', 'mutual', 'topology', 'dmd', 'transfer'],
                        help='Analysis type to perform')
    return parser.parse_args()

def animate_betti_numbers(trajectory, dx, dy, Lx, Ly, m=None, time_indices=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from ripser import ripser
    from matplotlib.colors import Normalize
    from matplotlib import cm
    
    nt, nx, ny = trajectory.shape
    
    if time_indices is None:
        time_indices = np.linspace(0, nt-1, min(10, nt), dtype=int)
    
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    
    fig.suptitle('Betti Numbers and Persistence Analysis', fontsize=16)
    
    field_plots = []
    max_val = np.max(np.abs(trajectory[time_indices]))
    norm = Normalize(vmin=-max_val, vmax=max_val)
    cmap = cm.coolwarm
    
    for t_idx in time_indices:
        field_plots.append(trajectory[t_idx])
    
    def analyze_frame(frame_idx):
        t_idx = time_indices[frame_idx]
        data = np.abs(trajectory[t_idx])
        
        stride = max(1, min(nx, ny) // 25)
        grid_x = np.arange(0, nx, stride)
        grid_y = np.arange(0, ny, stride)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        points_x = (X.flatten() * dx - Lx)
        points_y = (Y.flatten() * dy - Ly)
        values = data[X.flatten(), Y.flatten()]
        
        threshold = 0.1 * np.max(values)
        mask = values > threshold
        points_x = points_x[mask]
        points_y = points_y[mask]
        values = values[mask]
        
        max_points = min(200, len(values))
        if len(values) > max_points:
            indices = np.argsort(values)[-max_points:]
            points_x = points_x[indices]
            points_y = points_y[indices]
            values = values[indices]
        
        point_cloud = np.column_stack([points_x, points_y, values])
        
        results = ripser(point_cloud, maxdim=2, thresh=2.0)
        
        dgms = results['dgms']
        
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        
        im = ax1.imshow(field_plots[frame_idx], extent=[-Lx, Lx, -Ly, Ly], 
                     origin='lower', cmap=cmap, norm=norm)
        ax1.set_title(f'Field at t={t_idx}')
        
        ax2.scatter(points_x, points_y, c=values, cmap='viridis')
        ax2.set_title('Point Cloud (Top View)')
        ax2.set_xlim([-Lx, Lx])
        ax2.set_ylim([-Ly, Ly])
        
        for i, dgm in enumerate(dgms):
            if i == 0:
                if len(dgm) > 0:
                    ax3.scatter(dgm[:, 0], dgm[:, 1], c='blue', label=f'$H_0$')
            elif i == 1:
                if len(dgm) > 0:
                    ax3.scatter(dgm[:, 0], dgm[:, 1], c='orange', label=f'$H_1$')
        
        max_death = 2.0
        if len(dgms) > 0 and len(dgms[0]) > 0:
            dgm_max = np.nanmax(dgms[0][:, 1]) if np.nanmax(dgms[0][:, 1]) < float('inf') else 2.0
            max_death = max(max_death, dgm_max)
        if len(dgms) > 1 and len(dgms[1]) > 0:
            dgm_max = np.nanmax(dgms[1][:, 1]) if np.nanmax(dgms[1][:, 1]) < float('inf') else 2.0
            max_death = max(max_death, dgm_max)
            
        ax3.plot([0, max_death], [0, max_death], 'k--')
        ax3.set_title(f'Persistence Diagram at t={t_idx}')
        ax3.set_xlabel('Birth')
        ax3.set_ylabel('Death')
        ax3.legend()
        
        epsilon_range = np.linspace(0, max_death, 100)
        betti_0 = np.zeros_like(epsilon_range)
        betti_1 = np.zeros_like(epsilon_range)
        
        if len(dgms) > 0 and dgms[0].size > 0:
            births = dgms[0][:, 0]
            deaths = dgms[0][:, 1]
            finite_mask = np.isfinite(births) & np.isfinite(deaths)
            births = births[finite_mask]
            deaths = deaths[finite_mask]
            for i, epsilon in enumerate(epsilon_range):
                betti_0[i] = np.sum((births <= epsilon) & (deaths > epsilon))
                
        if len(dgms) > 1 and dgms[1].size > 0:
            births = dgms[1][:, 0]
            deaths = dgms[1][:, 1]
            finite_mask = np.isfinite(births) & np.isfinite(deaths)
            births = births[finite_mask]
            deaths = deaths[finite_mask]
            for i, epsilon in enumerate(epsilon_range):
                betti_1[i] = np.sum((births <= epsilon) & (deaths > epsilon))
        
        ax4.plot(epsilon_range, betti_0, 'b-', linewidth=2, label='$\\beta_0$')
        ax4.plot(epsilon_range, betti_1, 'orange', linewidth=2, label='$\\beta_1$')
        ax4.set_title(f'Betti Curves at t={t_idx}')
        ax4.set_xlabel('Filtration Value')
        ax4.set_ylabel('Betti Number')
        
        ax4.set_xlim([0, max_death])
        max_betti = max(np.nanmax(betti_0) if len(betti_0) > 0 else 0, 
                       np.nanmax(betti_1) if len(betti_1) > 0 else 0)
        if not np.isfinite(max_betti) or max_betti <= 0:
            max_betti = 1.0
        ax4.set_ylim([0, max_betti * 1.1])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        lifetimes_0 = []
        births_0 = []
        if len(dgms) > 0 and dgms[0].size > 0:
            lifetimes = dgms[0][:, 1] - dgms[0][:, 0]
            finite_mask = np.isfinite(lifetimes) & np.isfinite(dgms[0][:, 0])
            lifetimes_0 = lifetimes[finite_mask]
            births_0 = dgms[0][:, 0][finite_mask]
            
        lifetimes_1 = []
        births_1 = []
        if len(dgms) > 1 and dgms[1].size > 0:
            lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
            finite_mask = np.isfinite(lifetimes) & np.isfinite(dgms[1][:, 0])
            lifetimes_1 = lifetimes[finite_mask]
            births_1 = dgms[1][:, 0][finite_mask]
        
        if len(births_0) > 0:
            ax5.scatter(births_0, lifetimes_0, c='blue', label='$H_0$')
        if len(births_1) > 0:
            ax5.scatter(births_1, lifetimes_1, c='orange', label='$H_1$')
            
        ax5.set_title('Persistence vs Birth')
        ax5.set_xlabel('Birth')
        ax5.set_ylabel('Persistence')
        
        max_birth = 0
        if len(births_0) > 0:
            max_birth = max(max_birth, np.nanmax(births_0))
        if len(births_1) > 0:
            max_birth = max(max_birth, np.nanmax(births_1))
            
        max_persistence = 0
        if len(lifetimes_0) > 0:
            max_persistence = max(max_persistence, np.nanmax(lifetimes_0))
        if len(lifetimes_1) > 0:
            max_persistence = max(max_persistence, np.nanmax(lifetimes_1))
            
        if not np.isfinite(max_birth) or max_birth <= 0:
            max_birth = 1.0
        if not np.isfinite(max_persistence) or max_persistence <= 0:
            max_persistence = 1.0
            
        ax5.set_xlim([0, max_birth * 1.1])
        ax5.set_ylim([0, max_persistence * 1.1])
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        combined_hist_data = []
        combined_colors = []
        combined_labels = []
        
        if len(lifetimes_0) > 0:
            combined_hist_data.append(lifetimes_0)
            combined_colors.append('blue')
            combined_labels.append('$H_0$')
            
        if len(lifetimes_1) > 0:
            combined_hist_data.append(lifetimes_1)
            combined_colors.append('orange')
            combined_labels.append('$H_1$')
        
        if combined_hist_data:
            hist_min = min([np.nanmin(data) for data in combined_hist_data]) 
            hist_max = max([np.nanmax(data) for data in combined_hist_data])
            
            if hist_min < hist_max and np.isfinite(hist_min) and np.isfinite(hist_max):
                bins = np.linspace(hist_min, hist_max, 20)
                for data, color, label in zip(combined_hist_data, combined_colors, combined_labels):
                    ax6.hist(data, bins=bins, alpha=0.5, color=color, label=label)
        
        ax6.set_title('Persistence Histogram')
        ax6.set_xlabel('Persistence')
        ax6.set_ylabel('Count')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    anim = FuncAnimation(fig, analyze_frame, frames=len(time_indices), interval=500)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    trajectory = np.load(args.trajectory) 
    is_complex = np.iscomplexobj(trajectory)
    if is_complex:
        trajectory = np.abs(trajectory)
    m = None
    if args.m_file:
        m = np.laod(args.m_file)

    
    modal_entropy, freq_entropy, dominant_modes = modal_decomposition_entropy(
        trajectory, args.dx, args.dy, args.Lx, args.Ly, m)
    plt.figure(figsize=(10, 6))
    plt.plot(modal_entropy, label='Modal Entropy')
    plt.plot(freq_entropy, label='Frequency Entropy')
    plt.title('Modal Decomposition Entropy')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.legend() 
    plt.show()
    
    

    """
    mi_matrix, directed_mi, mi_network = spatiotemporal_mutual_information(
        trajectory, args.dx, args.dy, args.Lx, args.Ly, m)
      
    plt.figure(figsize=(10, 8))
    plt.imshow(mi_matrix, cmap='viridis')
    plt.colorbar(label='Mutual Information')
    plt.title('Spatiotemporal Mutual Information')
    plt.xlabel('Region Index')
    plt.ylabel('Region Index')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(mi_network)
    edges = mi_network.edges(data=True)
    weights = [d['weight'] * 5.0 / max(e[2]['weight'] for e in edges) if edges else 1.0 for _, _, d in edges]
    nx.draw(mi_network, pos, with_labels=True, node_color='skyblue', 
            node_size=500, font_size=10, font_weight='bold',
            width=weights, edge_color='gray', arrows=True)
    plt.title('Mutual Information Network')
    plt.show()
    """

    """
    nt = trajectory.shape[0]  
    time_indices = np.linspace(0, nt-1, min(10, nt), dtype=int)
    print("Persistent Homology Analysis")
    persistence_results, betti_curves, time_indices = persistent_homology_analysis(
        trajectory, args.dx, args.dy, args.Lx, args.Ly, m, time_indices)

    for i, t_idx in enumerate(time_indices):
        plt.figure(figsize=(10, 6))
        plot_diagrams(persistence_results[i], show=False)
        plt.title(f'Persistence Diagram at t={t_idx}')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        for dim in range(len(betti_curves[i])):
            plt.plot(betti_curves[i][dim], label=f'Dim {dim}')
        plt.title(f'Betti Curves at t={t_idx}')
        plt.xlabel('Filtration Value')
        plt.ylabel('Betti Number')
        plt.legend()
        plt.show()
    """

    """
    # useless
    fisher_matrix, eigenvalues, eigenvectors = fisher_information_matrix(trajectory, args.dx, args.dy, args.Lx, args.Ly, m=m)
    plt.figure(figsize=(8, 6))
    plt.imshow(fisher_matrix, cmap='coolwarm')
    plt.colorbar(label='Fisher Information')
    plt.title('Fisher Information Matrix')
    plt.xticks([0, 1], ['Amplitude', 'Frequency'])
    plt.yticks([0, 1], ['Amplitude', 'Frequency'])
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.bar([0, 1], eigenvalues)
    plt.title('Fisher Information Eigenvalues')
    plt.xticks([0, 1], ['Principal Direction 1', 'Principal Direction 2'])
    plt.ylabel('Sensitivity')
    plt.show()
    """

    """
    eigenvalues, spatial_modes, dmd_entropy = dynamic_mode_decomposition(
        trajectory, args.dx, args.dy, args.Lx, args.Ly, m)

    plt.figure(figsize=(8, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.3)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.title('DMD Eigenvalues')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(dmd_entropy)
    plt.title('DMD Entropy')
    plt.xlabel('Time step')
    plt.ylabel('Entropy')
    plt.show()
    """

    """
    te_matrices, te_networks, wavelet_coeffs = wavelet_transfer_entropy(trajectory,
            args.dx, args.dy, args.Lx, args.Ly, m=m, n_regions=2, wavelet_level=3)

    for level in range(len(te_matrices)):
        plt.figure(figsize=(8, 6))
        plt.imshow(te_matrices[level], cmap='viridis')
        plt.colorbar(label='Transfer Entropy')
        plt.title(f'Wavelet Transfer Entropy (Level {level+1})')
        plt.xlabel('Target Region')
        plt.ylabel('Source Region')
        plt.show()
    """

    animate_betti_numbers(trajectory, args.dx, args.dy, args.Lx, args.Ly, m=m)
