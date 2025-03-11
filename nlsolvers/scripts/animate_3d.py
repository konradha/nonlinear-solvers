import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import h5py
from skimage import measure
import os


def animate_isosurface(X, Y, Z, data, nt, name, is_complex=False, title="", 
                       iso_level=None, view_angles=(30, 45), alpha=0.7, fps=7,
                       colormap='viridis', show_progress=True):
    if len(data.shape) != 4:
        raise ValueError(f"Expected 4D data (nt, nx, ny, nz), got shape {data.shape}")
    
    processed_data = np.abs(data) if is_complex else data
    
    if iso_level is None:
        max_val = np.max(processed_data)
        min_val = np.min(processed_data)
        iso_level = min_val + 0.3 * (max_val - min_val)
    
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 1, height_ratios=[1, 9], hspace=0.05)
    
    title_ax = plt.subplot(grid[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=14, wrap=True)
    
    ax = plt.subplot(grid[1], projection='3d')
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    
    padding = 0.1
    ax.set_xlim([x_min - padding * (x_max - x_min), x_max + padding * (x_max - x_min)])
    ax.set_ylim([y_min - padding * (y_max - y_min), y_max + padding * (y_max - y_min)])
    ax.set_zlim([z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min)])
    
    verts, faces, _, _ = measure.marching_cubes(processed_data[0], level=iso_level)
    
    verts_scaled = np.zeros_like(verts)
    verts_scaled[:, 0] = x_min + verts[:, 0] * (x_max - x_min) / (processed_data.shape[1] - 1)
    verts_scaled[:, 1] = y_min + verts[:, 1] * (y_max - y_min) / (processed_data.shape[2] - 1)
    verts_scaled[:, 2] = z_min + verts[:, 2] * (z_max - z_min) / (processed_data.shape[3] - 1)
    
    mesh = Poly3DCollection([])
    ax.add_collection3d(mesh)
    
    progress_text = None
    if show_progress:
        progress_text = ax.text2D(0.05, 0.95, "Frame: 0/0", transform=ax.transAxes)
    
    cmap = plt.get_cmap(colormap)
    
    norm = mcolors.Normalize(vmin=0, vmax=np.max(processed_data))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Amplitude' if is_complex else 'Value')
    
    def update(frame):
        mesh.set_verts([])
        
        try:
            verts, faces, _, _ = measure.marching_cubes(processed_data[frame], level=iso_level)
            
            verts_scaled = np.zeros_like(verts)
            verts_scaled[:, 0] = x_min + verts[:, 0] * (x_max - x_min) / (processed_data.shape[1] - 1)
            verts_scaled[:, 1] = y_min + verts[:, 1] * (y_max - y_min) / (processed_data.shape[2] - 1)
            verts_scaled[:, 2] = z_min + verts[:, 2] * (z_max - z_min) / (processed_data.shape[3] - 1)
            
            mesh.set_verts([verts_scaled[faces[i]] for i in range(len(faces))])
            mesh.set_facecolor(cmap(norm(iso_level)))
            mesh.set_alpha(alpha)
        except ValueError:
            pass
        
        if show_progress:
            progress_text.set_text(f"Frame: {frame+1}/{nt}")
            
        return mesh,
    
    ani = FuncAnimation(fig, update, frames=nt, interval=1000/fps, blit=False)
    
    os.makedirs(os.path.dirname(os.path.abspath(name)) if os.path.dirname(name) else '.', exist_ok=True)
    
    ani.save(name, dpi=200)
    plt.close(fig)
    
    print(f"Animation saved to {name}")


def animate_isosurface_dual(X, Y, Z, data, nt, name, is_complex=False, title="", 
                           iso_levels=None, view_angles=(30, 45), alpha=0.7, fps=7,
                           colormaps=None, show_progress=True):
    if isinstance(data, tuple) and len(data) == 2:
        data1, data2 = data
        if is_complex:
            processed_data1 = np.abs(data1)
            processed_data2 = np.abs(data2)
        else:
            processed_data1 = data1
            processed_data2 = data2
    elif is_complex:
        processed_data1 = np.real(data)
        processed_data2 = np.imag(data)
    else:
        processed_data1 = data
        grad = np.zeros_like(data)
        for t in range(nt):
            grad_components = np.gradient(data[t])
            grad[t] = np.sqrt(grad_components[0]**2 + grad_components[1]**2 + grad_components[2]**2)
        processed_data2 = grad
    
    if colormaps is None:
        colormaps = ('viridis', 'plasma')
    
    if iso_levels is None:
        max_val1 = np.max(processed_data1)
        min_val1 = np.min(processed_data1)
        max_val2 = np.max(processed_data2)
        min_val2 = np.min(processed_data2)
        
        level1 = min_val1 + 0.3 * (max_val1 - min_val1)
        level2 = min_val2 + 0.3 * (max_val2 - min_val2)
        iso_levels = (level1, level2)
    
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 1, height_ratios=[1, 9], hspace=0.05)
    
    title_ax = plt.subplot(grid[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=14, wrap=True)
    
    ax = plt.subplot(grid[1], projection='3d')
    
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    
    padding = 0.1
    ax.set_xlim([x_min - padding * (x_max - x_min), x_max + padding * (x_max - x_min)])
    ax.set_ylim([y_min - padding * (y_max - y_min), y_max + padding * (y_max - y_min)])
    ax.set_zlim([z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min)])
    
    mesh1 = Poly3DCollection([])
    mesh2 = Poly3DCollection([])
    ax.add_collection3d(mesh1)
    ax.add_collection3d(mesh2)
    
    progress_text = None
    if show_progress:
        progress_text = ax.text2D(0.05, 0.95, "Frame: 0/0", transform=ax.transAxes)
    
    cmap1 = plt.get_cmap(colormaps[0])
    cmap2 = plt.get_cmap(colormaps[1])
    
    norm1 = mcolors.Normalize(vmin=0, vmax=np.max(processed_data1))
    norm2 = mcolors.Normalize(vmin=0, vmax=np.max(processed_data2))
    
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax, location='left', label='Isosurface 1')
    
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax, location='right', label='Isosurface 2')
    
    def update(frame):
        mesh1.set_verts([])
        mesh2.set_verts([])
        
        try:
            verts1, faces1, _, _ = measure.marching_cubes(processed_data1[frame], level=iso_levels[0])
            
            verts1_scaled = np.zeros_like(verts1)
            verts1_scaled[:, 0] = x_min + verts1[:, 0] * (x_max - x_min) / (processed_data1.shape[1] - 1)
            verts1_scaled[:, 1] = y_min + verts1[:, 1] * (y_max - y_min) / (processed_data1.shape[2] - 1)
            verts1_scaled[:, 2] = z_min + verts1[:, 2] * (z_max - z_min) / (processed_data1.shape[3] - 1)
            
            if len(faces1) > 0:
                mesh1.set_verts([verts1_scaled[faces1[i]] for i in range(len(faces1))])
                mesh1.set_facecolor(cmap1(norm1(iso_levels[0])))
                mesh1.set_alpha(alpha)
        except ValueError:
            pass
        
        try:
            verts2, faces2, _, _ = measure.marching_cubes(processed_data2[frame], level=iso_levels[1])
            
            verts2_scaled = np.zeros_like(verts2)
            verts2_scaled[:, 0] = x_min + verts2[:, 0] * (x_max - x_min) / (processed_data2.shape[1] - 1)
            verts2_scaled[:, 1] = y_min + verts2[:, 1] * (y_max - y_min) / (processed_data2.shape[2] - 1)
            verts2_scaled[:, 2] = z_min + verts2[:, 2] * (z_max - z_min) / (processed_data2.shape[3] - 1)
            
            if len(faces2) > 0:
                mesh2.set_verts([verts2_scaled[faces2[i]] for i in range(len(faces2))])
                mesh2.set_facecolor(cmap2(norm2(iso_levels[1])))
                mesh2.set_alpha(alpha)
        except ValueError:
            pass
        
        if show_progress:
            progress_text.set_text(f"Frame: {frame+1}/{nt}")
            
        return mesh1, mesh2,
    
    ani = FuncAnimation(fig, update, frames=nt, interval=1000/fps, blit=False)
    
    os.makedirs(os.path.dirname(os.path.abspath(name)) if os.path.dirname(name) else '.', exist_ok=True)
    
    ani.save(name, dpi=200)
    plt.close(fig)
    
    print(f"Animation saved to {name}")


def animate_isosurface_from_hdf5(h5_file, output_name, is_complex=True, iso_level=None, 
                                view_angles=(30, 45), alpha=0.7, fps=7):
    with h5py.File(h5_file, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:] if 'Z' in f else np.linspace(-f['grid'].attrs['Lz'], f['grid'].attrs['Lz'], f['grid'].attrs['nz'])
        u = f['u'][:]
        
        nx, ny = X.shape
        nz = Z.shape[0] if Z.ndim > 0 else f['grid'].attrs['nz']
        
        if X.ndim < 3 or Y.ndim < 3 or Z.ndim < 3:
            X, Y, Z = np.meshgrid(X[0,:], Y[:,0], Z, indexing='ij')
        
        nt = u.shape[0]
        num_snapshots = f['time'].attrs['num_snapshots'] if 'num_snapshots' in f['time'].attrs else nt
        
        title = f"NLSE 3D: {f['metadata'].attrs['problem_type']}\n"
        title += f"Domain: [{-f['grid'].attrs['Lx']:.2f}, {f['grid'].attrs['Lx']:.2f}] × "
        title += f"[{-f['grid'].attrs['Ly']:.2f}, {f['grid'].attrs['Ly']:.2f}] × "
        title += f"[{-f['grid'].attrs['Lz']:.2f}, {f['grid'].attrs['Lz']:.2f}]\n"
        title += f"Resolution: {f['grid'].attrs['nx']} × {f['grid'].attrs['ny']} × {f['grid'].attrs['nz']}\n"
        title += f"Time: T={f['time'].attrs['T']:.2f}, {f['time'].attrs['nt']} steps, {num_snapshots} snapshots"
        
        animate_isosurface(X, Y, Z, u, num_snapshots, output_name, 
                          is_complex=is_complex, title=title, 
                          iso_level=iso_level, view_angles=view_angles, 
                          alpha=alpha, fps=fps)


if __name__ == "__main__":
    nx, ny, nz = 32, 32, 32
    Lx, Ly, Lz = 5.0, 5.0, 5.0
    
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    z = np.linspace(-Lz, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    nt = 20
    data = np.zeros((nt, nx, ny, nz), dtype=np.complex128)
    
    for t in range(nt):
        center1 = np.array([-2.0 + 4.0*t/nt, 0, 0])
        center2 = np.array([2.0 - 4.0*t/nt, 0, 0])
        
        r1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2 + (Z - center1[2])**2)
        r2 = np.sqrt((X - center2[0])**2 + (Y - center2[1])**2 + (Z - center2[2])**2)
        
        wave1 = np.exp(-r1**2/2) * np.exp(1j * (X + 0.5*t))
        wave2 = np.exp(-r2**2/2) * np.exp(1j * (-X - 0.5*t))
        
        data[t] = wave1 + wave2
        
    data_real = np.real(data)
    data_imag = np.imag(data)
    
    if not os.path.exists('output'):
        os.makedirs('output')
    
    animate_isosurface(X, Y, Z, data, nt, 'output/isosurface_example.mp4', 
                      is_complex=True, title="Colliding wave packets", 
                      iso_level=0.3, view_angles=(30, 30))
    
    animate_isosurface_dual(X, Y, Z, (data_real, data_imag), nt, 'output/isosurface_dual_example.mp4',
                           is_complex=False, title="Colliding wave packets - Real & Imaginary Parts",
                           colormaps=('viridis', 'plasma'))
