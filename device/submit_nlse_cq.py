import h5py
import numpy as np
from mpi4py import MPI

import os
from pathlib import Path
import uuid
import subprocess


def save_trajectory(u_trajectory, X, Y, params, filename='nlse_trajectory.h5'):
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('u', data=u_trajectory,
                              compression='gzip',
                              compression_opts=9)

        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)

        for key, value in params.items():
            dset.attrs[key] = value



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
        u *= (1 + envelope * (np.exp(np.random.random() * 1j * phase) - 1))
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

def sampler_random_solitons_with_velocities(X, Y, num_solitons=5, soliton_width=0.5, amp_range=(0.5, 2.0),
                           vel_scale=5., seed=None):
    if seed is not None:
        np.random.seed(seed) 
    u = np.zeros_like(X, dtype=complex) 
    for _ in range(num_solitons):
        A = np.random.uniform(*amp_range) * (np.random.randn() + 1j*np.random.randn())
        x0 = np.random.uniform(X.min(), X.max())
        y0 = np.random.uniform(Y.min(), Y.max())
        theta = np.random.uniform(0, 2*np.pi)
        v_mag = vel_scale * np.random.uniform(0.5, 1.0)
        vx = v_mag * np.cos(theta)
        vy = v_mag * np.sin(theta)
        phase0 = np.random.uniform(0, 2*np.pi) 
        r = np.sqrt((X - x0)**2 + (Y - y0)**2) 
        phase = phase0 + 0.5*(vx*(X - x0) + vy*(Y - y0))
        soliton = A * (1/np.cosh(r/soliton_width)) * np.exp(1j * phase)
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

def sampler_lattice_structures(X, Y, structure_type='vortex',
                            spacing=1.0, modulation=None, seed=None):
   # cool, could get some actual randomness!
   if seed is not None:
       np.random.seed(seed) 
   Lx = X.max() - X.min()
   Ly = Y.max() - Y.min()
   
   positions = np.column_stack([X, Y]) 
   u = np.ones_like(X, dtype=complex)
   
   if modulation is not None:
       if modulation['type'] == 'spiral':
           theta = np.arctan2(positions[:, 1], positions[:, 0])
           r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
           positions[:, 0] += modulation['strength'] * r * np.cos(theta)
           positions[:, 1] += modulation['strength'] * r * np.sin(theta)
   
   for pos in positions:
       if structure_type == 'vortex':
           phase = np.arctan2(Y - pos[1], X - pos[0])
           r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
           core = np.tanh(r/0.5)
           u *= core * np.exp(1j * phase)
       elif structure_type == 'dark_soliton':
           r = np.abs(Y - pos[1])
           u *= np.tanh(r/0.5) 
   return u


def sampler_quasiperiodic(X, Y, num_waves=5, symmetry='penrose',
                        amplitude_variation=0.2, phase_variation=0.1, seed=None):
   # _really_ interesting periodic patterns to be observed!
   if seed is not None:
       np.random.seed(seed)
   
   u = np.zeros_like(X, dtype=complex)
   
   if symmetry == 'penrose':
       n_fold = 5
   elif symmetry == 'octagonal':
       n_fold = 8
   elif symmetry == 'dodecagonal':
       n_fold = 12
   
   base_k = 2 * np.pi
   
   for j in range(num_waves):
       for n in range(n_fold):
           angle = 2 * np.pi * n / n_fold + np.random.normal(0, phase_variation)
           kx = base_k * np.cos(angle)
           ky = base_k * np.sin(angle)
           
           amp = 1.0 + np.random.normal(0, amplitude_variation)
           phase = np.random.uniform(0, 2*np.pi)
           
           u += amp * np.exp(1j * (kx * X + ky * Y + phase)) 
   return u


def sampler_vortex_turbulence_transition(X, Y, 
                                       vortex_density=0.1,
                                       wave_energy=1.0,
                                       mixture_ratio=0.5,
                                       seed=None):
    """
    Mix vortices with "turbulence".
    """
    if seed is not None:
        np.random.seed(seed)
        
    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min()
    area = Lx * Ly
    
    num_vortices = int(vortex_density * area)
    vortex_field = np.ones_like(X, dtype=complex)
    
    positions = np.random.uniform(low=[X.min(), Y.min()], 
                                high=[X.max(), Y.max()],
                                size=(num_vortices, 2))
    
    charges = np.random.choice([-1, 1], size=num_vortices)
    
    for pos, charge in zip(positions, charges):
        r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
        phase = charge * np.arctan2(Y - pos[1], X - pos[0])
        vortex_field *= np.tanh(r/0.5) * np.exp(1j * phase)
    
    # turbulence-like background
    k_min = 2 * np.pi / max(Lx, Ly)
    k_max = 2 * np.pi / min(X[1,0] - X[0,0], Y[0,1] - Y[0,0])
    
    kx = 2 * np.pi * np.fft.fftfreq(X.shape[0], d=Lx/X.shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(X.shape[1], d=Ly/X.shape[1])
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    
    # Kolmogorov-like spectrum with cutoffs
    spectrum = np.zeros_like(K)
    k_mask = (K >= k_min) & (K <= k_max)
    spectrum[k_mask] = K[k_mask]**(-5/3) * np.exp(-K[k_mask]/k_max)
    
    phases = 2 * np.pi * np.random.random(K.shape)
    wave_field_k = np.sqrt(wave_energy * spectrum) * np.exp(1j * phases)
    wave_field = np.fft.ifft2(wave_field_k)
    
    combined = mixture_ratio * vortex_field + (1 - mixture_ratio) * wave_field
    return combined

def sampler_bdg_vortex_modes(X, Y, num_vortices=1,
                           excitation_spectrum='kelvin',
                           core_size=0.5, seed=None):
    """
    sampler inspired by vortices predicted in BdG formalism
    """
    if seed is not None:
        np.random.seed(seed)

    psi = np.ones_like(X, dtype=complex)

    for _ in range(num_vortices):
        x0 = np.random.uniform(X.min(), X.max())
        y0 = np.random.uniform(Y.min(), Y.max())
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        theta = np.arctan2(Y - y0, X - x0)

        core = np.tanh(r/core_size)
        psi *= core * np.exp(1j * theta)

    if excitation_spectrum == 'kelvin':
        for m in range(1, 4):
            amplitude = 0.1 / m
            phase = np.random.uniform(0, 2*np.pi)
            r = np.sqrt((X - x0)**2 + (Y - y0)**2)
            theta = np.arctan2(Y - y0, X - x0)

            mode = amplitude * (r/core_size)**m * np.exp(1j * (m*theta + phase))
            psi += mode * np.exp(-r/core_size)

    return psi


def sampler_bdg_excitations(X, Y, background_type='uniform',
                          excitation_strength=0.1,
                          excitation_type='phonon',
                          mu=1.0, seed=None):
    """
    more stuff "inspired" by BdG formalism

    background_type: 'uniform', 'thomas_fermi', 'stripe'
    excitation_type: 'phonon', 'breathing', 'dipole'
    """
    if seed is not None:
        np.random.seed(seed)

    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min()

    c1 = np.random.randint(0, 3)
    c2 = np.random.randint(0, 3)

    # background
    if c1==0:# background_type == 'uniform':
        psi_0 = np.sqrt(mu) * np.ones_like(X, dtype=complex)

    elif c1==1:#background_type == 'thomas_fermi':
        V_trap = 0.5 * (X**2 + Y**2)
        psi_0 = np.sqrt(np.maximum(mu - V_trap, 0))

    elif c1==2:#background_type == 'stripe':
        k0 = 2.0
        psi_0 = np.sqrt(mu) * np.exp(1j * k0 * X)

    if c2==0:#excitation_type == 'phonon':
        kx = np.random.uniform(0, 2*np.pi/Lx)
        ky = np.random.uniform(0, 2*np.pi/Ly)
        phase = kx * X + ky * Y
        u = np.exp(1j * phase)
        v = np.exp(-1j * phase)

    elif c2==1:#excitation_type == 'breathing':
        r2 = X**2 + Y**2
        u = r2 * np.exp(-r2/(2*Lx**2))
        v = u.conjugate()

    elif c2==2:#excitation_type == 'dipole':
        u = X + 1j*Y
        v = -u.conjugate()
    psi = psi_0 + excitation_strength * (u + v.conjugate())
    return psi


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    nx = ny = 256
    Lx = Ly = 10.0
    T = 3.
    nt = 1000
    num_snapshots = 100
    
    cwd = Path.cwd()
    ic_dir = cwd / "initial_conditions"
    traj_dir = cwd / "trajectories_new"

    
    if rank == 0:
        ic_dir.mkdir(exist_ok=True)
        traj_dir.mkdir(exist_ok=True)
    comm.Barrier()
    
    run_id = str(uuid.uuid4())[:8] if rank == 0 else None
    run_id = comm.bcast(run_id, root=0)
    
    rng = np.random.default_rng(seed=rank)
    xn, yn = np.linspace(-Lx, Lx, nx), np.linspace(-Ly, Ly, ny) 
    X, Y = np.meshgrid(xn, yn)
    ic = sampler_bdg_excitations(X, Y) 
    #ic = None 
    #choice = np.random.randint(0, 4) 
    #if choice == 0:
    #    ic = sampler_random_vortices(X, Y, num_vortices=np.random.randint(2, 15))
    #elif choice == 1:
    #    ic = sampler_random_solitons_with_velocities(X, Y, num_solitons=np.random.randint(2, 10)) 
    #elif choice == 2:
    #    ic = sampler_random_fourier_localized()
    #else:
    #    ic = sampler_random_solitons(X, Y)   
    
    input_file = ic_dir / f"ic_{run_id}_{rank:04d}.npy"
    output_file = traj_dir / f"traj_{run_id}_{rank:04d}.npy"
    
    np.save(input_file, ic)
    comm.Barrier()
    
    binary = Path.cwd() / "to_nlse_cq_call"
    cmd = [str(binary.absolute()), 
           str(nx), str(ny), 
           str(Lx), str(Ly),
           str(input_file.absolute()), 
           str(output_file.absolute()),
           str(T), str(nt), str(num_snapshots)]
    
    if rank == 0:
        print("Executing command:", " ".join(cmd))
    
    t_start = MPI.Wtime()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t_end = MPI.Wtime()
    t_elapsed = t_end - t_start
    all_times = comm.gather(t_elapsed, root=0)

    if rank == 0:
        print("All timings:", all_times)

    if result.returncode != 0:
        print(f"Rank {rank}: Error - {result.stderr}")
     
    comm.Barrier()
    print("rank", rank, " -- postprocessing")
    trajectory = np.load(output_file)  
    output_h5 = traj_dir / f"traj_{run_id}_{rank:04d}.h5"
    params = {'T': T}
    save_trajectory(trajectory, X, Y, params, filename=str(output_h5))
    os.unlink(output_file) 
