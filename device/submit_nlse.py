import numpy as np
from mpi4py import MPI
from pathlib import Path
import uuid
import subprocess

def generate_smooth_ic(nx, ny, Lx, Ly, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(x, y)
    k_max = 5
    kx = np.fft.rfftfreq(nx, d=2*Lx/nx)
    ky = np.fft.fftfreq(ny, d=2*Ly/ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    filt = np.exp(-(K/k_max)**2)
    noise_real = rng.standard_normal((ny, nx))
    noise_imag = rng.standard_normal((ny, nx))
    field_real_f = np.fft.rfft2(noise_real) * filt
    field_imag_f = np.fft.rfft2(noise_imag) * filt
    field_real = np.fft.irfft2(field_real_f)
    field_imag = np.fft.irfft2(field_imag_f)
    field = field_real + 1j * field_imag
    field = field / np.sqrt(np.sum(np.abs(field)**2) * (2*Lx/nx) * (2*Ly/ny))
    return field

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    nx = ny = 256
    Lx = Ly = 10.0
    T = 1.5
    nt = 500
    num_snapshots = 100
    
    cwd = Path.cwd()
    ic_dir = cwd / "initial_conditions"
    traj_dir = cwd / "trajectories"

    print("my rank is", rank)
    
    if rank == 0:
        ic_dir.mkdir(exist_ok=True)
        traj_dir.mkdir(exist_ok=True)
    comm.Barrier()
    
    run_id = str(uuid.uuid4())[:8] if rank == 0 else None
    run_id = comm.bcast(run_id, root=0)
    
    rng = np.random.default_rng(seed=42 + rank)
    ic = generate_smooth_ic(nx, ny, Lx, Ly, rng)
    
    
    input_file = ic_dir / f"ic_{run_id}_{rank:04d}.npy"
    output_file = traj_dir / f"traj_{run_id}_{rank:04d}.npy"
    
    np.save(input_file, ic)
    comm.Barrier()
    
    binary = Path.cwd() / "to_nlse_call"
    cmd = [str(binary.absolute()), 
           str(nx), str(ny), 
           str(Lx), str(Ly),
           str(input_file.absolute()), 
           str(output_file.absolute()),
           str(T), str(nt), str(num_snapshots)]
    
    if rank == 0:
        print("Executing command:", " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Rank {rank}: Error - {result.stderr}")
    
