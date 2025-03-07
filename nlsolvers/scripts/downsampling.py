import numpy as np
import torch
import matplotlib.pyplot as plt

from sys import argv


def downsample_fft(u: np.ndarray, target_shape: tuple) -> np.ndarray:
    assert len(u.shape) == 3
    assert len(target_shape) == 2
    is_complex = np.iscomplexobj(u)
    original_shape = u.shape[-2:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.complex128 if is_complex else torch.float64
    u_tensor = torch.from_numpy(u).to(device=device, dtype=dtype)

    ft = torch.fft.fft2(u_tensor, norm='ortho', dim=(-2, -1))
    ft_shifted = torch.fft.fftshift(ft, dim=(-2, -1))
    start_x = (original_shape[0] - target_shape[0]) // 2
    start_y = (original_shape[1] - target_shape[1]) // 2
    ft_cropped = ft_shifted[..., start_x:start_x+target_shape[0],
                             start_y:start_y+target_shape[1]]
    ft_cropped = torch.fft.ifftshift(ft_cropped, dim=(-2, -1))
    downsampled = torch.fft.ifft2(ft_cropped, norm='ortho', dim=(-2, -1))

    downsampled_np = downsampled.cpu().numpy()
    if not is_complex:
        downsampled_np = downsampled_np.real

    return downsampled_np

def reconstruct_fft(downsampled: np.ndarray, original_shape: tuple) -> np.ndarray:
    assert len(downsampled.shape) == 3
    assert len(original_shape) == 2

    is_complex = np.iscomplexobj(downsampled)
    current_shape = downsampled.shape[-2:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.complex128 if is_complex else torch.float64
    ds_tensor = torch.from_numpy(downsampled).to(device=device, dtype=dtype)
    ft = torch.fft.fft2(ds_tensor, norm='ortho', dim=(-2, -1))
    ft_shifted = torch.fft.fftshift(ft, dim=(-2, -1))
    padded_ft = torch.zeros(
        (*ds_tensor.shape[:-2], original_shape[0], original_shape[1]),
        dtype=dtype,
        device=device
    )
    start_x = (original_shape[0] - current_shape[0]) // 2
    start_y = (original_shape[1] - current_shape[1]) // 2
    padded_ft[...,
             start_x:start_x+current_shape[0],
             start_y:start_y+current_shape[1]] = ft_shifted

    padded_ft = torch.fft.ifftshift(padded_ft, dim=(-2, -1))
    reconstructed = torch.fft.ifft2(padded_ft, norm='ortho', dim=(-2, -1))
    reconstructed_np = reconstructed.cpu().numpy()
    if not is_complex:
        reconstructed_np = reconstructed_np.real
    return reconstructed_np


if __name__ == '__main__': 
    fname_u = str(argv[1])
    target_nx, target_ny = int(argv[2]), int(argv[3])

    u = np.load(fname_u)
    assert len(u.shape) == 3
    nt = u.shape[0]
    nx, ny = u.shape[1], u.shape[2]

    ts = np.random.randint(0, nt)

    is_complex = np.iscomplexobj(u)
    downsampled = downsample_fft(u, (target_nx, target_ny))
    reconstructed = reconstruct_fft(downsampled, (nx, ny)) 


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    im = ax1.imshow(np.abs(u[ts]) if is_complex else u[ts])
    ax1.set_title("Original")
    ax2.imshow(np.abs(downsampled[ts]) if is_complex else downsampled[ts])
    ax2.set_title("Downsampled")

    im_r = ax3.imshow(np.abs(u[ts] - reconstructed[ts]), cmap="seismic")
    ax3.set_title("Reconstructed to original diff")

    fig.colorbar(im, ax=[ax1, ax2], label='magnitude / [1]')
    fig.colorbar(im_r, ax=[ax3], label="Diff / [1]")

    plt.show()


