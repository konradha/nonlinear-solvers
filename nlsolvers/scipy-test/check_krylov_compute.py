import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse
import sys
import os

from tqdm import tqdm

def load_sparse_csr_components(basename):
    try:
        data = np.load(f"{basename}_data.npy")
        indices = np.load(f"{basename}_indices.npy")
        indptr = np.load(f"{basename}_indptr.npy")
        shape_val = np.load(f"{basename}_shape.npy")
        if shape_val.shape != (2,):
             raise ValueError("Shape file must contain 2 elements (rows, cols).")
        return sp.csr_matrix((data, indices, indptr), shape=tuple(shape_val))
    except FileNotFoundError as e:
        print(f"Error loading sparse matrix components from base {basename}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reconstructing sparse matrix from base {basename}: {e}", file=sys.stderr)
        sys.exit(1)

def compare_and_plot(L_sp, u0, t, dims_in, results_base, m_val, complex_mode):
    suffix = "complex" if complex_mode else "real"
    krylov_suffix = f"_m{m_val}"
    dtype = np.complex128 if complex_mode else np.float64

    # RAN bin/scipy_test_host 100 2. .1 ../scipy-test/data/

    try:
        y_cpp = np.load(f"{results_base}_y_host_{suffix}{krylov_suffix}.npy").astype(dtype)
    except FileNotFoundError:
        print(f"Error: C++ result file not found for m={m_val}, mode={suffix}", file=sys.stderr)
        return None, None, None

    if complex_mode:
        A = 1j * L_sp.astype(np.complex128)
        u0_sp = u0.astype(np.complex128)
    else:
        A = L_sp.astype(np.float64)
        u0_sp = u0.astype(np.float64)

    y_scipy_all = expm_multiply(A, u0_sp, start=0, stop=t, endpoint=True)
    y_scipy = y_scipy_all[-1]

    norm_scipy = norm(y_scipy)
    abs_diff = norm(y_scipy - y_cpp)
    rel_error_l2 = abs_diff / norm_scipy if norm_scipy > 1e-14 else abs_diff

    abs_diff_l1 = norm(y_scipy.ravel() - y_cpp.ravel(), ord=1)
    norm_scipy_l1 = norm(y_scipy.ravel(), ord=1)
    rel_error_l1 = abs_diff_l1 / norm_scipy_l1 if norm_scipy_l1 > 1e-14 else abs_diff_l1

    diff_vector = y_scipy - y_cpp
    diff_vector_reshaped = diff_vector.reshape(dims_in)

    return rel_error_l1, rel_error_l2, diff_vector_reshaped

def main():
    parser = argparse.ArgumentParser(description="Compare Host C++ Matrix Exp Action with SciPy.")
    parser.add_argument("data_base", help="Base name used for the .npy data files (e.g., 'test_data')")
    args = parser.parse_args()
    base = args.data_base

    try:
        L_real_sp = load_sparse_csr_components(f"{base}_L_real")
        L_complex_sp = load_sparse_csr_components(f"{base}_L_complex")
        u0_real = np.load(f"{base}_u0_real.npy")
        u0_complex = np.load(f"{base}_u0_complex.npy")
        t = 1e-2 # hardcode for now ...
        dims_in = np.load(f"{base}_dims.npy").astype(int)
        if len(dims_in) != 3: raise ValueError("Expected 3 dimensions (nz, ny, nx)")
        nz, ny, nx = dims_in
        Lx = 2.
        Ly = 2.
        Lz = 2.
    except Exception as e:
        print(f"Error loading base data: {e}", file=sys.stderr)
        sys.exit(1)

    krylov_dims = [10, 20, 30, 40, 50]
    results = {}

    for m in tqdm(krylov_dims):
        results[m] = {}
        l1_real, l2_real, diff_real = compare_and_plot(L_real_sp, u0_real, t, dims_in, base, m, complex_mode=False)
        if l1_real is not None: results[m]['real'] = {'l1': l1_real, 'l2': l2_real, 'diff': diff_real}

        l1_complex, l2_complex, diff_complex = compare_and_plot(L_complex_sp, u0_complex, t, dims_in, base, m, complex_mode=True)
        if l1_complex is not None: results[m]['complex'] = {'l1': l1_complex, 'l2': l2_complex, 'diff': diff_complex}

    print("computed diffs")


    fig, axes = plt.subplots(2, 6, figsize=(20, 10), squeeze=False)
    fig.suptitle(f"Host MatFunc Comparison (t={t:.3f}, n={nx})", fontsize=16)

    ax_err = axes[0, 0]
    m_values = np.array(krylov_dims)
    l1_errors_real = [results[m]['real']['l1'] for m in krylov_dims if m in results and 'real' in results[m]]
    l2_errors_real = [results[m]['real']['l2'] for m in krylov_dims if m in results and 'real' in results[m]]
    l1_errors_complex = [results[m]['complex']['l1'] for m in krylov_dims if m in results and 'complex' in results[m]]
    l2_errors_complex = [results[m]['complex']['l2'] for m in krylov_dims if m in results and 'complex' in results[m]]

    valid_m_real = [m for m in krylov_dims if m in results and 'real' in results[m]]
    valid_m_complex = [m for m in krylov_dims if m in results and 'complex' in results[m]]

    if l1_errors_real: ax_err.semilogy(valid_m_real, l1_errors_real, 'bo-', label='L1 Rel Err (Real)')
    if l2_errors_real: ax_err.semilogy(valid_m_real, l2_errors_real, 'ro-', label='L2 Rel Err (Real)')
    if l1_errors_complex: ax_err.semilogy(valid_m_complex, l1_errors_complex, 'bs--', label='L1 Rel Err (Complex)')
    if l2_errors_complex: ax_err.semilogy(valid_m_complex, l2_errors_complex, 'rs--', label='L2 Rel Err (Complex)')

    ax_err.set_xlabel("Krylov Dimension (m)")
    ax_err.set_ylabel("Relative Error")
    ax_err.set_title("Error vs Krylov Dimension")
    ax_err.legend()
    ax_err.grid(True, which='both', linestyle=':')
    ax_err.set_xticks(m_values)

    slice_idx = nz // 2
    x_grid = np.linspace(-Lx, Lx, nx)
    y_grid = np.linspace(-Ly, Ly, ny)
    X, Y = np.meshgrid(y_grid, x_grid, indexing='ij')

    for i, m in tqdm(enumerate(krylov_dims)):
        ax_diff_real = axes[0, i + 1]
        if m in results and 'real' in results[m] and results[m]['real']['diff'] is not None:
            diff_data_real = np.abs(results[m]['real']['diff'][slice_idx, :, :])
            im_real = ax_diff_real.pcolormesh(Y, X, diff_data_real, shading='auto', cmap='viridis')
            fig.colorbar(im_real, ax=ax_diff_real)
            ax_diff_real.set_title(f"|Difference| (Real, m={m}, z={slice_idx})")
        else: ax_diff_real.set_title(f"Data Missing (Real, m={m})")
        ax_diff_real.set_xlabel("x")
        ax_diff_real.set_ylabel("y")
        ax_diff_real.set_aspect('equal', adjustable='box')

    for i, m in enumerate(krylov_dims):
         ax_diff_complex = axes[1, i + 1]
         if m in results and 'complex' in results[m] and results[m]['complex']['diff'] is not None:
             diff_data_complex = np.abs(results[m]['complex']['diff'][slice_idx, :, :])
             im_complex = ax_diff_complex.pcolormesh(Y, X, diff_data_complex, shading='auto', cmap='viridis')
             fig.colorbar(im_complex, ax=ax_diff_complex)
             ax_diff_complex.set_title(f"|Difference| (Complex, m={m}, z={slice_idx})")
         else: ax_diff_complex.set_title(f"Data Missing (Complex, m={m})")
         ax_diff_complex.set_xlabel("x")
         ax_diff_complex.set_ylabel("y")
         ax_diff_complex.set_aspect('equal', adjustable='box')

    axes[1, 0].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"{base}_comparison.png"
    plt.savefig(plot_filename)

if __name__ == "__main__":
    main()
