#ifndef NLSE_SAT_SOLVER_HPP
#define NLSE_SAT_SOLVER_HPP


/*
 *
 * i u_t + (u_xx + u_yy) + (|u|²/ (1 + |u|² \kappa)u = 0
 * on rectangular domain with n_x x n_y gridpoints
 *
 */

#include "matfunc_complex.hpp"
#include "pragmas.hpp"
#include <thrust/complex.h>

__global__ void density(thrust::complex<double> *out,
                        const thrust::complex<double> *in, const uint32_t nx,
                        const uint32_t ny,
			const double kappa = 0.5) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    const auto val = in[idx];
    const auto d = thrust::abs(val) * thrust::abs(val);
    out[idx] = d / (1.f + kappa * d);
  }
}

__global__ void nonlin_part(thrust::complex<double> *out,
                            const thrust::complex<double> *in,
                            const thrust::complex<double> *density,
                            const thrust::complex<double> tau,
                            const uint32_t nx, const uint32_t ny) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    out[idx] = thrust::exp(tau * density[idx]) * in[idx];
  }
}

class NLSESaturatingSolver {
public:
  struct Parameters {
    uint32_t block_size;
    uint32_t num_snapshots;
    uint32_t snapshot_freq;
    uint32_t krylov_dim;
    Parameters(uint32_t ns = 100, uint32_t freq = 5, uint32_t m = 10)
        : block_size(256), num_snapshots(ns), snapshot_freq(freq),
          krylov_dim(m) {}
  };

  NLSESaturatingSolver(const Eigen::SparseMatrix<std::complex<double>> &L,
                   const std::complex<double> *host_u0,
                   const Parameters &params = Parameters())
      : n_(L.rows()), current_snapshot_(0), params_(params) {

    cudaMalloc(&d_u_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_density_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_u_trajectory_,
               n_ * params_.num_snapshots * sizeof(thrust::complex<double>));

    cudaMemcpy(d_u_, host_u0, n_ * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);

    matfunc_ = new MatrixFunctionApplicatorComplex(L, n_, params_.krylov_dim,
                                                   L.nonZeros());

    nx_ = sqrt(n_);
    ny_ = nx_;
    uint32_t threads = 256;
    block_dim_ = dim3(threads / 16, threads / 16);
    grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                     (ny_ + block_dim_.y - 1) / block_dim_.y);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    store_snapshot(0);
  }

  ~NLSESaturatingSolver() {
    delete matfunc_;
    cudaFree(d_u_);
    cudaFree(d_buf_);
    cudaFree(d_density_);
    cudaFree(d_u_trajectory_);
  }

  void step(const std::complex<double> tau, const uint32_t step_number) {
    density<<<grid_dim_, block_dim_>>>(d_density_, d_u_, nx_, ny_);
    nonlin_part<<<grid_dim_, block_dim_>>>(d_buf_, d_u_, d_density_, -.5 * tau,
                                           nx_, ny_);
    matfunc_->apply(d_u_, d_buf_, -tau);

    density<<<grid_dim_, block_dim_>>>(d_density_, d_u_, nx_, ny_);
    nonlin_part<<<grid_dim_, block_dim_>>>(d_u_, d_u_, d_density_, -.5 * tau,
                                           nx_, ny_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    if (step_number % params_.snapshot_freq == 0) {
      store_snapshot(step_number);
    }
  }

  void transfer_snapshots(std::complex<double> *dst) {
    cudaMemcpy(dst, d_u_trajectory_,
               params_.num_snapshots * n_ * sizeof(thrust::complex<double>),
               cudaMemcpyDeviceToHost);
  }

private:
  void store_snapshot(const uint32_t step_number) {
    cudaMemcpy(d_u_trajectory_ + current_snapshot_ * n_, d_u_,
               n_ * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
    current_snapshot_++;
  }

  MatrixFunctionApplicatorComplex *matfunc_;
  thrust::complex<double> *d_u_;
  thrust::complex<double> *d_buf_;
  thrust::complex<double> *d_density_;
  thrust::complex<double> *d_u_trajectory_;

  uint32_t nx_, ny_;
  uint32_t current_snapshot_;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;
};

#endif
