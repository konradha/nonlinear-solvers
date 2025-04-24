#ifndef NLSE_DEV_HPP
#define NLSE_DEV_HPP

#include "boundaries.cuh"
#include "matfunc_complex.hpp"
#include "nlse.cuh"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace device {

class NLSESolverDevice {
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

  NLSESolverDevice(const Eigen::SparseMatrix<std::complex<double>> &L,
                   const std::complex<double> *host_u0, const double *host_m,
                   const Parameters &params = Parameters())
      : n_(L.rows()), current_snapshot_(0), params_(params) {

    cudaMalloc(&d_u_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_density_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf2_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf3_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf4_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf5_, n_ * sizeof(thrust::complex<double>));

    cudaMalloc(&d_m_, n_ * sizeof(double));
    cudaMalloc(&d_u_trajectory_,
               n_ * params_.num_snapshots * sizeof(thrust::complex<double>));

    cudaMemcpy(d_u_, host_u0, n_ * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_, host_m, n_ * sizeof(double), cudaMemcpyHostToDevice);

    matfunc_ = new MatrixFunctionApplicatorComplex(L, n_, params_.krylov_dim,
                                                   L.nonZeros());

    nx_ = sqrt(n_);
    ny_ = nx_;
    if (nx_ * ny_ != n_) {
      is_3d_ = true;
      nx_ = ny_ = nz_ = std::cbrt(n_);
      assert(nx_ * ny_ * nz_ == n_);
    }

    // TODO check dims here
    uint32_t threads = 256;
    if (!is_3d_) {
      block_dim_ = dim3(16, 16, 1);
      grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                       (ny_ + block_dim_.y - 1) / block_dim_.y);
    } else {
      block_dim_ = dim3(8, 8, 4);
      grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                       (ny_ + block_dim_.y - 1) / block_dim_.y,
                       (nz_ + block_dim_.z - 1) / block_dim_.z);
    }
    // block_dim_ = dim3(threads / 16, threads / 16);
    // grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
    //                  (ny_ + block_dim_.y - 1) / block_dim_.y);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    store_snapshot(0);
  }

  ~NLSESolverDevice() {
    delete matfunc_;
    cudaFree(d_u_);
    cudaFree(d_buf_);
    cudaFree(d_buf2_);
    cudaFree(d_buf3_);
    cudaFree(d_buf4_);
    cudaFree(d_buf5_);
    cudaFree(d_density_);
    cudaFree(d_m_);
    cudaFree(d_u_trajectory_);
    if (d_u_prev_ != nullptr)
      cudaFree(d_u_prev_);
  }

  void apply_bc() {
    if (!is_3d_) {
      neumann_bc_no_velocity_blocking<thrust::complex<double>>(d_u_, nx_, ny_);
    } else {
      neumann_bc_no_velocity_blocking_3d<thrust::complex<double>>(d_u_, nx_,
                                                                  ny_, nz_);
    }
  }

  void step(const std::complex<double> tau, const uint32_t step_number) {
    if (!is_3d_) {
      device::density<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
                                                 ny_);
      device::nonlin_part<<<grid_dim_, block_dim_>>>(d_buf_, d_u_, d_density_,
                                                     .5 * tau, nx_, ny_);
      matfunc_->apply(d_u_, d_buf_, tau, "exp");

      device::density<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
                                                 ny_);
      device::nonlin_part<<<grid_dim_, block_dim_>>>(d_u_, d_u_, d_density_,
                                                     .5 * tau, nx_, ny_);
    } else {
      device::density_3d<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
                                                 ny_, nz_);
      device::nonlin_part_3d<<<grid_dim_, block_dim_>>>(d_buf_, d_u_, d_density_,
                                                     .5 * tau, nx_, ny_, nz_);
      matfunc_->apply(d_u_, d_buf_, tau, "exp");

      device::density_3d<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
                                                 ny_, nz_);
      device::nonlin_part_3d<<<grid_dim_, block_dim_>>>(d_u_, d_u_, d_density_,
                                                     .5 * tau, nx_, ny_, nz_);
    }

    if (step_number % params_.snapshot_freq == 0) {
      store_snapshot(step_number);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }

  void step_sewi(const std::complex<double> tau, const uint32_t step_number) {
    // initial step is symmetric SS2, see publication for reference
    if (step_number == 1) {
      cudaMalloc(&d_u_prev_, n_ * sizeof(thrust::complex<double>));
      cudaMemcpy(d_u_prev_, d_u_, n_ * sizeof(thrust::complex<double>),
                 cudaMemcpyDeviceToDevice);
      step(tau, 1);
    } else {
      cudaMemcpy(d_buf_, d_u_, n_ * sizeof(thrust::complex<double>),
                 cudaMemcpyDeviceToDevice); // bookkeeping prev
      if (!is_3d_) {
        B<<<grid_dim_, block_dim_>>>(d_buf2_, d_m_, d_u_, nx_, ny_); 
        thrust::complex<double> real_tau(tau.imag(), 0.0);
	matfunc_->apply(d_buf3_ /* out */, d_buf2_ /* in */, real_tau, "sinc");
        matfunc_->apply(d_buf2_, d_buf3_, tau, "exp"); // re-use buffer from before 
        matfunc_->apply(d_buf3_, d_u_prev_, 2. * tau, "exp");
        // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
        sewi<<<grid_dim_, block_dim_>>>(d_u_, d_buf2_, d_buf3_, tau, nx_, ny_); 
      } else {
        B_3d<<<grid_dim_, block_dim_>>>(d_buf2_, d_m_, d_u_, nx_, ny_, nz_); 
        thrust::complex<double> real_tau(tau.imag(), 0.0);
	matfunc_->apply(d_buf3_ /* out */, d_buf2_ /* in */, real_tau, "sinc");
        matfunc_->apply(d_buf2_, d_buf3_, tau, "exp"); // re-use buffer from before 
        matfunc_->apply(d_buf3_, d_u_prev_, 2. * tau, "exp");
        // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
        sewi_3d<<<grid_dim_, block_dim_>>>(d_u_, d_buf2_, d_buf3_, tau, nx_, ny_, nz_);
      }
      cudaMemcpy(d_u_prev_, d_buf_, n_ * sizeof(thrust::complex<double>),
                 cudaMemcpyDeviceToDevice); // attach "prev" to prev
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
  thrust::complex<double> *d_u_prev_ =
      nullptr; // for first step set to nullptr?
  thrust::complex<double> *d_buf_;
  thrust::complex<double> *d_buf2_;
  thrust::complex<double> *d_buf3_;
  thrust::complex<double> *d_buf4_;
  thrust::complex<double> *d_buf5_;
  thrust::complex<double> *d_density_;
  thrust::complex<double> *d_u_trajectory_;
  double *d_m_;

  uint32_t nx_, ny_, nz_;
  bool is_3d_ = false;
  uint32_t current_snapshot_;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;
};
} // namespace device
#endif
