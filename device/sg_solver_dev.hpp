#ifndef SG_SOLVER_DEV_HPP
#define SG_SOLVER_DEV_HPP

#include "matfunc_real.hpp"
#include "pragmas.hpp"

__global__ void check_nans_kernel(int *has_nan, const double *arr,
                                  const uint32_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (isnan(arr[idx]))
      atomicAdd(has_nan, 1);
  }
}

__global__ void neg_sin_kernel_2d(double *out, const double *in,
                                  const uint32_t nx, const uint32_t ny) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    out[idx] = -sin(in[idx]);
  }
}

__global__ void curr_velocity_2d(double *v, const double *u,
                                 const double *u_past, const double tau,
                                 const uint32_t nx, const uint32_t ny) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    v[idx] = (u[idx] - u_past[idx]) / tau;
  }
}

__global__ void gautschi_kernel_2d(double *u_next, const double *u_past,
                                   const double *costu,
                                   const double *filtered_sinc,
                                   const double tau, const uint32_t nx,
                                   const uint32_t ny) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    u_next[idx] =
        2.0 * costu[idx] - u_past[idx] + tau * tau * filtered_sinc[idx];
  }
}

__global__ void calculate_energy_kernel_2d(double *energy, const double *u,
                                           const double *v, const double *Lu,
                                           const double *c, const double *m,
                                           const uint32_t nx,
                                           const uint32_t ny) {
  extern __shared__ double sdata[];
  const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  double local_sum = 0.0;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    local_sum = 0.5 * (v[idx] * v[idx] + c[idx] * u[idx] * Lu[idx] -
                       2.0 * m[idx] * cos(u[idx]));
  }

  sdata[tid] = local_sum;
  __syncthreads();

  for (uint32_t s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    energy[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
}

__global__ void validate_coverage_2d(bool *coverage_map, const uint32_t nx,
                                     const uint32_t ny) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const int idx = y * nx + x;
    coverage_map[idx] = true;
  }
}

class SGESolverDevice {
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

  SGESolverDevice(const int *d_row_ptr, const int *d_col_ind,
                  const double *d_values, const double *h_c, const double *h_m,
                  uint32_t n, uint32_t nnz, const double *host_u0,
                  const double *host_v0, const double *host_u_past,
                  const Parameters &params = Parameters())
      : n_(n), current_snapshot_(0), params_(params) {

    cudaMalloc(&d_u_, n * sizeof(double));
    cudaMalloc(&d_v_, n * sizeof(double));
    cudaMalloc(&d_u_past_, n * sizeof(double));

    cudaMemcpy(d_u_, host_u0, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_, host_v0, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_past_, host_u_past, n * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_buf_, n * sizeof(double));
    cudaMalloc(&d_buf2_, n * sizeof(double));
    cudaMalloc(&d_buf3_, n * sizeof(double));
    cudaMalloc(&d_c_, n * sizeof(double));
    cudaMalloc(&d_m_, n * sizeof(double));
    cudaMalloc(&d_work_buffer_, n * sizeof(double));

    cudaMalloc(&d_u_trajectory_, n * params_.num_snapshots * sizeof(double));
    cudaMalloc(&d_v_trajectory_, n * params_.num_snapshots * sizeof(double));
    cudaMalloc(&d_e_trajectory_, params_.num_snapshots * sizeof(double));

    cudaMemcpy(d_c_, h_c, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_, h_m, n * sizeof(double), cudaMemcpyHostToDevice);

    spmv_ = new DeviceSpMV<double>(d_row_ptr, d_col_ind, d_values, n, nnz);
    matfunc_ = new MatrixFunctionApplicatorReal(d_row_ptr, d_col_ind, d_values,
                                                n, params_.krylov_dim, nnz);

    // TODO refactor to actually put in nx, ny
    // AND of course: actual dimension if we want to push this into 3d
    nx_ = sqrt(n);
    ny_ = nx_;
    uint32_t threads = 256;
    block_dim_ = dim3(threads / 16, threads / 16);
    grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                     (ny_ + block_dim_.y - 1) / block_dim_.y);

    store_snapshot(0);
    verify_domain_coverage();
  }

  ~SGESolverDevice() {
    delete spmv_;
    delete matfunc_;
    cudaFree(d_u_);
    cudaFree(d_v_);
    cudaFree(d_u_past_);
    cudaFree(d_buf_);
    cudaFree(d_buf2_);
    cudaFree(d_buf3_);
    cudaFree(d_c_);
    cudaFree(d_m_);
    cudaFree(d_u_trajectory_);
    cudaFree(d_v_trajectory_);
    cudaFree(d_e_trajectory_);
    cudaFree(d_work_buffer_);
  }

  void step(const double tau, const uint32_t step_number) {
    // reset_buffers();
    cudaMemcpy(d_buf_, d_u_, n_ * sizeof(double), cudaMemcpyDeviceToDevice);

    matfunc_->apply(d_work_buffer_, d_u_, tau,
                    MatrixFunctionApplicatorReal::FunctionType::ID_SQRT);
    neg_sin_kernel_2d<<<grid_dim_, block_dim_>>>(d_work_buffer_, d_work_buffer_,
                                                 nx_, ny_);
    matfunc_->apply(d_buf2_, d_work_buffer_, tau,
                    MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);

    matfunc_->apply(d_buf3_, d_u_, tau,
                    MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);

    gautschi_kernel_2d<<<grid_dim_, block_dim_>>>(d_u_, d_u_past_, d_buf3_,
                                                  d_buf2_, tau, nx_, ny_);
    cudaMemcpy(d_u_past_, d_buf_, n_ * sizeof(double),
               cudaMemcpyDeviceToDevice);

    curr_velocity_2d<<<grid_dim_, block_dim_>>>(d_v_, d_u_, d_u_past_, tau, nx_,
                                                ny_);

    if (step_number % params_.snapshot_freq == 0) {
      store_snapshot(step_number);
    }
  }

  void transfer_snapshots(double *dst, char which = 'u') {
    if (which == 'u')
      cudaMemcpy(dst, d_u_trajectory_,
                 params_.num_snapshots * n_ * sizeof(double),
                 cudaMemcpyDeviceToHost);
    else if (which == 'v')
      cudaMemcpy(dst, d_v_trajectory_,
                 params_.num_snapshots * n_ * sizeof(double),
                 cudaMemcpyDeviceToHost);
    else if (which == 'e')
      cudaMemcpy(dst, d_e_trajectory_, params_.num_snapshots * sizeof(double),
                 cudaMemcpyDeviceToHost);
    else
      throw std::runtime_error("Invalid observable requested");
  }

private:
  void store_snapshot(const uint32_t step_number) {
    // calculate_energy_kernel<<<grid_dim_, block_dim_.x * sizeof(double)>>>(
    //     d_e_trajectory_ + current_snapshot_, d_u_, d_buf_,
    //     d_buf_, d_c_, d_m_, n_
    //);
    cudaMemcpy(d_u_trajectory_ + current_snapshot_ * n_, d_u_,
               n_ * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_v_trajectory_ + current_snapshot_ * n_, d_v_,
               n_ * sizeof(double), cudaMemcpyDeviceToDevice);
    current_snapshot_++;
  }

  void reset_buffers() {
    cudaMemset(d_buf_, 0.0, n_ * sizeof(double));
    cudaMemset(d_buf2_, 0.0, n_ * sizeof(double));
    cudaMemset(d_buf3_, 0.0, n_ * sizeof(double));
    cudaMemset(d_work_buffer_, 0.0, n_ * sizeof(double));
  }

  DeviceSpMV<double> *spmv_;
  MatrixFunctionApplicatorReal *matfunc_;
  double *d_u_;
  double *d_v_;
  double *d_u_past_;

  // suppose equation to be
  // u_tt = c (u_xx + u_yy) - m sin(u)
  // hence, update will look like TODO -- incorporating c implies changing L!
  // u_{n+1} = 2 cos (tau \Omega) u_{n} - u_{n-1} + tau² sinc²(tau / 2 \Omega) x
  // g(\phi(tau \Omegau_{n}))
  double *d_c_;
  double *d_m_;

  double *d_buf_;
  double *d_buf2_;
  double *d_buf3_;
  double *d_work_buffer_;

  double *d_u_trajectory_;
  double *d_v_trajectory_;
  double *d_e_trajectory_;

  uint32_t nx_, ny_;

  uint32_t current_snapshot_;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;

  void verify_domain_coverage() {
    bool *d_coverage;
    cudaMalloc(&d_coverage, n_ * sizeof(bool));
    cudaMemset(d_coverage, 0, n_ * sizeof(bool));

    validate_coverage_2d<<<grid_dim_, block_dim_>>>(d_coverage, nx_, ny_);

    bool *h_coverage = new bool[n_];
    cudaMemcpy(h_coverage, d_coverage, n_ * sizeof(bool),
               cudaMemcpyDeviceToHost);

    bool complete_coverage = true;
    for (uint32_t i = 0; i < n_; ++i) {
      if (!h_coverage[i]) {
        const uint32_t y = i / nx_;
        const uint32_t x = i % nx_;
        complete_coverage = false;
      }
    }

    if (complete_coverage) {
      printf("Complete domain coverage verified: all %u x %u points are "
             "processed\n",
             nx_, ny_);
    } else {
      throw std::runtime_error(
          "Kernels will not traverse entire grid correctly\n");
    }

    delete[] h_coverage;
    cudaFree(d_coverage);
  }
};

#endif
