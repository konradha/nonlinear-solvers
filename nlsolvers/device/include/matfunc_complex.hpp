#ifndef MATFUNC_COMPLEX_HPP
#define MATFUNC_COMPLEX_HPP

#include "lanczos_complex.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/complex.h>

__global__ void transform_eigenvals_exp(thrust::complex<double> *out,
                                        const double *eigvals,
                                        thrust::complex<double> dt,
                                        const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    out[i] = thrust::exp(dt * eigvals[i]);
  }
}

__global__ void matrix_multiply_QDQ(const thrust::complex<double> *Q,
                                    const thrust::complex<double> *D,
                                    thrust::complex<double> *result,
                                    const uint32_t m) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < m) {
    thrust::complex<double> sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += Q[k * m + row] * D[k] * Q[k * m + col];
    }
    result[col * m + row] = sum;
  }
}

__global__ void matrix_multiply_VK(const thrust::complex<double> *V,
                                   const thrust::complex<double> *K,
                                   thrust::complex<double> *result,
                                   const uint32_t n, const uint32_t m) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < m) {
    thrust::complex<double> sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += V[k * n + row] * K[col * m + k];
    }
    result[col * n + row] = sum;
  }
}

__global__ void scale_first_col(const thrust::complex<double> *X,
                                thrust::complex<double> *result,
                                const double beta, const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = beta * X[idx];
  }
}

class MatrixFunctionApplicatorComplex {
public:
  struct Parameters {
    uint32_t block_size_1d;
    Parameters() : block_size_1d(256) {}
  };

  MatrixFunctionApplicatorComplex(
      const Eigen::SparseMatrix<std::complex<double>> &A, uint32_t n,
      uint32_t m, uint32_t nnz, const Parameters &params = Parameters())
      : n_(n), m_(m), params_(params) {
    const int *row_ptr = A.outerIndexPtr();
    const int *col_ind = A.innerIndexPtr();
    const std::complex<double> *values = A.valuePtr();
    cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_ind_, A.nonZeros() * sizeof(int));
    cudaMalloc(&d_values_, A.nonZeros() * sizeof(thrust::complex<double>));

    cudaMemcpy(d_row_ptr_, row_ptr, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind_, col_ind, A.nonZeros() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_, values,
               A.nonZeros() * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);

    spmv_ = new DeviceSpMV<thrust::complex<double>>(d_row_ptr_, d_col_ind_,
                                                    d_values_, n, nnz);

    cudaMalloc(&krylov_.V, n * m * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.T, m * m * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.buf1, n * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.buf2, n * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.d_beta, sizeof(double));
    cudaMalloc(&krylov_.reconstruct_beta, sizeof(double));
    cublasCreate(&krylov_.handle);
    krylov_.n = n;
    krylov_.m = m;

    cudaMalloc(&d_eigenvalues_, m * sizeof(double));

    cudaMalloc(&d_diag_, m * sizeof(thrust::complex<double>));
    cudaMalloc(&d_work_, m * m * sizeof(thrust::complex<double>));
    cudaMalloc(&d_work_large_, n * m * sizeof(thrust::complex<double>));
    cudaMalloc(&d_info_, sizeof(int));

    cusolverDnCreate(&solver_handle_);
    cublasCreate(&cublas_handle_);

    int lwork;
    cusolverDnZheevd_bufferSize(
        solver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m,
        reinterpret_cast<cuDoubleComplex *>(d_eigenvectors_), m, d_eigenvalues_,
        &lwork);
    cudaMalloc(&d_eigenvectors_, m * m * sizeof(cuDoubleComplex));
    cudaMalloc(&d_solver_work_, lwork * sizeof(cuDoubleComplex));
    solver_work_size_ = lwork;

    block_dim_1d_ = dim3(256);
    block_2d_ = dim3(16, 16);
    grid_VK_ = dim3((m_ + block_2d_.x - 1) / block_2d_.x,
                    (n_ + block_2d_.y - 1) / block_2d_.y);
    grid_QDQ_ = dim3((m_ + block_2d_.x - 1) / block_2d_.x,
                     (m_ + block_2d_.y - 1) / block_2d_.y);
    block_1d_ = dim3(256);
    grid_1d_ = dim3((n_ + block_1d_.x - 1) / block_1d_.x);
  }

  ~MatrixFunctionApplicatorComplex() {
    cudaFree(d_row_ptr_);
    cudaFree(d_col_ind_);
    cudaFree(d_values_);
    cudaFree(krylov_.V);
    cudaFree(krylov_.T);
    cudaFree(krylov_.buf1);
    cudaFree(krylov_.buf2);
    cudaFree(krylov_.d_beta);
    cudaFree(krylov_.reconstruct_beta);
    cublasDestroy(krylov_.handle);
    cudaFree(d_eigenvalues_);
    cudaFree(d_eigenvectors_);
    cudaFree(d_diag_);
    cudaFree(d_work_);
    cudaFree(d_work_large_);
    cudaFree(d_solver_work_);
    cudaFree(d_info_);
    cusolverDnDestroy(solver_handle_);
    cublasDestroy(cublas_handle_);
    delete spmv_;
  }

  void apply(thrust::complex<double> *result,
             const thrust::complex<double> *input, std::complex<double> dt) {
    reset_data();
    lanczos_iteration_complex(spmv_, &krylov_, input);
    // cudaDeviceSynchronize();
    double beta;
    cudaMemcpy(&beta, krylov_.reconstruct_beta, sizeof(double),
               cudaMemcpyDeviceToHost);

    compute_eigen_decomposition();

    // thrust::apply
    transform_eigenvals_exp<<<grid_1d_, block_1d_>>>(d_diag_, d_eigenvalues_,
                                                     dt, m_);

    matrix_multiply_QDQ<<<grid_QDQ_, block_2d_>>>(d_eigenvectors_, d_diag_,
                                                  d_work_, m_);
    // gemmZ
    matrix_multiply_VK<<<grid_VK_, block_2d_>>>(krylov_.V, d_work_,
                                                d_work_large_, n_, m_);
    // dscalZ
    scale_first_col<<<grid_1d_, block_1d_>>>(d_work_large_, result, beta, n_);
  }

private:
  void compute_eigen_decomposition() {
    cudaMemcpy(d_eigenvectors_, krylov_.T, m_ * m_ * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToDevice);
    cusolverDnZheevd(
        solver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m_,
        reinterpret_cast<cuDoubleComplex *>(d_eigenvectors_), m_,
        d_eigenvalues_, reinterpret_cast<cuDoubleComplex *>(d_solver_work_),
        solver_work_size_, d_info_);
  }

  void reset_data() {
    // maybe this can become shorter
    cudaMemset(krylov_.T, 0, m_ * m_ * sizeof(thrust::complex<double>));
    cudaMemset(krylov_.V, 0, n_ * m_ * sizeof(thrust::complex<double>));
    cudaMemset(krylov_.buf1, 0, n_ * sizeof(thrust::complex<double>));
    cudaMemset(krylov_.buf2, 0, n_ * sizeof(thrust::complex<double>));

    cudaMemset(krylov_.d_beta, 0, sizeof(double));
    cudaMemset(krylov_.reconstruct_beta, 0, sizeof(double));
    cudaMemset(d_eigenvectors_, 0, m_ * m_ * sizeof(thrust::complex<double>));
    cudaMemset(d_eigenvalues_, 0, m_ * sizeof(double));
  }

  DeviceSpMV<thrust::complex<double>> *spmv_;
  int *d_row_ptr_, *d_col_ind_;
  thrust::complex<double> *d_values_;

  KrylovInfoComplex krylov_;
  cusolverDnHandle_t solver_handle_;
  cublasHandle_t cublas_handle_;
  double *d_eigenvalues_;
  thrust::complex<double> *d_eigenvectors_;
  thrust::complex<double> *d_diag_;
  thrust::complex<double> *d_work_;
  thrust::complex<double> *d_work_large_;
  thrust::complex<double> *d_solver_work_;
  int *d_info_;
  int solver_work_size_;
  uint32_t n_;
  uint32_t m_;
  Parameters params_;

  dim3 block_dim_1d_;
  dim3 block_2d_;

  dim3 grid_VK_;
  dim3 grid_QDQ_;

  dim3 block_1d_;
  dim3 grid_1d_;
};

#endif
