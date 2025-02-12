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

class MatrixFunctionApplicatorComplex {
public:
  struct Parameters {
    uint32_t block_size_1d;
    Parameters() : block_size_1d(256) {}
  };

  MatrixFunctionApplicatorComplex(const int *d_row_ptr, const int *d_col_ind,
                                  const cuDoubleComplex *d_values, uint32_t n,
                                  uint32_t m, uint32_t nnz,
                                  const Parameters &params = Parameters())
      : n_(n), m_(m), params_(params) {

    thrust::complex<double> *thrust_values;
    cudaMalloc(&thrust_values, nnz * sizeof(thrust::complex<double>));
    cudaMemcpy(thrust_values, d_values, nnz * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToDevice);

    spmv_ = new DeviceSpMV<thrust::complex<double>>(d_row_ptr, d_col_ind,
                                                    thrust_values, n, nnz);
    cudaFree(thrust_values);

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
    cudaMalloc(&d_eigenvectors_, m * m * sizeof(thrust::complex<double>));
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
    cudaMalloc(&d_solver_work_, lwork * sizeof(thrust::complex<double>));
    solver_work_size_ = lwork;
  }

  ~MatrixFunctionApplicatorComplex() {
    delete spmv_;
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
  }

  void apply(cuDoubleComplex *result, const cuDoubleComplex *input,
             thrust::complex<double> dt) {
    auto input_thrust =
        reinterpret_cast<const thrust::complex<double> *>(input);
    lanczos_iteration_complex(spmv_, &krylov_, input_thrust);
    double beta;
    cudaMemcpy(&beta, krylov_.reconstruct_beta, sizeof(double),
               cudaMemcpyDeviceToHost);
    compute_eigen_decomposition();

    Eigen::MatrixX<std::complex<double>> T =
        Eigen::MatrixX<std::complex<double>>::Zero(m_, m_);
    Eigen::MatrixX<std::complex<double>> V =
        Eigen::MatrixX<std::complex<double>>::Zero(n_, m_);

    cudaMemcpy(T.data(), krylov_.T, m_ * m_ * sizeof(std::complex<double>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(V.data(), krylov_.V, n_ * m_ * sizeof(std::complex<double>),
               cudaMemcpyDeviceToHost);

    Eigen::MatrixX<std::complex<double>> evs =
        Eigen::MatrixX<std::complex<double>>::Zero(m_, m_);
    Eigen::VectorX<std::complex<double>> lambdas =
        Eigen::VectorX<std::complex<double>>::Zero(T.rows());

    cudaMemcpy(evs.data(), d_eigenvectors_,
               m_ * m_ * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    cudaMemcpy(lambdas.data(), d_eigenvalues_,
               m_ * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    std::complex<double> tau(dt.real(), dt.imag());
    Eigen::MatrixX<std::complex<double>> exp_T =
        evs *
        (tau * lambdas.array().abs().unaryExpr(
                   [](std::complex<double> x) { return std::exp(x); }))
            .matrix()
            .asDiagonal() *
        evs.transpose();
    Eigen::VectorX<std::complex<double>> e1 =
        Eigen::VectorX<std::complex<double>>::Zero(T.rows());
    e1(0) = 1.0;
    e1 = beta * V * exp_T * e1;
    cudaMemcpy(result, e1.data(), n_ * sizeof(std::complex<double>),
               cudaMemcpyHostToDevice);

    /*
     * Eigen::MatrixX<Float> exp_T =
          (es.eigenvectors() *
           (t * es.eigenvalues().array().abs())
               .unaryExpr([](Float x) { return std::exp(x); })
               .matrix()
               .asDiagonal() *
           es.eigenvectors().transpose());
      Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
      e1(0) = 1.0;
      return beta * V * exp_T * e1;
      */
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

  DeviceSpMV<thrust::complex<double>> *spmv_;
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
};

#endif
