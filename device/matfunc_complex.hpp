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

__global__ void matrix_multiply_QDQ(const thrust::complex<double> *Q, const double *D,
                                    thrust::complex<double> *result, const uint32_t m) {
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

__global__ void matrix_multiply_VK(const thrust::complex<double> *V, const thrust::complex<double> *K,
                                   thrust::complex<double> *result, const uint32_t n,
                                   const uint32_t m) {
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

__global__ void scale_first_col(const thrust::complex<double> *X, thrust::complex<double> *result,
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

  MatrixFunctionApplicatorComplex(const Eigen::SparseMatrix<std::complex<double>> & A,
		                  uint32_t n,
                                  uint32_t m, uint32_t nnz,
                                  const Parameters &params = Parameters())
      : n_(n), m_(m), params_(params) { 
    const int *row_ptr = A.outerIndexPtr();
    const int *col_ind = A.innerIndexPtr();
    const std::complex<double> *values = A.valuePtr(); 
    cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_ind_, A.nonZeros() * sizeof(int));
    cudaMalloc(&d_values_, A.nonZeros() * sizeof(thrust::complex<double>));

    cudaMemcpy(d_row_ptr_, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind_, col_ind, A.nonZeros() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_, values, A.nonZeros() * sizeof(thrust::complex<double>),
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

  void apply(thrust::complex<double> *result, const thrust::complex<double> *input,
             std::complex<double> dt) { 
    reset_data();
    lanczos_iteration_complex(spmv_, &krylov_, input); 
    cudaDeviceSynchronize();
    double beta;
    cudaMemcpy(&beta, krylov_.reconstruct_beta, sizeof(double),
               cudaMemcpyDeviceToHost);
    //Eigen::MatrixX<std::complex<double>> T(m_, m_);
    //cudaMemcpy(T.data(), krylov_.T,
    //           m_ * m_ * sizeof(std::complex<double>),
    //           cudaMemcpyDeviceToHost); 
    //Eigen::MatrixX<std::complex<double>> V(n_, m_);
    //cudaMemcpy(V.data(), krylov_.V,
    //           n_ * m_ * sizeof(std::complex<double>),
    //           cudaMemcpyDeviceToHost);

    //std::cout << "Device beta (hostcpy):\n" << beta << "\n";
    //std::cout << "Device T (hostcpy):\n" << T << "\n";
    //std::cout << "Device V (hostcpy):\n" << V << "\n";

    
    compute_eigen_decomposition();
    dim3 block_dim_1d_ = dim3(256);
    
    dim3 block_2d(16, 16);
    dim3 grid_VK((m_ + block_2d.x - 1) / block_2d.x,
                 (n_ + block_2d.y - 1) / block_2d.y);
    dim3 grid_QDQ((m_ + block_2d.x - 1) / block_2d.x,
                  (m_ + block_2d.y - 1) / block_2d.y);
    dim3 block_1d(256);
    dim3 grid_1d((n_ + block_1d.x - 1) / block_1d.x);

    matrix_multiply_QDQ<<<grid_QDQ, block_2d>>>(d_eigenvectors_, d_eigenvalues_,
                                                d_work_, m_);
    matrix_multiply_VK<<<grid_VK, block_2d>>>(krylov_.V, d_work_, d_work_large_,
                                              n_, m_);
    scale_first_col<<<grid_1d, block_1d>>>(d_work_large_, result, beta, n_);



    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<std::complex<double>>> es(T);
    ////std::cout << "Device T eigenvalues\n";
    ////std::cout << es.eigenvalues() << "\n";
    //cudaDeviceSynchronize();
    //Eigen::MatrixX<std::complex<double>> exp_T =
    //       (es.eigenvectors() *
    //        (dt * es.eigenvalues().array().abs())
    //            .unaryExpr([](std::complex<double> x) { return std::exp(x); })
    //            .matrix()
    //            .asDiagonal() *
    //        es.eigenvectors().transpose());
    //Eigen::VectorX<std::complex<double>> e1 = Eigen::VectorX<std::complex<double>>::Zero(T.rows());
    //e1(0) = 1.0;
    //Eigen::VectorX<std::complex<double>> r = beta * V * exp_T * e1;
    //cudaDeviceSynchronize(); 
    //cudaMemcpy(result, r.data(), 
    //           n_ * sizeof(thrust::complex<double>),
    //           cudaMemcpyHostToDevice);

    /*
     * Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
       Eigen::MatrixX<Float> exp_T =
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

  void reset_data() {
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
};

#endif
