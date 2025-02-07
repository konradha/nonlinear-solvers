#ifndef MATFUNC_REAL_HPP
#define MATFUNC_REAL_HPP

#include "lanczos.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <iostream>

__global__ void transform_cos_sqrt(double *transform, const double *eigenvalues,
                                   double t, uint32_t m) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    transform[idx] = cos(t * sqrt(abs(eigenvalues[idx])));
  }
}

__global__ void transform_sinc2_sqrt(double *transform,
                                     const double *eigenvalues, double t,
                                     uint32_t m, double threshold) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    double x = t * sqrt(abs(eigenvalues[idx]));
    transform[idx] = abs(x) < threshold ? 1.0 : (sin(x) / x) * (sin(x) / x);
  }
}

__global__ void transform_id_sqrt(double *transform, const double *eigenvalues,
                                  double t, uint32_t m) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    transform[idx] = t * sqrt(abs(eigenvalues[idx]));
  }
}

__global__ void compute_matrix_function_kernel(double *result, const double *Q,
                                               const double *transform,
                                               uint32_t m) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < m) {
    double sum = 0;
    for (uint32_t k = 0; k < m; k++) {
      sum += Q[row + k * m] * transform[k] * Q[col + k * m];
    }
    result[row + col * m] = sum;
  }
}

__global__ void apply_to_first_column_kernel(double *result, const double *V,
                                             const double *matrix, double beta,
                                             uint32_t n, uint32_t m) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    double sum = 0;
    for (uint32_t j = 0; j < m; j++) {
      sum += V[j * n + idx] * matrix[j];
    }
    result[idx] = beta * sum;
  }
}

class MatrixFunctionApplicatorReal {
public:
  enum class FunctionType { COS_SQRT, SINC2_SQRT, ID_SQRT };

  struct Parameters {
    uint32_t block_size_1d;
    uint32_t block_size_2d;
    double sinc_threshold;
    Parameters() : block_size_1d(64), block_size_2d(8), sinc_threshold(1e-8) {}
  };

  MatrixFunctionApplicatorReal(const int *d_row_ptr, const int *d_col_ind,
                               const double *d_values, uint32_t n, uint32_t m,
                               uint32_t nnz,
                               const Parameters &params = Parameters())
      : n_(n), m_(m), params_(params) {

    spmv_ = new DeviceSpMV<double>(d_row_ptr, d_col_ind, d_values, n, nnz);

    cudaMalloc((void **)&krylov_.V, n * m * sizeof(double));
    cudaMalloc((void **)&krylov_.T, m * m * sizeof(double));
    cudaMalloc((void **)&krylov_.buf1, n * sizeof(double));
    cudaMalloc((void **)&krylov_.buf2, n * sizeof(double));
    cudaMalloc((void **)&krylov_.d_beta, sizeof(double));
    krylov_.n = n;
    krylov_.m = m;

    cudaMalloc((void **)&d_eigenvalues_, m * sizeof(double));
    cudaMalloc((void **)&d_eigenvectors_, m * m * sizeof(double));
    cudaMalloc((void **)&d_transform_, m * sizeof(double));
    cudaMalloc((void **)&d_work_, m * m * sizeof(double));
    cudaMalloc((void **)&d_info_, sizeof(int));

    cusolverDnCreate(&solver_handle_);
    int lwork;
    cusolverDnDsyevd_bufferSize(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER, m, d_eigenvectors_, m,
                                d_eigenvalues_, &lwork);
    cudaMalloc((void **)&d_solver_work_, lwork * sizeof(double));
    solver_work_size_ = lwork;

    block_dim_2d_ = dim3(params_.block_size_2d, params_.block_size_2d);
    grid_dim_2d_ = dim3((m + block_dim_2d_.x - 1) / block_dim_2d_.x,
                        (m + block_dim_2d_.y - 1) / block_dim_2d_.y);
    block_dim_1d_ = dim3(params_.block_size_1d);
    grid_dim_1d_ = dim3(block_dim_1d_.x);
  }

  ~MatrixFunctionApplicatorReal() {
    delete spmv_;
    cudaFree(krylov_.V);
    cudaFree(krylov_.T);
    cudaFree(krylov_.buf1);
    cudaFree(krylov_.buf2);
    cudaFree(krylov_.d_beta);
    cudaFree(d_eigenvalues_);
    cudaFree(d_eigenvectors_);
    cudaFree(d_transform_);
    cudaFree(d_work_);
    cudaFree(d_solver_work_);
    cudaFree(d_info_);
    cusolverDnDestroy(solver_handle_);
  }

  void apply(double *result, const double *input, double t, FunctionType type) {
    Eigen::MatrixX<double> V = Eigen::MatrixX<double>::Zero(n_, m_);
    Eigen::MatrixX<double> T = Eigen::MatrixX<double>::Zero(m_, m_);

    // double beta;
    // Eigen::VectorX<double> eigenvalues(m_);
    // Eigen::MatrixX<double> eigenvectors(m_, m_);

    lanczos_iteration(spmv_, &krylov_, input);
    cudaDeviceSynchronize();

    compute_eigen_decomposition();

    /*
    // Working on porting this to device
    cudaMemcpy(&beta, krylov_.d_beta, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V.data(), krylov_.V, n_ * m_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(T.data(), krylov_.T, m_ * m_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvalues.data(), d_eigenvalues_, m_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvectors.data(), d_eigenvectors_, m_ * m_ * sizeof(double),
               cudaMemcpyDeviceToHost);

    //std::cout << "V.col(0) device\n";
    //std::cout << V.col(0) << "\n";
    //std::cout << "T device\n";
    //std::cout << T << "\n";

    //eigenvectors = eigenvectors.transpose();
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<double>> es(T);
    //eigenvalues = es.eigenvalues();
    //eigenvectors = es.eigenvectors();

    Eigen::MatrixX<double> sinc_sqrt_T =
        (eigenvectors *
         (t / 2. * eigenvalues.array().abs().sqrt())
             .unaryExpr([](double x) {
               return std::abs(x) < 1e-8 ? 1.0 : std::sin(x) / x;
             })
             .square()
             .matrix()
             .asDiagonal() *
         eigenvectors.transpose());
    Eigen::VectorX<double> e1 = Eigen::VectorX<double>::Zero(T.rows());
    e1(0) = 1.0;
    Eigen::VectorX<double> res = beta * V * sinc_sqrt_T * e1;
    cudaMemcpy(result, res.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
    */

    // TODO: figure out error in these kernels!
    compute_transform(t / 2, type);
    compute_matrix_function();
    apply_to_first_column(result);
  }

private:
  void compute_eigen_decomposition() {
    cudaMemcpy(d_eigenvectors_, krylov_.T, m_ * m_ * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                     CUBLAS_FILL_MODE_LOWER, m_, d_eigenvectors_, m_,
                     d_eigenvalues_, d_solver_work_, solver_work_size_,
                     d_info_);
  }

  void compute_transform(double t, FunctionType type) {
    switch (type) {
    case FunctionType::COS_SQRT:
      transform_cos_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
          d_transform_, d_eigenvalues_, t, m_);
      return;
    case FunctionType::SINC2_SQRT:
      transform_sinc2_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
          d_transform_, d_eigenvalues_, t, m_, params_.sinc_threshold);
      return;
    case FunctionType::ID_SQRT:
      transform_id_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(d_transform_,
                                                         d_eigenvalues_, t, m_);
      return;
    }
  }

  void compute_matrix_function() {
    compute_matrix_function_kernel<<<grid_dim_2d_, block_dim_2d_>>>(
        d_work_, d_eigenvectors_, d_transform_, m_);
  }

  void apply_to_first_column(double *result) {
    double beta;
    cudaMemcpy(&beta, (krylov_.d_beta), sizeof(double), cudaMemcpyDeviceToHost);
    apply_to_first_column_kernel<<<grid_dim_1d_, block_dim_1d_>>>(
        result, krylov_.V, d_work_, beta, n_, m_);
  }

  DeviceSpMV<double> *spmv_;
  KrylovInfo krylov_;
  cusolverDnHandle_t solver_handle_;
  double *d_eigenvalues_;
  double *d_eigenvectors_;
  double *d_transform_;
  double *d_work_;
  double *d_solver_work_;
  int *d_info_;
  int solver_work_size_;
  uint32_t n_;
  uint32_t m_;
  dim3 grid_dim_2d_;
  dim3 block_dim_2d_;
  dim3 grid_dim_1d_;
  dim3 block_dim_1d_;
  Parameters params_;
};

#endif
