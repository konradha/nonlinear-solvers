#ifndef MATFUNC_REAL_HPP
#define MATFUNC_REAL_HPP

#include "lanczos.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <iostream>

// K1: f(t sqrt(λ_i))

// sinc²
__global__ void transform_eigenvals_sinc2_sqrt(double *out, const double *eigvals,
                                    const double t, const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    double x = t * sqrt(abs(eigvals[i]));
    out[i] = (abs(x) < 1e-8) ? 1.0 : pow(sin(x) / x, 2);
  }
}

// cos
__global__ void transform_eigenvals_cos_sqrt(double *out, const double *eigvals,
                                    const double t, const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    double x = t * sqrt(abs(eigvals[i]));
    out[i] = cos(x);
  }
}

// Id
__global__ void transform_eigenvals_id_sqrt(double *out, const double *eigvals,
                                    const double t, const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    double x = t * sqrt(abs(eigvals[i]));
    out[i] = x;
  }
}

// K2: Q f(λ) Q^T
__global__ void eigvec_transform(double *out, const double *Q,
                                 const double *f_lambda, const uint32_t m) {
  // let's take a single block -- might become problematic for large m!
  int i = threadIdx.x;
  int j = threadIdx.y;
  if (i < m && j < m) {
    double sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += Q[k * m + i] * f_lambda[k] * Q[k * m + j];
    }
    out[i * m + j] = sum;
  }
}

// K3: β V(Q f(λ) Q^T)e_1
__global__ void final_multiply(double *result, const double *V,
                               const double *transform, const double beta,
                               const uint32_t n, const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    double sum = 0.0;
    for (int j = 0; j < m; j++) {
      double v_sum = 0.0;
      for (int k = 0; k < m; k++) {
        v_sum += V[k * n + i] * transform[k * m + j];
      }
      sum = (j == 0) ? v_sum : sum;
    }
    result[i] = beta * sum;
  }
}

class MatrixFunctionApplicatorReal {
public:
  enum class FunctionType { COS_SQRT, SINC2_SQRT, ID_SQRT };

  void print_type(FunctionType t) const {
	  switch (t) {
		  case FunctionType::COS_SQRT:
			  std::cout << "type is cos sqrt\n";
			  break;
		  case FunctionType::SINC2_SQRT:
			  std::cout << "type is sinc2 sqrt\n";
			  break;
		  case FunctionType::ID_SQRT:
			  std::cout << "type is Id sqrt\n";
			  break;
		  default:
			  throw std::runtime_error("Invalid matrix func call");
	  }
  }

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

    cudaMalloc(&d_diag_, m * sizeof(double));
    cudaMalloc(&d_small_result_, m * m * sizeof(double));

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
    cudaFree(d_diag_);
    cudaFree(d_small_result_);
    cudaFree(d_transform_);
    cudaFree(d_work_);
    cudaFree(d_solver_work_);
    cudaFree(d_info_);
    cusolverDnDestroy(solver_handle_);
  }

  void apply(double *result, const double *input, double t, FunctionType type) {
    lanczos_iteration(spmv_, &krylov_, input);
    cudaDeviceSynchronize();

    double beta;
    cudaMemcpy(&beta, krylov_.d_beta, sizeof(double), cudaMemcpyDeviceToHost);
    compute_eigen_decomposition();

    cudaMemset(d_diag_, 0.0, m_ * sizeof(double)); 
    block_dim_1d_ = dim3(256);
    // TODO this might need better parametrization depending on sqrt(n) or even n^(1/3) (if 3d case)
    grid_dim_1d_ = dim3((n_ + block_dim_1d_.x - 1) / block_dim_1d_.x);
    block_dim_2d_ = dim3(16, 16);
    grid_dim_2d_ = dim3((m_ + block_dim_2d_.x - 1) / block_dim_2d_.x,
                        (m_ + block_dim_2d_.y - 1) / block_dim_2d_.y);

    // TODO check if the values _actually_ check out -- ie. if time stepping agrees let's just keep it at first
    // and optimize kernels and kernel tuning later

    // print_type(type);
    switch (type) {
	    case FunctionType::SINC2_SQRT:
		transform_eigenvals_sinc2_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
        	d_diag_, d_eigenvalues_, t, m_);
		break;
	    case FunctionType::COS_SQRT:
		transform_eigenvals_cos_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
        	d_diag_, d_eigenvalues_, t, m_);
		break;	
	    case FunctionType::ID_SQRT:
		transform_eigenvals_id_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
        	d_diag_, d_eigenvalues_, t, m_);
		break;

	default:
		throw std::runtime_error("Invalid Matfunc call");
    }
    

    eigvec_transform<<<grid_dim_2d_, block_dim_2d_>>>(
        d_small_result_, d_eigenvectors_, d_diag_, m_);

    final_multiply<<<grid_dim_1d_, block_dim_1d_>>>(
        result, krylov_.V, d_small_result_, beta, n_, m_);
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

  double *d_diag_;
  double *d_small_result_;

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

  dim3 block_dim_large_;
  dim3 grid_dim_large_;
  Parameters params_;
};

#endif
