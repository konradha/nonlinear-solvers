// matfunc.hpp
#include "pragmas.hpp"
#include "lanczos.hpp"
#include "spmv.hpp"


template<typename ScalarType, typename TimeType>
class MatrixFunctionApplicator {
private:
   DeviceSpMV* spmv_;
   KrylovInfo* krylov_;
   cusolverDnHandle_t solver_handle_;
   double* d_eigenvalues_;
   double* d_eigenvectors_;
   double* d_work_;
   ScalarType* d_transform_;
   int* d_info_;
   uint32_t n_;
   uint32_t m_;
   int lwork_;
   dim3 grid_dim_2d_;
   dim3 block_dim_2d_;
   dim3 grid_dim_1d_;
   dim3 block_dim_1d_;

   using function_type = ScalarType(*)(TimeType, double);

   void compute_eigen_decomposition() {
       cudaMemcpy(d_eigenvectors_, krylov_->T, m_ * m_ * sizeof(double), 
                 cudaMemcpyDeviceToDevice);
       
       cusolverDnDsyevd_bufferSize(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER, m_,
                                  d_eigenvectors_, m_, d_eigenvalues_, &lwork_);
       
       cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                       CUBLAS_FILL_MODE_LOWER, m_,
                       d_eigenvectors_, m_, d_eigenvalues_,
                       d_work_, lwork_, d_info_);
   }

public:
   enum class TransformType {
       DIRECT,
       SQRT
   };

   MatrixFunctionApplicator(DeviceSpMV* spmv, KrylovInfo* krylov,
                           cusolverDnHandle_t handle, uint32_t n, uint32_t m)
       : spmv_(spmv), krylov_(krylov), solver_handle_(handle), 
         n_(n), m_(m) {
       cudaMalloc(&d_eigenvalues_, m * sizeof(double));
       cudaMalloc(&d_eigenvectors_, m * m * sizeof(double));
       cudaMalloc(&d_transform_, m * sizeof(ScalarType));
       cudaMalloc(&d_info_, sizeof(int));

       block_dim_2d_ = dim3(16, 16);
       grid_dim_2d_ = dim3((m + block_dim_2d_.x - 1) / block_dim_2d_.x,
                          (m + block_dim_2d_.y - 1) / block_dim_2d_.y);
       block_dim_1d_ = dim3(256);
       grid_dim_1d_ = dim3((n + block_dim_1d_.x - 1) / block_dim_1d_.x);
   }

   ~MatrixFunctionApplicator() {
       cudaFree(d_eigenvalues_);
       cudaFree(d_eigenvectors_);
       cudaFree(d_transform_);
       cudaFree(d_info_);
       if(d_work_) cudaFree(d_work_);
   }

   void apply(ScalarType* result, const double* u, TimeType t,
             function_type f, TransformType transform = TransformType::DIRECT) {
       lanczos_iteration(spmv_, krylov_);
       compute_eigen_decomposition();
       
       if(transform == TransformType::SQRT) {
           transform_eigenvalues_sqrt_kernel<<<grid_dim_1d_, block_dim_1d_>>>
               (d_transform_, d_eigenvalues_, t, f, m_);
       } else {
           transform_eigenvalues_kernel<<<grid_dim_1d_, block_dim_1d_>>>
               (d_transform_, d_eigenvalues_, t, f, m_);
       }
                    
       compute_matrix_function_kernel<<<grid_dim_2d_, block_dim_2d_>>>
           (krylov_->buf1, d_eigenvectors_, d_transform_, m_);
           
       apply_to_first_column_kernel<<<grid_dim_1d_, block_dim_1d_>>>
           (result, krylov_->V, krylov_->buf1, krylov_->beta, n_, m_);
   }
};

template<typename ScalarType, typename TimeType>
__global__ void transform_eigenvalues_kernel(
   ScalarType* transformed, const double* eigenvalues,
   TimeType t, ScalarType(*f)(TimeType, double), uint32_t m) {
   
   const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < m) {
       transformed[idx] = f(t, eigenvalues[idx]);
   }
}

template<typename ScalarType, typename TimeType>
__global__ void transform_eigenvalues_sqrt_kernel(
   ScalarType* transformed, const double* eigenvalues,
   TimeType t, ScalarType(*f)(TimeType, double), uint32_t m) {
   
   const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < m) {
       transformed[idx] = f(t, sqrt(abs(eigenvalues[idx])));
   }
}

template<typename ScalarType>
__global__ void compute_matrix_function_kernel(
   ScalarType* result_matrix,
   const double* Q,
   const ScalarType* transformed_eigenvalues,
   uint32_t m) {
   
   const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
   const uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;
   
   if(row < m && col < m) {
       ScalarType sum = 0;
       for(uint32_t k = 0; k < m; k++) {
           sum += Q[row + k * m] * transformed_eigenvalues[k] * Q[col + k * m];
       }
       result_matrix[row + col * m] = sum;
   }
}

template<typename ScalarType>
__global__ void apply_to_first_column_kernel(
   ScalarType* result,
   const double* V,
   const ScalarType* matrix,
   const double beta,
   uint32_t n,
   uint32_t m) {
   
   const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if(idx < n) {
       ScalarType sum = 0;
       for(uint32_t j = 0; j < m; j++) {
           sum += V[j * n + idx] * matrix[j];
       }
       result[idx] = beta * sum;
   }
}

template<typename ScalarType, typename TimeType>
ScalarType exp_func(TimeType t, double x) { return exp(t * x); }

template<typename ScalarType, typename TimeType>
ScalarType cos_func(TimeType t, double x) { return cos(t * x); }

template<typename ScalarType, typename TimeType>
ScalarType sin_func(TimeType t, double x) { return sin(t * x); }

template<typename ScalarType, typename TimeType>
ScalarType sinc_func(TimeType t, double x) { 
   auto tx = t * x;
   return abs(tx) < 1e-14 ? ScalarType(1.0) : sin(tx)/tx; 
}
