#ifndef LANCZOS_COMPLEX_HPP
#define LANCZOS_COMPLEX_HPP

#define COUNTER 0

#include "pragmas.hpp"
#include "spmv.hpp"
#include <thrust/complex.h>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <cublas_v2.h>

#define NUM_BLOCKS 256
#define BLOCK_SIZE 1024
#define MAX_BLOCKS_PER_DIM 65535

struct KrylovInfoComplex {
  thrust::complex<double> * T;
  thrust::complex<double> * V;
  thrust::complex<double> * buf1;
  thrust::complex<double> * buf2;
  double * d_beta;
  double * reconstruct_beta;
  uint32_t n;
  uint32_t m;
  cublasHandle_t handle;
};


__global__ void norm_complex(const thrust::complex<double> *__restrict__ vec,
                             double *__restrict__ result, const uint32_t n) {
  __shared__ double shared_mem[BLOCK_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0.0;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    thrust::complex<double> val = vec[i];
    sum += val.real() * val.real() + val.imag() * val.imag();
  }

  shared_mem[tid] = sum;
  __syncthreads();

  for (uint32_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_mem[tid] += shared_mem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, shared_mem[0]); 
  } 

  __syncthreads();
  if(tid == 0 && blockIdx.x == 0) { 
    __threadfence();
    *result = sqrt(*result);
  }
}


__global__ void inv_scale_complex(thrust::complex<double> *__restrict__ vec,
                                  const double * scalar, const uint32_t n) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  double local_scalar = *scalar;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    vec[i] = vec[i] / local_scalar;
  }
}

__global__ void dot_complex(const thrust::complex<double> *__restrict__ v1,
                            const thrust::complex<double> *__restrict__ v2,
                            thrust::complex<double> *__restrict__ result,
                            const uint32_t n) {
  __shared__ thrust::complex<double> shared_mem[BLOCK_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  thrust::complex<double> sum(0.0, 0.0);
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    sum += thrust::conj(v2[i]) * v1[i];
  }

  shared_mem[tid] = sum;
  __syncthreads();

  for (uint32_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_mem[tid] += shared_mem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    double *result_ptr = reinterpret_cast<double *>(result);
    atomicAdd(&result_ptr[0], shared_mem[0].real());
    atomicAdd(&result_ptr[1], shared_mem[0].imag());
  }
}

__global__ void axpby_complex(thrust::complex<double> *__restrict__ v1,
                              const thrust::complex<double> *__restrict__ v2,
                              thrust::complex<double>* alpha,
                              thrust::complex<double>* beta,
                              const uint32_t n, bool neg_beta=true) {   
  thrust::complex<double> local_alpha = alpha[0];
  thrust::complex<double> local_beta = neg_beta? -beta[0] : beta[0];
  
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    v1[i] = local_alpha * v1[i] + local_beta * v2[i];
  }
}

__global__ void copyDoubleToComplexReal(thrust::complex<double>* dest, const double* src) {
    *dest = thrust::complex<double>(*src, 0.0);
}

/*
void lanczos_iteration_complex(
    DeviceSpMV<thrust::complex<double>> *spmv, KrylovInfoComplex *krylov,
    const thrust::complex<double> *__restrict__ u) {
  const uint32_t n = krylov->n;
  const uint32_t m = krylov->m;
 
  const uint32_t total_threads_needed = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const uint32_t blocks_x = std::min(total_threads_needed,
                                   static_cast<uint32_t>(MAX_BLOCKS_PER_DIM));
  const uint32_t blocks_y = (total_threads_needed + blocks_x - 1) / blocks_x;
  
  dim3 grid(blocks_x, blocks_y);
  dim3 block(BLOCK_SIZE);

  //printf("Total threads %d\n", total_threads_needed);
  //printf("Total blocks_x %d\n", blocks_x);
  //printf("Total blocks_y %d\n", blocks_y);

  cudaMemcpy(krylov->V, u, n * sizeof(thrust::complex<double>),
             cudaMemcpyDeviceToDevice);

  cudaMemset(krylov->d_beta, 0, sizeof(double));
  norm_complex<<<grid, block>>>(krylov->V, krylov->d_beta, n);
  cudaMemcpy(krylov->reconstruct_beta, krylov->d_beta,
               sizeof(double), cudaMemcpyDeviceToDevice);
  inv_scale_complex<<<grid, block>>>(krylov->V, krylov->d_beta, n);

  cudaMemset(krylov->T, 0, m * m * sizeof(thrust::complex<double>));

  thrust::complex<double> one(1., 0);
  thrust::complex<double>* one_dev;
  cudaMalloc(&one_dev, sizeof(thrust::complex<double>));
  cudaMemcpy(one_dev, &one, sizeof(thrust::complex<double>), cudaMemcpyHostToDevice); 

  for (uint32_t j = 0; j < m - 1; j++) {
    spmv->multiply(&krylov->V[j * n], krylov->buf1);

      
    if (j > 0) {
      axpby_complex<<<grid, block>>>(
          krylov->buf1, &krylov->V[(j - 1) * n],
          one_dev,
	  &krylov->T[m * (j - 1) + j],
          n, true);
    }
     
    
    cudaMemset(&krylov->T[m * j + j], 0, sizeof(thrust::complex<double>));
    dot_complex<<<grid, block>>>(krylov->buf1, &krylov->V[j * n],
                                 &krylov->T[m * j + j], n);

    axpby_complex<<<grid, block>>>(
        krylov->buf1, &krylov->V[j * n], one_dev,
        &krylov->T[m * j + j], n, true);

    for (uint32_t i = 0; i <= j; i++) {
      cudaMemset(krylov->buf2, 0, sizeof(thrust::complex<double>));
      dot_complex<<<grid, block>>>(krylov->buf1, &krylov->V[i * n],
                                   krylov->buf2, n);
      axpby_complex<<<grid, block>>>(
          krylov->buf1, &krylov->V[i * n], one_dev,
          krylov->buf2, n, true);
    }

    cudaMemset(krylov->d_beta, 0, sizeof(double));
    norm_complex<<<grid, block>>>(krylov->buf1, krylov->d_beta, n);
    
    copyDoubleToComplexReal<<<1, 1>>>(&(krylov->T[m * j + j + 1]), krylov->d_beta);
    copyDoubleToComplexReal<<<1, 1>>>(&(krylov->T[m * (j+1) + j]), krylov->d_beta); 
    inv_scale_complex<<<grid, block>>>(krylov->buf1, krylov->d_beta, n);
    cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1,
               n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice); 
  }
  cudaFree(one_dev); 
}*/

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <cublas_v2.h>
#include <thrust/iterator/counting_iterator.h>


#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>


#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

void lanczos_iteration_complex(
    DeviceSpMV<thrust::complex<double>> *spmv,
    KrylovInfoComplex *krylov,
    const thrust::complex<double> *__restrict__ u) {
#if COUNTER
    cudaEvent_t start_data_mov, stop_data_mov;
    cudaEventCreate(&start_data_mov);
    cudaEventCreate(&stop_data_mov);

    cudaEvent_t start_spmv, stop_spmv;
    cudaEventCreate(&start_spmv);
    cudaEventCreate(&stop_spmv);

    cudaEvent_t start_beg_loop, stop_beg_loop;
    cudaEventCreate(&start_beg_loop);
    cudaEventCreate(&stop_beg_loop);

    cudaEvent_t start_inner_loop, stop_inner_loop;
    cudaEventCreate(&start_inner_loop);
    cudaEventCreate(&stop_inner_loop);

    cudaEvent_t start_rest_loop, stop_rest_loop;
    cudaEventCreate(&start_rest_loop);
    cudaEventCreate(&stop_rest_loop);
    float t = 0.f, t_data_mov = 0.f, t_spmv = 0.f, t_beg = 0.f, t_inner = 0.f, t_rest = 0.f;
#endif

    const uint32_t n = krylov->n;
    const uint32_t m = krylov->m;

    auto handle = krylov->handle;
#if COUNTER
    cudaEventRecord(start_data_mov);
#endif
    cudaMemcpy(krylov->V, u, n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
    cudaMemset(krylov->T, 0, m * m * sizeof(thrust::complex<double>));
#if COUNTER
    cudaEventRecord(stop_data_mov);
    cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
    t_data_mov += t;
    t = 0;
#endif
    double beta;
    cublasNrm2Ex(handle, n,
                 krylov->V, CUDA_C_64F, 1,
                 &beta, CUDA_R_64F, CUDA_R_64F);
#if COUNTER
    cudaEventRecord(start_data_mov);
#endif
    cudaMemcpy(krylov->d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(krylov->reconstruct_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
#if COUNTER
    cudaEventRecord(stop_data_mov);
    cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
    t_data_mov += t;
    t = 0;
#endif
    cuDoubleComplex scale = make_cuDoubleComplex(1.0/beta, 0.0);
    cublasZscal(handle, n, &scale, reinterpret_cast<cuDoubleComplex*>(krylov->V), 1);

    cuDoubleComplex dot_result; 
    cuDoubleComplex* h;
    cudaMalloc(&h, m*sizeof(cuDoubleComplex));
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
    for (uint32_t j = 0; j < m - 1; j++) {
#if COUNTER
	cudaEventRecord(start_spmv);
#endif
        spmv->multiply(&krylov->V[j * n], krylov->buf1);
#if COUNTER
        cudaEventRecord(stop_spmv);
        cudaEventSynchronize(stop_spmv);
        cudaEventElapsedTime(&t, start_spmv, stop_spmv);
	t_spmv += t;
	t = 0;
	cudaEventRecord(start_beg_loop);
#endif
        if (j > 0) {
            cublasZdotc(handle, n,
                       reinterpret_cast<const cuDoubleComplex*>(krylov->buf1), 1,
                       reinterpret_cast<const cuDoubleComplex*>(&krylov->V[(j-1) * n]), 1,
                       &dot_result);
#if COUNTER
	    cudaEventRecord(start_data_mov);
#endif
            cudaMemcpy(&krylov->T[m * (j-1) + j], &dot_result, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
#if COUNTER
            cudaEventRecord(stop_data_mov);
            cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
            t_data_mov += t;
            t = 0;
#endif
            scale = make_cuDoubleComplex(-dot_result.x, -dot_result.y);
            cublasZaxpy(handle, n, &scale,
                       reinterpret_cast<const cuDoubleComplex*>(&krylov->V[(j-1) * n]), 1,
                       reinterpret_cast<cuDoubleComplex*>(krylov->buf1), 1);
        }

        cublasZdotc(handle, n,
                   reinterpret_cast<const cuDoubleComplex*>(krylov->buf1), 1,
                   reinterpret_cast<const cuDoubleComplex*>(&krylov->V[j * n]), 1,
                   &dot_result);
#if COUNTER
        cudaEventRecord(start_data_mov);
#endif
        cudaMemcpy(&krylov->T[m * j + j], &dot_result, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
#if COUNTER
        cudaEventRecord(stop_data_mov);
        cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
        t_data_mov += t;
        t = 0;
#endif
        scale = make_cuDoubleComplex(-dot_result.x, -dot_result.y);
        cublasZaxpy(handle, n, &scale,
                   reinterpret_cast<const cuDoubleComplex*>(&krylov->V[j * n]), 1,
                   reinterpret_cast<cuDoubleComplex*>(krylov->buf1), 1);
#if COUNTER
        cudaEventRecord(stop_beg_loop);
        cudaEventSynchronize(stop_beg_loop);
        cudaEventElapsedTime(&t, start_beg_loop, stop_beg_loop);
	t_beg += t;
	t = 0;

        cudaEventRecord(start_inner_loop);	     
#endif
        cublasZgemv(handle, CUBLAS_OP_C,
                    n, j+1,  // V: n rows, j+1 cols
                    &one,
                    reinterpret_cast<const cuDoubleComplex*>(krylov->V), n,
                    reinterpret_cast<const cuDoubleComplex*>(krylov->buf1), 1,
                    &zero,
                    h, 1); 
        cublasZgemv(handle, CUBLAS_OP_N,
                    n, j+1,
                    &neg_one,
                    reinterpret_cast<const cuDoubleComplex*>(krylov->V), n,
                    h, 1,
                    &one, reinterpret_cast<cuDoubleComplex*>(krylov->buf1), 1);
	
        //for (uint32_t i = 0; i <= j; i++) {
        //    cublasZdotc(handle, n,
        //               reinterpret_cast<const cuDoubleComplex*>(krylov->buf1), 1,
        //               reinterpret_cast<const cuDoubleComplex*>(&krylov->V[i * n]), 1,
        //               &dot_result);

        //    scale = make_cuDoubleComplex(-dot_result.x, -dot_result.y);
        //    cublasZaxpy(handle, n, &scale,
        //               reinterpret_cast<const cuDoubleComplex*>(&krylov->V[i * n]), 1,
        //               reinterpret_cast<cuDoubleComplex*>(krylov->buf1), 1);
        //}	
#if COUNTER
        cudaEventRecord(stop_inner_loop);
        cudaEventSynchronize(stop_inner_loop);
        cudaEventElapsedTime(&t, start_inner_loop, stop_inner_loop);
	t_inner += t;
	t = 0;
        cudaEventRecord(start_rest_loop);
#endif
        cublasNrm2Ex(handle, n,
                     krylov->buf1, CUDA_C_64F, 1,
                     &beta, CUDA_R_64F, CUDA_R_64F);

        cuDoubleComplex beta_complex = make_cuDoubleComplex(beta, 0.0);
#if COUNTER
        cudaEventRecord(start_data_mov);
#endif
        cudaMemcpy(&krylov->T[m * j + j + 1], &beta_complex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(&krylov->T[m * (j+1) + j], &beta_complex, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
#if COUNTER
        cudaEventRecord(stop_data_mov);
        cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
        t_data_mov += t;
        t = 0;
#endif
        if (beta > 0.0) {
            scale = make_cuDoubleComplex(1.0/beta, 0.0);
            cublasZscal(handle, n, &scale, reinterpret_cast<cuDoubleComplex*>(krylov->buf1), 1);
        }
#if COUNTER
	cudaEventRecord(start_data_mov);
#endif
        cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1, n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
#if COUNTER
        cudaEventRecord(stop_data_mov);
        cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
        t_data_mov += t;
        t = 0;
        cudaEventRecord(stop_rest_loop);
        cudaEventSynchronize(stop_rest_loop);
        cudaEventElapsedTime(&t, start_rest_loop, stop_rest_loop);
	t_rest += t;
	t = 0;
#endif
    }
    cudaFree(h);
#if COUNTER
    float s = t_spmv + t_data_mov + t_inner + t_rest + t_beg;
    printf("%-12s | %-8s | %-10s | %-10s | %-10s\n", 
       "data movement", "spmv", "begin loop", "inner loop", "rest loop");
    printf("%12.6f | %8.6f | %10.6f | %10.6f | %10.6f\n", 
       t_data_mov/s, t_spmv/s, t_beg/s, t_inner/s, t_rest/s);
#endif
    //cublasDestroy(handle);
}

#endif
