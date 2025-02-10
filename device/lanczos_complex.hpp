#ifndef LANCZOS_COMPLEX_HPP
#define LANCZOS_COMPLEX_HPP

#include "pragmas.hpp"
#include "spmv.hpp"
#include <thrust/complex.h>

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
}

__global__ void scalar_sqrt_complex(double *__restrict__ x) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0)
    x[0] = sqrt(x[0]);
}

__global__ void inv_scale_complex(thrust::complex<double> *__restrict__ vec,
                                  const double scalar, const uint32_t n) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    vec[i] = vec[i] / scalar;
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
                              const thrust::complex<double> alpha,
                              const thrust::complex<double> beta,
                              const uint32_t n) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    v1[i] = alpha * v1[i] + beta * v2[i];
  }
}


void lanczos_iteration_complex(
    const Eigen::SparseMatrix<std::complex<double>> &L,
    DeviceSpMV<thrust::complex<double>> *spmv, KrylovInfoComplex *krylov,
    const thrust::complex<double> *__restrict__ u,
    const Eigen::VectorX<std::complex<double>> u_cpu) {
  const uint32_t n = krylov->n;
  const uint32_t m = krylov->m;
  //dim3 grid(NUM_BLOCKS);
  //dim3 block(BLOCK_SIZE);
  const uint32_t total_threads_needed = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const uint32_t blocks_x = std::min(total_threads_needed,
                                   static_cast<uint32_t>(MAX_BLOCKS_PER_DIM));
  const uint32_t blocks_y = (total_threads_needed + blocks_x - 1) / blocks_x;
  
  dim3 grid(blocks_x, blocks_y);
  dim3 block(BLOCK_SIZE);


  cudaMemcpy(krylov->V, u, n * sizeof(thrust::complex<double>),
             cudaMemcpyDeviceToDevice);

  cudaMemset(krylov->d_beta, 0, sizeof(double));
  norm_complex<<<grid, block>>>(krylov->V, krylov->d_beta, n);
  scalar_sqrt_complex<<<1, 1>>>(krylov->d_beta);

  double beta;
  cudaMemcpy(&beta, krylov->d_beta, sizeof(double), cudaMemcpyDeviceToHost);
  inv_scale_complex<<<grid, block>>>(krylov->V, beta, n);

  Eigen::MatrixX<std::complex<double>> T(m, m);
  cudaMemset(krylov->T, 0, m * m * sizeof(thrust::complex<double>));

  for (uint32_t j = 0; j < m - 1; j++) {
    spmv->multiply(&krylov->V[j * n], krylov->buf1);

    if (j > 0) {
      axpby_complex<<<grid, block>>>(
          krylov->buf1, &krylov->V[(j - 1) * n],
          thrust::complex<double>(1.0, 0.0),
          thrust::complex<double>(-T(j - 1, j).real(), -T(j - 1, j).imag()), n);
    }

    cudaMemset(&krylov->T[m * j + j], 0, sizeof(thrust::complex<double>));
    dot_complex<<<grid, block>>>(krylov->buf1, &krylov->V[j * n],
                                 &krylov->T[m * j + j], n);
    cudaMemcpy(&T(j, j), &krylov->T[m * j + j], sizeof(std::complex<double>),
               cudaMemcpyDeviceToHost);

    axpby_complex<<<grid, block>>>(
        krylov->buf1, &krylov->V[j * n], thrust::complex<double>(1.0, 0.0),
        thrust::complex<double>(-T(j, j).real(), -T(j, j).imag()), n);

    for (uint32_t i = 0; i <= j; i++) {
      cudaMemset(krylov->buf2, 0, sizeof(thrust::complex<double>));
      dot_complex<<<grid, block>>>(krylov->buf1, &krylov->V[i * n],
                                   krylov->buf2, n);
      thrust::complex<double> coeff;
      cudaMemcpy(&coeff, krylov->buf2, sizeof(thrust::complex<double>),
                 cudaMemcpyDeviceToHost);
      axpby_complex<<<grid, block>>>(
          krylov->buf1, &krylov->V[i * n], thrust::complex<double>(1.0, 0.0),
          thrust::complex<double>(-coeff.real(), -coeff.imag()), n);
    }

    cudaMemset(krylov->d_beta, 0, sizeof(double));
    norm_complex<<<grid, block>>>(krylov->buf1, krylov->d_beta, n);
    scalar_sqrt_complex<<<1, 1>>>(krylov->d_beta);
    cudaDeviceSynchronize();

    double norm_val;
    cudaMemcpy(&norm_val, krylov->d_beta, sizeof(double),
               cudaMemcpyDeviceToHost);
    T(j + 1, j) = std::complex<double>(norm_val, 0.0);
    T(j, j + 1) = T(j + 1, j);
    cudaMemcpy(&krylov->T[m * j + j + 1], &T(j + 1, j),
               sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(&krylov->T[m * (j + 1) + j], &T(j + 1, j),
               sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

    inv_scale_complex<<<grid, block>>>(krylov->buf1, norm_val, n);
    cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1,
               n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(krylov->reconstruct_beta, &beta,
               sizeof(double), cudaMemcpyHostToDevice);

}

#endif
