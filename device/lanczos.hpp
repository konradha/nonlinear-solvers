// lanczos.hpp
#include "pragmas.hpp"

__global__ void dot_product_kernel(const double* x, const double* y, 
                                 double* result, const uint32_t n) {
    __shared__ double temp[BLOCK_SIZE];
    const uint32_t tid = threadIdx.x;
    double sum = 0.0;

    for(uint32_t i = tid + blockIdx.x * blockDim.x; i < n; 
        i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    temp[tid] = sum;
    __syncthreads();

    for(uint32_t s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if(tid < s) temp[tid] += temp[tid + s];
        __syncthreads();
    }
    if(tid == 0) atomicAdd(result, temp[0]);
}

__global__ void scale_normalize_kernel(double* v, const double* x,
                                     const double* norm, const uint32_t n) {
    const double inv_norm = 1.0 / sqrt(*norm);
    for(uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        v[i] = x[i] * inv_norm;
    }
}

__global__ void lanczos_update_kernel(double* w, const double* v, 
                                    const double* coeffs,
                                    const uint32_t j, const uint32_t n) {
    for(uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
        i += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for(uint32_t k = 0; k <= j; k++) {
            sum += coeffs[k] * v[k * n + i];
        }
        w[i] -= sum;
    }
}

void lanczos_iteration(DeviceSpMV* spmv, KrylovInfo* krylov) {
    cudaMemset(krylov->buf1, 0, sizeof(double));
    dot_product_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(krylov->V, krylov->V, 
                                                 krylov->buf1, krylov->n);
    scale_normalize_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(krylov->V, krylov->V, 
                                                     krylov->buf1, krylov->n);

    for(uint32_t j = 0; j < krylov->m - 1; j++) {
        spmv->multiply(krylov->V + j * krylov->n, krylov->buf2);
        
        cudaMemset(krylov->buf1, 0, (j + 1) * sizeof(double));
        for(uint32_t i = 0; i <= j; i++) {
            dot_product_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>
                (krylov->buf2, krylov->V + i * krylov->n, 
                 krylov->buf1 + i, krylov->n);
        }
        
        lanczos_update_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>
            (krylov->buf2, krylov->V, krylov->buf1, j, krylov->n);

        cudaMemset(krylov->buf1, 0, sizeof(double));
        dot_product_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>
            (krylov->buf2, krylov->buf2, krylov->buf1, krylov->n);
            
        scale_normalize_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>
            (krylov->V + (j + 1) * krylov->n, krylov->buf2, 
             krylov->buf1, krylov->n);
             
        cudaMemcpy(krylov->T + j * krylov->m, krylov->buf1, 
                  (j + 1) * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(krylov->T + (j + 1) + j * krylov->m, krylov->buf1, 
                  sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(krylov->T + j + (j + 1) * krylov->m, krylov->buf1, 
                  sizeof(double), cudaMemcpyDeviceToDevice);
    }
}
