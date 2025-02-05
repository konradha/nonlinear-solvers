#ifndef LANCZOS_HPP
#define LANCZOS_HPP

#include "pragmas.hpp"
#include "spmv.hpp"

#define NUM_BLOCKS 256
#define BLOCK_SIZE 256

struct KrylovInfo {
    double *__restrict__ T;        
    double *__restrict__ V;       
    double *__restrict__ buf1;     
    double *__restrict__ buf2;     
    double *__restrict__ d_beta;
    uint32_t n;                    
    uint32_t m;                    
};


__global__ void vector_norm_kernel(const double* __restrict__ vec,
                                 double* __restrict__ result,
                                 const uint32_t n) {
    __shared__ double shared_mem[BLOCK_SIZE];
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
        double val = vec[i];
        sum += val * val;
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (uint32_t s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

__global__ void vector_scale_kernel(double* __restrict__ vec,
                                  const double scalar,
                                  const uint32_t n) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
        vec[i] /= scalar;
    }
}

__global__ void dot_product_kernel(const double* __restrict__ v1,
                                 const double* __restrict__ v2,
                                 double* __restrict__ result,
                                 const uint32_t n) {
    __shared__ double shared_mem[BLOCK_SIZE];
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
        sum += v1[i] * v2[i];
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (uint32_t s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

__global__ void vector_axpby_kernel(double* __restrict__ v1,
                                  const double* __restrict__ v2,
                                  const double alpha,
                                  const double beta,
                                  const uint32_t n) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
        v1[i] = alpha * v1[i] + beta * v2[i];
    }
}

__global__ void scalar_sqrt(double * __restrict__ x) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0)
    x[0] = sqrt(x[0]);
}


void lanczos_iteration(const Eigen::SparseMatrix<double> &L,
                      DeviceSpMV<double> *spmv,
                      KrylovInfo *krylov,
                      const double* __restrict__ u,
		      const Eigen::VectorX<double> u_cpu) {

    const uint32_t n = krylov->n;
    const uint32_t m = krylov->m;
    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);
 
    cudaMemcpy(krylov->V, (void*)u, n * sizeof(double), cudaMemcpyDeviceToDevice);

    // \beta = ||u||
    cudaMemset(krylov->d_beta, 0, sizeof(double));
    vector_norm_kernel<<<grid, block>>>(krylov->V, krylov->d_beta, n);

    double beta;
    cudaMemcpy(&beta, krylov->d_beta, sizeof(double), cudaMemcpyDeviceToHost);
    beta = std::sqrt(beta);
    vector_scale_kernel<<<grid, block>>>(krylov->V, beta, n);

    Eigen::MatrixX<double> V(n, m);
    Eigen::MatrixX<double> T(m, m); 

    cudaMemcpy(V.data(), krylov->V, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (uint32_t j = 0; j < m - 1; j++) {
	cudaMemcpy(&krylov->V[j * n], V.col(j).data(), n * sizeof(double), cudaMemcpyHostToDevice);	
	spmv->multiply(&krylov->V[j*n], krylov->buf1);
	 
	if (j> 0) {
	  vector_axpby_kernel<<<grid, block>>>(krylov->buf1, &krylov->V[(j-1)*n], 1.0, -T(j-1,j), n);           
	}
        	
	
	cudaMemset(krylov->T + m * j + j, 0, sizeof(double));
	dot_product_kernel<<<grid, block>>>(krylov->buf1, &(krylov->V[j * n]), krylov->T + m * j + j, n);
	cudaMemcpy(&T(j, j), krylov->T + m * j + j, sizeof(double), cudaMemcpyDeviceToHost);
        vector_axpby_kernel<<<grid, block>>>(krylov->buf1, &krylov->V[j*n], 1.0, -T(j,j), n);

	Eigen::VectorX<double> w(n);
        cudaMemcpy(w.data(), krylov->buf1, n * sizeof(double), cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i <= j; i++) {
	      double coeff = w.dot(V.col(i));
	      // CGS
	      // w -= coeff * V.col(i);

	      // MGS for better behavior
	      w.noalias() -= coeff * V.col(i);
	    }

       
        //for (uint32_t i = 0; i <= j; i++) {
	//    dot_product_kernel<<<grid, block>>>(krylov->buf1, &(krylov->V[i * n]), krylov->buf2, n);    
	//    double coeff;
	//    cudaMemcpy(&coeff, krylov->buf2, sizeof(double), cudaMemcpyDeviceToHost);
        //    vector_axpby_kernel<<<grid, block>>>(krylov->buf1, &krylov->V[i*n], 1.0, -coeff, n);
        //}
	
	

        //cudaMemset(krylov->T + m * j + j + 1, 0, sizeof(double));
        
	//vector_norm_kernel<<<grid, block>>>(krylov->buf1, krylov->T + m * j + j + 1, n);
	//scalar_sqrt<<<1, 1>>>(krylov->T + m * j + j + 1);
	//cudaMemcpy(krylov->T + m * (j+1) + j, krylov->T + m * j + j + 1,  sizeof(double), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(&T(j+1,j), krylov->T + m * j + j + 1,  sizeof(double), cudaMemcpyDeviceToHost);
	
	//printf("w norm (host): %f\n", w.norm()); 
	//printf("w norm (dev):  %f\n", T(j+1,j)); 
	T(j + 1, j) = w.norm();
    	T(j, j + 1) = T(j + 1, j);

	
	//vector_scale_kernel<<<grid, block>>>(krylov->buf1, T(j + 1, j), n);
	//cudaMemcpy(&krylov->V[(j+1) * n], krylov->buf1, n * sizeof(double), cudaMemcpyDeviceToHost);
                                   
        V.col(j + 1) = w / T(j + 1, j);
    }


    cudaMemcpy(krylov->V, V.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(krylov->T, T.data(), m * m * sizeof(double), cudaMemcpyHostToDevice);  
}

#endif
