#ifndef SG_SINGLE_CUH
#define SG_SINGLE_CUH

#include "matfunc_real.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>

namespace device {

namespace SGESolver {

__global__ void neg_sin_kernel(double *out, const double *in,
                               const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = -sin(in[idx]);
  }
}

__global__ void gautschi_kernel(double *u_next, const double *u,
                                const double *u_past, const double *costu,
                                const double *filtered_sinc, const double tau,
                                const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    u_next[idx] =
        2.0 * costu[idx] - u_past[idx] + tau * tau * filtered_sinc[idx];
  }
}

__global__ void stormer_verlet_kernel(double *u_next, const double *u,
                                      const double *u_past,
                                      const double *lapl_sin_term,
                                      const double tau, const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    u_next[idx] = 2.0 * u[idx] - u_past[idx] + tau * tau * lapl_sin_term[idx];
  }
}

__global__ void sin_term_kernel(double *out, const double *u, const double *m,
                                const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = m[idx] * sin(u[idx]);
  }
}

__global__ void compute_lapl_sin_kernel(double *out, const double *Lu,
                                        const double *sin_term,
                                        const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = Lu[idx] + sin_term[idx];
  }
}

void step_sv(double *d_u, double *d_u_past, double *d_buf, double *d_buf2,
             DeviceSpMV<double> *spmv, const double *d_m, const double tau,
             const uint32_t n, const dim3 grid, const dim3 block) {
  sin_term_kernel<<<grid, block>>>(d_buf2, d_u, d_m, n);
  // TODO: This apparently is broken here. Needs fixing to test if this approach works as well in CUDA!
  //spmv->multiply(d_buf, d_u);
  compute_lapl_sin_kernel<<<grid, block>>>(d_buf, d_buf, d_buf2, n);
  double *d_u_temp;
  cudaMalloc(&d_u_temp, n * sizeof(double));
  cudaMemcpy(d_u_temp, d_u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  stormer_verlet_kernel<<<grid, block>>>(d_u, d_u, d_u_past, d_buf, tau, n);
  cudaMemcpy(d_u_past, d_u_temp, n * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaFree(d_u_temp);
}

void step(double *d_u, double *d_u_past, double *d_buf, double *d_buf2,
          double *d_buf3, MatrixFunctionApplicatorReal *matfunc,
          const double *d_m, const double tau, const uint32_t n,
          const dim3 grid, const dim3 block) {
  cudaMemcpy(d_buf, d_u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  matfunc->apply(d_buf2, d_u, tau,
                 MatrixFunctionApplicatorReal::FunctionType::ID_SQRT);
  neg_sin_kernel<<<grid, block>>>(d_buf2, d_buf2, n);
  matfunc->apply(d_buf3, d_buf2, tau,
                 MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
  matfunc->apply(d_buf2, d_u, tau,
                 MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);
  gautschi_kernel<<<grid, block>>>(d_u, d_u, d_u_past, d_buf2, d_buf3, tau, n);
  cudaMemcpy(d_u_past, d_buf, n * sizeof(double), cudaMemcpyDeviceToDevice);
}

} // namespace SGESolver

} // namespace device

#endif // SG_SINGLE_CUH
