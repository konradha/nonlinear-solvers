#ifndef NLSE_CUH
#define NLSE_CUH

#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace device {

__global__ void density(thrust::complex<double> *rho,
                        const thrust::complex<double> *u, const double *m,
                        const uint32_t nx, const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = u[idx];
    rho[idx] = m[idx] * (val.real() * val.real() + val.imag() * val.imag());
  }
}

__global__ void nonlin_part(thrust::complex<double> *out,
                            const thrust::complex<double> *in,
                            const thrust::complex<double> *rho,
                            const thrust::complex<double> tau,
                            const uint32_t nx, const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = rho[idx];
    out[idx] = in[idx] * exp(tau * val);
  }
}

} // namespace device

#endif
