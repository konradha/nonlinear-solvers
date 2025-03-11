#ifndef BOUNDARIES_3D_HPP
#define BOUNDARIES_3D_HPP

#include <Eigen/Dense>
 
// naive port to 3d
template <typename Float>
void neumann_bc_no_velocity_3d(Eigen::VectorX<Float> & u,
                const uint32_t nx, 
                const uint32_t ny,
                const uint32_t nz) {
  const uint32_t slice_size = nx * ny;
  for (uint32_t k = 1; k < nz-1; ++k) {
      for (uint32_t j = 1; j < ny-1; ++j) {
          u[j*nx + k*slice_size] = u[j*nx + 1 + k*slice_size];
          u[j*nx + (nx-1) + k*slice_size] = u[j*nx + (nx-2) + k*slice_size];
      }
  }
  for (uint32_t k = 1; k < nz-1; ++k) {
      for (uint32_t i = 0; i < nx; ++i) {
          u[i + k*slice_size] = u[i + nx + k*slice_size];
          u[i + (ny-1)*nx + k*slice_size] = u[i + (ny-2)*nx + k*slice_size];
      }
  }
  for (uint32_t j = 0; j < ny; ++j) {
      for (uint32_t i = 0; i < nx; ++i) {
          u[i + j*nx] = u[i + j*nx + slice_size];
          u[i + j*nx + (nz-1)*slice_size] = u[i + j*nx + (nz-2)*slice_size];
      }
  }
}
#endif
