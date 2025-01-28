#include "eigen_krylov_real.hpp"
#include "laplacians.hpp"
namespace BSolver {
/*
    u_tt - \Delta u + 3 (u²)_xx - u_xxxx = 0
    -> u_tt + L u = g(u)

    with   L = -\Delta u - u_xxxx
        g(u) = - 3 (u²)_xx

*/

template <typename Float_t>
void u_xx(Eigen::VectorX<Float_t> &buf, const Eigen::VectorX<Float_t> &u,
          const uint32_t nx, const uint32_t ny, const Float_t dx) {
  // Eigen assumes ColMajor!
  assert(buf.size() == u.size());
  assert(u.size() == nx * ny);
  const Float_t dx2_inv = static_cast<Float_t>(1.0) / (dx * dx);

  // left
  for (uint32_t j = 0; j < ny; ++j) {
    const uint32_t idx = j * nx;
    buf[idx] = (u[idx + 1] - u[idx]) * dx2_inv;
  }

  // right
  for (uint32_t j = 0; j < ny; ++j) {
    const uint32_t idx = j * nx + (nx - 1);
    buf[idx] = (u[idx - 1] - u[idx]) * dx2_inv;
  }

  // interior
  for (uint32_t j = 0; j < ny; ++j) {
    const uint32_t row_start = j * nx + 1;
    const uint32_t row_end = j * nx + (nx - 1);
    for (uint32_t idx = row_start; idx < row_end; ++idx) {
      buf[idx] = (u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) * dx2_inv;
    }
  }
}

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &c, const Eigen::VectorX<Scalar_t> &m,
          const Scalar_t tau, const uint32_t nx, const uint32_t ny,
          const Scalar_t dx) {

  // g (tau \Omega u)
  auto nonlin = id_sqrt_multiply(L, u, tau);
  nonlin = nonlin.cwiseProduct(nonlin);
  u_xx(buf, nonlin, nx, ny, dx);

  // sinc²(tau / 2 \Omega) -3 * (u²)_xx
  // see reasoning above for the sign
  
  buf = -3. * buf;
  const auto s2 = sinc2_sqrt_multiply(L, buf, .5 * tau);
  const auto cos = cos_sqrt_multiply(L, u, tau);

  auto u_cpy = u;
  u = 2 * cos - u_past + tau * tau * s2;
  u_past = u_cpy;
}
}; // namespace BSolver
