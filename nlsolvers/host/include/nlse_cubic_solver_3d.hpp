#ifndef NLSE_CUBIC_SOLVER_HPP
#define NLSE_CUBIC_SOLVER_HPP

#include "eigen_krylov_complex.hpp"

namespace NLSESolver3d {
// scalar type is either
// std::complex<float>, std::complex<double>
template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &buf, Eigen::VectorX<Scalar_t> &rho_buf,
          Eigen::VectorX<Scalar_t> &u, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau)
// tau = 1j * dt; important to propagate this in complex time --
// else we're computing an expensive equilibrium
{

  rho_buf =
      (m.cwiseProduct((u.real().cwiseProduct(u.real())) +
                      (u.imag()).cwiseProduct(u.imag())))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(.5 * tau * x); })
          .cwiseProduct(u);

  buf = expm_multiply(L, rho_buf, tau);
  u = (m.cwiseProduct((buf.real().cwiseProduct(buf.real())) +
                      (buf.imag()).cwiseProduct(buf.imag())))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(.5 * tau * x); })
          .cwiseProduct(buf);
}
}; // namespace NLSESolver3d
#endif
