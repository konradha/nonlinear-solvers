#ifndef NLSE_CUBIC_QUINTIC_SOLVER_HPP
#define NLSE_CUBIC_QUINTIC_SOLVER_HPP

#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"

// i u_t + \Delta u + m(\sigma_1 |u| + \sigma_2 |u|²) u = 0

namespace NLSEQuinticSolver {
template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &buf, Eigen::VectorX<Scalar_t> &rho_buf,
          Eigen::VectorX<Scalar_t> &u, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> & m,
          const Scalar_t tau,
          const Scalar_t s1, const Scalar_t s2)
{
  rho_buf = (u.real().cwiseProduct(u.real())) + (u.imag()).cwiseProduct(u.imag());
  rho_buf = m.cwiseProduct(s1 * rho_buf + s2 * rho_buf.cwiseProduct(rho_buf))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(-.5 * tau * x); })
          .cwiseProduct(u); 

  buf = expm_multiply(L, rho_buf, -tau);
  rho_buf = (u.real().cwiseProduct(u.real())) + (u.imag()).cwiseProduct(u.imag());

  u = m.cwiseProduct(s1 * rho_buf + s2 * rho_buf.cwiseProduct(rho_buf))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(-.5 * tau * x); })
          .cwiseProduct(buf);
}
};
#endif
