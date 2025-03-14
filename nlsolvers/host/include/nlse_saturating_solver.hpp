#ifndef NLSE_SATURATING_SOLVER_HPP
#define NLSE_SATURATING_SOLVER_HPP

#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"

// i u_t + (u_xx + u_yy) + m (|u|²/ (1 + \kappa |u|²)u

namespace NLSESaturatingSolver {
template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &buf, Eigen::VectorX<Scalar_t> &rho_buf,
          Eigen::VectorX<Scalar_t> &u, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<double> &m, const Scalar_t tau,
          const Scalar_t kappa) {
  Eigen::VectorX<Scalar_t> ones_buf = Eigen::VectorX<Scalar_t>::Ones(L.rows());
  ones_buf = (ones_buf + kappa * u).cwiseInverse();

  rho_buf =
      (u.real().cwiseProduct(u.real())) + (u.imag()).cwiseProduct(u.imag());
  rho_buf =
      m.cwiseProduct(rho_buf.cwiseProduct(ones_buf))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(-.5 * tau * x); })
          .cwiseProduct(u);

  buf = expm_multiply(L, rho_buf, -tau);
  ones_buf =
      (Eigen::VectorX<Scalar_t>::Ones(L.rows()) + kappa * buf).cwiseInverse();
  rho_buf =
      (u.real().cwiseProduct(u.real())) + (u.imag()).cwiseProduct(u.imag());
  u = m.cwiseProduct(rho_buf.cwiseProduct(ones_buf))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(-.5 * tau * x); })
          .cwiseProduct(buf);
}
}; // namespace NLSESaturatingSolver
#endif
