#ifndef NLSE_CUBIC_GAUTSCHI_SOLVER_HPP
#define NLSE_CUBIC_GAUTSCHI_SOLVER_HPP

#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"

/*
 *
 * This solver has issues: It's not an actually Gautschi-like integraor.
 * Formally, we can see that the time stepping needs to be adjusted much
 * more carefully than, say, the SS2 integrator's.
 * Hence, we'll use it for comparison purposes, not really for data generation!
 */


namespace NLSECubicGautschiSolver {
// scalar type is either
// std::complex<float>, std::complex<double>
template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &buf, Eigen::VectorX<Scalar_t> &rho_buf,
          Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_prev,
          const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<double> &m, const Scalar_t tau)
{
  rho_buf = u;
  // B(u^n) = -m(x,y)V(|u|^2)u
  auto compute_B = [&m](const Eigen::VectorX<Scalar_t> & u) {
    auto u_abs_squared = u.real().cwiseProduct(u.real()) + 
                        u.imag().cwiseProduct(u.imag());
    return (-m.cwiseProduct(u_abs_squared)).cwiseProduct(u);
  };
  buf = compute_B(u);
  // e^{-i*tau*Delta} on phi_s(tau*Delta)B(u^n)
  Eigen::VectorX<Scalar_t> phi_s_B_un = sincm_multiply(L, buf, -tau);
  Eigen::VectorX<Scalar_t> exp_phi_s_B_un = expm_multiply(L, phi_s_B_un, -tau);

  // e^{-2i*tau*Delta} on u^{n-1}
  Eigen::VectorX<Scalar_t> exp_u_prev = expm_multiply(L, u_prev, -2.*tau);

  // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
  u = exp_u_prev -2. * tau * exp_phi_s_B_un;
  u_prev = rho_buf;
}
}; // namespace NLSECubicGautschiSolver
#endif
