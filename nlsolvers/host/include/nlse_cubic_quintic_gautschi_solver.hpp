#ifndef NLSE_CUBIC_QUINTIC_GAUTSCHI_SOLVER_HPP
#define NLSE_CUBIC_QUINTIC_GAUTSCHI_SOLVER_HPP

#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"


// i u_t + \Delta u + m(\sigma_1 |u| + \sigma_2 |u|²) u = 0

namespace NLSEQuinticGautschiSolver {
template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &buf, Eigen::VectorX<Scalar_t> &rho_buf,
          Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_prev,
          const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<double> &m, const Scalar_t tau,
          const double s1, const double s2) {
  rho_buf = u;
  // B(u^n) = -m(x,y)V(|u|^2)u
  auto compute_B = [&m, &s1, &s2](const Eigen::VectorX<Scalar_t> & u) {
    auto u_abs_squared = u.real().cwiseProduct(u.real()) +
                        u.imag().cwiseProduct(u.imag());
    return -(m.cwiseProduct(s1 * u_abs_squared + s2 * u_abs_squared.cwiseProduct(u_abs_squared))).cwiseProduct(u);
  };
  buf = compute_B(u);
  // e^{-i*tau*Delta} on phi_s(tau*Delta)B(u^n)
  std::complex<double> real_tau(tau.imag(), 0.0);
  assert(std::abs(tau.real()) < 1e-12); 
  assert(std::abs(tau.imag()) > 1e-7);  

  Eigen::VectorX<Scalar_t> phi_s_B_un = sincm_multiply(L, buf, real_tau);
  Eigen::VectorX<Scalar_t> exp_phi_s_B_un = expm_multiply(L, phi_s_B_un, tau);

  // e^{-2i*tau*Delta} on u^{n-1}
  Eigen::VectorX<Scalar_t> exp_u_prev = expm_multiply(L, u_prev, 2.*tau);

  // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
  u = exp_u_prev - 2. * tau * exp_phi_s_B_un;
  u_prev = rho_buf; 
}
}; // namespace NLSEQuinticSolver
#endif
