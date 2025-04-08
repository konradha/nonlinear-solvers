#ifndef NLSE_CUBIC_GAUTSCHI_SOLVER_HPP
#define NLSE_CUBIC_GAUTSCHI_SOLVER_HPP

#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"

#include <cassert>

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
  // B(u^n) = -m(x,y)|u|^2u
  auto compute_B = [&m](const Eigen::VectorX<Scalar_t> & u) {
      auto u_abs_squared = u.real().cwiseProduct(u.real()) +
                          u.imag().cwiseProduct(u.imag());
      return -1.0 * (m.cwiseProduct(u_abs_squared)).cwiseProduct(u);
  };
  buf = compute_B(u);
  
  std::complex<double> real_tau(tau.imag(), 0.0);
  Eigen::VectorX<Scalar_t> phi_s_B_un = sincm_multiply(L, buf, real_tau);
  Eigen::VectorX<Scalar_t> exp_phi_s_B_un = expm_multiply(L, phi_s_B_un, -tau);
  Eigen::VectorX<Scalar_t> exp_u_prev = expm_multiply(L, u_prev, -2.*tau); 
  u = exp_u_prev + 2. * tau * exp_phi_s_B_un;
  u_prev = rho_buf;
  /* V2
   *
  rho_buf = u;
  // B(u^n) = -m(x,y)|u|^2u
  auto compute_B = [&m](const Eigen::VectorX<Scalar_t> & u) {
      auto u_abs_squared = u.real().cwiseProduct(u.real()) +
                          u.imag().cwiseProduct(u.imag());
      return -1.0 * (m.cwiseProduct(u_abs_squared)).cwiseProduct(u);
  };
  buf = compute_B(u);
  
  std::complex<double> real_tau(tau.imag(), 0.0);
  assert(std::abs(tau.real()) < 1e-12); // real part should be close to 0
  assert(std::abs(tau.imag()) > 1e-7);  // imag part should carry our time step
  
  // phi_s(dt*Delta)B(u^n)
  Eigen::VectorX<Scalar_t> phi_s_B_un = sincm_multiply(L, buf, real_tau);
  
  // e^{-i*dt*Delta}phi_s(dt*Delta)B(u^n)
  Eigen::VectorX<Scalar_t> exp_phi_s_B_un = expm_multiply(L, phi_s_B_un, -tau);
  
  // e^{-2i*dt*Delta}u^{n-1}
  Eigen::VectorX<Scalar_t> exp_u_prev = expm_multiply(L, u_prev, -2.*tau);
  
  // u^{n+1} = e^{-2i*dt*Delta}u^{n-1} + 2*dt*e^{-i*dt*Delta}*phi_s(dt*Delta)*B(u^n)
  u = exp_u_prev + 2. * tau * exp_phi_s_B_un;
  u_prev = rho_buf;
  */  



  /*
  // B(u^n) = -m(x,y)V(|u|^2)u
  auto compute_B = [&m](const Eigen::VectorX<Scalar_t> & u) {
    auto u_abs_squared = u.real().cwiseProduct(u.real()) + 
                        u.imag().cwiseProduct(u.imag());
    return -(m.cwiseProduct(u_abs_squared)).cwiseProduct(u); // sign unclear from paper
  };
  buf = compute_B(u);
  // e^{-i*tau*Delta} on phi_s(tau*Delta)B(u^n)
  std::complex<double> real_tau(tau.imag(), 0.0);
  assert(std::abs(tau.real()) < 1e-12); // real part should be close to 0
  assert(std::abs(tau.imag()) > 1e-7);  // imag part should carry our time step
  
  Eigen::VectorX<Scalar_t> phi_s_B_un = sincm_multiply(L, buf, real_tau);
  Eigen::VectorX<Scalar_t> exp_phi_s_B_un = expm_multiply(L, phi_s_B_un, tau);

  // e^{-2i*tau*Delta} on u^{n-1}
  Eigen::VectorX<Scalar_t> exp_u_prev = expm_multiply(L, u_prev, 2.*tau);

  // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
  u = exp_u_prev - 2. * tau * exp_phi_s_B_un;
  u_prev = rho_buf;
  */
}

// (exp(tau*L)*v - v)/(tau*L)
// for solid initialization
template <typename Scalar_t>
Eigen::VectorX<Scalar_t> phi1m_multiply(const Eigen::SparseLU<Eigen::SparseMatrix<Scalar_t>>& solver,
                                        const Eigen::SparseMatrix<Scalar_t> &L,
                                        const Eigen::VectorX<Scalar_t> &v,
                                        const Scalar_t tau) {
    Eigen::VectorX<Scalar_t> exp_v = expm_multiply(L, v, tau);
    Eigen::VectorX<Scalar_t> diff = exp_v - v;
    return solver.solve(diff);
}
}; // namespace NLSECubicGautschiSolver
#endif
