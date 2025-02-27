#ifndef NLSE_CUBIC_SOLVER_HPP
#define NLSE_CUBIC_SOLVER_HPP

#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"

/*
 * This file implements the famous SS2 timestepping on a square domain
 * discretized with finite differences.
 * Assume the NLSE in the form
 *
 *        i u_t + (u_xx + u_yy) + |u|²u = 0
 *
 * which we can of course extend into de/focussing:
 *
 *        i u_t + \bar{\Delta} u + m(x,y)|u|²u = 0
 *
 * Assume    H = (d/dx)² + (d/dy)² + |u|²
 *             = L + N(u)
 *             = H_1(u) + H_2(u)
 *   where
 *           L = H_1
 *        N(u) = H_2(u)
 *
 * Then write the Operator Splitting as
 * u(t + dt) = exp(-i dt/2 H_2(t)) exp(-i dt H_1(t)) exp(-i dt/2 H_2(t)) u(t) +
 * O((dt)³ [L, N])
 *
 * where we can express the operations for the nonlinear part of
 * the equation as point-wise scalar multiplication.
 * The only expensive part is evaluating the matrix function
 *
 *             exp(i dt L) u
 *
 * for which we can apply a Krylov subspace method. This implies m SpMV
 * which we have to perform to approximate this operator evaluation.
 * There are methods to lower the number of m, or to split
 *
 *             exp(i dt L) u = cos(dt L) u + i sin(dt L) u
 *
 * as elaborated by Al-Mohy and Higham. We however restrict ourselves first
 * and foremost in implementing a very basic Krylov subspace iteration to check
 * correctness in terms of CPU and GPU time stepping.
 *
 * This integrator is structure-preserving, allowing for nice properties
 * wrt. e.g. energy conservation. Further implementing a CUDA-enabled integrator
 * in this manner will accelerate solving trajectories of the NLSE quite a bit.
 * cuSPARSE allows for _very_ rapid SpMV evals.
 */

namespace NLSESolver {
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
          .unaryExpr([&tau](Scalar_t x) { return std::exp(-.5 * tau * x); })
          .cwiseProduct(u);

  buf = expm_multiply(L, rho_buf, -tau);

  u = (m.cwiseProduct((buf.real().cwiseProduct(buf.real())) +
                      (buf.imag()).cwiseProduct(buf.imag())))
          .unaryExpr([&tau](Scalar_t x) { return std::exp(-.5 * tau * x); })
          .cwiseProduct(buf);
}
}; // namespace NLSESolver
#endif
