#ifndef SG_SINGLE_SOLVER_HPP
#define SG_SINGLE_SOLVER_HPP

#include "eigen_krylov_real.hpp"
#include "laplacians.hpp"

/*
 Here we implement time stepping for the sine-Gordon equation (simple):
      u_tt + c(x, y) * (u_xx + u_yy) + m(x, y) * sin(u) = 0


 Stormer-Verlet as a geometric integrator is very good. We only give the
 Gautschi-type integrator for comparison.

   Stormer-Verlet:

      u_{n+1} = 2 * u_{n} - u_{n-1} + .5 * tau² * (c(x,y) * (u_xx + u_yy) + m(x,
 y) * sin(u))

   Gautschi:
   We start from a system of ODEs: y_tt + Au = g(u) where A is a linear operator
 and g is a constant heterogeneity.

      u_{n+1} = 2 * cos (tau \Omega) * u_{n} - u_{n-1} + tau² * sinc²(tau/2
 \Omega) * \phi(g(u_{n}))


   Write \Omega² = L, ie. we face the serious problem of having to compute the
 square root of a sparse matrix and then compute a trigonometric matrix
 function. Fortunately, we can rely on a standard Lanczos iteration for this. Of
 course, this can be accelerated (for instance, Al-Mohy and Higham made good
 progress giving us access to iterations that allow for faster computation for
 these trig functions. For now (CPU), we rely on standard computation to make
   the method more readable.

   \phi is a filter function, usually tau * \Omega for our case.

 */

namespace SGESolver {

template <typename Scalar_t>
void step_sv(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
             Eigen::VectorX<Scalar_t> &buf,
             const Eigen::SparseMatrix<Scalar_t> &L,
             const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau) {
  Eigen::VectorX<Scalar_t> buf2 =
      u.unaryExpr([&](Scalar_t x) { return std::sin(x); });
  buf = L * u + m.cwiseProduct(buf2);

  buf2 = u;
  u = 2 * u - u_past + tau * tau * buf;
  u_past = buf2;
}

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau) {
  // u_tt + Au = g(u)
  // u_{n+1} =
  // 2 cos (tau \Omega) u_{n} - u_{n-1} + tau² sinc²(tau / 2 \Omega)
  // x g(\phi(tau \Omega)u_{n}))

  Eigen::VectorX<Scalar_t> buf2 = id_sqrt_multiply(L, u, tau);
  buf2 =
      m.cwiseProduct(buf2.unaryExpr([](Scalar_t x) { return -std::sin(x); }));
  buf2 = sinc2_sqrt_half(L, buf2, tau);
  Eigen::VectorX<Scalar_t> u_cpy = u;
  u = 2 * cos_sqrt_multiply(L, u, tau) - u_past + tau * tau * buf2;
  u_past = u_cpy;
}

}; // namespace SGESolver
#endif
