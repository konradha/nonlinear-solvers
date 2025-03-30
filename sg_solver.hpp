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
             const Eigen::VectorX<Scalar_t> &c,
             const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau) {
  Eigen::VectorX<Scalar_t> sin_u = u.unaryExpr([](Scalar_t x) { return std::sin(x); });
  buf = L * u - m.cwiseProduct(sin_u);
  Eigen::VectorX<Scalar_t> u_cpy = u;
  u = 2 * u - u_past + tau * tau * buf;
  u_past = u_cpy;
}

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &c, const Eigen::VectorX<Scalar_t> &m,
          const Scalar_t tau) {
  uint32_t subspace_dim = 10;
  // u_tt + Au = g(u)
  // u_{n+1} =
  // 2 cos (tau \Omega) u_{n} - u_{n-1} + tau² sinc²(tau / 2 \Omega)
  // x g(\phi(tau \Omega)u_{n}))

  // filter
  Eigen::VectorX<Scalar_t> filtered_u = id_sqrt_multiply(L, u, tau, subspace_dim);
  Eigen::VectorX<Scalar_t> sin_filtered_u = filtered_u.unaryExpr([](Scalar_t x) { return -std::sin(x); });
  Eigen::VectorX<Scalar_t> m_sin_filtered_u = m.cwiseProduct(sin_filtered_u); 
  Eigen::VectorX<Scalar_t> sinc2_term = sinc2_sqrt_half(L, m_sin_filtered_u, tau, subspace_dim);
  Eigen::VectorX<Scalar_t> cos_term = cos_sqrt_multiply(L, u, tau, subspace_dim);
  Eigen::VectorX<Scalar_t> u_cpy = u;
  u = 2 * cos_term - u_past - tau * tau * sinc2_term;
  u_past = u_cpy;
}

}; // namespace SGESolver
