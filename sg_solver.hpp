#include "laplacians.hpp"
//#include "eigen_krylov.hpp" TODO: implement the methods needed (simple float
//krylov iteration)

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

// TODO template step type; implement Gautschi step of course
template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &c, const Eigen::VectorX<Scalar_t> &m,
          const Scalar_t tau) {
  buf = c.cwiseProduct(L * u) +
        m.cwiseProduct(u.unaryExpr([&](Scalar_t x) { return std::sin(x); }));
  const auto buf2 = u;
  u = 2 * u - u_past + .5 * tau * tau * buf;
  u_past = buf2;
}
}; // namespace SGESolver
