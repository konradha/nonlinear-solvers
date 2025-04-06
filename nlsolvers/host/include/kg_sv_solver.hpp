#ifndef KG_SV_SOLVER_HPP
#define KG_SV_SOLVER_HPP

#include "eigen_krylov_real.hpp"
#include "laplacians.hpp"

/*
      u_tt - (u_xx + u_yy) + m(x, y) * u = 0
 */
namespace KGESVSolver {

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau) {
  Eigen::VectorX<Scalar_t> buf2 =
      u.unaryExpr([&](Scalar_t x) { return x; });
  buf = L * u - m.cwiseProduct(buf2);
  buf2 = u;
  u = 2 * u - u_past + tau * tau * buf;
  u_past = buf2;

}

}; // namespace KGESVSolver
#endif
