#ifndef SG_SINGLE_SOLVER_HPP
#define SG_SINGLE_SOLVER_HPP

#include "eigen_krylov_real.hpp"
#include "laplacians.hpp"


namespace SGESolverSV {

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
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
}; // namespace SGESolverSV
#endif
