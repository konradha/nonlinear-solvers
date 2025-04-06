#ifndef SG_DOUBLE_SV_SOLVER_HPP
#define SG_DOUBLE_SV_SOLVER_HPP

#include "eigen_krylov_real.hpp"
#include "laplacians.hpp"

namespace SGEDoubleSVSolver {

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau) {
  Eigen::VectorX<Scalar_t> buf2 = L * u -  m.cwiseProduct(buf2.unaryExpr(
      [](Scalar_t x) { return (std::sin(x) + .6 * std::sin(2. * x)); })); 
  Eigen::VectorX<Scalar_t> u_cpy = u;
  u = 2 * u - u_past + tau * tau * buf2;
  u_past = u_cpy;
}

}; // namespace SGEDoubleSVSolver
#endif
