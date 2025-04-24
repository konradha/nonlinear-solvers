#ifndef KG_SV_SOLVER_3D_HPP
#define KG_SV_SOLVER_3D_HPP

#include "eigen_krylov_real.hpp"

/*
      u_tt - div(c(x,y,z) grad(u)) + m(x, y, z) * u = 0
      - anisotropy field c implicit through operator L
 */
namespace KGESVSolver3d {

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &m, const Scalar_t tau) {
  Eigen::VectorX<Scalar_t> buf2 = u.unaryExpr([&](Scalar_t x) { return x; });
  buf = L * u - m.cwiseProduct(buf2);
  buf2 = u;
  u = 2 * u - u_past + tau * tau * buf;
  u_past = buf2;
}

}; // namespace KGESVSolver3d
#endif
