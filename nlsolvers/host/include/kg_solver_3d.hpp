#ifndef KG_SOLVER_3D_HPP
#define KG_SOLVER_3D_HPP

#include "eigen_krylov_real.hpp"
/*
      u_tt - div(c(x,y,z) grad(u)) + m(x, y, z) * u = 0
      - anisotropy field implicit in operator L
 */
namespace KGESolver3d {

template <typename Scalar_t>
void step(Eigen::VectorX<Scalar_t> &u, Eigen::VectorX<Scalar_t> &u_past,
          Eigen::VectorX<Scalar_t> &buf, const Eigen::SparseMatrix<Scalar_t> &L,
          const Eigen::VectorX<Scalar_t> &m,
          const Scalar_t tau) {
  Eigen::VectorX<Scalar_t> buf2 = id_sqrt_multiply(L, u, tau);
  buf2 = -m.cwiseProduct(buf2.unaryExpr(
      [](Scalar_t x) { return x; }));
  buf2 = sinc2_sqrt_half(L, buf2, tau);
  Eigen::VectorX<Scalar_t> u_cpy = u;
  u = 2 * cos_sqrt_multiply(L, u, tau) - u_past + tau * tau * buf2;
  u_past = u_cpy;
}

}; // namespace KGESolver3d
#endif
