#include <Eigen/Dense>
#include <Eigen/Sparse>

template <typename Float>
std::tuple<Eigen::MatrixX<Float>, Eigen::MatrixX<Float>, Float>
lanczos_L(const Eigen::SparseMatrix<Float> &L, const Eigen::VectorX<Float> &u,
          const uint32_t m) {
  const uint32_t n = L.rows();
  Eigen::MatrixX<Float> V = Eigen::MatrixX<Float>::Zero(n, m);
  Eigen::MatrixX<Float> T = Eigen::MatrixX<Float>::Zero(m, m);

  Float beta = u.norm();
  V.col(0) = u / beta;

  for (uint32_t j = 0; j < m - 1; j++) {
    Eigen::VectorX<Float> w = L * V.col(j);
    if (j > 0)
      w -= T(j - 1, j) * V.col(j - 1);

    T(j, j) = w.dot(V.col(j));
    w -= T(j, j) * V.col(j);

    for (uint32_t i = 0; i <= j; i++) {
      Float coeff = w.dot(V.col(i));
      // CGS
      // w -= coeff * V.col(i);

      // MGS for better behavior
      w.noalias() -= coeff * V.col(i);
    }
    T(j + 1, j) = w.norm();
    // use symmetry
    T(j, j + 1) = T(j + 1, j);
    if (std::abs(T(j + 1, j)) < 1e-8) {
      V.conservativeResize(Eigen::NoChange, j + 1);
      T.conservativeResize(j + 1, j + 1);
      break;
    }
    V.col(j + 1) = w / T(j + 1, j);
  }
  return {V, T, beta};
}

template <typename Float>
Eigen::VectorX<Float> cosm_sqrt_multiply(const Eigen::SparseMatrix<Float> &L,
                                         const Eigen::VectorX<Float> &u,
                                         Float t, const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> cos_sqrt_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs().sqrt()
           .unaryExpr([](Float x) {
    return std::cos(x); })
           .matrix()
           .asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * cos_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> sinc2_sqrt_multiply(const Eigen::SparseMatrix<Float> &L,
                                          const Eigen::VectorX<Float> &u,
                                          Float t, const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> sinc2_sqrt_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs().sqrt()
           .unaryExpr([](Float x) {
    return std::sinc(x) * std::sinc(x); })
           .matrix()
           .asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * sinc2_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> id_sqrt_multiply(const Eigen::SparseMatrix<Float> &L,
                                       const Eigen::VectorX<Float> &u, Float t,
                                       const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> id_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs().sqrt()
           .unaryExpr([](Float x) {
    return x; })
           .matrix()
           .asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * id_T * e1;
}
