#ifndef EIGEN_KRYLOV_REAL_HPP
#define EIGEN_KRYLOV_REAL_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

template <typename Float>
std::tuple<Eigen::MatrixX<Float>, Eigen::MatrixX<Float>, Float>
lanczos_L(const Eigen::SparseMatrix<Float> &L, const Eigen::VectorX<Float> &u,
          const uint32_t m) {
  const uint32_t n = L.rows();
  Eigen::MatrixX<Float> V = Eigen::MatrixX<Float>::Zero(n, m);
  Eigen::MatrixX<Float> T = Eigen::MatrixX<Float>::Zero(m, m);

  Float beta = u.norm();
  V.col(0) = u / beta;

  // std::cout << "Host Lanczos beta: " << beta << "\n";
  V.col(0) = u / beta;
  // std::cout << "Host V first col norm: " << V.col(0).norm() << "\n";

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
    // Lanczos breakdown check
    // if (std::abs(T(j + 1, j)) < 1e-8) {
    //   V.conservativeResize(Eigen::NoChange, j + 1);
    //   T.conservativeResize(j + 1, j + 1);
    //   break;
    // }
    V.col(j + 1) = w / T(j + 1, j);
  }
  // std::cout << "Host T diagonal: ";
  // for(int i = 0; i < m; ++i) std::cout << T(i,i) << " ";
  // std::cout << "\n";

  return {V, T, beta};
}

template <typename Float>
Eigen::VectorX<Float> cos_sqrt_multiply(const Eigen::SparseMatrix<Float> &L,
                                        const Eigen::VectorX<Float> &u, Float t,
                                        const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);

  // std::cout << "Host (cos) V.row(0)\n";
  // std::cout << V.row(0) << "\n";
  // std::cout << "\n";

  // std::cout << "Host (cos) T\n";
  // std::cout << T << "\n";
  // std::cout << "\n";

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  // std::cout << "Host (cos) eigenvalues of T\n";
  // std::cout << es.eigenvalues() << "\n";
  // std::cout << "Host (cos) Eigenvectors of T\n";
  // std::cout << es.eigenvectors() << "\n";

  Eigen::MatrixX<Float> cos_sqrt_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs().sqrt()).cos().matrix().asDiagonal() *
       es.eigenvectors().transpose());

  // find decomposition T = Q D Q.T
  // then apply Q t * sqrt({d_i}) * Q.T
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * cos_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> sinc2_sqrt_multiply(const Eigen::SparseMatrix<Float> &L,
                                          const Eigen::VectorX<Float> &u,
                                          Float t, const uint32_t m = 10) {
  auto sinc = [](Float x) {
    return std::abs(x) < 1e-8 ? Float(1) : std::sin(x) / x;
  };
  const auto [V, T, beta] = lanczos_L(L, u, m);

  // std::cout << "Host (sinc2) V.row(0)\n";
  // std::cout << V.row(0) << "\n";
  // std::cout << "\n";

  // std::cout << "Host (sinc2) T\n";
  // std::cout << T << "\n";
  // std::cout << "\n";

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  // std::cout << "Host (sinc2) eigenvalues of T\n";
  // std::cout << es.eigenvalues() << "\n";
  // std::cout << "Host (sinc2) Eigenvectors of T\n";
  // std::cout << es.eigenvectors() << "\n";

  Eigen::MatrixX<Float> sinc2_sqrt_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs().sqrt())
           .unaryExpr(sinc)
           .square()
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

  // std::cout << "Host (id) V.row(0)\n";
  // std::cout << V.row(0) << "\n";
  // std::cout << "\n";

  // std::cout << "Host (id) T\n";
  // std::cout << T << "\n";
  // std::cout << "\n";

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  // std::cout << "Host (id) eigenvalues of T\n";
  // std::cout << es.eigenvalues() << "\n";
  // std::cout << "Host (id) Eigenvectors of T\n";
  // std::cout << es.eigenvectors() << "\n";

  Eigen::MatrixX<Float> id_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs().sqrt()).matrix().asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * id_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> sinc2_sqrt_half(const Eigen::SparseMatrix<Float> &L,
                                      const Eigen::VectorX<Float> &u, Float t,
                                      const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);

  // std::cout << "Eigen host beta\n";
  // std::cout << beta << "\n";
  // std::cout << "Eigen host eigenvectors\n";
  // std::cout << es.eigenvectors() << "\n";
  // std::cout << "Eigen host eigenvalues\n";
  // std::cout << es.eigenvalues() << "\n";
  // std::cout << "Eigen host V.row(0)\n";
  // std::cout << V.row(0) << "\n";

  Eigen::MatrixX<Float> sinc_sqrt_T =
      (es.eigenvectors() *
       (t / 2. * es.eigenvalues().array().abs().sqrt())
           .unaryExpr([](Float x) {
             return std::abs(x) < 1e-8 ? Float(1) : std::sin(x) / x;
           })
           .square()
           .matrix()
           .asDiagonal() *
       es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * sinc_sqrt_T * e1;
}

template <typename Float>
Eigen::VectorX<Float> mod_cosine_multiply(const Eigen::SparseMatrix<Float> &L,
                                          const Eigen::VectorX<Float> &u,
                                          Float t, const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  Eigen::MatrixX<Float> mod_T = (es.eigenvectors() *
                                 (t * es.eigenvalues().array().abs().sqrt())
                                     .unaryExpr([](Float x) {
                                       const auto theta = x;
                                       if (std::abs(theta) < 1e-12)
                                         return 1.0;
                                       double half_theta = theta / 2.0;
                                       return std::cos(half_theta) *
                                              std::cos(half_theta) *
                                              std::sin(theta) / theta;
                                     })
                                     .square()
                                     .matrix()
                                     .asDiagonal() *
                                 es.eigenvectors().transpose());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  e1(0) = 1.0;
  return beta * V * mod_T * e1;
}

#endif
