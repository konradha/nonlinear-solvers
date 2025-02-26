#ifndef EIGEN_KRYLOV_COMPLEX_HPP
#define EIGEN_KRYLOV_COMPLEX_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#define DEBUG 1

#if DEBUG
#include <iostream>
#endif

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
    T(j, j) = V.col(j).adjoint() * w;
    w -= T(j, j) * V.col(j);

    // This could be matmuls. See cuda implementation.
    for (uint32_t i = 0; i <= j; i++) {
      Float coeff = V.col(i).adjoint() * w;
      w.noalias() -= coeff * V.col(i);
    }
    T(j + 1, j) = w.norm();
    T(j, j + 1) = T(j + 1, j);
    V.col(j + 1) = w / T(j + 1, j);
  }
  //std::cout << "Eigen beta: " << beta << "\n";
  //std::cout << "Eigen T: " << T << "\n";
  //std::cout << "Eigen V: " << V << "\n";
  return {V, T, beta};
}

template <typename Float>
Eigen::VectorX<Float> expm_multiply(const Eigen::SparseMatrix<Float> &L,
                                    const Eigen::VectorX<Float> &u, Float t,
                                    const uint32_t m = 10) {
  const auto [V, T, beta] = lanczos_L(L, u, m);
  //std::cout << "Host beta\n" << beta << "\n";
  //std::cout << "Host T\n" << T << "\n";
  //std::cout << "Host V\n" << V << "\n";
  //std::cout << "Host dt: " << t << "\n";
//#if DEBUG
//  std::cout << "V: " << V << "\n";
//  std::cout << "T: " << T << "\n";
//  std::cin.get();
//#endif
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Float>> es(T);
  //std::cout << "Host eigenvalues\n" << es.eigenvalues() << "\n";
  Eigen::MatrixX<Float> exp_T =
      (es.eigenvectors() *
       (t * es.eigenvalues().array().abs())
           .unaryExpr([](Float x) { return std::exp(x); })
           .matrix()
           .asDiagonal() *
       es.eigenvectors().adjoint());
  Eigen::VectorX<Float> e1 = Eigen::VectorX<Float>::Zero(T.rows());
  //printf("Host beta=%f\n", beta);
  //std::cout << "Host eigenvalues=" << es.eigenvalues() << "\n";
  //std::cout << "Host eigenvectors=" << es.eigenvectors() << "\n";
  e1(0) = 1.0;
  return beta * V * exp_T * e1;
}

#endif
