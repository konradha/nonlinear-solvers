#include "../eigen_krylov_real.hpp"
#include "../laplacians.hpp"

#include "lanczos.hpp"
#include "spmv.hpp"

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void test_lanczos(const Eigen::SparseMatrix<double> &A, const uint32_t m,
                  uint32_t num_trials = 10) {
  const uint32_t n = A.rows();
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "n = " << n << ", m = " << m << "\n";

  const int *row_ptr = A.outerIndexPtr();
  const int *col_ind = A.innerIndexPtr();
  const double *values = A.valuePtr();

  int *d_row_ptr, *d_col_ind;
  double *d_values;

  cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
  cudaMalloc(&d_col_ind, A.nonZeros() * sizeof(int));
  cudaMalloc(&d_values, A.nonZeros() * sizeof(double));

  cudaMemcpy(d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, col_ind, A.nonZeros() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, A.nonZeros() * sizeof(double),
             cudaMemcpyHostToDevice);

  DeviceSpMV<double> spmv(d_row_ptr, d_col_ind, d_values, n, A.nonZeros());

  KrylovInfo krylov;
  cudaMalloc((void **)&krylov.V, n * m * sizeof(double));
  cudaMalloc((void **)&krylov.T, m * m * sizeof(double));
  cudaMalloc((void **)&krylov.buf1, n * sizeof(double));
  cudaMalloc((void **)&krylov.buf2, n * sizeof(double));
  cudaMalloc((void **)&krylov.d_beta, sizeof(double));
  krylov.n = n;
  krylov.m = m;

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  double avg_eigen_time = 0.0;
  float avg_gpu_time = 0.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (uint32_t trial = 0; trial < num_trials; trial++) {
    Eigen::VectorXd u = Eigen::VectorXd::NullaryExpr(
        n, [&gen, &dist](Eigen::Index) { return dist(gen); });
    u.normalize();

    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto [V_eigen, T_eigen, beta_eigen] = lanczos_L(A, u, m);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    avg_eigen_time += std::chrono::duration_cast<std::chrono::microseconds>(
                          cpu_end - cpu_start)
                          .count();

    cudaMemcpy(krylov.buf1, u.data(), n * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    lanczos_iteration(&spmv, &krylov, krylov.buf1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_gpu_time += milliseconds;

    // To check if V and T agree, print for all trials!
    if (trial == 0) {
      std::vector<double> V_gpu(n * m);
      std::vector<double> T_gpu(m * m);
      double beta;

      cudaMemcpy(V_gpu.data(), krylov.V, n * m * sizeof(double),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(T_gpu.data(), krylov.T, m * m * sizeof(double),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&beta, krylov.d_beta, sizeof(double), cudaMemcpyDeviceToHost);

      Eigen::Map<Eigen::MatrixXd> V_gpu_map(V_gpu.data(), n, m);
      Eigen::Map<Eigen::MatrixXd> T_gpu_map(T_gpu.data(), m, m);

      // std::cout << "V contains NaN: " << V_gpu_map.hasNaN() << "\n";
      // std::cout << "T contains NaN: " << T_gpu_map.hasNaN() << "\n";

      if (V_gpu_map.hasNaN() || T_gpu_map.hasNaN()) {
        std::cout << "First column of V norm: " << V_gpu_map.col(0).norm()
                  << "\n";
        std::cout << "T diagonal elements:\n";
        for (int i = 0; i < std::min(5, (int)m); ++i) {
          std::cout << T_gpu_map(i, i) << " ";
        }
        std::cout << "\n";
      }

      Eigen::MatrixXd V_diff = V_eigen - V_gpu_map;
      Eigen::MatrixXd T_diff = T_eigen - T_gpu_map;

      std::cout << "V diff: L1 = " << V_diff.lpNorm<1>()
                << ", L2 = " << V_diff.norm() << "\n";
      std::cout << "T diff: L1 = " << T_diff.lpNorm<1>()
                << ", L2 = " << T_diff.norm() << "\n";

      std::cout << "beta Eigen: " << beta_eigen << "\n";
      std::cout << "beta dev:   " << beta << "\n";
      std::cout << "\n";
    }
  }

  avg_eigen_time /= num_trials;
  avg_gpu_time /= num_trials;

  std::cout << "Average Eigen time: " << avg_eigen_time << " us\n";
  std::cout << "Average GPU time:   " << avg_gpu_time * 1000.0 << " us\n";
  std::cout << "Speedup: " << avg_eigen_time / (avg_gpu_time * 1000.0)
            << "x\n\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);
  cudaFree(krylov.V);
  cudaFree(krylov.T);
  cudaFree(krylov.buf1);
  cudaFree(krylov.buf2);
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  auto ns = {50, 100, 200};
  std::vector<uint32_t> krylov_dims = {10, 20, 30};

  for (auto ni : ns) {
    const uint32_t nx = ni;
    const uint32_t ny = ni;
    uint32_t n = nx * ny;
    double Lx = 5., Ly = 5.;
    double dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
    Eigen::SparseMatrix<double> A =
        build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);
    // the 3d finite differences matrix is huge. long walltime needed
    // build_laplacian_noflux_3d<double>(nx - 2, ny - 2, nx - 2,  dx, dy, dx);

    for (auto m : krylov_dims) {
      test_lanczos(A, m);
    }
  }
  return 0;
}
