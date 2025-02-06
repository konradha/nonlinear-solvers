#include "../eigen_krylov_complex.hpp"
#include "../laplacians.hpp"

#include "lanczos_complex.hpp"
#include "spmv.hpp"

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void test_lanczos_complex(const Eigen::SparseMatrix<std::complex<double>> &A, 
                         const uint32_t m, uint32_t num_trials = 10) {
  const uint32_t n = A.rows();
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "n = " << n << ", m = " << m << "\n";

  const int *row_ptr = A.outerIndexPtr();
  const int *col_ind = A.innerIndexPtr();
  const std::complex<double> *values = A.valuePtr();

  int *d_row_ptr, *d_col_ind;
  thrust::complex<double> *d_values;

  cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
  cudaMalloc(&d_col_ind, A.nonZeros() * sizeof(int));
  cudaMalloc(&d_values, A.nonZeros() * sizeof(thrust::complex<double>));

  cudaMemcpy(d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, col_ind, A.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, A.nonZeros() * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  DeviceSpMV<thrust::complex<double>> spmv(d_row_ptr, d_col_ind, d_values, n, A.nonZeros());

  KrylovInfoComplex krylov;
  cudaMalloc((void **)&krylov.V, n * m * sizeof(thrust::complex<double>));
  cudaMalloc((void **)&krylov.T, m * m * sizeof(thrust::complex<double>));
  cudaMalloc((void **)&krylov.buf1, n * sizeof(thrust::complex<double>));
  cudaMalloc((void **)&krylov.buf2, n * sizeof(thrust::complex<double>));
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
    Eigen::VectorX<std::complex<double>> u(n);
    for (uint32_t i = 0; i < n; i++) {
      u[i] = std::complex<double>(dist(gen), dist(gen));
    }
    u.normalize();

    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto [V_eigen, T_eigen, beta_eigen] = lanczos_L(A, u, m);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    avg_eigen_time += std::chrono::duration_cast<std::chrono::microseconds>(
                          cpu_end - cpu_start).count();

    cudaMemcpy(krylov.buf1, u.data(), n * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    lanczos_iteration_complex(A, &spmv, &krylov, krylov.buf1, u);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_gpu_time += milliseconds;

    /*if (trial == 0)*/ {
      std::vector<std::complex<double>> V_gpu(n * m);
      std::vector<std::complex<double>> T_gpu(m * m);

      cudaMemcpy(V_gpu.data(), krylov.V, n * m * sizeof(thrust::complex<double>),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(T_gpu.data(), krylov.T, m * m * sizeof(thrust::complex<double>),
                 cudaMemcpyDeviceToHost);

      Eigen::Map<Eigen::MatrixX<std::complex<double>>> V_gpu_map(V_gpu.data(), n, m);
      Eigen::Map<Eigen::MatrixX<std::complex<double>>> T_gpu_map(T_gpu.data(), m, m);

      if (V_gpu_map.hasNaN() || T_gpu_map.hasNaN()) {
        std::cout << "First column of V norm: " << V_gpu_map.col(0).norm() << "\n";
        std::cout << "T diagonal elements:\n";
        for (int i = 0; i < std::min(5, (int)m); ++i) {
          std::cout << T_gpu_map(i, i) << " ";
        }
        std::cout << "\n";
      }

      Eigen::MatrixX<std::complex<double>> V_diff = V_eigen - V_gpu_map;
      Eigen::MatrixX<std::complex<double>> T_diff = T_eigen - T_gpu_map;

      std::cout << "V diff: L1 = " << V_diff.lpNorm<1>()
                << ", L2 = " << V_diff.norm() << "\n";
      std::cout << "T diff: L1 = " << T_diff.lpNorm<1>()
                << ", L2 = " << T_diff.norm() << "\n";
      std::cout << "\n";
    }
  }

  avg_eigen_time /= num_trials;
  avg_gpu_time /= num_trials;

  std::cout << "Average Eigen time: " << avg_eigen_time << " us\n";
  std::cout << "Average GPU time:   " << avg_gpu_time * 1000.0 << " us\n";
  std::cout << "Speedup: " << avg_eigen_time / (avg_gpu_time * 1000.0) << "x\n\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);
  cudaFree(krylov.V);
  cudaFree(krylov.T);
  cudaFree(krylov.buf1);
  cudaFree(krylov.buf2);
  cudaFree(krylov.d_beta);
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
    Eigen::SparseMatrix<std::complex<double>> A =
        build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);
    
    // Another nice test case
    //for (int k = 0; k < A.nonZeros(); k++) {
    //  A.valuePtr()[k] *= std::complex<double>(1.0, 1.0);
    //}

    for (auto m : krylov_dims) {
      test_lanczos_complex(A, m);
    }
  }
  return 0;
}
