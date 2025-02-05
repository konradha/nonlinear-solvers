#include "../laplacians.hpp"
#include "spmv.hpp"

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void test_spmv(const Eigen::SparseMatrix<double> &A, uint32_t num_trials = 10) {
  const uint32_t n = A.rows();
  std::cout << "n = " << n << "\n";
  const int *row_ptr = A.outerIndexPtr();
  const int *col_ind = A.innerIndexPtr();
  const double *values = A.valuePtr();

  int *d_row_ptr, *d_col_ind;
  double *d_values, *d_x, *d_y;

  cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
  cudaMalloc(&d_col_ind, A.nonZeros() * sizeof(int));
  cudaMalloc(&d_values, A.nonZeros() * sizeof(double));
  cudaMalloc(&d_x, n * sizeof(double));
  cudaMalloc(&d_y, n * sizeof(double));

  cudaMemcpy(d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, col_ind, A.nonZeros() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, A.nonZeros() * sizeof(double),
             cudaMemcpyHostToDevice);

  DeviceSpMV spmv(d_row_ptr, d_col_ind, d_values, n, A.nonZeros());

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  double max_diff = 0.0;
  double avg_eigen_time = 0.0;
  float avg_gpu_time = 0.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (uint32_t trial = 0; trial < num_trials; trial++) {
    Eigen::VectorXd x = Eigen::VectorXd::NullaryExpr(
        n, [&gen, &dist](Eigen::Index) { return dist(gen); });

    auto cpu_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd y_eigen = A * x;
    auto cpu_end = std::chrono::high_resolution_clock::now();
    avg_eigen_time += std::chrono::duration_cast<std::chrono::microseconds>(
                          cpu_end - cpu_start)
                          .count();

    cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    spmv.multiply(d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_gpu_time += milliseconds;

    std::vector<double> y_gpu_vec(n);
    cudaMemcpy(y_gpu_vec.data(), d_y, n * sizeof(double),
               cudaMemcpyDeviceToHost);

    Eigen::Map<Eigen::VectorXd> y_gpu(y_gpu_vec.data(), n);
    Eigen::VectorXd diff = y_eigen - y_gpu;

    if (trial == 0) {
      std::cout << "L1 diff: " << diff.template lpNorm<1>() << "\n";
      std::cout << "L2 diff: " << diff.template lpNorm<2>() << "\n";
    }
  }

  avg_eigen_time /= num_trials;
  avg_gpu_time /= num_trials;

  std::cout << std::scientific << std::setprecision(4);

  std::cout << "Average Eigen time: " << avg_eigen_time << " us\n";
  std::cout << "Average GPU time:   " << avg_gpu_time * 1000.0 << " us\n";
  std::cout << "Speedup: " << avg_eigen_time / (avg_gpu_time * 1000.0) << "x\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);
}

int main(int argc, char **argv) {
  auto ns = {128, 256, 512, 1024};
  for (auto ni : ns) {
    const uint32_t nx = ni;
    const uint32_t ny = ni;
    uint32_t n = nx * ny;
    double Lx = 5., Ly = 5.;
    double dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
    Eigen::SparseMatrix<double> A =
        build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);
    test_spmv(A);
  }
  return 0;
}
