#include "../laplacians.hpp"
#include "spmv.hpp"
#include <thrust/complex.h>

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
void test_spmv(const Eigen::SparseMatrix<T> &A, uint32_t num_trials = 10) {
  using device_type = std::conditional_t<std::is_same_v<T, double>, double,
                                         thrust::complex<double>>;

  const uint32_t n = A.rows();
  std::cout << "n = " << n << "\n";
  const int *row_ptr = A.outerIndexPtr();
  const int *col_ind = A.innerIndexPtr();
  const T *values = A.valuePtr();

  int *d_row_ptr, *d_col_ind;
  device_type *d_values, *d_x, *d_y;

  cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
  cudaMalloc(&d_col_ind, A.nonZeros() * sizeof(int));
  cudaMalloc(&d_values, A.nonZeros() * sizeof(device_type));
  cudaMalloc(&d_x, n * sizeof(device_type));
  cudaMalloc(&d_y, n * sizeof(device_type));

  cudaMemcpy(d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, col_ind, A.nonZeros() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, values, A.nonZeros() * sizeof(device_type),
             cudaMemcpyHostToDevice);

  DeviceSpMV<device_type> spmv(d_row_ptr, d_col_ind, d_values, n, A.nonZeros());

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  double max_diff = 0.0;
  double avg_eigen_time = 0.0;
  float avg_gpu_time = 0.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (uint32_t trial = 0; trial < num_trials; trial++) {
    Eigen::Vector<T, Eigen::Dynamic> x(n);
    if constexpr (std::is_same_v<T, double>) {
      x = Eigen::VectorXd::NullaryExpr(
          n, [&gen, &dist](Eigen::Index) { return dist(gen); });
    } else {
      for (uint32_t i = 0; i < n; i++) {
        x[i] = std::complex<double>(dist(gen), dist(gen));
      }
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    Eigen::Vector<T, Eigen::Dynamic> y_eigen = A * x;
    auto cpu_end = std::chrono::high_resolution_clock::now();
    avg_eigen_time += std::chrono::duration_cast<std::chrono::microseconds>(
                          cpu_end - cpu_start)
                          .count();

    cudaMemcpy(d_x, x.data(), n * sizeof(device_type), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    spmv.multiply(d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_gpu_time += milliseconds;

    std::vector<T> y_gpu_vec(n);
    cudaMemcpy(y_gpu_vec.data(), d_y, n * sizeof(device_type),
               cudaMemcpyDeviceToHost);

    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> y_gpu(y_gpu_vec.data(), n);
    Eigen::Vector<T, Eigen::Dynamic> diff = y_eigen - y_gpu;

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
  setbuf(stdout, NULL);
  auto ns = {128, 256, 512, 1024};
  for (auto ni : ns) {
    const uint32_t nx = ni;
    const uint32_t ny = ni;
    uint32_t n = nx * ny;
    double Lx = 5., Ly = 5.;
    double dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);

    std::cout << "\nReal SpMV:\n";
    Eigen::SparseMatrix<double> A_real =
        build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);
    test_spmv(A_real);

    std::cout << "\nUnreal SpMV:\n";
    Eigen::SparseMatrix<std::complex<double>> A_complex =
        build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);
    for (int k = 0; k < A_complex.nonZeros(); k++) {
      A_complex.valuePtr()[k] *= std::complex<double>(1.0, 1.0);
    }
    test_spmv(A_complex);
  }
  return 0;
}
