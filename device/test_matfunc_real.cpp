#include "../eigen_krylov_real.hpp"
#include "../laplacians.hpp"
#include "matfunc_real.hpp"

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void test_matfunc(const Eigen::SparseMatrix<double> &A, uint32_t m,
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

  MatrixFunctionApplicatorReal::Parameters params;
  MatrixFunctionApplicatorReal matfunc(d_row_ptr, d_col_ind, d_values, n, m,
                                       A.nonZeros(), params);

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  double *d_input, *d_result;
  cudaMalloc(&d_input, n * sizeof(double));
  cudaMalloc(&d_result, n * sizeof(double));

  std::vector<double> input(n), result(n);
  double dt = 0.1;

  double avg_gpu_time = 0.0;
  double avg_cpu_time = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (uint32_t trial = 0; trial < num_trials; trial++) {
    for (uint32_t i = 0; i < n; i++) {
      input[i] = dist(gen);
    }
    Eigen::Map<Eigen::VectorXd> input_vec(input.data(), n);
    input_vec.normalize();

    cudaMemcpy(d_input, input.data(), n * sizeof(double),
               cudaMemcpyHostToDevice);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd ref_sinc2 = sinc2_sqrt_multiply(A, input_vec, dt, m);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    avg_cpu_time += std::chrono::duration_cast<std::chrono::microseconds>(
                        cpu_end - cpu_start)
                        .count();

    cudaEventRecord(start);
    matfunc.apply(d_result, d_input, dt / 2,
                  MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_gpu_time += milliseconds * 1000.0;

    cudaMemcpy(result.data(), d_result, n * sizeof(double),
               cudaMemcpyDeviceToHost);
    Eigen::Map<Eigen::VectorXd> gpu_sinc2(result.data(), n);
    Eigen::VectorXd diff_sinc2 = ref_sinc2 - gpu_sinc2;

    if (trial == 0) {
      std::cout << "sinc2_sqrt diff: L1 = " << diff_sinc2.lpNorm<1>()
                << ", L2 = " << diff_sinc2.norm() << "\n";
    }
  }

  avg_cpu_time /= num_trials;
  avg_gpu_time /= num_trials;

  std::cout << "Average CPU time: " << avg_cpu_time << " us\n";
  std::cout << "Average GPU time: " << avg_gpu_time << " us\n";
  std::cout << "Speedup: " << avg_cpu_time / avg_gpu_time << "x\n\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);
  cudaFree(d_input);
  cudaFree(d_result);
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

    for (auto m : krylov_dims) {
      test_matfunc(A, m);
    }
  }
  return 0;
}
