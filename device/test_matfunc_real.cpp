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
  double dt = 1e-2;

  double avg_gpu_time = 0.0;
  double avg_cpu_time = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto fun_types = {MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT,
                    MatrixFunctionApplicatorReal::FunctionType::COS_SQRT,
                    MatrixFunctionApplicatorReal::FunctionType::ID_SQRT};
  for (const auto &fun : fun_types) {
    for (uint32_t trial = 0; trial < num_trials; trial++) {
      for (uint32_t i = 0; i < n; i++) {
        input[i] = dist(gen);
      }
      Eigen::Map<Eigen::VectorXd> input_vec(input.data(), n);
      input_vec.normalize();

      cudaMemcpy(d_input, input.data(), n * sizeof(double),
                 cudaMemcpyHostToDevice);

      Eigen::VectorXd ref_fun(n);
      auto cpu_start = std::chrono::high_resolution_clock::now();
      switch (fun) {
      case MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT:
        ref_fun = sinc2_sqrt_half<double>(A, input_vec, dt, m);
        break;
      case MatrixFunctionApplicatorReal::FunctionType::COS_SQRT:
        ref_fun = cos_sqrt_multiply<double>(A, input_vec, dt, m);
        break;
      case MatrixFunctionApplicatorReal::FunctionType::ID_SQRT:
        ref_fun = id_sqrt_multiply<double>(A, input_vec, dt, m);
        break;
      }
      auto cpu_end = std::chrono::high_resolution_clock::now();
      avg_cpu_time += std::chrono::duration_cast<std::chrono::microseconds>(
                          cpu_end - cpu_start)
                          .count();

      cudaEventRecord(start);
      if (fun == MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT)
        matfunc.apply(d_result, d_input, dt / 2, fun);
      else
        matfunc.apply(d_result, d_input, dt, fun);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      avg_gpu_time += milliseconds * 1000.0;

      cudaMemcpy(result.data(), d_result, n * sizeof(double),
                 cudaMemcpyDeviceToHost);
      Eigen::Map<Eigen::VectorXd> gpu_result(result.data(), n);
      Eigen::VectorXd diff = ref_fun - gpu_result;

      /*if (trial == 0)*/ {
        matfunc.print_type(fun);
        std::cout << "L1 = " << diff.lpNorm<1>() << ", L2 = " << diff.norm()
                  << "\n";
      }
    }
  }

  avg_cpu_time /= (num_trials * 3);
  avg_gpu_time /= (num_trials * 3);

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
  
  auto ns = {50, 1000};
  std::vector<uint32_t> krylov_dims = {5, 10, 20};

  for (auto ni : ns) {
    const uint32_t nx = ni;
    const uint32_t ny = ni;
    uint32_t n = nx * ny;
    double Lx = 5., Ly = 5.;
    double dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
    Eigen::SparseMatrix<double> A =
        build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);

    for (auto m : krylov_dims) {
      test_matfunc(-A, m);
    }
  }
  return 0;
}
