#include "../eigen_krylov_complex.hpp"
#include "../laplacians.hpp"
#include "matfunc_complex.hpp"

#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void test_matfunc_complex(const Eigen::SparseMatrix<std::complex<double>> &A,
                          uint32_t m, uint32_t num_trials = 10) {
  const uint32_t n = A.rows();
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "n = " << n << ", m = " << m << "\n";



  MatrixFunctionApplicatorComplex::Parameters params;
  MatrixFunctionApplicatorComplex matfunc(A, n, m,
                                          A.nonZeros(), params);

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);

  thrust::complex<double> *d_input, *d_result;
  cudaMalloc(&d_input, n * sizeof(thrust::complex<double>));
  cudaMalloc(&d_result, n * sizeof(thrust::complex<double>));
  cudaMemset(d_input, 0, n * sizeof(thrust::complex<double>));
  
  std::vector<std::complex<double>> input(n), result(n);
  double dt = 1e-2;

  double avg_gpu_time = 0.0;
  double avg_cpu_time = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (uint32_t trial = 0; trial < num_trials; trial++) {
    // Generate random complex input
    for (uint32_t i = 0; i < n; i++) {
      input[i] = std::complex<double>(dist(gen), dist(gen));
    }
    Eigen::Map<Eigen::VectorX<std::complex<double>>> input_vec(input.data(), n);

    
    std::complex<double> tau(0, dt);
    Eigen::VectorXcd ref_fun;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    ref_fun = expm_multiply<std::complex<double>>(A, input_vec, tau, m);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    avg_cpu_time += std::chrono::duration_cast<std::chrono::microseconds>(
                        cpu_end - cpu_start)
                        .count();

    cudaMemcpy(d_input, input_vec.data(), n * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    
    matfunc.apply(d_result, d_input, tau);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avg_gpu_time += milliseconds * 1000.0;

    cudaMemcpy(result.data(), d_result, n * sizeof(std::complex<double>),
               cudaMemcpyDeviceToHost);
    Eigen::Map<Eigen::VectorX<std::complex<double>>> gpu_result(result.data(), n);
    Eigen::VectorXcd diff = ref_fun - gpu_result;


    std::cout << "L1 = " << diff.lpNorm<1>() << ", L2 = " << diff.norm()
              << "\n";

    if (diff.lpNorm<1>() > 1e-1){
      std::cout << "Correct values\n";
      for(uint32_t i=0;i<5;++i)
        std::cout << ref_fun(i) << " ";
      std::cout << "\n";
      std::cout << "Device values\n";
      for(uint32_t i=0;i<5;++i)
        std::cout << gpu_result(i) << " ";
      std::cout << "\n";
    }
  }

  avg_cpu_time /= num_trials;
  avg_gpu_time /= num_trials;

  std::cout << "Average CPU time: " << avg_cpu_time << " us\n";
  std::cout << "Average GPU time: " << avg_gpu_time << " us\n";
  std::cout << "Speedup: " << avg_cpu_time / avg_gpu_time << "x\n\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_input);
  cudaFree(d_result);
}

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  // debug sizes
  auto ns = {5};
  std::vector<uint32_t> krylov_dims = {2};

  // benchmark sizes
  // auto ns = {50, 132, 500, 877};
  // std::vector<uint32_t> krylov_dims = {5, 10, 20};

  for (auto ni : ns) {
    const uint32_t nx = ni;
    const uint32_t ny = ni;
    uint32_t n = nx * ny;
    double Lx = 5., Ly = 5.;
    double dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);

    Eigen::SparseMatrix<std::complex<double>> A =
        build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);

    for (auto m : krylov_dims) {
      test_matfunc_complex(-A, m);
    }
  }
  return 0;
}
