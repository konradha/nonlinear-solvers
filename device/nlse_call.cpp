#include "../laplacians.hpp"
#include "../util.hpp"

#include "nlse_solver_dev.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <iomanip>
#include <string>

void print_usage(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " nx ny Lx Ly input_u0.npy output_traj.npy T nt num_snapshots\n";
  std::cerr << "Example: " << program_name
            << " 256 256 10.0 10.0 initial.npy evolution.npy 1.5 500 100\n";
}

int main(int argc, char **argv) {
  if (argc != 10) {
    print_usage(argv[0]);
    return 1;
  }
  const uint32_t nx = std::stoul(argv[1]);
  const uint32_t ny = std::stoul(argv[2]);
  const double Lx = std::stod(argv[3]);
  const double Ly = std::stod(argv[4]);
  const std::string input_file = argv[5];
  const std::string output_file = argv[6];
  const double T = std::stod(argv[7]);
  const uint32_t nt = std::stoul(argv[8]);
  const uint32_t num_snapshots = std::stoul(argv[9]);

  const double dx = 2 * Lx / (nx - 1);
  const double dy = 2 * Ly / (ny - 1);
  const double dt = T / nt;
  const auto freq = nt / num_snapshots;
  const std::complex<double> dti(0, dt);

  std::vector<uint32_t> input_shape;
  Eigen::VectorXcd u0 =
      read_from_npy<std::complex<double>>(input_file, input_shape);

  if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
    std::cerr << "Error: Input array dimensions mismatch\n";
    std::cerr << "Expected: " << ny << "x" << nx << "\n";
    std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
    return 1;
  }

  const Eigen::SparseMatrix<std::complex<double>> L =
      build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);

  Eigen::VectorXcd u_save(num_snapshots * nx * ny);

  auto start = std::chrono::high_resolution_clock::now();

  {
    NLSESolverDevice::Parameters params(num_snapshots, freq);
    NLSESolverDevice solver(L, u0.data(), params);

    for (uint32_t i = 1; i < nt; ++i) {
      solver.step(dti, i);
    }

    solver.transfer_snapshots(u_save.data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto compute_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  const std::vector<uint32_t> shape = {num_snapshots, ny, nx};
  save_to_npy(output_file, u_save, shape);

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "Trajectory took: " << compute_time / 1.e6 << "s\n";

  return 0;
}
