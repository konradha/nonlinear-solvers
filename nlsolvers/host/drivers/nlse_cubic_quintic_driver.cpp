#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"
#include "nlse_cubic_quintic_solver.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 12 && argc != 13) {
    std::cerr << "Usage: " << argv[0]
              << " nx ny Lx Ly s1 s2 input_u0.npy output_traj.npy T nt "
                 "num_snapshots [input_m.npy]\n";
    std::cerr << "Example: " << argv[0]
              << " 256 256 10.0 10.0 0.6 -0.6 initial.npy evolution.npy 1.5 "
                 "500 100\n";
    std::cerr << "Example with m(x,y): " << argv[0]
              << " 256 256 10.0 10.0 0.6 -0.6 initial.npy evolution.npy 1.5 "
                 "500 100 focusing.npy\n";
    return 1;
  }

  const uint32_t nx = std::stoul(argv[1]);
  const uint32_t ny = std::stoul(argv[2]);
  const double Lx = std::stod(argv[3]);
  const double Ly = std::stod(argv[4]);
  const double s1 = std::stod(argv[5]);
  const double s2 = std::stod(argv[6]);
  const std::string input_file = argv[7];
  const std::string output_file = argv[8];
  const double T = std::stod(argv[9]);
  const uint32_t nt = std::stoul(argv[10]);
  const uint32_t num_snapshots = std::stoul(argv[11]);

  std::optional<std::string> m_file;
  if (argc == 13) {
    m_file = argv[12];
  }

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

  Eigen::VectorXcd m = Eigen::VectorXcd::Zero(nx * ny);

  if (m_file) {
    try {
      std::vector<uint32_t> m_shape;
      m = read_from_npy<std::complex<double>>(m_file.value(), m_shape);
      if (m_shape.size() != 2 || m_shape[0] != ny || m_shape[1] != nx) {
        std::cerr << "Error: Focusing array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << m_shape[0] << "x" << m_shape[1] << "\n";
        std::cerr << "Using default m=1.0 everywhere\n";
      }
    } catch (const std::exception &e) {
      std::cerr << "Error loading m(x,y): " << e.what() << "\n";
      throw std::runtime_error("Aborting");
    }
  }

  const Eigen::SparseMatrix<std::complex<double>> L =
      build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);

  Eigen::VectorXcd u_save(num_snapshots * nx * ny);
  Eigen::Map<Eigen::Matrix<std::complex<double>, -1, -1, Eigen::RowMajor>>
      u_save_mat(u_save.data(), num_snapshots, nx * ny);

  u_save_mat.row(0) = u0.transpose();

  auto start = std::chrono::high_resolution_clock::now();

  Eigen::VectorXcd u = u0;
  Eigen::VectorXcd buf = u0;
  Eigen::VectorXcd rho_buf(nx * ny);

  for (uint32_t i = 1; i < nt; ++i) {
    NLSEQuinticSolver::step<std::complex<double>>(buf, rho_buf, u, L, m, dti,
                                                  s1, s2);
    if (i % freq == 0) {
      uint32_t snapshot_idx = i / freq;
      if (snapshot_idx < num_snapshots) {
        u_save_mat.row(snapshot_idx) = u.transpose();
      }
    }
    // PROGRESS_BAR(i, nt);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto compute_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  const std::vector<uint32_t> shape = {num_snapshots, ny, nx};
  save_to_npy(output_file, u_save, shape);

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "walltime: " << compute_time / 1.e6 << " seconds\n";
  return 0;
}
