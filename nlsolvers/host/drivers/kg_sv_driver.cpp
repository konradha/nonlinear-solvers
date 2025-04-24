#include "boundaries.hpp"
#include "eigen_krylov_real.hpp"
#include "kg_sv_solver.hpp"
#include "laplacians.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// We'll assume to be taking first-order approximants for the velocity for now
// for all equations in u_tt

int main(int argc, char **argv) {
  if (argc != 12 && argc != 13) {
    std::cerr << "Usage: " << argv[0]
              << " nx ny Lx Ly input_u0.npy input_v0.npy output_traj.npy "
                 "output_vel.npy T nt "
                 "num_snapshots [input_m.npy]\n";
    std::cerr << "Example: " << argv[0]
              << " 256 256 10.0 10.0 initial.npy velocity.npy evolution_u.npy "
                 "evolution_v.npy "
                 "1.5 500 100\n";
    std::cerr << "Example with m(x,y): " << argv[0]
              << " 256 256 10.0 10.0 initial.npy velocity.npy evolution_u.npy "
                 "evolution_v.npy "
                 "1.5 500 100 coupling.npy\n";
    return 1;
  }

  const uint32_t nx = std::stoul(argv[1]);
  const uint32_t ny = std::stoul(argv[2]);
  const double Lx = std::stod(argv[3]);
  const double Ly = std::stod(argv[4]);
  const std::string input_file = argv[5];
  const std::string input_velocity = argv[6];
  const std::string output_file = argv[7];
  const std::string output_vel = argv[8];
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

  std::vector<uint32_t> input_shape;
  Eigen::VectorXd u0 = read_from_npy<double>(input_file, input_shape);
  Eigen::VectorXd v0 = read_from_npy<double>(input_velocity, input_shape);

  if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
    std::cerr << "Error: Input array dimensions mismatch\n";
    std::cerr << "Expected: " << ny << "x" << nx << "\n";
    std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
    return 1;
  }

  Eigen::VectorXd u_past = u0 - dt * v0;
  Eigen::VectorXd m;

  if (m_file) {
    try {
      std::vector<uint32_t> m_shape;
      m = read_from_npy<double>(m_file.value(), m_shape);

      if (m_shape.size() != 2 || m_shape[0] != ny || m_shape[1] != nx) {
        std::cerr << "Error: Coupling array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << m_shape[0] << "x" << m_shape[1] << "\n";
        std::cerr << "Using default m=1.0 everywhere\n";
        m = Eigen::VectorXd::Ones(nx * ny);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error loading m(x,y): " << e.what() << "\n";
      std::cerr << "Using default m=1.0 everywhere\n";
      throw std::runtime_error("Faulty m");
    }
  }

  const Eigen::SparseMatrix<double> L =
      build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);

  Eigen::VectorXd u_save(num_snapshots * nx * ny);
  Eigen::VectorXd v_save(num_snapshots * nx * ny);

  Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> u_save_mat(
      u_save.data(), num_snapshots, nx * ny);
  Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> v_save_mat(
      v_save.data(), num_snapshots, nx * ny);

  u_save_mat.row(0) = u0.transpose();
  v_save_mat.row(0) = v0.transpose();

  Eigen::VectorXd u = u0;
  Eigen::VectorXd v = v0;
  Eigen::VectorXd buf(nx * ny);

  for (uint32_t i = 1; i < nt; ++i) {
    KGESVSolver::step<double>(u, u_past, buf, L, m, dt);
    neumann_bc_no_velocity<double>(u, nx, ny);
    v = (u - u_past) / dt;
    if (i % freq == 0) {
      uint32_t snapshot_idx = i / freq;
      if (snapshot_idx < num_snapshots) {
        u_save_mat.row(snapshot_idx) = u.transpose();
        v_save_mat.row(snapshot_idx) = v.transpose();
      }
    }
  }
  const std::vector<uint32_t> shape = {num_snapshots, ny, nx};
  save_to_npy(output_file, u_save, shape);
  save_to_npy(output_vel, v_save, shape);
  return 0;
}
